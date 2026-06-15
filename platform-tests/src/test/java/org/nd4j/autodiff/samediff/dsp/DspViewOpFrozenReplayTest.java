/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.autodiff.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Regression + coverage test for the DSP view-op wrapper-swap lifecycle guard.
 *
 * <p>Reproduces and validates the fix for the BGE-base encoder crash:
 * <pre>
 *   LIFECYCLE VIOLATION: NDArray pointer replacement at slot 2
 *   (tag=view-op-install) during frozen phase (phase=1 execCount=1).
 *   old=... new=... oldDb=0x... newDb=0x...   (oldDb == newDb)
 * </pre>
 * A reshape op legitimately produces a fresh {@code NDArray} wrapper over the
 * same underlying {@code DataBuffer} on the second execution, but the old guard
 * in {@code NativeDynamicShapePlan::writeOutputSlot()} rejected the pointer
 * swap because the plan was already in {@code SHAPES_FROZEN}. The fix relaxes
 * the guard: when {@code old->dataBuffer() == value->dataBuffer()} AND offset,
 * shape, stride, and dtype all match, the swap is a pure wrapper replacement
 * and is allowed (with a {@code WRITE_SLOT_WRAPPER_SWAP} diagnostic).
 *
 * <p>To prove the fix is actually exercised, each test runs the graph under
 * SLOT_BY_SLOT to capture a reference, then under the target execution mode
 * for multiple replays (5) so that the second and subsequent calls land on
 * the frozen-phase path in {@code writeOutputSlot()}. A {@code
 * WRITE_SLOT_WRAPPER_SWAP} diagnostic line in the test log confirms the
 * wrapper-swap branch fired.
 *
 * <p>Coverage is parameterized over <b>view-op fixture × execution mode</b>.
 * Fixtures intentionally include the BGE-specific pattern of a reshape whose
 * input is a plan-owned intermediate (not a placeholder) so the upstream
 * buffer pointer is stable across runs and the reshape really does mint a
 * fresh wrapper over the exact same buffer.
 *
 * <p><b>Run everything (default):</b>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=DspViewOpFrozenReplayTest \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
 *       2&gt;&amp;1 | tee /tmp/view-op-coverage-test.log
 * </pre>
 *
 * <p><b>Run a single combination:</b>
 * <pre>
 *   mvn test -Dtest='DspViewOpFrozenReplayTest#testViewOpFrozenReplay[reshapeOfMmul-CUDA_GRAPHS]'
 * </pre>
 */
@Slf4j
@Tag("dsp")
@DisplayName("DSP view-op frozen-phase replay: wrapper-swap must not trip LIFECYCLE VIOLATION "
        + "across reshape/permute/slice/squeeze/expandDims under all CUDA execution modes")
public class DspViewOpFrozenReplayTest {

    // ─── Tolerances ─────────────────────────────────────────────────────────
    private static final double FP32_RTOL = 1e-4;
    private static final double FP32_ATOL = 1e-5;
    private static final double LOOSE_RTOL = 1e-2;
    private static final double LOOSE_ATOL = 1e-3;

    // ─── Replay count ──────────────────────────────────────────────────────
    // The bug fires on the SECOND call (first = freeze, second = replay).
    // Five replays give enough margin that the AUTO_SEAL transition has
    // definitely happened and every subsequent call is on the frozen-phase
    // writeOutputSlot() path.
    private static final int REPLAYS = 5;

    // ─── Current SameDiff under test (cleaned up in @AfterEach) ─────────────
    private SameDiff sd;

    // ──────────────────────────────────────────────────────────────────────
    // Fixtures: one record per view-op graph we want to validate
    // ──────────────────────────────────────────────────────────────────────

    /**
     * A named, buildable view-op graph fixture plus its canned input map and
     * output name. The input map is regenerated on demand so each ref/test
     * call gets its own copies and can safely close them afterwards.
     */
    private static final class ViewOpFixture {
        final String name;
        final Supplier<SameDiff> graphBuilder;
        final Supplier<Map<String, INDArray>> inputBuilder;
        final String outputName;

        ViewOpFixture(String name, Supplier<SameDiff> graphBuilder,
                      Supplier<Map<String, INDArray>> inputBuilder, String outputName) {
            this.name = name;
            this.graphBuilder = graphBuilder;
            this.inputBuilder = inputBuilder;
            this.outputName = outputName;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // JUnit lifecycle
    // ──────────────────────────────────────────────────────────────────────

    @BeforeEach
    public void setUp() {
        // DSP must be enabled at the system level for native plan compilation.
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    public void tearDown() {
        if (sd != null) {
            try {
                sd.close();
            } catch (Throwable t) {
                log.warn("sd.close() failed in tearDown", t);
            }
            sd = null;
        }
        Nd4j.getExecutioner().commit();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Parameter providers
    // ──────────────────────────────────────────────────────────────────────

    /**
     * All CUDA-testable execution modes. Platform-locked backends (MLX, NNAPI,
     * ARM_HYBRID, HIP_GRAPHS, LEVEL_ZERO, VULKAN, METAL, TPU, HEXAGON,
     * OPENVINO, etc.) are intentionally excluded — the BGE crash is
     * CUDA-specific and those backends do not run DSP frozen replay on the
     * CI machine. TRITON is conditionally available at runtime.
     */
    static List<GraphExecutionMode> executionModes() {
        List<GraphExecutionMode> modes = new ArrayList<>();
        modes.add(GraphExecutionMode.AUTO);
        modes.add(GraphExecutionMode.SLOT_BY_SLOT);
        modes.add(GraphExecutionMode.CUDA_GRAPHS);
        modes.add(GraphExecutionMode.NVRTC_JIT);
        modes.add(GraphExecutionMode.PTX_JIT);
        if (isTritonAvailable()) {
            modes.add(GraphExecutionMode.TRITON);
        }
        modes.add(GraphExecutionMode.EMULATED_REPLAY);
        return modes;
    }

    private static boolean isTritonAvailable() {
        try {
            return Nd4j.getNativeOps().isTritonAvailable();
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * The full list of view-op fixtures. Ordered so the BGE-specific cases
     * (reshape-of-mmul, double-reshape, reshape-in-larger-graph,
     * reshape-at-slot2) come first, followed by the original four unit-style
     * graphs kept for regression coverage.
     */
    private static List<ViewOpFixture> viewOpFixtures() {
        List<ViewOpFixture> fixtures = new ArrayList<>();

        // ─── BGE-shaped coverage — reshape/permute consume a plan-owned
        //    intermediate, which is the exact pattern that triggered the
        //    real user crash.
        fixtures.add(new ViewOpFixture(
                "reshapeOfMmul",
                DspViewOpFrozenReplayTest::buildReshapeOfMmulOutput,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -0.5, 0.01, 4 * 16).reshape(4, 16)),
                "out"));

        fixtures.add(new ViewOpFixture(
                "permuteOfMmul",
                DspViewOpFrozenReplayTest::buildPermuteOfMmulOutput,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -0.4, 0.01, 2 * 3 * 4).reshape(2, 3, 4)),
                "out"));

        fixtures.add(new ViewOpFixture(
                "doubleReshape",
                DspViewOpFrozenReplayTest::buildDoubleReshape,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -0.3, 0.01, 4 * 16).reshape(4, 16)),
                "out"));

        fixtures.add(new ViewOpFixture(
                "reshapeInsideLargerGraph",
                DspViewOpFrozenReplayTest::buildReshapeInsideLargerGraph,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -0.25, 0.01, 4 * 16).reshape(4, 16)),
                "out"));

        fixtures.add(new ViewOpFixture(
                "reshapeAtExactSlot2",
                DspViewOpFrozenReplayTest::buildReshapeAtExactSlot2,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -0.2, 0.01, 4 * 16).reshape(4, 16)),
                "out"));

        // ─── Original four fixtures kept for regression coverage. These
        //    exercise the shape-only view ops on simpler graphs where the
        //    view op may consume a placeholder directly.
        fixtures.add(new ViewOpFixture(
                "reshapePlaceholder",
                DspViewOpFrozenReplayTest::buildReshapePlaceholderGraph,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -0.5, 0.01, 4 * 16).reshape(4, 16)),
                "out"));

        fixtures.add(new ViewOpFixture(
                "permutePlaceholder",
                DspViewOpFrozenReplayTest::buildPermutePlaceholderGraph,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -1.0, 0.05, 2 * 3 * 4).reshape(2, 3, 4)),
                "out"));

        fixtures.add(new ViewOpFixture(
                "slice",
                DspViewOpFrozenReplayTest::buildSliceGraph,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -0.4, 0.01, 8 * 8).reshape(8, 8)),
                "out"));

        fixtures.add(new ViewOpFixture(
                "squeezeExpand",
                DspViewOpFrozenReplayTest::buildSqueezeExpandGraph,
                () -> singletonInput("x",
                        Nd4j.linspace(DataType.FLOAT, -0.3, 0.005, 4 * 16).reshape(1, 4, 1, 16)),
                "out"));

        return fixtures;
    }

    /** JUnit {@code @MethodSource} for the cross product. */
    static Stream<Arguments> viewOpAndMode() {
        List<ViewOpFixture> fixtures = viewOpFixtures();
        List<GraphExecutionMode> modes = executionModes();
        List<Arguments> args = new ArrayList<>(fixtures.size() * modes.size());
        for (ViewOpFixture f : fixtures) {
            for (GraphExecutionMode m : modes) {
                args.add(Arguments.of(f, m));
            }
        }
        return args.stream();
    }

    // ──────────────────────────────────────────────────────────────────────
    // The parameterized entry point
    // ──────────────────────────────────────────────────────────────────────

    /**
     * For every (fixture, mode) combination:
     *
     * <ol>
     *   <li>Build the graph and run it once under SLOT_BY_SLOT to capture a
     *       reference output. Close that SameDiff.</li>
     *   <li>Rebuild the same graph fresh and configure it for the target
     *       execution mode.</li>
     *   <li>Run the graph {@link #REPLAYS} times. The first run freezes the
     *       plan; every subsequent run lands on the frozen-phase path in
     *       {@code writeOutputSlot()} and must not throw
     *       {@code LIFECYCLE VIOLATION}. Each replay is compared element-wise
     *       against the reference.</li>
     * </ol>
     */
    @ParameterizedTest(name = "{0}-{1}")
    @MethodSource("viewOpAndMode")
    public void testViewOpFrozenReplay(ViewOpFixture fixture, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        // 1. Reference pass (SLOT_BY_SLOT)
        INDArray reference = captureReference(fixture);

        try {
            // 2. Build a fresh SameDiff for the test mode.
            sd = fixture.graphBuilder.get();
            configureDsp(sd, mode);

            // 3. Five replays — a LIFECYCLE VIOLATION fires on the second+
            // run in the broken code, so REPLAYS>=2 is enough to reproduce.
            // REPLAYS=5 gives margin for AUTO_SEAL cycles and ensures the
            // frozen-phase writeOutputSlot() path is exercised many times.
            double[] tol = tolerances(mode);
            for (int i = 0; i < REPLAYS; i++) {
                final int iter = i;
                Map<String, INDArray> inputs = fixture.inputBuilder.get();
                INDArray actual;
                try {
                    actual = sd.output(inputs, fixture.outputName).get(fixture.outputName);
                } catch (Throwable t) {
                    fail(fixture.name + " / " + mode
                            + ": replay #" + iter
                            + " threw — almost certainly a LIFECYCLE VIOLATION. "
                            + "Root cause: " + t.getMessage(), t);
                    return; // unreachable — fail() throws
                }
                assertNotNull(actual,
                        fixture.name + " / " + mode + ": replay #" + iter + " produced null output");
                assertTrue(Arrays.equals(reference.shape(), actual.shape()),
                        fixture.name + " / " + mode
                                + ": replay #" + iter + " shape mismatch: ref="
                                + Arrays.toString(reference.shape())
                                + " actual=" + Arrays.toString(actual.shape()));
                compareArrays(reference, actual, tol[0], tol[1],
                        fixture.name + " / " + mode + " replay #" + iter);
                // Close per-run input copies.
                closeAll(inputs);
            }
        } finally {
            if (reference != null && reference.closeable() && !reference.wasClosed()) {
                reference.close();
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Graph fixtures — BGE-shaped reshape/permute of an intermediate
    // ──────────────────────────────────────────────────────────────────────

    /**
     * <b>BGE pattern #1 — {@code reshapeOfMmul}.</b>
     * Matches the real user crash: a reshape whose INPUT is a plan-owned
     * intermediate (the output of an {@code mmul}), not a placeholder. On
     * the second run the upstream buffer pointer is stable, so the reshape
     * really does mint a fresh {@code NDArray} wrapper over the exact same
     * {@code DataBuffer}.
     *
     * <pre>
     *   x [4,16] → mmul(w[16,16]) [4,16] → relu → RESHAPE[2,32] → add 0.25 → out
     * </pre>
     */
    private static SameDiff buildReshapeOfMmulOutput() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w",
                Nd4j.linspace(DataType.FLOAT, 0.01, 0.005, 16 * 16).reshape(16, 16));
        SDVariable h = g.mmul("h", x, w);            // intermediate [4,16]
        SDVariable a = g.nn.relu("a", h, 0);         // intermediate [4,16]
        SDVariable r = g.reshape("r", a, 2, 32);     // VIEW over a's buffer
        SDVariable out = g.math.add("out", r, 0.25); // [2,32]
        g.setOutputs("out");
        return g;
    }

    /**
     * <b>BGE pattern #2 — {@code permuteOfMmul}.</b>
     * Permute consumes an intermediate produced by a prior compute op, so
     * the wrapper-swap path fires for permutes as well. Also exercises the
     * {@code buildPermutedViewShapeInfo} path.
     *
     * <pre>
     *   x [2,3,4] → add(bias) [2,3,4] → relu → PERMUTE[0,2,1][2,4,3] → mul 2.0 → out
     * </pre>
     */
    private static SameDiff buildPermuteOfMmulOutput() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable bias = g.var("bias", Nd4j.linspace(DataType.FLOAT, 0.0, 0.1, 4));
        SDVariable shifted = x.add("shifted", bias);      // intermediate [2,3,4]
        SDVariable a = g.nn.relu("a", shifted, 0);        // intermediate [2,3,4]
        SDVariable p = g.permute("p", a, 0, 2, 1);        // VIEW over a's buffer, [2,4,3]
        SDVariable out = g.math.mul("out", p, 2.0);       // [2,4,3]
        g.setOutputs("out");
        return g;
    }

    /**
     * <b>BGE pattern #3 — {@code doubleReshape}.</b>
     * Two reshapes back-to-back. Both are view ops whose output triggers
     * {@code writeOutputSlot()} with the wrapper-swap condition, from two
     * different op instances in the same replay. Catches bugs where the
     * relaxed guard only checks the first view-op output in the plan.
     *
     * <pre>
     *   x [4,16] → mmul(w1) [4,16] → RESHAPE[2,32] → RESHAPE[8,8] → add → out
     * </pre>
     */
    private static SameDiff buildDoubleReshape() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w",
                Nd4j.linspace(DataType.FLOAT, 0.02, 0.004, 16 * 16).reshape(16, 16));
        SDVariable h = g.mmul("h", x, w);              // intermediate [4,16]
        SDVariable r1 = g.reshape("r1", h, 2, 32);     // VIEW over h's buffer
        SDVariable r2 = g.reshape("r2", r1, 8, 8);     // VIEW over r1's buffer
        SDVariable out = g.math.add("out", r2, 0.1);   // [8,8]
        g.setOutputs("out");
        return g;
    }

    /**
     * <b>BGE pattern #4 — {@code reshapeInsideLargerGraph}.</b>
     * A reshape whose OUTPUT feeds a downstream matmul, simulating a
     * Transformer feed-forward block where the reshape sits in the middle of
     * a longer dataflow. The downstream matmul reads the reshape's wrapper,
     * so any wrapper-swap bug propagates into a mismatched output. Runs 5
     * replays (via the outer parameterized loop) to verify the frozen-phase
     * path stays correct over multiple iterations.
     *
     * <pre>
     *   x [4,16] → mmul(w1) [4,16] → RESHAPE[2,32] → mmul(w2) [2,32] → relu → out
     * </pre>
     */
    private static SameDiff buildReshapeInsideLargerGraph() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w1 = g.var("w1",
                Nd4j.linspace(DataType.FLOAT, 0.01, 0.003, 16 * 16).reshape(16, 16));
        SDVariable w2 = g.var("w2",
                Nd4j.linspace(DataType.FLOAT, 0.005, 0.002, 32 * 32).reshape(32, 32));
        SDVariable h = g.mmul("h", x, w1);             // intermediate [4,16]
        SDVariable r = g.reshape("r", h, 2, 32);       // VIEW [2,32]
        SDVariable h2 = g.mmul("h2", r, w2);           // downstream consumer [2,32]
        SDVariable out = g.nn.relu("out", h2, 0);      // [2,32]
        g.setOutputs("out");
        return g;
    }

    /**
     * <b>BGE pattern #5 — {@code reshapeAtExactSlot2}.</b>
     * The user's crash report pinpointed {@code slot 2} specifically. This
     * fixture builds a minimal 3-op graph (placeholder + one compute op + one
     * reshape + one trailing op) so that the reshape's output lands at or
     * near slot index 2, mirroring the exact layout from the BGE stack trace.
     *
     * <pre>
     *   x [4,16] → add 1.0 [4,16] → RESHAPE[2,32] → mul 2.0 → out
     * </pre>
     *
     * The add produces slot 0 or 1 (depending on constants), the reshape
     * produces slot ~2, and the mul produces the trailing output.
     */
    private static SameDiff buildReshapeAtExactSlot2() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable shifted = g.math.add("shifted", x, 1.0);  // slot ~1
        SDVariable r = g.reshape("r", shifted, 2, 32);       // slot ~2 (VIEW)
        SDVariable out = g.math.mul("out", r, 2.0);          // slot ~3
        g.setOutputs("out");
        return g;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Graph fixtures — the original 4 placeholder-based view-op graphs
    // (kept for regression coverage of the shape-only view ops)
    // ──────────────────────────────────────────────────────────────────────

    private static SameDiff buildReshapePlaceholderGraph() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w",
                Nd4j.linspace(DataType.FLOAT, 0.01, 0.005, 16 * 16).reshape(16, 16));
        SDVariable h = g.mmul("h", x, w);
        SDVariable a = g.nn.relu("a", h, 0);
        SDVariable r = g.reshape("r", a, 2, 32);
        SDVariable out = g.math.add("out", r, 0.25);
        g.setOutputs("out");
        return g;
    }

    private static SameDiff buildPermutePlaceholderGraph() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable bias = g.var("bias", Nd4j.linspace(DataType.FLOAT, 0.0, 0.1, 4));
        SDVariable shifted = x.add("shifted", bias);
        SDVariable p = g.permute("p", shifted, 0, 2, 1);
        SDVariable out = g.math.mul("out", p, 2.0);
        g.setOutputs("out");
        return g;
    }

    private static SameDiff buildSliceGraph() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 8, 8);
        SDVariable w = g.var("w",
                Nd4j.linspace(DataType.FLOAT, 0.0, 0.01, 8 * 8).reshape(8, 8));
        SDVariable h = g.mmul("h", x, w);
        SDVariable a = g.nn.relu("a", h, 0);
        SDVariable s = g.slice("s", a, new int[]{0, 0}, new int[]{4, 8});
        SDVariable out = g.math.add("out", s, 0.5);
        g.setOutputs("out");
        return g;
    }

    private static SameDiff buildSqueezeExpandGraph() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 4, 1, 16);
        SDVariable sq = g.squeeze("sq", x, 2);
        SDVariable sq2 = g.squeeze("sq2", sq, 0);
        SDVariable w = g.var("w",
                Nd4j.linspace(DataType.FLOAT, 0.0, 0.01, 16 * 8).reshape(16, 8));
        SDVariable h = g.mmul("h", sq2, w);
        SDVariable ex = g.expandDims("ex", h, 0);
        SDVariable out = g.math.add("out", ex, 0.1);
        g.setOutputs("out");
        return g;
    }

    // ──────────────────────────────────────────────────────────────────────
    // DSP/mode configuration helpers
    // ──────────────────────────────────────────────────────────────────────

    private static void configureDsp(SameDiff target, GraphExecutionMode mode) {
        if (mode == GraphExecutionMode.SLOT_BY_SLOT) {
            target.setDspAutoCompileEnabled(false);
            target.setDspNativeAutoCompileEnabled(false);
        } else {
            target.setDspAutoCompileEnabled(true);
            target.setDspNativeAutoCompileEnabled(true);
        }
        target.setGraphExecutionMode(mode);
    }

    private static void disableDsp(SameDiff target) {
        target.setDspAutoCompileEnabled(false);
        target.setDspNativeAutoCompileEnabled(false);
        target.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
    }

    /** Skip configs whose backend isn't available at runtime. */
    private static void assumeBackendAvailable(GraphExecutionMode mode) {
        if (mode == GraphExecutionMode.TRITON) {
            assumeTrue(isTritonAvailable(),
                    "Triton unavailable on this build — skipping TRITON config");
        }
    }

    /**
     * Tolerance selection mirrors {@code DspLifecycleValidationTest}: stricter
     * FP32 tolerances for CPU-compatible pure-FP32 modes, looser for modes
     * that may internally use TF32 or FP16.
     */
    private static double[] tolerances(GraphExecutionMode mode) {
        // CUDA_GRAPHS, NVRTC_JIT, PTX_JIT, TRITON may use TF32 via cuBLAS —
        // be permissive. SLOT_BY_SLOT, AUTO, and EMULATED_REPLAY mirror the
        // reference path so FP32 tolerances apply.
        switch (mode) {
            case CUDA_GRAPHS:
            case NVRTC_JIT:
            case PTX_JIT:
            case TRITON:
                return new double[]{LOOSE_RTOL, LOOSE_ATOL};
            default:
                return new double[]{FP32_RTOL, FP32_ATOL};
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Reference capture + comparison helpers
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Build the graph, run it once under SLOT_BY_SLOT, dup the output, and
     * close the SameDiff. The dup() is critical — the session will free the
     * original output buffer when it closes.
     */
    private INDArray captureReference(ViewOpFixture fixture) {
        SameDiff ref = fixture.graphBuilder.get();
        try {
            disableDsp(ref);
            Map<String, INDArray> inputs = fixture.inputBuilder.get();
            try {
                Map<String, INDArray> out = ref.output(inputs, fixture.outputName);
                INDArray arr = out.get(fixture.outputName);
                assertNotNull(arr,
                        fixture.name + ": reference output '" + fixture.outputName + "' must be non-null");
                return arr.dup();
            } finally {
                closeAll(inputs);
            }
        } finally {
            try {
                ref.close();
            } catch (Throwable ignored) {
            }
        }
    }

    /**
     * Element-wise compare using the {@code DspLifecycleValidationTest}
     * pattern: cast both sides to DOUBLE to avoid FP32 compare-eps noise,
     * track the first divergent index, and fail with worst-abs/worst-rel
     * and the first-bad index if any element exceeds both absolute and
     * relative tolerance.
     */
    private static void compareArrays(INDArray ref, INDArray actual,
                                      double rtol, double atol, String label) {
        assertTrue(Arrays.equals(ref.shape(), actual.shape()),
                label + ": shape mismatch ref=" + Arrays.toString(ref.shape())
                        + " actual=" + Arrays.toString(actual.shape()));
        INDArray refD = ref.castTo(DataType.DOUBLE);
        INDArray actD = actual.castTo(DataType.DOUBLE);
        long n = refD.length();
        int firstBad = -1;
        double worstAbs = 0;
        double worstRel = 0;
        for (long i = 0; i < n; i++) {
            double rv = refD.getDouble(i);
            double tv = actD.getDouble(i);
            double absDiff = Math.abs(rv - tv);
            double relDiff = absDiff / (Math.abs(rv) + 1e-12);
            if (absDiff > atol && relDiff > rtol) {
                if (firstBad < 0) firstBad = (int) i;
            }
            if (absDiff > worstAbs) worstAbs = absDiff;
            if (relDiff > worstRel) worstRel = relDiff;
        }
        if (firstBad >= 0) {
            fail(label + ": diverges at index " + firstBad
                    + " ref=" + refD.getDouble(firstBad)
                    + " test=" + actD.getDouble(firstBad)
                    + " (worstAbs=" + worstAbs + " worstRel=" + worstRel
                    + " atol=" + atol + " rtol=" + rtol + ")");
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Small utilities
    // ──────────────────────────────────────────────────────────────────────

    private static Map<String, INDArray> singletonInput(String name, INDArray value) {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put(name, value);
        return m;
    }

    private static void closeAll(Map<String, INDArray> arrays) {
        if (arrays == null) return;
        for (INDArray arr : arrays.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) {
                arr.close();
            }
        }
    }
}
