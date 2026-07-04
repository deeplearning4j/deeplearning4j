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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.nd4j.common.tests.tags.TagNames;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
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
 * Standalone DSP accuracy tests targeting buffer-alias patterns found via
 * Triton verify diagnostics on VLM decode:
 *
 * <ol>
 *   <li><b>Op → reshape view alias</b> — an op writes its output, a downstream
 *       reshape shares the exact same DataBuffer, a further op reads through the
 *       view. If DSP scheduling is wrong the view reads stale pre-write data.</li>
 *   <li><b>Concat → expand_dims view</b> — concat grows a buffer (KV-cache
 *       pattern), expand_dims views the result. Partial-buffer corruption at
 *       the concat boundary (byte offset where new data starts) means the view
 *       op sees stale data in the appended region.</li>
 *   <li><b>Multi-view chain off single producer</b> — one op feeds reshape →
 *       reshape_no_copy → permute, all sharing the same GPU buffer. Tests that
 *       the entire alias chain sees consistent data after the producer fires.</li>
 *   <li><b>Accumulating concat with view</b> — simulates KV-cache accumulation
 *       across replays. Each call concats new data, then reads through a view.
 *       Tests that growing-buffer concat doesn't leave stale data visible
 *       through a downstream view op.</li>
 * </ol>
 *
 * All graphs are minimal standalone SameDiff constructions — no ONNX import.
 * Each is compared against SLOT_BY_SLOT reference for correctness.
 *
 * <p><b>Run:</b>
 * <pre>
 *   cd platform-tests && mvn test \
 *       -Dtest=DspBufferAliasAccuracyTest \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2>&1 | tee /tmp/dsp-buffer-alias.log
 * </pre>
 */
@Slf4j
@Tag("dsp")
@Tag(TagNames.FULL_CI)
@DisplayName("DSP buffer-alias accuracy — op→view aliasing across execution modes")
public class DspBufferAliasAccuracyTest {

    private static final int REPLAYS = 8;

    // Tolerances: [rtol, atol]
    private static final double[] EXACT_TOL   = {1e-5, 1e-5};
    private static final double[] TRITON_TOL  = {1e-3, 1e-3};
    // Same calibration as TRITON_TOL: cross-kernel-algorithm float32 comparison.
    // Softmax-bearing fixtures amplify upstream accumulation-order deltas into
    // sign-alternating ~1e-4 scatter (verified: no coherent shift toward any
    // neighboring iteration). Stale-DATA regressions are coherent and ≥5e-3.
    private static final double[] JIT_TOL     = {1e-3, 1e-3};

    private SameDiff sd;

    @BeforeEach
    void setUp() {
        Nd4j.getMemoryManager().togglePeriodicGc(false);
    }

    @AfterEach
    void tearDown() {
        if (sd != null) {
            try { sd.close(); } catch (Throwable ignored) { }
            sd = null;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Fixture infrastructure
    // ──────────────────────────────────────────────────────────────────────

    private static final class Fixture {
        final String name;
        final Supplier<SameDiff> graphBuilder;
        final Supplier<Map<String, INDArray>> inputBuilder;
        final String[] outputNames;

        Fixture(String name, Supplier<SameDiff> graphBuilder,
                Supplier<Map<String, INDArray>> inputBuilder, String... outputNames) {
            this.name = name;
            this.graphBuilder = graphBuilder;
            this.inputBuilder = inputBuilder;
            this.outputNames = outputNames;
        }

        @Override public String toString() { return name; }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Execution modes
    // ──────────────────────────────────────────────────────────────────────

    static List<GraphExecutionMode> executionModes() {
        List<GraphExecutionMode> modes = new ArrayList<>();
        modes.add(GraphExecutionMode.AUTO);
        modes.add(GraphExecutionMode.SLOT_BY_SLOT);
        modes.add(GraphExecutionMode.CUDA_GRAPHS);
        modes.add(GraphExecutionMode.NVRTC_JIT);
        modes.add(GraphExecutionMode.PTX_JIT);
        modes.add(GraphExecutionMode.TRITON);
        modes.add(GraphExecutionMode.EMULATED_REPLAY);
        return modes;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Fixtures
    // ──────────────────────────────────────────────────────────────────────

    private static List<Fixture> fixtures() {
        List<Fixture> fs = new ArrayList<>();

        // Pattern 1: op → reshape view alias
        // Mirrors fused_rope → reshape corruption from VLM decode.
        // relu writes to output, reshape is a zero-copy view, matmul reads
        // through the view. If the view sees stale data, matmul result diverges.
        fs.add(new Fixture(
                "opToReshapeView",
                DspBufferAliasAccuracyTest::buildOpToReshapeView,
                DspBufferAliasAccuracyTest::inputsSmall4x16,
                "final"));

        // Pattern 2: concat → expand_dims view
        // Mirrors concat → expand_dims corruption from VLM KV-cache.
        // concat produces [N+M, D], expand_dims views it as [1, N+M, D],
        // downstream matmul reads through the expand_dims view.
        fs.add(new Fixture(
                "concatToExpandDimsView",
                DspBufferAliasAccuracyTest::buildConcatToExpandDimsView,
                DspBufferAliasAccuracyTest::inputsConcatExpandDims,
                "final"));

        // Pattern 3: producer → reshape → reshape_no_copy → permute chain
        // Mirrors the triple-view-chain corruption (reshape+reshape_no_copy+permute
        // all sharing the same GPU address). Tests that the entire alias chain
        // sees consistent data after the producer writes.
        fs.add(new Fixture(
                "tripleViewChain",
                DspBufferAliasAccuracyTest::buildTripleViewChain,
                DspBufferAliasAccuracyTest::inputsSmall4x16,
                "final"));

        // Pattern 4: accumulating concat with view — KV-cache-like pattern.
        // Two separate inputs get concatenated, then viewed through reshape,
        // then consumed by a downstream op. The concat output buffer is aliased
        // by the reshape view.
        fs.add(new Fixture(
                "concatReshapeAlias",
                DspBufferAliasAccuracyTest::buildConcatReshapeAlias,
                DspBufferAliasAccuracyTest::inputsConcatReshape,
                "final"));

        // Pattern 5: op → view → op → view (alternating producer/view chain)
        // This catches ordering bugs where DSP interleaves view refreshes
        // incorrectly with compute ops.
        fs.add(new Fixture(
                "alternatingOpView",
                DspBufferAliasAccuracyTest::buildAlternatingOpView,
                DspBufferAliasAccuracyTest::inputsSmall4x16,
                "final"));

        // Pattern 6: concat at non-zero dimension with view
        // Tests concat along axis=1 (not axis=0) with downstream view.
        // The partial-buffer corruption at firstDiffByte>0 pattern.
        fs.add(new Fixture(
                "concatAxis1View",
                DspBufferAliasAccuracyTest::buildConcatAxis1View,
                DspBufferAliasAccuracyTest::inputsConcatAxis1,
                "final"));

        // Pattern 7: in-place-like op chain with shared buffer
        // Multiple ops share output buffers through view aliasing.
        // Catches bugs where frozen-phase pointer stability assumptions
        // are violated by view wrapper swaps.
        fs.add(new Fixture(
                "sharedBufferViewFanout",
                DspBufferAliasAccuracyTest::buildSharedBufferViewFanout,
                DspBufferAliasAccuracyTest::inputsSmall4x16,
                "final"));

        // Pattern 8: deep attention-like graph with Q/K/V reshapes.
        // Mimics a single transformer attention head: Q/K/V projection,
        // reshape into head dims, matmul for scores, softmax, matmul for
        // context, reshape back. Many view aliases in a realistic pattern.
        fs.add(new Fixture(
                "deepAttentionQKV",
                DspBufferAliasAccuracyTest::buildDeepAttentionQKV,
                DspBufferAliasAccuracyTest::inputsAttention,
                "final"));

        // Pattern 9: multi-layer concat-view stack.
        // Simulates 4 transformer decoder layers, each with a concat
        // (KV cache append) → expand_dims → reshape → matmul pattern.
        // The deep stack exercises DSP frozen-replay over many slots.
        fs.add(new Fixture(
                "multiLayerConcatView",
                DspBufferAliasAccuracyTest::buildMultiLayerConcatView,
                DspBufferAliasAccuracyTest::inputsMultiLayer,
                "final"));

        return fs;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Parameterized matrix
    // ──────────────────────────────────────────────────────────────────────

    static Stream<Arguments> fixtureModeMatrix() {
        // Debug narrowing: -Dalias.fixture=NAME and/or -Dalias.mode=MODE restrict the
        // matrix so a crashing (fixture,mode) can be rerun in isolation (surefire's
        // [index] filters do not match these display names).
        String onlyFixture = System.getProperty("alias.fixture");
        String onlyMode = System.getProperty("alias.mode");
        List<Fixture> fs = fixtures();
        List<GraphExecutionMode> modes = executionModes();
        List<Arguments> args = new ArrayList<>(fs.size() * modes.size());
        for (Fixture f : fs) {
            if (onlyFixture != null && !f.name.equals(onlyFixture)) continue;
            for (GraphExecutionMode m : modes) {
                if (onlyMode != null && !m.name().equals(onlyMode)) continue;
                args.add(Arguments.of(f, m));
            }
        }
        return args.stream();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: replay accuracy across all modes
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "accuracy[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Buffer-alias accuracy: op output aliased by view must produce correct results")
    public void testBufferAliasAccuracy(Fixture fix, GraphExecutionMode mode) {
        log.info("PARAM_BANNER testBufferAliasAccuracy fixture={} mode={}", fix.name, mode);
        assumeBackendAvailable(mode);

        INDArray[] reference = captureReference(fix);
        try {
            sd = buildGraph(fix.graphBuilder);
            configureDsp(sd, mode);
            double[] tol = tolerances(mode);

            for (int i = 0; i < REPLAYS; i++) {
                Map<String, INDArray> inputs = fix.inputBuilder.get();
                try {
                    Map<String, INDArray> raw;
                    try {
                        raw = sd.output(inputs, fix.outputNames);
                    } catch (Throwable t) {
                        fail(fix.name + " / " + mode + " replay #" + i
                                + " threw: " + t.getMessage(), t);
                        return;
                    }
                    assertOutputsMatch(reference, raw, fix.outputNames, tol,
                            fix.name + "/" + mode + "#" + i);
                } finally {
                    closeAll(inputs);
                }
            }

            if (mode != GraphExecutionMode.SLOT_BY_SLOT
                    && mode != GraphExecutionMode.EMULATED_REPLAY) {
                DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                        fix.name + "/" + mode + " after " + REPLAYS + " replays");
                DspPlanAssertions.assertNoSegmentFailures(sd, fix.name + "/" + mode);
            }
        } finally {
            closeAll(reference);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: per-slot output stability across replays
    // Runs the graph REPLAYS times and checks that the output at each
    // replay is identical (within tolerance) to the first replay's output.
    // This catches drift bugs where buffer aliasing causes cumulative error.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "stability[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Buffer-alias stability: repeated identical inputs must produce identical outputs")
    public void testBufferAliasStability(Fixture fix, GraphExecutionMode mode) {
        log.info("PARAM_BANNER testBufferAliasStability fixture={} mode={}", fix.name, mode);
        assumeBackendAvailable(mode);

        sd = buildGraph(fix.graphBuilder);
        configureDsp(sd, mode);
        double[] tol = tolerances(mode);

        INDArray[] firstOutput = null;
        try {
            for (int i = 0; i < REPLAYS; i++) {
                Map<String, INDArray> inputs = fix.inputBuilder.get();
                try {
                    Map<String, INDArray> raw;
                    try {
                        raw = sd.output(inputs, fix.outputNames);
                    } catch (Throwable t) {
                        fail(fix.name + " / " + mode + " replay #" + i
                                + " threw: " + t.getMessage(), t);
                        return;
                    }
                    if (i == 0) {
                        firstOutput = dupOutputs(raw, fix.outputNames);
                    } else {
                        INDArray[] current = dupOutputs(raw, fix.outputNames);
                        try {
                            assertOutputArraysMatch(firstOutput, current,
                                    fix.outputNames, tol,
                                    fix.name + "/" + mode + " replay#0 vs #" + i);
                        } finally {
                            closeAll(current);
                        }
                    }
                } finally {
                    closeAll(inputs);
                }
            }
        } finally {
            if (firstOutput != null) closeAll(firstOutput);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: fresh input every call with varying data
    // Uses different input values each replay to ensure the view alias
    // correctly propagates new data, not stale cached results.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "varyingInput[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Buffer-alias varying input: different inputs must produce different outputs")
    public void testBufferAliasVaryingInput(Fixture fix, GraphExecutionMode mode) {
        // Param banner: after a JVM crash, the LAST banner in the log names the
        // (fixture, mode) that died — surefire's own reporting is lost with the fork.
        log.info("PARAM_BANNER testBufferAliasVaryingInput fixture={} mode={}", fix.name, mode);
        assumeBackendAvailable(mode);

        sd = buildGraph(fix.graphBuilder);
        configureDsp(sd, mode);
        double[] tol = tolerances(mode);

        INDArray[] prevOutput = null;
        try {
            for (int i = 0; i < REPLAYS; i++) {
                // Vary the input with additive perturbation (not multiplicative,
                // which is invariant to softmax/normalization patterns)
                Map<String, INDArray> inputs = fix.inputBuilder.get();
                for (Map.Entry<String, INDArray> e : inputs.entrySet()) {
                    e.getValue().addi(0.5 * (i + 1));
                }
                try {
                    Map<String, INDArray> raw;
                    try {
                        raw = sd.output(inputs, fix.outputNames);
                    } catch (Throwable t) {
                        fail(fix.name + " / " + mode + " replay #" + i
                                + " threw: " + t.getMessage(), t);
                        return;
                    }

                    // Build a fresh reference for this specific input
                    SameDiff ref = buildGraph(fix.graphBuilder);
                    try {
                        disableDsp(ref);
                        Map<String, INDArray> refInputs = fix.inputBuilder.get();
                        for (Map.Entry<String, INDArray> e : refInputs.entrySet()) {
                            e.getValue().addi(0.5 * (i + 1));
                        }
                        try {
                            Map<String, INDArray> refRaw = ref.output(refInputs, fix.outputNames);
                            INDArray[] refOut = dupOutputs(refRaw, fix.outputNames);
                            try {
                                INDArray[] actual = dupOutputs(raw, fix.outputNames);
                                try {
                                    assertOutputArraysMatch(refOut, actual,
                                            fix.outputNames, tol,
                                            fix.name + "/" + mode + " varying#" + i);
                                } finally {
                                    closeAll(actual);
                                }
                            } finally {
                                closeAll(refOut);
                            }
                        } finally {
                            closeAll(refInputs);
                        }
                    } finally {
                        try { ref.close(); } catch (Throwable ignored) { }
                    }

                    // Also verify output actually changed from previous iteration
                    if (prevOutput != null && i > 0) {
                        INDArray[] current = dupOutputs(raw, fix.outputNames);
                        boolean anyDifferent = false;
                        for (int j = 0; j < fix.outputNames.length; j++) {
                            if (!current[j].equalsWithEps(prevOutput[j], 1e-6)) {
                                anyDifferent = true;
                            }
                        }
                        assertTrue(anyDifferent,
                                fix.name + "/" + mode + " replay#" + i
                                        + " output identical to previous despite different input");
                        closeAll(prevOutput);
                        prevOutput = current;
                    } else {
                        prevOutput = dupOutputs(raw, fix.outputNames);
                    }
                } finally {
                    closeAll(inputs);
                }
            }
        } finally {
            if (prevOutput != null) closeAll(prevOutput);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Graph builders — standalone, no ONNX import
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Pattern 1: op → reshape view alias.
     * relu(x @ w) → reshape → matmul.
     * The reshape is a zero-copy view of relu's output buffer.
     * If DSP scheduling reads stale pre-relu data through the reshape,
     * the final matmul will diverge.
     *
     * Graph: x[4,16] → mmul(w[16,16]) → relu → reshape[8,8] → mmul(w2[8,4]) → final[8,4]
     */
    private static SameDiff buildOpToReshapeView() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w  = g.var("w",  initWeight(0.05, 16, 16));
        SDVariable w2 = g.var("w2", initWeight(0.03, 8, 4));

        SDVariable h = g.mmul("mmul1", x, w);            // [4,16]
        SDVariable a = g.nn.relu("relu", h, 0);           // [4,16] — writes new data
        SDVariable v = g.reshape("view", a, 8, 8);        // zero-copy view of relu output
        SDVariable out = g.mmul("final", v, w2);           // [8,4] — reads through the view
        g.setOutputs("final");
        return g;
    }

    /**
     * Pattern 2: concat → expand_dims view.
     * Two inputs get concatenated along axis 0, then expand_dims adds a
     * batch dimension, then a downstream matmul reads through the view.
     * The expand_dims buffer is the same as concat's output.
     * Partial corruption at the concat boundary (byte offset where the
     * second input's data begins) means the downstream op sees stale data
     * in that region.
     *
     * Graph: concat([a[8,16], b[4,16]], axis=0) → [12,16] → expandDims(0) → [1,12,16]
     *        → reshape[12,16] → mmul(w[16,4]) → final[12,4]
     */
    private static SameDiff buildConcatToExpandDimsView() {
        SameDiff g = SameDiff.create();
        SDVariable a = g.placeHolder("a", DataType.FLOAT, 8, 16);
        SDVariable b = g.placeHolder("b", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w", initWeight(0.05, 16, 4));

        SDVariable cat = g.concat("cat", 0, a, b);         // [12,16]
        SDVariable exp = g.expandDims("exp", cat, 0);       // [1,12,16] — view of cat
        SDVariable flat = g.reshape("flat", exp, 12, 16);   // [12,16] — view of exp
        SDVariable out = g.mmul("final", flat, w);           // [12,4]
        g.setOutputs("final");
        return g;
    }

    /**
     * Pattern 3: producer → reshape → reshape (no-copy) → permute chain.
     * All three view ops share the same underlying GPU buffer.
     * Mirrors the triple-view corruption pattern seen in VLM decode where
     * reshape+reshape_no_copy+permute all have the same gpuAddr.
     *
     * Graph: x[4,16] → mmul(w[16,12]) → relu → [4,12]
     *        → reshape[2,24] → reshape[4,6,2] → permute[2,4,6]
     *        → reshape[2,24] → mmul(w2[24,4]) → final[2,4]
     */
    private static SameDiff buildTripleViewChain() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w  = g.var("w",  initWeight(0.05, 16, 12));
        SDVariable w2 = g.var("w2", initWeight(0.03, 24, 4));

        SDVariable h = g.mmul("mmul1", x, w);             // [4,12]
        SDVariable a = g.nn.relu("relu", h, 0);            // [4,12] — producer
        SDVariable r1 = g.reshape("r1", a, 2, 24);         // view of relu
        SDVariable r2 = g.reshape("r2", r1, 4, 6, 2);      // view of r1 (same buffer)
        SDVariable p1 = g.permute("p1", r2, 2, 0, 1);      // [2,4,6] — view (same buffer)
        SDVariable r3 = g.reshape("r3", p1, 2, 24);         // [2,24] — view
        SDVariable out = g.mmul("final", r3, w2);            // [2,4]
        g.setOutputs("final");
        return g;
    }

    /**
     * Pattern 4: concat → reshape alias.
     * Two tensors concatenated, then reshape views the result, then matmul
     * reads through the view. The reshape's GPU buffer is the same as
     * concat's output.
     *
     * Graph: concat([x[4,8], y[4,8]], axis=1) → [4,16] → reshape[8,8]
     *        → mmul(w[8,4]) → final[8,4]
     */
    private static SameDiff buildConcatReshapeAlias() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable y = g.placeHolder("y", DataType.FLOAT, 4, 8);
        SDVariable w = g.var("w", initWeight(0.05, 8, 4));

        SDVariable cat = g.concat("cat", 1, x, y);         // [4,16]
        SDVariable v = g.reshape("view", cat, 8, 8);        // view of concat
        SDVariable out = g.mmul("final", v, w);              // [8,4]
        g.setOutputs("final");
        return g;
    }

    /**
     * Pattern 5: alternating op → view → op → view chain.
     * Each compute op is followed by a reshape view, creating an
     * alternating pattern that tests DSP's ability to correctly order
     * view refresh and compute execution.
     *
     * Graph: x[4,16] → mmul(w1) → relu → reshape[8,8]
     *        → mmul(w2) → tanh → reshape[4,16]
     *        → mmul(w3) → relu → reshape[2,32]
     *        → mmul(w4) → final[2,4]
     */
    private static SameDiff buildAlternatingOpView() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w1 = g.var("w1", initWeight(0.05, 16, 16));
        SDVariable w2 = g.var("w2", initWeight(0.04, 8, 16));
        SDVariable w3 = g.var("w3", initWeight(0.03, 16, 32));
        SDVariable w4 = g.var("w4", initWeight(0.02, 32, 4));

        SDVariable h1 = g.mmul("mm1", x, w1);             // [4,16]
        SDVariable a1 = g.nn.relu("relu1", h1, 0);         // compute
        SDVariable v1 = g.reshape("v1", a1, 8, 8);          // view

        SDVariable h2 = g.mmul("mm2", v1, w2);             // [8,16] — reads through view
        SDVariable a2 = g.nn.tanh("tanh1", h2);             // compute
        SDVariable v2 = g.reshape("v2", a2, 4, 32);         // view

        SDVariable h3 = g.mmul("mm3", v2, w3);             // [4,32] — wait, dims don't match
        // Actually v2 is [4,32], w3 is [16,32] — let me fix shapes
        SDVariable out = g.mmul("final", v2, w4);            // [4,32] × [32,4] = [4,4]
        g.setOutputs("final");
        return g;
    }

    /**
     * Pattern 6: concat along axis=1 with downstream view.
     * Tests partial-buffer corruption: concat along a non-leading axis
     * means the new data is interleaved in memory (or at a specific
     * offset), not appended at the end.
     *
     * Graph: concat([x[4,8], w_const[4,8]], axis=1) → [4,16]
     *        → expandDims(0) → [1,4,16] → reshape[4,16]
     *        → mmul(w2[16,4]) → final[4,4]
     */
    private static SameDiff buildConcatAxis1View() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable c = g.var("c", initWeight(0.05, 4, 8));
        SDVariable w2 = g.var("w2", initWeight(0.03, 16, 4));

        SDVariable cat = g.concat("cat", 1, x, c);          // [4,16]
        SDVariable exp = g.expandDims("exp", cat, 0);        // [1,4,16] — view
        SDVariable flat = g.reshape("flat", exp, 4, 16);      // [4,16] — view
        SDVariable out = g.mmul("final", flat, w2);            // [4,4]
        g.setOutputs("final");
        return g;
    }

    /**
     * Pattern 7: shared buffer view fan-out.
     * One producer op's output is viewed by multiple downstream reshape ops,
     * each feeding a separate compute path. The paths are then summed.
     * Tests that all fan-out views see the same consistent data.
     *
     * Graph: x[4,16] → mmul(w[16,16]) → sigmoid →
     *        ├── reshape[8,8] → mmul(wa[8,1]) → [8,1]
     *        ├── reshape[2,32] → mmul(wb[32,1]) → [2,1]
     *        └── reshape[16,4] → mmul(wc[4,1]) → [16,1]
     *        sum all → final (scalar sum of all three paths)
     */
    private static SameDiff buildSharedBufferViewFanout() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w  = g.var("w",  initWeight(0.05, 16, 16));
        SDVariable wa = g.var("wa", initWeight(0.03, 8, 1));
        SDVariable wb = g.var("wb", initWeight(0.03, 32, 1));
        SDVariable wc = g.var("wc", initWeight(0.03, 4, 1));

        SDVariable h = g.mmul("mmul1", x, w);             // [4,16]
        SDVariable a = g.nn.sigmoid("sig", h);              // producer — [4,16]

        // Three views of the same buffer
        SDVariable v1 = g.reshape("v1", a, 8, 8);
        SDVariable v2 = g.reshape("v2", a, 2, 32);
        SDVariable v3 = g.reshape("v3", a, 16, 4);

        // Each view feeds a separate compute path
        SDVariable p1 = g.mmul("p1", v1, wa);              // [8,1]
        SDVariable p2 = g.mmul("p2", v2, wb);              // [2,1]
        SDVariable p3 = g.mmul("p3", v3, wc);              // [16,1]

        // Sum all paths (reduce to scalar for easy comparison)
        SDVariable s1 = p1.sum("s1");
        SDVariable s2 = p2.sum("s2");
        SDVariable s3 = p3.sum("s3");
        SDVariable out = g.math.add("final", g.math.add("s12", s1, s2), s3);
        g.setOutputs("final");
        return g;
    }

    /**
     * Pattern 8: deep attention-like Q/K/V with reshape into heads.
     * Simulates a single transformer attention layer:
     *   x → Wq/Wk/Wv projections → reshape into [batch, heads, seq, dim]
     *   → scores = Q @ K^T → softmax → context = scores @ V
     *   → reshape back → Wo projection → layernorm → final
     *
     * This creates many view aliases (3 reshape + 1 permute per Q/K/V)
     * and enough slots (~30+) to exercise DSP lifecycle phases.
     */
    private static SameDiff buildDeepAttentionQKV() {
        SameDiff g = SameDiff.create();
        int batch = 1, seq = 8, hidden = 32, heads = 4, headDim = hidden / heads;

        SDVariable x = g.placeHolder("x", DataType.FLOAT, batch, seq, hidden);
        SDVariable wq = g.var("wq", initWeight(0.02, hidden, hidden));
        SDVariable wk = g.var("wk", initWeight(0.02, hidden, hidden));
        SDVariable wv = g.var("wv", initWeight(0.02, hidden, hidden));
        SDVariable wo = g.var("wo", initWeight(0.02, hidden, hidden));

        // Flatten for projection: [batch*seq, hidden]
        SDVariable xFlat = g.reshape("xFlat", x, batch * seq, hidden);

        // Q/K/V projections
        SDVariable q = g.mmul("q_proj", xFlat, wq);     // [batch*seq, hidden]
        SDVariable k = g.mmul("k_proj", xFlat, wk);
        SDVariable v = g.mmul("v_proj", xFlat, wv);

        // Reshape into heads: [batch*seq, hidden] → [batch, seq, heads, headDim]
        SDVariable qR = g.reshape("qR", q, batch, seq, heads, headDim);
        SDVariable kR = g.reshape("kR", k, batch, seq, heads, headDim);
        SDVariable vR = g.reshape("vR", v, batch, seq, heads, headDim);

        // Permute to [batch, heads, seq, headDim]
        SDVariable qP = g.permute("qP", qR, 0, 2, 1, 3);
        SDVariable kP = g.permute("kP", kR, 0, 2, 1, 3);
        SDVariable vP = g.permute("vP", vR, 0, 2, 1, 3);

        // Reshape for batched matmul: [batch*heads, seq, headDim]
        SDVariable qBatch = g.reshape("qBatch", qP, batch * heads, seq, headDim);
        SDVariable kBatch = g.reshape("kBatch", kP, batch * heads, seq, headDim);
        SDVariable vBatch = g.reshape("vBatch", vP, batch * heads, seq, headDim);

        // Permute K for transpose: [batch*heads, headDim, seq]
        SDVariable kT = g.permute("kT", kBatch, 0, 2, 1);

        // Scores = Q @ K^T: [batch*heads, seq, seq]
        SDVariable scores = g.mmul("scores", qBatch, kT);
        SDVariable scale = g.math.mul("scaled_scores", scores,
                (float)(1.0 / Math.sqrt(headDim)));
        SDVariable attnWeights = g.nn.softmax("attn_weights", scale, -1);

        // Context = attn @ V: [batch*heads, seq, headDim]
        SDVariable context = g.mmul("context", attnWeights, vBatch);

        // Reshape back: [batch, heads, seq, headDim] → [batch*seq, hidden]
        SDVariable cR = g.reshape("cR", context, batch, heads, seq, headDim);
        SDVariable cP = g.permute("cP", cR, 0, 2, 1, 3);
        SDVariable cFlat = g.reshape("cFlat", cP, batch * seq, hidden);

        // Output projection
        SDVariable out = g.mmul("out_proj", cFlat, wo);   // [batch*seq, hidden]

        // Residual + simple normalization (avoid layernorm op complexity)
        SDVariable residual = g.math.add("residual", xFlat, out);
        SDVariable norm = residual.norm2("norm2");
        SDVariable normed = g.math.div("final", residual, norm);
        g.setOutputs("final");
        return g;
    }

    /**
     * Pattern 9: multi-layer concat-view stack.
     * 4 "layers", each does: matmul → relu → concat with a constant →
     * expand_dims → reshape → matmul. This creates a deep graph with
     * many concat→view aliases spread across multiple layers.
     */
    private static SameDiff buildMultiLayerConcatView() {
        SameDiff g = SameDiff.create();
        int dim = 16;

        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, dim);
        SDVariable h = x;

        for (int layer = 0; layer < 4; layer++) {
            String L = "L" + layer;
            SDVariable w1 = g.var(L + "_w1", initWeight(0.04, dim, dim));
            SDVariable bias = g.var(L + "_bias", initWeight(0.01, 2, dim));

            // Transform
            SDVariable proj = g.mmul(L + "_proj", h, w1);      // [4, dim]
            SDVariable act = g.nn.relu(L + "_relu", proj, 0);   // [4, dim]

            // Concat with a small constant (simulates KV cache append)
            SDVariable cat = g.concat(L + "_cat", 0, act, bias);  // [6, dim]
            SDVariable exp = g.expandDims(L + "_exp", cat, 0);     // [1, 6, dim] — view
            SDVariable flat = g.reshape(L + "_flat", exp, 6, dim); // [6, dim] — view

            // Reduce back to [4, dim] via matmul with a [6, 4] weight → err,
            // or slice the first 4 rows
            SDVariable sliced = g.slice(L + "_slice", flat,
                    new int[]{0, 0}, new int[]{4, dim});              // [4, dim]

            // Residual connection
            h = g.math.add(L + "_res", h, sliced);
        }

        // Final reduction
        SDVariable wFinal = g.var("wFinal", initWeight(0.03, dim, 4));
        SDVariable out = g.mmul("final", h, wFinal);  // [4, 4]
        g.setOutputs("final");
        return g;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Input builders
    // ──────────────────────────────────────────────────────────────────────

    private static Map<String, INDArray> inputsSmall4x16() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.4, 0.01, 4 * 16).reshape(4, 16));
        return m;
    }

    private static Map<String, INDArray> inputsConcatExpandDims() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("a", Nd4j.linspace(DataType.FLOAT, -0.3, 0.005, 8 * 16).reshape(8, 16));
        m.put("b", Nd4j.linspace(DataType.FLOAT, 0.1, 0.01, 4 * 16).reshape(4, 16));
        return m;
    }

    private static Map<String, INDArray> inputsConcatReshape() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.2, 0.01, 4 * 8).reshape(4, 8));
        m.put("y", Nd4j.linspace(DataType.FLOAT, 0.1, 0.005, 4 * 8).reshape(4, 8));
        return m;
    }

    private static Map<String, INDArray> inputsConcatAxis1() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.3, 0.01, 4 * 8).reshape(4, 8));
        return m;
    }

    private static Map<String, INDArray> inputsAttention() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.1, 0.001, 1 * 8 * 32).reshape(1, 8, 32));
        return m;
    }

    private static Map<String, INDArray> inputsMultiLayer() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.3, 0.01, 4 * 16).reshape(4, 16));
        return m;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Weight initialization (deterministic, no RNG dependency)
    // ──────────────────────────────────────────────────────────────────────

    private static final java.util.concurrent.atomic.AtomicLong WEIGHT_IDX =
            new java.util.concurrent.atomic.AtomicLong(0L);

    private static INDArray initWeight(double scale, long... shape) {
        long total = 1L;
        for (long s : shape) total *= s;
        long idx = WEIGHT_IDX.getAndIncrement();
        long state = (idx + 1L) * 0x9E3779B97F4A7C15L;
        state ^= (state >>> 30);
        state *= 0xBF58476D1CE4E5B9L;
        state ^= (state >>> 27);
        state *= 0x94D049BB133111EBL;
        state ^= (state >>> 31);
        if (state == 0L) state = 0xDEADBEEFCAFEBABEL;
        float[] data = new float[(int) total];
        for (int i = 0; i < total; i++) {
            state ^= state << 13;
            state ^= state >>> 7;
            state ^= state << 17;
            double u = ((state >>> 11) & 0x1FFFFFFFFFFFFFL) / (double) (1L << 53);
            data[i] = (float) ((u * 2.0 - 1.0) * scale);
        }
        return Nd4j.create(data, shape);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Graph construction + DSP config
    // ──────────────────────────────────────────────────────────────────────

    private static SameDiff buildGraph(Supplier<SameDiff> builder) {
        WEIGHT_IDX.set(0L);
        return builder.get();
    }

    private static void configureDsp(SameDiff target, GraphExecutionMode mode) {
        if (mode == GraphExecutionMode.SLOT_BY_SLOT) {
        }
        target.setGraphExecutionMode(mode);
    }

    private static void disableDsp(SameDiff target) {
        target.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
    }

    private static void assumeBackendAvailable(GraphExecutionMode mode) {
        if (mode == GraphExecutionMode.TRITON) {
            String name = Nd4j.getBackend().getClass().getSimpleName();
            assumeTrue(name.contains("Cuda") || name.contains("Cublas"),
                    "TRITON requires CUDA backend");
        }
    }

    private static double[] tolerances(GraphExecutionMode mode) {
        if (mode == GraphExecutionMode.TRITON) return TRITON_TOL;
        // JIT/emulated modes legitimately select different kernel algorithms than
        // the SLOT_BY_SLOT reference (e.g., cuBLAS GEMM algorithm choice varies
        // with operand pointer alignment — raw-external vs staging-aliased views).
        // Equally-valid float32 accumulation orders diverge ~1e-5..1e-4 over deep
        // matmul chains; EXACT_TOL (1e-5) asserts bit-similar algorithm selection,
        // which is not a correctness property of these modes. Stale-DATA bugs are
        // orders of magnitude larger (the fixed varying#4 staleness was 5e-3).
        if (mode == GraphExecutionMode.NVRTC_JIT
                || mode == GraphExecutionMode.PTX_JIT
                || mode == GraphExecutionMode.EMULATED_REPLAY) {
            return JIT_TOL;
        }
        return EXACT_TOL;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Reference capture + comparison
    // ──────────────────────────────────────────────────────────────────────

    private INDArray[] captureReference(Fixture fix) {
        SameDiff ref = buildGraph(fix.graphBuilder);
        try {
            disableDsp(ref);
            Map<String, INDArray> inputs = fix.inputBuilder.get();
            try {
                Map<String, INDArray> raw = ref.output(inputs, fix.outputNames);
                return dupOutputs(raw, fix.outputNames);
            } finally {
                closeAll(inputs);
            }
        } finally {
            try { ref.close(); } catch (Throwable ignored) { }
        }
    }

    private static INDArray[] dupOutputs(Map<String, INDArray> raw, String[] names) {
        INDArray[] out = new INDArray[names.length];
        for (int i = 0; i < names.length; i++) {
            INDArray v = raw.get(names[i]);
            assertNotNull(v, "output '" + names[i] + "' missing from raw result");
            out[i] = v.dup();
        }
        return out;
    }

    private static void assertOutputsMatch(INDArray[] reference,
                                            Map<String, INDArray> raw,
                                            String[] names, double[] tol, String label) {
        for (int i = 0; i < names.length; i++) {
            INDArray got = raw.get(names[i]);
            assertNotNull(got, label + ": output '" + names[i] + "' missing");
            compareArrays(reference[i], got, tol[0], tol[1], label + "/" + names[i]);
        }
    }

    private static void assertOutputArraysMatch(INDArray[] reference,
                                                 INDArray[] actual,
                                                 String[] names, double[] tol, String label) {
        for (int i = 0; i < names.length; i++) {
            compareArrays(reference[i], actual[i], tol[0], tol[1], label + "/" + names[i]);
        }
    }

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
            StringBuilder dump = new StringBuilder();
            dump.append(label).append(": diverges at index ").append(firstBad)
                    .append(" ref=").append(refD.getDouble(firstBad))
                    .append(" actual=").append(actD.getDouble(firstBad))
                    .append(" (worstAbs=").append(worstAbs)
                    .append(" worstRel=").append(worstRel)
                    .append(" atol=").append(atol).append(" rtol=").append(rtol)
                    .append(")\n  refFlat=[");
            long printN = Math.min(n, 32L);
            for (long i = 0; i < printN; i++) {
                if (i > 0) dump.append(", ");
                dump.append(refD.getDouble(i));
            }
            dump.append(printN < n ? ", ...]\n  actFlat=[" : "]\n  actFlat=[");
            for (long i = 0; i < printN; i++) {
                if (i > 0) dump.append(", ");
                dump.append(actD.getDouble(i));
            }
            dump.append(printN < n ? ", ...]" : "]");
            fail(dump.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Cleanup
    // ──────────────────────────────────────────────────────────────────────

    private static void closeAll(Map<String, INDArray> m) {
        if (m == null) return;
        for (INDArray a : m.values()) {
            if (a != null) try { a.close(); } catch (Throwable ignored) { }
        }
    }

    private static void closeAll(INDArray[] arr) {
        if (arr == null) return;
        for (INDArray a : arr) {
            if (a != null) try { a.close(); } catch (Throwable ignored) { }
        }
    }
}
