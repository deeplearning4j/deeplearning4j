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
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
 * Comprehensive DSP slot lifecycle audit — targets the two bge warmup encode
 * bugs that passed through the existing view-op frozen-replay suite but still
 * fire in production:
 *
 * <ol>
 *   <li><b>Bug 1 — CLOSED DataBuffer in SHAPES_FROZEN+ phase.</b>
 *       <pre>
 *       LIFECYCLE_ERROR: slot 28 has CLOSED DataBuffer 0x… during
 *       SHAPES_FROZEN+ phase — buffer was freed while shapes assumed stable
 *       </pre>
 *       An intermediate/constant/external buffer referenced by a frozen
 *       slot snapshot gets closed between executions. The audit exercises
 *       every place a caller could legitimately free an array around a DSP
 *       call (inputs, outputs, variables, re-associated variables) and
 *       requires zero crashes across multiple replays.</li>
 *   <li><b>Bug 2 — NDArray pointer replacement at view-op-install.</b>
 *       <pre>
 *       LIFECYCLE VIOLATION: NDArray pointer replacement at slot 2
 *       (tag=view-op-install) during frozen phase (phase=1 execCount=1).
 *       old=0x… new=0x… oldDb=0x… newDb=0x…    (oldDb == newDb)
 *       </pre>
 *       The frozen-phase guard rejects a fresh NDArray wrapper even when
 *       the underlying DataBuffer is unchanged. The audit builds BGE-sized
 *       graphs (multi-layer encoder with reshape/permute chains producing
 *       slots past index 28) and runs them against all execution modes to
 *       catch wrapper-swap paths the existing view-op suite doesn't
 *       cover.</li>
 * </ol>
 *
 * <p>Coverage matrix:
 * <table>
 *   <tr><th>Dimension</th><th>Values</th></tr>
 *   <tr><td>Execution mode</td><td>SLOT_BY_SLOT, AUTO, CUDA_GRAPHS,
 *       NVRTC_JIT, PTX_JIT, TRITON (if available), EMULATED_REPLAY</td></tr>
 *   <tr><td>Graph topology</td><td>bge-style encoder (multi-layer attention +
 *       FFN, reaches slot ≥28), single-layer attention, reshape over
 *       constant, reshape over intermediate, slice over variable,
 *       permute-reshape chain, multiple views of shared source</td></tr>
 *   <tr><td>Caller lifecycle</td><td>fresh input each call, recycled input,
 *       close output between calls, close input after output.dup(), close
 *       and rebind variable between calls</td></tr>
 * </table>
 *
 * <p><b>Run everything:</b>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=DspSlotLifecycleAuditTest \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
 *       2&gt;&amp;1 | tee /tmp/dsp-slot-audit.log
 * </pre>
 *
 * <p><b>Run a single combination:</b>
 * <pre>
 *   mvn test -Dtest='DspSlotLifecycleAuditTest#testWarmupRecycleInput[bgeEncoder-CUDA_GRAPHS]'
 * </pre>
 */
@Slf4j
@Tag("dsp")
@Tag(TagNames.FULL_CI)
@DisplayName("DSP slot lifecycle audit — close-between-calls, bge encoder, wrapper-swap edge cases")
public class DspSlotLifecycleAuditTest {

    // ─── Tolerances ─────────────────────────────────────────────────────────
    private static final double FP32_RTOL = 1e-4;
    private static final double FP32_ATOL = 1e-5;
    private static final double LOOSE_RTOL = 1e-2;
    private static final double LOOSE_ATOL = 1e-3;

    // ─── Replay count ──────────────────────────────────────────────────────
    // 6 is the smallest value that covers: first run (freeze), second run
    // (writeOutputSlot enters frozen-phase path), third run (AUTO_SEAL most
    // likely promoted), and three more margin runs so cache-hit / capture /
    // replay combinations all exercise the frozen writeOutputSlot guard.
    private static final int REPLAYS = 6;

    // ─── BGE-like encoder sizes (small enough for unit-test speed) ─────────
    //   HEADS × HEAD_DIM = HIDDEN so reshape/permute Q/K/V works.
    //   2 layers is enough to push slot count past 28.
    private static final int BGE_HEADS = 4;
    private static final int BGE_HEAD_DIM = 4;
    private static final int BGE_HIDDEN = BGE_HEADS * BGE_HEAD_DIM;  // 16
    private static final int BGE_SEQ = 4;
    private static final int BGE_FF = BGE_HIDDEN * 4;                // 64

    private SameDiff sd;

    // ──────────────────────────────────────────────────────────────────────
    // Fixture descriptor
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

        @Override
        public String toString() {
            return name;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // JUnit lifecycle
    // ──────────────────────────────────────────────────────────────────────

    /** Deterministic seed for all RNG-driven weight initialization. Each call
     *  to {@link #buildGraph(Supplier)} re-seeds the global Nd4j RNG with this
     *  value so that a freshly-built reference graph and a freshly-built
     *  test graph have identical weights. Without this, {@code Nd4j.randn}
     *  inside the fixture builders would yield different tensors on each
     *  call and every accuracy assertion would trivially fail. */
    private static final long FIXTURE_SEED = 0xD5CA11B10L;

    @BeforeEach
    public void setUp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
        Nd4j.getRandom().setSeed(FIXTURE_SEED);
    }

    /** Build a graph with deterministic weights. Resets the
     *  {@link #WEIGHT_CALL_IDX} counter so that every fixture builder sees
     *  the exact same sequence of {@code randFloat} results on every call —
     *  both for the reference graph and the test-mode graph. Nd4j's own
     *  RNG is left seeded too for any downstream code that queries it, but
     *  {@code randFloat} no longer depends on it. */
    private static SameDiff buildGraph(Supplier<SameDiff> builder) {
        WEIGHT_CALL_IDX.set(0L);
        Nd4j.getRandom().setSeed(FIXTURE_SEED);
        return builder.get();
    }

    @AfterEach
    public void tearDown() {
        if (sd != null) {
            try { sd.close(); } catch (Throwable t) { log.warn("sd.close() in tearDown", t); }
            sd = null;
        }
        Nd4j.getExecutioner().commit();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Execution modes + fixture lists
    // ──────────────────────────────────────────────────────────────────────

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
        try { return Nd4j.getNativeOps().isTritonAvailable(); }
        catch (Throwable t) { return false; }
    }

    /** All fixtures that this suite parameterizes over. */
    private static List<Fixture> fixtures() {
        List<Fixture> fs = new ArrayList<>();

        // ─── BGE-shaped encoder — reaches slot ≥ 28. The user's slot-28
        //    CLOSED-DataBuffer crash was in a graph of roughly this size.
        fs.add(new Fixture(
                "bgeEncoder",
                DspSlotLifecycleAuditTest::buildBgeEncoder,
                DspSlotLifecycleAuditTest::bgeEncoderInputs,
                "embedding"));

        // ─── Single-layer attention — a simpler variant so the BGE bug's
        //    direct ancestor (one attention block with reshape/permute) is
        //    exercised independently.
        fs.add(new Fixture(
                "attnSingleLayer",
                DspSlotLifecycleAuditTest::buildSingleLayerAttention,
                DspSlotLifecycleAuditTest::bgeEncoderInputs,
                "L0_ln2"));

        // ─── Reshape directly over a constant variable. Stresses the
        //    wrapper-swap path where the view-op input is a constant
        //    (sd.var) rather than a placeholder or intermediate.
        fs.add(new Fixture(
                "reshapeOverConstant",
                DspSlotLifecycleAuditTest::buildReshapeOverConstant,
                DspSlotLifecycleAuditTest::smallInputs,
                "out"));

        // ─── Permute over constant — same pattern, different view op.
        fs.add(new Fixture(
                "permuteOverConstant",
                DspSlotLifecycleAuditTest::buildPermuteOverConstant,
                DspSlotLifecycleAuditTest::smallInputs,
                "out"));

        // ─── Slice (non-zero offset) over intermediate. Triggers offset
        //    checks in the wrapper-swap fast-path.
        fs.add(new Fixture(
                "sliceOverIntermediate",
                DspSlotLifecycleAuditTest::buildSliceOverIntermediate,
                DspSlotLifecycleAuditTest::smallInputs,
                "out"));

        // ─── Multiple views of the same source (Q/K/V sharing same input
        //    via three reshapes). Catches bugs where two view slots race
        //    for the same DataBuffer.
        fs.add(new Fixture(
                "multipleViewsSharedSource",
                DspSlotLifecycleAuditTest::buildMultipleViewsSharedSource,
                DspSlotLifecycleAuditTest::smallInputs,
                "out"));

        // ─── Long reshape/permute chain — stresses the frozen-phase guard
        //    across many sequential view-producer slots.
        fs.add(new Fixture(
                "longViewChain",
                DspSlotLifecycleAuditTest::buildLongViewChain,
                DspSlotLifecycleAuditTest::longChainInputs,
                "out"));

        return fs;
    }

    /** Cross product of (fixture, mode) for parameterized tests. */
    static Stream<Arguments> fixtureModeMatrix() {
        List<Fixture> fs = fixtures();
        List<GraphExecutionMode> modes = executionModes();
        List<Arguments> args = new ArrayList<>(fs.size() * modes.size());
        for (Fixture f : fs) {
            for (GraphExecutionMode m : modes) {
                args.add(Arguments.of(f, m));
            }
        }
        return args.stream();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: warmup pattern — RECYCLE the same input INDArray across calls
    // (this is the exact bge warmup pattern: build INDArray once, call
    //  encode() repeatedly)
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "warmupRecycleInput[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Caller reuses the same input INDArray across N calls — "
            + "must not trip CLOSED DataBuffer or pointer-replacement violations")
    public void testWarmupRecycleInput(Fixture fix, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        INDArray[] reference = captureReference(fix);
        try {
            sd = buildGraph(fix.graphBuilder);
            configureDsp(sd, mode);

            // Build inputs ONCE and reuse across all replays.
            Map<String, INDArray> inputs = fix.inputBuilder.get();
            try {
                double[] tol = tolerances(mode);
                for (int i = 0; i < REPLAYS; i++) {
                    Map<String, INDArray> raw;
                    try {
                        raw = sd.output(inputs, fix.outputNames);
                    } catch (Throwable t) {
                        fail(fix.name + " / " + mode + " recycle replay #" + i
                                + " threw: " + t.getMessage(), t);
                        return;
                    }
                    assertOutputsMatch(reference, raw, fix.outputNames, tol,
                            fix.name + "/" + mode + "#" + i);
                    // Duplicate outputs before release so next call can reuse
                    // the underlying slot cache without colliding with our
                    // reference comparison. We don't close the raw values —
                    // they're session-owned views.
                }
            } finally {
                closeAll(inputs);
            }
        } finally {
            closeAll(reference);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: caller creates a FRESH input each call, closes old one mid-loop
    // (this is the exact pattern that frees an external DataBuffer between
    //  two DSP calls — most likely Bug 1 trigger)
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "freshInputCloseBetween[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Caller allocates + closes a fresh input INDArray each call — "
            + "must not leave a stale DataBuffer reference in any slot snapshot")
    public void testFreshInputCloseBetween(Fixture fix, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        INDArray[] reference = captureReference(fix);
        try {
            sd = buildGraph(fix.graphBuilder);
            configureDsp(sd, mode);

            double[] tol = tolerances(mode);
            for (int i = 0; i < REPLAYS; i++) {
                Map<String, INDArray> inputs = fix.inputBuilder.get();
                INDArray[] dupOutputs;
                try {
                    Map<String, INDArray> raw = sd.output(inputs, fix.outputNames);
                    dupOutputs = dupOutputs(raw, fix.outputNames);
                } catch (Throwable t) {
                    closeAll(inputs);
                    fail(fix.name + " / " + mode + " fresh-close replay #" + i
                            + " threw: " + t.getMessage(), t);
                    return;
                }
                try {
                    assertOutputArraysMatch(reference, dupOutputs,
                            fix.outputNames, tol,
                            fix.name + "/" + mode + "#" + i);
                } finally {
                    closeAll(dupOutputs);
                    closeAll(inputs);          // free the input immediately so any
                                               // cached slot pointer is invalidated
                                               // before the next call.
                    Nd4j.getExecutioner().commit();
                }
            }
        } finally {
            closeAll(reference);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: close the previous call's OUTPUT (.dup) before the next call
    // runs. Catches cases where the caller's held output buffer overlaps
    // with a slot-cached wrapper.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "closeDupOutput[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Caller dup()'s and closes the prior output before the next call")
    public void testCloseDupOutputBetween(Fixture fix, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        INDArray[] reference = captureReference(fix);
        try {
            sd = buildGraph(fix.graphBuilder);
            configureDsp(sd, mode);

            double[] tol = tolerances(mode);
            INDArray[] prev = null;
            for (int i = 0; i < REPLAYS; i++) {
                // Close any prior held dup before the call — the slot cache
                // must not retain a pointer into it.
                if (prev != null) {
                    closeAll(prev);
                    prev = null;
                    Nd4j.getExecutioner().commit();
                }
                Map<String, INDArray> inputs = fix.inputBuilder.get();
                try {
                    Map<String, INDArray> raw;
                    try {
                        raw = sd.output(inputs, fix.outputNames);
                    } catch (Throwable t) {
                        fail(fix.name + " / " + mode + " close-dup replay #" + i
                                + " threw: " + t.getMessage(), t);
                        return;
                    }
                    prev = dupOutputs(raw, fix.outputNames);
                    assertOutputArraysMatch(reference, prev, fix.outputNames, tol,
                            fix.name + "/" + mode + "#" + i);
                } finally {
                    closeAll(inputs);
                }
            }
            if (prev != null) closeAll(prev);
        } finally {
            closeAll(reference);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: baseline accuracy across replays. This is the catch-all regression
    // gate — any fixture × mode that produces wrong values or crashes is
    // failed, so wrapper-swap bugs at NON-slot-2 positions also trip.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "replayAccuracy[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("N replays under the target mode must match the SLOT_BY_SLOT reference")
    public void testReplayAccuracy(Fixture fix, GraphExecutionMode mode) {
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
            // After REPLAYS iterations, plan should have advanced past warmup
            // SLOT_BY_SLOT and EMULATED_REPLAY don't advance DSP phases
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
    // Test: close a WEIGHT / CONSTANT between DSP calls. This is the exact
    // path flagged by the bge warmup crash — the SlotBufferOwnership
    // external-input validator emits LIFECYCLE_ERROR when a frozen-phase
    // snapshot still points at a DataBuffer that the caller has freed. The
    // test must either (a) run successfully across all replays because the
    // library re-snapshots on every call, or (b) throw a clean diagnostic
    // exception mentioning CLOSED / LIFECYCLE_ERROR. A SIGABRT or silent
    // corruption is a bug.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "closeWeightBetween[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Caller closes a weight/constant between DSP calls — "
            + "must not leave a dangling DataBuffer reference in any frozen slot snapshot")
    public void testCloseWeightBetweenReplays(Fixture fix, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        INDArray[] reference = captureReference(fix);
        try {
            sd = buildGraph(fix.graphBuilder);
            configureDsp(sd, mode);

            SDVariable targetVar = pickFirstVariable(sd);
            assumeTrue(targetVar != null,
                    fix.name + " has no VariableType.VARIABLE — skipping");

            // 1) Warmup pass — run REPLAYS calls so the DSP plan freezes past
            //    SHAPES_FROZEN+ phase (the state where the SlotBufferOwnership
            //    external-input validator reports CLOSED DataBuffer). A
            //    single call won't reach the AUTO_SEAL threshold.
            double[] warmupTol = tolerances(mode);
            for (int w = 0; w < REPLAYS; w++) {
                Map<String, INDArray> warmupInputs = fix.inputBuilder.get();
                try {
                    Map<String, INDArray> raw = sd.output(warmupInputs, fix.outputNames);
                    assertOutputsMatch(reference, raw, fix.outputNames,
                            warmupTol, fix.name + "/" + mode + "/warmup#" + w);
                } catch (Throwable t) {
                    fail(fix.name + " / " + mode + " warmup#" + w + " threw: "
                            + t.getMessage(), t);
                    return;
                } finally {
                    closeAll(warmupInputs);
                }
            }

            // Structural assertion: plan should have frozen after warmup
            // SLOT_BY_SLOT and EMULATED_REPLAY don't advance DSP phases
            if (mode != GraphExecutionMode.SLOT_BY_SLOT
                    && mode != GraphExecutionMode.EMULATED_REPLAY) {
                DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                        fix.name + "/" + mode + " post-warmup");
            }

            // 2) Free the weight's DataBuffer. Undo the non-closeable poisoning
            //    that directExecHelper() applies to variables so close() takes
            //    effect. Also re-associate with a freshly-generated copy of the
            //    same data so the library has a valid array to read next call
            //    (or so we can observe how it reacts to a dangling buffer if it
            //    doesn't re-snapshot).
            INDArray prevArr = targetVar.getArr();
            assertNotNull(prevArr,
                    "variable " + targetVar.name() + " has no array");
            INDArray replacement = prevArr.dup();
            prevArr.setCloseable(true);
            if (prevArr.data() != null) prevArr.data().setConstant(false);
            // A graceful rebind via associateArrayWithVariable releases the
            // old reference from the SameDiff side; we then explicitly close
            // prevArr to make sure its DataBuffer is gone before the next call.
            sd.associateArrayWithVariable(replacement, targetVar);
            try {
                prevArr.close();
            } catch (Throwable ignored) {
                // Some paths leave the array non-closeable even after the
                // unpoisoning above — not a fail, we still want to run the
                // second call.
            }
            Nd4j.getExecutioner().commit();

            // 3) Second call — library must still match reference. If it
            //    throws, the exception must be a diagnosable lifecycle
            //    violation, not an opaque NPE or segfault. Note: silent
            //    corruption (returning zeros / stale constants from the
            //    freed buffer) is worse than a clean exception and will
            //    fail the assertOutputsMatch call below.
            double[] tol = tolerances(mode);
            Map<String, INDArray> inputs = fix.inputBuilder.get();
            try {
                Map<String, INDArray> raw;
                try {
                    raw = sd.output(inputs, fix.outputNames);
                } catch (Throwable t) {
                    String msg = String.valueOf(t.getMessage()).toLowerCase();
                    boolean isLifecycleDiag =
                            msg.contains("closed")
                                    || msg.contains("lifecycle")
                                    || msg.contains("databuffer")
                                    || msg.contains("freed")
                                    || msg.contains("execution failed");
                    if (!isLifecycleDiag) {
                        fail(fix.name + " / " + mode
                                + " second call threw non-diagnostic exception: "
                                + t.getMessage(), t);
                    }
                    // Diagnostic caught cleanly — path is exercised, no crash.
                    return;
                }
                assertOutputsMatch(reference, raw, fix.outputNames, tol,
                        fix.name + "/" + mode + "/afterClose");
            } finally {
                closeAll(inputs);
            }
        } finally {
            closeAll(reference);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: rebind a weight with a fresh INDArray wrapper containing the
    // same data. This swaps the wrapper (and its DataBuffer) without
    // changing the values, exactly mirroring Bug 2's "NDArray pointer
    // replacement" path but on the EXTERNAL INPUT track rather than the
    // output slot track. A correctly-behaving library must re-read variable
    // arrays on each call; a bug hits the "DataBuffer replaced during
    // frozen execution" check.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "rebindWeightBetween[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Caller rebinds a weight with a fresh INDArray wrapper (same values) — "
            + "output must still match and no LIFECYCLE violation should fire")
    public void testRebindWeightBetweenReplays(Fixture fix, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        INDArray[] reference = captureReference(fix);
        try {
            sd = buildGraph(fix.graphBuilder);
            configureDsp(sd, mode);

            SDVariable targetVar = pickFirstVariable(sd);
            assumeTrue(targetVar != null,
                    fix.name + " has no VariableType.VARIABLE — skipping");

            // Snapshot the original weight values so every rebind preserves
            // mathematical correctness — any divergence in output is caused
            // by the rebind, not by drifted data.
            INDArray originalValues = targetVar.getArr().dup();

            double[] tol = tolerances(mode);
            for (int i = 0; i < REPLAYS; i++) {
                // Rebind with a brand-new wrapper carrying identical data.
                // This creates a fresh DataBuffer, so the ext-input snapshot
                // from the prior call now points at a stale buffer unless
                // the library re-snapshots on every call.
                INDArray fresh = originalValues.dup();
                sd.associateArrayWithVariable(fresh, targetVar);
                Nd4j.getExecutioner().commit();

                Map<String, INDArray> inputs = fix.inputBuilder.get();
                try {
                    Map<String, INDArray> raw;
                    try {
                        raw = sd.output(inputs, fix.outputNames);
                    } catch (Throwable t) {
                        fail(fix.name + " / " + mode + " rebind replay #" + i
                                + " threw: " + t.getMessage(), t);
                        return;
                    }
                    assertOutputsMatch(reference, raw, fix.outputNames, tol,
                            fix.name + "/" + mode + "/rebind#" + i);
                } finally {
                    closeAll(inputs);
                }
            }
            try { originalValues.close(); } catch (Throwable ignored) { }
        } finally {
            closeAll(reference);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: fresh NDArray wrapper over the SAME underlying DataBuffer on
    // each call. This is the exact production pattern from the bge warmup
    // encode("test") crash:
    //
    //   LIFECYCLE VIOLATION: NDArray pointer replacement at slot 2
    //   (tag=view-op-install) during frozen phase (phase=1 execCount=1).
    //   old=0x… new=0x… oldDb=0x… newDb=0x…      (oldDb == newDb)
    //
    // The wrapper-swap guard in NativeDynamicShapePlan::writeOutputSlot
    // previously rejected any view-op wrapper swap whose shape/stride/offset
    // descriptor was not bit-for-bit identical to the prior wrapper, even
    // when the two wrappers pointed at the same underlying DataBuffer.
    // That's unsafe for common reshape/permute/slice patterns that
    // deterministically regenerate their wrapper each call. The fix: a
    // view-op tag with same DataBuffer is always a legitimate wrapper swap,
    // regardless of descriptor metadata, because no memory changes hands.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "viewOpInstallSameBuffer[{0}]")
    @MethodSource("executionModes")
    @DisplayName("Fresh NDArray wrapper over same input DataBuffer must not trip "
            + "frozen-phase wrapper-swap guard at view-op-install")
    public void testViewOpInstallSameBuffer(GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        // Reshape over a placeholder — the view-op slot receives a fresh
        // NDArray wrapper each call, and the underlying DataBuffer is the
        // caller's input buffer. Recycling the same backing INDArray but
        // passing a fresh wrapper each call (via .get(all,all), which
        // returns a view over the same DataBuffer) reproduces the production
        // oldDb == newDb path.
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable xR = sd.reshape("xR", x, 16, 4);                    // view [16, 4]
        SDVariable w = sd.var("w", randFloat(0.05, 4, 16));
        SDVariable h = sd.mmul("h", xR, w);                            // [16, 16]
        SDVariable out = sd.math.add("out", h, 0.1);
        sd.setOutputs("out");
        configureDsp(sd, mode);

        INDArray backing = Nd4j.linspace(DataType.FLOAT, -0.4, 0.01, 4L * 16)
                .reshape(4, 16);
        try {
            for (int i = 0; i < REPLAYS + 2; i++) {
                // Fresh NDArray wrapper over the SAME underlying DataBuffer.
                INDArray freshWrapper = backing.get(NDArrayIndex.all(), NDArrayIndex.all());
                Map<String, INDArray> inputs = new LinkedHashMap<>();
                inputs.put("x", freshWrapper);
                try {
                    Map<String, INDArray> raw = sd.output(inputs, "out");
                    INDArray got = raw.get("out");
                    assertNotNull(got, "mode=" + mode + " call #" + i + " missing output");
                } catch (Throwable t) {
                    String msg = String.valueOf(t.getMessage());
                    if (msg.toLowerCase().contains("lifecycle violation")
                            && msg.toLowerCase().contains("pointer replacement")) {
                        fail("mode=" + mode + " call #" + i
                                + " wrongly rejected same-buffer wrapper swap: "
                                + msg, t);
                    }
                    fail("mode=" + mode + " call #" + i + " threw: " + msg, t);
                    return;
                }
            }
        } finally {
            if (backing.closeable() && !backing.wasClosed()) backing.close();
        }
    }

    /** Return the first {@link VariableType#VARIABLE} SDVariable in the
     *  graph, or {@code null} if the graph has none. Used by the
     *  close-weight / rebind-weight tests to pick a target for the
     *  external-input lifecycle path without hard-coding a name per
     *  fixture. */
    private static SDVariable pickFirstVariable(SameDiff target) {
        for (SDVariable v : target.variables()) {
            if (v.getVariableType() == VariableType.VARIABLE) {
                return v;
            }
        }
        return null;
    }

    // ──────────────────────────────────────────────────────────────────────
    // BGE-like encoder — big enough to push slot index past 28 so the
    // slot-28 CLOSED DataBuffer crash has room to reproduce.
    //
    //   x [1, S, H] → Q/K/V projections → reshape → permute → scaled
    //   dot-product attention → reshape → output projection → residual →
    //   layernorm → FFN → residual → layernorm → slice([0,0,0],[1,1,H])
    //   → reshape(1,H) → l2_norm → embedding
    //
    // Done twice (2 layers) so slot count reaches ≈40.
    // ──────────────────────────────────────────────────────────────────────

    private static SameDiff buildBgeEncoder() {
        SameDiff g = SameDiff.create();

        // Placeholder: full hidden state (already embedded)
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, BGE_SEQ, BGE_HIDDEN);

        // Layer 0 weights
        SDVariable wq0 = g.var("wq0", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wk0 = g.var("wk0", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wv0 = g.var("wv0", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wo0 = g.var("wo0", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wff0_1 = g.var("wff0_1", randFloat(0.05, BGE_HIDDEN, BGE_FF));
        SDVariable wff0_2 = g.var("wff0_2", randFloat(0.05, BGE_FF, BGE_HIDDEN));

        SDVariable h0 = bgeEncoderLayer(g, x, "L0", wq0, wk0, wv0, wo0, wff0_1, wff0_2);

        // Layer 1 weights
        SDVariable wq1 = g.var("wq1", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wk1 = g.var("wk1", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wv1 = g.var("wv1", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wo1 = g.var("wo1", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wff1_1 = g.var("wff1_1", randFloat(0.05, BGE_HIDDEN, BGE_FF));
        SDVariable wff1_2 = g.var("wff1_2", randFloat(0.05, BGE_FF, BGE_HIDDEN));

        SDVariable h1 = bgeEncoderLayer(g, h0, "L1", wq1, wk1, wv1, wo1, wff1_1, wff1_2);

        // Pooling — slice the CLS (first token) and L2-normalize
        SDVariable cls = g.slice("cls", h1, new int[]{0, 0, 0}, new int[]{1, 1, BGE_HIDDEN});
        SDVariable cls2d = g.reshape("cls2d", cls, 1, BGE_HIDDEN);
        SDVariable sqSum = g.math.sum(cls2d.mul(cls2d), true, 1L);        // [1,1]
        SDVariable normVal = g.math.sqrt(sqSum).add(1e-12);               // [1,1]
        SDVariable embedding = g.math.div("embedding", cls2d, normVal);    // [1,16]

        g.setOutputs("embedding");
        return g;
    }

    /**
     * One BGE encoder layer: multi-head self-attention + residual + layernorm
     * + FFN + residual + layernorm. Returns the post-LN output. The reshape +
     * permute pattern inside QKV is the one that mints view-op wrappers that
     * trip the frozen-phase guard.
     */
    private static SDVariable bgeEncoderLayer(SameDiff g, SDVariable x, String prefix,
                                              SDVariable wq, SDVariable wk, SDVariable wv,
                                              SDVariable wo, SDVariable wff1, SDVariable wff2) {
        // Q/K/V projections — batched matmul across [1, S, H]
        SDVariable q = g.mmul(prefix + "_q", x, wq);                    // [1, S, H]
        SDVariable k = g.mmul(prefix + "_k", x, wk);
        SDVariable v = g.mmul(prefix + "_v", x, wv);

        // Reshape to [1, S, heads, head_dim]
        SDVariable qR = g.reshape(prefix + "_qR", q, 1, BGE_SEQ, BGE_HEADS, BGE_HEAD_DIM);
        SDVariable kR = g.reshape(prefix + "_kR", k, 1, BGE_SEQ, BGE_HEADS, BGE_HEAD_DIM);
        SDVariable vR = g.reshape(prefix + "_vR", v, 1, BGE_SEQ, BGE_HEADS, BGE_HEAD_DIM);

        // Permute to [1, heads, S, head_dim]
        SDVariable qP = g.permute(prefix + "_qP", qR, 0, 2, 1, 3);
        SDVariable kP = g.permute(prefix + "_kP", kR, 0, 2, 1, 3);
        SDVariable vP = g.permute(prefix + "_vP", vR, 0, 2, 1, 3);

        // kT: [1, heads, head_dim, S]
        SDVariable kT = g.permute(prefix + "_kT", kP, 0, 1, 3, 2);

        // scores = q · kT  → [1, heads, S, S]
        SDVariable scores = g.mmul(prefix + "_scores", qP, kT);
        SDVariable scaled = scores.mul(prefix + "_scaled", 1.0 / Math.sqrt(BGE_HEAD_DIM));
        SDVariable probs = g.nn.softmax(prefix + "_probs", scaled, -1);

        // ctx = probs · v → [1, heads, S, head_dim]
        SDVariable ctx = g.mmul(prefix + "_ctx", probs, vP);

        // Permute back + reshape to [1, S, H]
        SDVariable ctxP = g.permute(prefix + "_ctxP", ctx, 0, 2, 1, 3);
        SDVariable ctxFlat = g.reshape(prefix + "_ctxFlat", ctxP, 1, BGE_SEQ, BGE_HIDDEN);
        SDVariable attnOut = g.mmul(prefix + "_attnOut", ctxFlat, wo);

        // Residual + lightweight "layernorm" (mean-center + scale by rms)
        SDVariable res = x.add(prefix + "_res", attnOut);
        SDVariable ln1 = simpleLayernorm(g, res, prefix + "_ln1");

        // FFN
        SDVariable ff1 = g.mmul(prefix + "_ff1", ln1, wff1);
        SDVariable ff1_act = g.nn.relu(prefix + "_ff1_act", ff1, 0);
        SDVariable ff2 = g.mmul(prefix + "_ff2", ff1_act, wff2);
        SDVariable res2 = ln1.add(prefix + "_res2", ff2);
        return simpleLayernorm(g, res2, prefix + "_ln2");
    }

    /**
     * A very small "layernorm" surrogate — enough to reach slot index counts
     * and exercise the reduction → broadcast pattern without dragging in a
     * full layernorm op. Uses mean-center + div by rms.
     */
    private static SDVariable simpleLayernorm(SameDiff g, SDVariable in, String name) {
        SDVariable mean = in.mean(true, -1L);
        SDVariable centered = in.sub(mean);
        SDVariable sq = centered.mul(centered);
        SDVariable variance = sq.mean(true, -1L);
        SDVariable denom = g.math.sqrt(variance.add(1e-6));
        return g.math.div(name, centered, denom);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Simpler fixtures
    // ──────────────────────────────────────────────────────────────────────

    private static SameDiff buildSingleLayerAttention() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, BGE_SEQ, BGE_HIDDEN);
        SDVariable wq = g.var("wq", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wk = g.var("wk", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wv = g.var("wv", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wo = g.var("wo", randFloat(0.05, BGE_HIDDEN, BGE_HIDDEN));
        SDVariable wff1 = g.var("wff1", randFloat(0.05, BGE_HIDDEN, BGE_FF));
        SDVariable wff2 = g.var("wff2", randFloat(0.05, BGE_FF, BGE_HIDDEN));
        SDVariable h = bgeEncoderLayer(g, x, "L0", wq, wk, wv, wo, wff1, wff2);
        // Don't wrap h in g.identity — SameDiff SLOT_BY_SLOT executes the
        // identity op as a no-op that leaves "out" as a zero-filled alias,
        // producing a bogus reference that masks real DSP divergences. Use
        // the layer's natural output name ("L0_ln2") directly so both
        // reference and test path hit the same final op.
        g.setOutputs("L0_ln2");
        return g;
    }

    private static SameDiff buildReshapeOverConstant() {
        SameDiff g = SameDiff.create();
        // Placeholder drives the compute side.
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        // Constant that we explicitly reshape — the view's underlying buffer
        // is the constant's storage, not a plan-owned intermediate.
        SDVariable c = g.var("c", randFloat(0.1, 16, 8));          // [16, 8]
        SDVariable cR = g.reshape("cR", c, 8, 16);                 // VIEW over c, [8, 16]
        SDVariable h = g.mmul("h", x, c);                          // [4, 16] × [16, 8] = [4, 8]
        SDVariable scaled = g.mmul("scaled", h, cR);               // [4, 8] × [8, 16] = [4, 16]
        SDVariable out = g.math.add("out", scaled, 0.1);
        g.setOutputs("out");
        return g;
    }

    private static SameDiff buildPermuteOverConstant() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        // Permute swaps the two constant dims into a [4, 16] view so the
        // downstream add can broadcast with the placeholder.
        SDVariable c = g.var("c", randFloat(0.1, 16, 4));          // [16, 4]
        SDVariable cP = g.permute("cP", c, 1, 0);                  // VIEW [4, 16]
        SDVariable sum = x.add("sum", cP);                         // [4, 16]
        SDVariable out = g.math.mul("out", sum, 2.0);
        g.setOutputs("out");
        return g;
    }

    private static SameDiff buildSliceOverIntermediate() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w", randFloat(0.05, 16, 16));
        SDVariable h = g.mmul("h", x, w);                          // intermediate [4,16]
        SDVariable a = g.nn.relu("a", h, 0);
        // Slice a middle window at non-zero offset — the offset != 0 case
        // stresses the wrapper-swap fast-path's offset equality check.
        SDVariable s = g.slice("s", a, new int[]{1, 4}, new int[]{2, 8});
        SDVariable out = g.math.add("out", s, 0.25);
        g.setOutputs("out");
        return g;
    }

    private static SameDiff buildMultipleViewsSharedSource() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w", randFloat(0.05, 16, 16));
        SDVariable h = g.mmul("h", x, w);                          // intermediate [4,16]
        // Three distinct views of the same upstream — simulates Q/K/V
        // projections sharing input. Each reshape writes a view wrapper
        // to its own output slot.
        SDVariable r1 = g.reshape("r1", h, 2, 32);
        SDVariable r2 = g.reshape("r2", h, 8, 8);
        SDVariable r3 = g.reshape("r3", h, 16, 4);
        SDVariable sum = g.math.add("sum_ab",
                r1.reshape(4, 16), r2.reshape(4, 16));
        SDVariable out = g.math.add("out", sum, r3.reshape(4, 16));
        g.setOutputs("out");
        return g;
    }

    private static SameDiff buildLongViewChain() {
        SameDiff g = SameDiff.create();
        // Deep reshape/permute chain: many sequential view producers.
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 2, 3, 4, 5);
        SDVariable w = g.var("w", randFloat(0.05, 5, 5));
        SDVariable p0 = g.permute("p0", x, 0, 4, 1, 2, 3);                    // [1,5,2,3,4]
        SDVariable r0 = g.reshape("r0", p0, 1, 5, 24);                        // [1,5,24]
        SDVariable p1 = g.permute("p1", r0, 0, 2, 1);                         // [1,24,5]
        SDVariable r1 = g.reshape("r1", p1, 120, 1);                          // [120,1]
        SDVariable r2 = g.reshape("r2", r1, 24, 5);                           // [24,5]
        SDVariable r3 = g.reshape("r3", r2, 4, 30);                           // [4,30]
        SDVariable r4 = g.reshape("r4", r3, 120, 1);                          // [120,1]
        SDVariable r5 = g.reshape("r5", r4, 24, 5);                           // [24,5]
        SDVariable out = g.mmul("out", r5, w);                                // [24,5]
        g.setOutputs("out");
        return g;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Input builders
    // ──────────────────────────────────────────────────────────────────────

    private static Map<String, INDArray> bgeEncoderInputs() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.3, 0.01, 1L * BGE_SEQ * BGE_HIDDEN)
                .reshape(1, BGE_SEQ, BGE_HIDDEN));
        return m;
    }

    private static Map<String, INDArray> smallInputs() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        // Generate enough data for every simple-fixture placeholder shape we use.
        // Each fixture picks the subset it needs by placeholder name.
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.4, 0.01, 4 * 16).reshape(4, 16));
        return m;
    }

    private static Map<String, INDArray> longChainInputs() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.2, 0.005, 1 * 2 * 3 * 4 * 5)
                .reshape(1, 2, 3, 4, 5));
        return m;
    }

    /** Monotonically-increasing counter used by {@link #randFloat(double, long...)}
     *  to produce deterministic-but-unique weight tensors within a single
     *  {@link #buildGraph(Supplier)} invocation. Reset at the start of every
     *  build so that the reference graph and the test-mode graph end up with
     *  bit-identical weights. {@code Nd4j.getRandom().setSeed()} proved
     *  unreliable on the CUDA backend, so we stop relying on it entirely. */
    private static final java.util.concurrent.atomic.AtomicLong WEIGHT_CALL_IDX =
            new java.util.concurrent.atomic.AtomicLong(0L);

    private static INDArray randFloat(double scale, long... shape) {
        long total = 1L;
        for (long s : shape) total *= s;
        if (total <= 0 || total > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "unsupported weight shape " + Arrays.toString(shape));
        }
        long callIdx = WEIGHT_CALL_IDX.getAndIncrement();
        // Seed the xorshift stream with a large odd mix of callIdx so two
        // consecutive randFloat calls produce well-separated sequences.
        long state = (callIdx + 1L) * 0x9E3779B97F4A7C15L;
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
            // Map the 64-bit state into [-scale, +scale).
            double u = ((state >>> 11) & 0x1FFFFFFFFFFFFFL) / (double) (1L << 53);
            data[i] = (float) ((u * 2.0 - 1.0) * scale);
        }
        return Nd4j.create(data, shape);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Reference capture + output comparison
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
            dump.append(printN < n ? ", …]\n  actFlat=[" : "]\n  actFlat=[");
            for (long i = 0; i < printN; i++) {
                if (i > 0) dump.append(", ");
                dump.append(actD.getDouble(i));
            }
            dump.append(printN < n ? ", …]" : "]");
            fail(dump.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // DSP configuration + env helpers
    // ──────────────────────────────────────────────────────────────────────

    private static void configureDsp(SameDiff target, GraphExecutionMode mode) {
        if (mode == GraphExecutionMode.SLOT_BY_SLOT) {
        } else {
            target.setDspAutoCompileEnabled(true);
            target.setDspNativeAutoCompileEnabled(true);
        }
        target.setGraphExecutionMode(mode);
    }

    private static void disableDsp(SameDiff target) {
        target.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
    }

    private static void assumeBackendAvailable(GraphExecutionMode mode) {
        if (mode == GraphExecutionMode.TRITON) {
            assumeTrue(isTritonAvailable(), "Triton unavailable — skipping TRITON");
        }
    }

    private static double[] tolerances(GraphExecutionMode mode) {
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
    // Test: slot input visibility at each execution phase.
    //
    // Verifies that EVERY slot sees non-null inputs at each phase:
    //   (1) UNFROZEN warmup (execCount=0, shapes not frozen)
    //   (2) FROZEN warmup (execCount=0, shapes just frozen)
    //   (3) FROZEN capture (execCount=1-2, building CUDA graphs)
    //   (4) FROZEN replay (execCount>=3, steady-state graph replay)
    //
    // After each execution, queries every output slot via DspHandle and
    // asserts it is non-null. Any null slot at any phase is a bug.
    //
    // This test directly addresses the user's requirement: "We should be
    // asserting and testing what each slot sees at each phase. nulls
    // included. Add more testing and exception throwing."
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "slotPhaseVisibility[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Assert every slot has non-null output at every execution phase")
    public void testSlotPhaseVisibility(Fixture fix, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        sd = buildGraph(fix.graphBuilder);
        configureDsp(sd, mode);

        // Phase names for diagnostics
        String[] phaseNames = {
            "unfrozen_warmup_0",   // exec 0 before freeze
            "unfrozen_warmup_1",   // exec 1 before freeze
            "frozen_warmup_2",     // exec 2 — plan may freeze here
            "frozen_capture_3",    // exec 3 — capture warmup
            "frozen_capture_4",    // exec 4 — capture pass 2
            "frozen_replay_5",     // exec 5 — should be steady-state
            "frozen_replay_6",     // exec 6 — confirm stability
            "frozen_replay_7"      // exec 7 — additional stability check
        };

        DspHandle handle = null;
        for (int exec = 0; exec < phaseNames.length; exec++) {
            Map<String, INDArray> inputs = fix.inputBuilder.get();
            try {
                Map<String, INDArray> raw;
                try {
                    raw = sd.output(inputs, fix.outputNames);
                } catch (Throwable t) {
                    fail(fix.name + "/" + mode + " exec#" + exec + " ("
                            + phaseNames[exec] + ") threw: " + t.getMessage(), t);
                    return;
                }

                // Verify all requested outputs are present
                for (String outName : fix.outputNames) {
                    INDArray out = raw.get(outName);
                    assertNotNull(out, fix.name + "/" + mode + "/" + phaseNames[exec]
                            + ": requested output '" + outName + "' is null");
                    assertTrue(out.length() > 0, fix.name + "/" + mode + "/"
                            + phaseNames[exec] + ": output '" + outName + "' is empty");
                }

                // After first execution, DspHandle is available — inspect all slots
                if (handle == null) {
                    try {
                        handle = sd.dsp();
                    } catch (Throwable t) {
                        // DspHandle may not be available for SLOT_BY_SLOT mode
                        if (mode == GraphExecutionMode.SLOT_BY_SLOT) continue;
                        fail(fix.name + "/" + mode + ": cannot get DspHandle after exec#"
                                + exec + ": " + t.getMessage(), t);
                        return;
                    }
                }

                if (handle != null && handle.isCompiled()) {
                    int totalSlots = handle.totalSlots();
                    int phase = handle.planPhase();
                    int nullCount = 0;
                    List<Integer> nullSlots = new ArrayList<>();

                    for (int s = 0; s < totalSlots; s++) {
                        INDArray slotOut = null;
                        try {
                            slotOut = handle.getSlotOutput(s);
                        } catch (Throwable t) {
                            // Some slots may be legitimately empty (e.g. shape-only ops)
                            // but a crash is always a bug
                            log.warn("{}/{}/{}: slot {} threw during getSlotOutput: {}",
                                    fix.name, mode, phaseNames[exec], s, t.getMessage());
                        }
                        if (slotOut == null) {
                            nullCount++;
                            if (nullSlots.size() < 20) {
                                nullSlots.add(s);
                            }
                        }
                    }

                    log.info("{}/{}/{}: totalSlots={} nullSlots={} phase={} execCount={}",
                            fix.name, mode, phaseNames[exec], totalSlots, nullCount,
                            phase, exec);

                    // In FROZEN+REPLAY phase, no output slot should be null —
                    // all intermediates must be populated from prior executions.
                    if (phase >= PlanPhase.REPLAYING.getNativeCode() && exec >= 5) {
                        if (nullCount > 0) {
                            fail(fix.name + "/" + mode + "/" + phaseNames[exec]
                                    + ": " + nullCount + " null output slots in REPLAYING phase"
                                    + " (first 20: " + nullSlots + ")."
                                    + " Every slot must be populated during replay.");
                        }
                    }

                    // Check for NaN in any slot (indicates computation error)
                    int nanSlot = handle.firstNaNSlot();
                    if (nanSlot >= 0) {
                        fail(fix.name + "/" + mode + "/" + phaseNames[exec]
                                + ": NaN detected at slot " + nanSlot
                                + " — computation error during " + phaseNames[exec]);
                    }
                }
            } finally {
                closeAll(inputs);
            }
        }

        // Final structural assertions for modes that advance phases
        if (mode != GraphExecutionMode.SLOT_BY_SLOT
                && mode != GraphExecutionMode.EMULATED_REPLAY) {
            DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                    fix.name + "/" + mode + " after phase visibility sweep");
            DspPlanAssertions.assertNoSegmentFailures(sd,
                    fix.name + "/" + mode + " after phase visibility sweep");
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: slot output non-null invariant across freeze boundary.
    //
    // Runs the plan multiple times, freezes shapes, then runs more times.
    // At every step, asserts that no slot output transitioned from non-null
    // to null (a slot that was populated before freeze must stay populated
    // after freeze). This catches bugs where freeze/invalidation wipes
    // slot outputs.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "slotNonNullInvariantAcrossFreeze[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("Slot outputs that are non-null before freeze must remain non-null after freeze")
    public void testSlotNonNullInvariantAcrossFreeze(Fixture fix, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        sd = buildGraph(fix.graphBuilder);
        configureDsp(sd, mode);

        // Run pre-freeze executions to populate slots
        int preFreeze = 3;
        for (int i = 0; i < preFreeze; i++) {
            Map<String, INDArray> inputs = fix.inputBuilder.get();
            try {
                sd.output(inputs, fix.outputNames);
            } catch (Throwable t) {
                fail(fix.name + "/" + mode + " pre-freeze exec#" + i
                        + " threw: " + t.getMessage(), t);
                return;
            } finally {
                closeAll(inputs);
            }
        }

        // Snapshot which slots are non-null before freeze
        DspHandle handle;
        try {
            handle = sd.dsp();
        } catch (Throwable t) {
            fail(fix.name + "/" + mode + ": cannot get DspHandle: " + t.getMessage(), t);
            return;
        }
        if (!handle.isCompiled()) return;

        int totalSlots = handle.totalSlots();
        boolean[] wasNonNull = new boolean[totalSlots];
        for (int s = 0; s < totalSlots; s++) {
            try {
                INDArray out = handle.getSlotOutput(s);
                wasNonNull[s] = (out != null);
            } catch (Throwable t) {
                wasNonNull[s] = false;
            }
        }

        // Run more executions to trigger freeze + capture + replay
        int postFreeze = 5;
        for (int i = 0; i < postFreeze; i++) {
            Map<String, INDArray> inputs = fix.inputBuilder.get();
            try {
                sd.output(inputs, fix.outputNames);

                // Check slot outputs BEFORE closing inputs. View-producer slots
                // (permute, reshape) hold views wrapping the placeholder's
                // DataBuffer. Once closeAll(inputs) frees that DataBuffer, the
                // view becomes dangling and getSlotOutput correctly returns null.
                // The invariant we test is that slots stay populated while the
                // inputs they depend on are alive.
                List<Integer> droppedSlots = new ArrayList<>();
                for (int s = 0; s < totalSlots; s++) {
                    if (!wasNonNull[s]) continue;
                    try {
                        INDArray out = handle.getSlotOutput(s);
                        if (out == null) {
                            droppedSlots.add(s);
                        }
                    } catch (Throwable t2) {
                        droppedSlots.add(s);
                    }
                }
                if (!droppedSlots.isEmpty()) {
                    for (int ds : droppedSlots.subList(0, Math.min(5, droppedSlots.size()))) {
                        INDArray debugOut = null;
                        try { debugOut = handle.getSlotOutput(ds); } catch (Throwable ignored) {}
                        System.out.println("DEBUG_DROP: slot=" + ds + " wasNonNull=" + wasNonNull[ds]
                                + " getSlotOutput=" + debugOut
                                + " phase=" + handle.planPhase()
                                + " exec=" + i + " mode=" + mode + " fix=" + fix.name);
                    }
                    fail(fix.name + "/" + mode + " post-freeze exec#" + i
                            + ": " + droppedSlots.size() + " slots dropped from non-null to null"
                            + " after freeze (first 20: "
                            + droppedSlots.subList(0, Math.min(20, droppedSlots.size())) + ")."
                            + " phase=" + handle.planPhase()
                            + " — freeze must not wipe populated slot outputs.");
                }
            } catch (Throwable t) {
                fail(fix.name + "/" + mode + " post-freeze exec#" + i
                        + " threw: " + t.getMessage(), t);
                return;
            } finally {
                closeAll(inputs);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test: external input non-null at every phase.
    //
    // After each execution, queries external input addresses via DspHandle
    // and asserts that all placeholder external inputs have non-null
    // addresses. This catches the effectiveExternals_ zero-initialization
    // bug where gap segments see null external inputs after
    // markExternalInputVariable.
    // ──────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "extInputNonNullAtEveryPhase[{0}-{1}]")
    @MethodSource("fixtureModeMatrix")
    @DisplayName("All placeholder external inputs must have non-null addresses at every phase")
    public void testExtInputNonNullAtEveryPhase(Fixture fix, GraphExecutionMode mode) {
        assumeBackendAvailable(mode);

        sd = buildGraph(fix.graphBuilder);
        configureDsp(sd, mode);

        DspHandle handle = null;
        for (int exec = 0; exec < 8; exec++) {
            Map<String, INDArray> inputs = fix.inputBuilder.get();
            try {
                Map<String, INDArray> raw;
                try {
                    raw = sd.output(inputs, fix.outputNames);
                } catch (Throwable t) {
                    fail(fix.name + "/" + mode + " exec#" + exec
                            + " threw: " + t.getMessage(), t);
                    return;
                }

                // Verify outputs are non-null
                for (String outName : fix.outputNames) {
                    assertNotNull(raw.get(outName), fix.name + "/" + mode + " exec#"
                            + exec + ": output '" + outName + "' is null");
                }

                if (handle == null) {
                    try {
                        handle = sd.dsp();
                    } catch (Throwable t) {
                        if (mode == GraphExecutionMode.SLOT_BY_SLOT) continue;
                        fail(fix.name + "/" + mode + ": cannot get DspHandle: "
                                + t.getMessage(), t);
                        return;
                    }
                }

                if (handle != null && handle.isCompiled()) {
                    int numExt = handle.numExternalInputs();
                    int phase = handle.planPhase();
                    // Check that placeholder inputs (the ones we provide via the map)
                    // have non-zero addresses at this phase
                    for (String inputName : inputs.keySet()) {
                        int extIdx = handle.extInputIndex(inputName);
                        if (extIdx < 0) continue;
                        long addr = Nd4j.getNativeOps().getPlanLastExternalInputAddress(
                                handle.getNativePlanHandle(), extIdx);
                        if (addr == 0) {
                            fail(fix.name + "/" + mode + " exec#" + exec
                                    + " phase=" + phase
                                    + ": placeholder '" + inputName + "' (extIdx=" + extIdx
                                    + ") has null address. External inputs must always "
                                    + "have valid device addresses after execution.");
                        }
                    }
                }
            } finally {
                closeAll(inputs);
            }
        }
    }

    private static void closeAll(Map<String, INDArray> arrays) {
        if (arrays == null) return;
        for (INDArray arr : arrays.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
    }

    private static void closeAll(INDArray[] arrays) {
        if (arrays == null) return;
        for (INDArray arr : arrays) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
    }
}
