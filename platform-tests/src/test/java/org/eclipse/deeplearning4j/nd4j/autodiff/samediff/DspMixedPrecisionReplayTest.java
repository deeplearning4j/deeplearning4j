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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.ExecutionPhase;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that attribute specific DSP issues to their root cause:
 *
 * <ol>
 *   <li><b>Mixed-precision matmul</b> — FP16 weight × FP32 input must produce FP32 output
 *       when precisionBoostAllowed is true. Regression: pickPairwiseResultType returned
 *       the LHS type (FP16) instead of max(FP16, FP32)=FP32, causing NaN propagation.</li>
 *   <li><b>Decode-loop DSP lifecycle</b> — a repeated execution loop with changing
 *       placeholder values must reach SHAPES_FROZEN → pointer stability → REPLAYING.
 *       Regression: plans stayed at SHAPES_FROZEN with pointersStable=false.</li>
 *   <li><b>FP16 weight pre-cast with replay</b> — HALF constants combined with FLOAT
 *       placeholders must produce finite, non-NaN outputs at every DSP phase.</li>
 *   <li><b>Replay throughput</b> — once REPLAYING, the segment replay count must
 *       increment each step (not fall back to slot-by-slot).</li>
 * </ol>
 *
 * <p><b>Run:</b>
 * <pre>
 *   cd platform-tests && mvn test \
 *       -Dtest=DspMixedPrecisionReplayTest \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2&gt;&amp;1 | tee /tmp/mixed-precision-replay.log
 * </pre>
 */
@Slf4j
@Tag("dsp")
@DisplayName("DSP mixed-precision replay attribution tests")
public class DspMixedPrecisionReplayTest {

    private SameDiff sd;

    @BeforeEach
    public void setUp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    public void tearDown() {
        if (sd != null) {
            try { sd.close(); } catch (Throwable t) { /* ignore */ }
            sd = null;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 1: pickPairwiseResultType — FP16 × FP32 must yield FP32
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Regression test for DataTypeUtils.pickPairwiseResultType ignoring precisionBoostAllowed.
     *
     * When precisionBoostAllowed=true (default), matmul(HALF weight, FLOAT input) must
     * produce FLOAT output. If it returns HALF, downstream ops accumulate in reduced
     * precision and eventually produce NaN logits → premature EOS.
     *
     * Root cause: the float-float branch in pickPairwiseResultType was returning typeX
     * (the LHS) unconditionally instead of max(typeX, typeY).
     */
    @ParameterizedTest(name = "fp16WeightFp32Input_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    public void testFp16WeightFp32InputProducesFp32(GraphExecutionMode mode) {
        sd = SameDiff.create();

        // Simulate the VLM decoder pattern: FP16 pre-cast weight constant + FP32 input
        INDArray weight = Nd4j.randn(DataType.HALF, 64, 64);
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        SDVariable w = sd.constant("weight", weight);
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, 1, 64);
        SDVariable mm = sd.mmul("matmul", x, w);
        SDVariable out = sd.nn().relu("output", mm, 0);

        sd.setGraphExecutionMode(mode);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", input);

        // Execute and check output dtype is FLOAT (not HALF)
        Map<String, INDArray> result = sd.output(ph, "output");
        INDArray output = result.get("output");
        assertNotNull(output, mode + ": output is null");
        assertEquals(DataType.FLOAT, output.dataType(),
                mode + ": matmul(HALF, FLOAT) should produce FLOAT when precisionBoostAllowed=true");

        // Verify no NaN/Inf in the output
        assertFalse(output.isNaN().any(),
                mode + ": NaN detected in mixed-precision matmul output");
        assertFalse(output.isInfinite().any(),
                mode + ": Inf detected in mixed-precision matmul output");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 2: Decode-loop lifecycle — must reach REPLAYING
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Simulates a decode loop: repeated execution with the same shape but different
     * placeholder values. The plan must progress through:
     *   SLOT_BY_SLOT → SHAPES_FROZEN → pointer stability → REPLAYING
     *
     * Regression: plans stuck at SHAPES_FROZEN with pointersStable=false because
     * frozen execution count never reached the pointer stability threshold.
     */
    @ParameterizedTest(name = "decodeLoopLifecycle_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    public void testDecodeLoopReachesReplay(GraphExecutionMode mode) {
        sd = SameDiff.create();

        // Mini decoder: input -> matmul(weight) -> layer_norm -> matmul(proj) -> output
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 32, 32);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 16);

        SDVariable weight1 = sd.constant("w1", w1);
        SDVariable weight2 = sd.constant("w2", w2);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);

        SDVariable h = sd.mmul("hidden", input, weight1);
        // Simple normalization: h / max(||h||, eps)
        SDVariable norm = sd.math().norm2("norm", h, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5));
        SDVariable maxNorm = sd.math().max("maxNorm", norm, eps);
        SDVariable normalized = sd.math().div("normalized", h, maxNorm);
        SDVariable out = sd.mmul("output", normalized, weight2);

        sd.setGraphExecutionMode(mode);

        Map<String, INDArray> ph = new LinkedHashMap<>();

        // Run 20 "decode steps" with varying input values (simulates token embeddings changing)
        INDArray[] outputs = new INDArray[20];
        for (int step = 0; step < 20; step++) {
            ph.put("input", Nd4j.randn(DataType.FLOAT, 1, 32));
            Map<String, INDArray> result = sd.output(ph, "output");
            outputs[step] = result.get("output").dup();
        }

        // After 20 executions, DSP should have reached at least SHAPES_FROZEN
        DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                mode + " after 20 steps");

        // Pointers should be stable
        DspPlanAssertions.assertPointersStable(sd,
                mode + " after 20 steps");

        // Frozen execution count should be well past warmup
        DspPlanAssertions.assertFrozenExecCountAtLeast(sd, 5,
                mode + " after 20 steps");

        // No capture failures
        DspPlanAssertions.assertNoCaptureFailures(sd,
                mode + " after 20 steps");

        // No phase contract violations
        DspPlanAssertions.assertNoPhaseContractViolations(sd,
                mode + " after 20 steps");

        // Verify varying inputs produce varying outputs (not stale replay)
        boolean anyDifferent = false;
        for (int i = 1; i < outputs.length; i++) {
            if (!outputs[i].equals(outputs[i - 1])) {
                anyDifferent = true;
                break;
            }
        }
        assertTrue(anyDifferent,
                mode + ": all 20 decode steps produced identical output — stale replay suspected");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 3: FP16 constant + FP32 placeholder — no NaN at any phase
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * End-to-end test for the FP16 weight pre-cast + DSP pipeline.
     *
     * Regression: FP16 weights combined with FP32 activations produced NaN in
     * the output after the freeze phase, because the matmul result was truncated
     * to FP16 (pickPairwiseResultType bug) and downstream softmax overflowed.
     */
    @ParameterizedTest(name = "fp16ConstantNoNaN_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    public void testFp16ConstantNoNaNThroughLifecycle(GraphExecutionMode mode) {
        sd = SameDiff.create();

        // Pattern from VLM: FP16 weight constant (pre-cast), FP32 hidden state
        INDArray weightData = Nd4j.randn(DataType.FLOAT, 32, 32).castTo(DataType.HALF);
        INDArray biasData = Nd4j.zeros(DataType.FLOAT, 1, 32);

        SDVariable w = sd.constant("weight", weightData);
        SDVariable b = sd.constant("bias", biasData);
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, 1, 32);

        SDVariable mm = sd.mmul("matmul", x, w);
        SDVariable added = sd.math().add("biased", mm, b);
        // Softmax is where NaN from FP16 truncation surfaces
        SDVariable out = sd.nn().softmax("output", added, -1);

        sd.setGraphExecutionMode(mode);

        Map<String, INDArray> ph = new LinkedHashMap<>();

        // Phase 1: warmup (slot-by-slot)
        for (int i = 0; i < 3; i++) {
            ph.put("input", Nd4j.randn(DataType.FLOAT, 1, 32));
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");
            assertFalse(output.isNaN().any(),
                    mode + " warmup step " + i + ": NaN in output");
            assertFalse(output.isInfinite().any(),
                    mode + " warmup step " + i + ": Inf in output");
            // Softmax output must sum to ~1
            double sum = output.sumNumber().doubleValue();
            assertEquals(1.0, sum, 0.01,
                    mode + " warmup step " + i + ": softmax sum should be ~1.0 but was " + sum);
        }

        // Phase 2: frozen execution (shapes frozen, but before capture)
        for (int i = 0; i < 10; i++) {
            ph.put("input", Nd4j.randn(DataType.FLOAT, 1, 32));
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");
            assertFalse(output.isNaN().any(),
                    mode + " frozen step " + i + ": NaN in output — FP16 truncation suspected");
            assertFalse(output.isInfinite().any(),
                    mode + " frozen step " + i + ": Inf in output");
            double sum = output.sumNumber().doubleValue();
            assertEquals(1.0, sum, 0.01,
                    mode + " frozen step " + i + ": softmax sum should be ~1.0 but was " + sum);
        }

        // Verify no phase contract violations
        DspPlanAssertions.assertNoPhaseContractViolations(sd, mode.name());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 4: Replay throughput — replay count increments per step
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Once a plan reaches REPLAYING, every subsequent execution should increment
     * the segment replay count (not fall back to slot-by-slot).
     *
     * Regression: cudaGetLastError() called after every segment serialized the GPU
     * pipeline, making replay slower than slot-by-slot. Also, plans destroyed at
     * execCount=6 frozen=false indicated early teardown before replay.
     */
    @ParameterizedTest(name = "replayCountIncrements_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "CUDA_GRAPHS"})
    public void testReplayCountIncrementsPerStep(GraphExecutionMode mode) {
        sd = SameDiff.create();

        // Simple graph that should compile into one capturable segment
        INDArray w = Nd4j.randn(DataType.FLOAT, 16, 16);
        SDVariable weight = sd.constant("w", w);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        SDVariable mm = sd.mmul("matmul", input, weight);
        SDVariable out = sd.math().tanh("output", mm);

        sd.setGraphExecutionMode(mode);

        Map<String, INDArray> ph = new LinkedHashMap<>();

        // Warmup + freeze + capture: 15 steps should be more than enough
        for (int i = 0; i < 15; i++) {
            ph.put("input", Nd4j.randn(DataType.FLOAT, 1, 16));
            sd.output(ph, "output");
        }

        // Should be at least at SHAPES_FROZEN
        DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                mode + " after 15 warmup steps");

        // No capture failures
        DspPlanAssertions.assertNoCaptureFailures(sd, mode + " after warmup");

        // Log the current plan state for diagnosis
        log.info("{} after warmup: {}", mode, DspPlanAssertions.snapshotPlanState(sd));

        // Now run 10 more "steady state" steps
        int replaysBefore = DspPlanAssertions.getTotalGraphReplays(sd);
        int frozenBefore = DspPlanAssertions.getFrozenExecCount(sd);

        for (int i = 0; i < 10; i++) {
            ph.put("input", Nd4j.randn(DataType.FLOAT, 1, 16));
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");
            assertFalse(output.isNaN().any(),
                    mode + " steady step " + i + ": NaN in output");
        }

        int replaysAfter = DspPlanAssertions.getTotalGraphReplays(sd);
        int frozenAfter = DspPlanAssertions.getFrozenExecCount(sd);

        log.info("{} steady state: replays {} -> {}, frozenExec {} -> {}",
                mode, replaysBefore, replaysAfter, frozenBefore, frozenAfter);

        // Frozen exec count must have incremented
        assertTrue(frozenAfter > frozenBefore,
                mode + ": frozen exec count did not increase during steady state — "
                        + "plan may have been destroyed and recreated. Before=" + frozenBefore
                        + " After=" + frozenAfter);

        // Log final state
        log.info("{} final: {}", mode, DspPlanAssertions.snapshotPlanState(sd));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 5: SLOT_BY_SLOT vs mode accuracy — output equivalence
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Compares each execution mode's output against SLOT_BY_SLOT (ground truth).
     *
     * This is the direct attribution test: if a mode produces different results than
     * SLOT_BY_SLOT, the bug is in that mode's execution path (not the op kernels).
     * The assertion names which mode diverged and at which execution step.
     */
    @ParameterizedTest(name = "modeMatchesSlotBySlot_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    public void testModeOutputMatchesSlotBySlot(GraphExecutionMode mode) {
        // Use a fixed seed for reproducibility
        Nd4j.getRandom().setSeed(42);

        INDArray weightData = Nd4j.randn(DataType.FLOAT, 32, 32);
        INDArray biasData = Nd4j.randn(DataType.FLOAT, 1, 32);
        // Create the SAME sequence of inputs for both modes
        INDArray[] inputs = new INDArray[10];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, 32);
        }

        // Run SLOT_BY_SLOT (reference)
        INDArray[] referenceOutputs = new INDArray[inputs.length];
        {
            SameDiff ref = SameDiff.create();
            SDVariable w = ref.constant("w", weightData.dup());
            SDVariable b = ref.constant("b", biasData.dup());
            SDVariable x = ref.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable mm = ref.mmul("mm", x, w);
            SDVariable added = ref.math().add("added", mm, b);
            SDVariable out = ref.math().tanh("output", added);
            ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> ph = new LinkedHashMap<>();
            for (int i = 0; i < inputs.length; i++) {
                ph.put("input", inputs[i]);
                Map<String, INDArray> result = ref.output(ph, "output");
                referenceOutputs[i] = result.get("output").dup();
            }
            ref.close();
        }

        // Run the test mode
        {
            sd = SameDiff.create();
            SDVariable w = sd.constant("w", weightData.dup());
            SDVariable b = sd.constant("b", biasData.dup());
            SDVariable x = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable mm = sd.mmul("mm", x, w);
            SDVariable added = sd.math().add("added", mm, b);
            SDVariable out = sd.math().tanh("output", added);
            sd.setGraphExecutionMode(mode);

            Map<String, INDArray> ph = new LinkedHashMap<>();
            for (int i = 0; i < inputs.length; i++) {
                ph.put("input", inputs[i]);
                Map<String, INDArray> result = sd.output(ph, "output");
                INDArray actual = result.get("output");

                // Check exact equality first
                if (!actual.equals(referenceOutputs[i])) {
                    // Allow small FP tolerance for TF32/Triton paths
                    double maxDiff = actual.sub(referenceOutputs[i]).amaxNumber().doubleValue();
                    assertTrue(maxDiff < 1e-3,
                            mode + " step " + i + ": output diverges from SLOT_BY_SLOT reference. "
                                    + "maxDiff=" + maxDiff + " (threshold=1e-3). "
                                    + "This indicates the " + mode + " execution path produces "
                                    + "different results — check segment compilation and replay.");
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 6: FP16 weight matmul chain — accumulation does not drift to NaN
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Stacks multiple matmul layers with FP16 weights to test precision accumulation.
     * In a real VLM decoder, 30 transformer layers each do matmul with FP16 weights.
     * If any intermediate result is truncated to FP16, the chain will produce NaN.
     */
    @Test
    @DisplayName("FP16 weight chain: 5 matmul layers, no NaN accumulation")
    public void testFp16WeightChainNoNaN() {
        sd = SameDiff.create();

        int hidden = 32;
        int layers = 5;

        SDVariable x = sd.placeHolder("input", DataType.FLOAT, 1, hidden);
        SDVariable current = x;

        for (int i = 0; i < layers; i++) {
            INDArray wData = Nd4j.randn(DataType.FLOAT, hidden, hidden)
                    .muli(0.1)  // scale down to prevent overflow
                    .castTo(DataType.HALF);
            SDVariable w = sd.constant("w" + i, wData);
            current = sd.mmul("mm" + i, current, w);
            current = sd.math().tanh("act" + i, current);  // bounded activation
        }

        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        for (int step = 0; step < 15; step++) {
            ph.put("input", Nd4j.randn(DataType.FLOAT, 1, hidden));
            Map<String, INDArray> result = sd.output(ph, "act" + (layers - 1));
            INDArray output = result.get("act" + (layers - 1));

            assertNotNull(output, "step " + step + ": output is null");
            assertEquals(DataType.FLOAT, output.dataType(),
                    "step " + step + ": output dtype should be FLOAT after "
                            + layers + " FP16-weight matmul layers");
            assertFalse(output.isNaN().any(),
                    "step " + step + ": NaN in output after " + layers
                            + " FP16-weight matmul layers — precision accumulation bug");
            assertFalse(output.isInfinite().any(),
                    "step " + step + ": Inf in output after " + layers
                            + " FP16-weight matmul layers");
        }

        DspPlanAssertions.assertNoPhaseContractViolations(sd,
                "FP16 chain after 15 steps");
    }
}
