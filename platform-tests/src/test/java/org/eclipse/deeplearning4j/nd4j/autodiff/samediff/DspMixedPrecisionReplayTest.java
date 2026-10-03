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
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
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
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

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

    /** Triton GPU segments and device-buffer contracts exist only on the CUDA backend. */
    private static void assumeCuda() {
        assumeTrue(Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA,
                "Triton GPU and device-buffer contracts need the CUDA backend");
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

    @Test
    @DisplayName("DSP shape pre-pass preserves explicit HALF to FLOAT cast dtype")
    public void testShapePrePassPreservesExplicitCastDtype() {
        sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.HALF, 1, 1024);
        SDVariable castFloat = input.castTo("cast_float", DataType.FLOAT);
        SDVariable weightA = sd.constant("weight_a", Nd4j.randn(DataType.FLOAT, 1024, 16));
        SDVariable weightB = sd.constant("weight_b", Nd4j.randn(DataType.FLOAT, 1024, 16));
        SDVariable projectedA = sd.mmul("projected_a", castFloat, weightA);
        SDVariable projectedB = sd.mmul("projected_b", castFloat, weightB);
        SDVariable output = projectedA.add("output", projectedB);

        sd.setGraphExecutionMode(GraphExecutionMode.TRITON);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", Nd4j.randn(DataType.HALF, 1, 1024));

        for (int step = 0; step < 2; step++) {
            Map<String, INDArray> result = sd.output(ph, "cast_float", "output");
            assertEquals(DataType.FLOAT, result.get("cast_float").dataType(),
                    "DSP shape pre-pass must retain the cast op's declared FLOAT output");
            assertEquals(DataType.FLOAT, result.get("output").dataType(),
                    "Downstream batched matmuls must use the cast op's FLOAT dtype");
            assertFalse(result.get("output").isNaN().any(),
                    "Mixed-precision output must remain finite at step " + step);
        }

        DspPlanAssertions.assertNoSegmentFailures(sd, "explicitHalfToFloatCast");
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
        // Reuse the same INDArray object — DSP pointer stability requires stable buffer
        // addresses between executions. Creating new INDArray objects each step means
        // argTableStable can never become true, blocking CUDA graph capture/replay.
        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 32);
        ph.put("input", inputArr);

        // Run 30 "decode steps" with varying input values (simulates token embeddings changing).
        // norm2/div/max ops need extra warmup steps for pointer stability compared to
        // simple matmul-only graphs (the reduction ops cause intermediate buffer reallocation).
        int totalSteps = 30;
        INDArray[] outputs = new INDArray[totalSteps];
        for (int step = 0; step < totalSteps; step++) {
            inputArr.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
            Map<String, INDArray> result = sd.output(ph, "output");
            outputs[step] = result.get("output").dup();
        }

        // After sufficient executions, DSP should have reached at least SHAPES_FROZEN
        DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                mode + " after " + totalSteps + " steps");

        // Pointers should be stable
        DspPlanAssertions.assertPointersStable(sd,
                mode + " after " + totalSteps + " steps");

        // Frozen execution count should be well past warmup
        DspPlanAssertions.assertFrozenExecCountAtLeast(sd, 5,
                mode + " after " + totalSteps + " steps");

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

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 7: Triton reduction must reproduce native CUDA accumulation order
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Regression for the fixed-buffer reuse divergence first observed at Qwen
     * {@code gdn_k_normsq_0}: 16 rows each reduce 64 FLOAT values. Native CUDA
     * uses 32 strided partial sums followed by a fixed binary tree, while the
     * Triton section used to perform a sequential Kahan sum. Both are stable,
     * but they differ by one ULP for this cancellation-sensitive exponent pattern.
     */
    @Test
    @DisplayName("Triton reduce_sum [16,64] matches native CUDA raw bits")
    public void testTritonReductionMatchesNativeTreeExactly() {
        assumeCuda();
        final int rows = 16;
        final int reductionSize = 64;
        float[] values = new float[rows * reductionSize];
        for (int row = 0; row < rows; row++) {
            for (int k = 0; k < reductionSize; k++) {
                values[row * reductionSize + k] =
                        Math.scalb(1.0f, ((k * 7) & 31) - 16);
            }
        }
        INDArray inputData = Nd4j.createFromArray(values).reshape(rows, reductionSize);

        INDArray reference;
        try (SameDiff ref = SameDiff.create()) {
            SDVariable input = ref.placeHolder("input", DataType.FLOAT, rows, reductionSize);
            SDVariable summed = ref.math().sum("summed", input, 1);
            SDVariable zero = ref.constant("zero", Nd4j.scalar(DataType.FLOAT, 0.0f));
            ref.math().add("output", summed, zero);
            ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("input", inputData);
            reference = ref.output(placeholders, "output").get("output").dup();
        }

        for (int row = 0; row < rows; row++) {
            int referenceBits = Float.floatToRawIntBits(reference.getFloat(row));
            assertEquals(0x47ffffff, referenceBits,
                    "row " + row + ": discriminator no longer exercises the native 32-lane tree");
        }

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            // This is the same REDUCTION inclusion used by the production OPTIMAL
            // configuration that exposed the Qwen reuse divergence.
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("REDUCTION,ELEMENTWISE");

            sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, rows, reductionSize);
            SDVariable summed = sd.math().sum("summed", input, 1);
            SDVariable zero = sd.constant("zero", Nd4j.scalar(DataType.FLOAT, 0.0f));
            sd.math().add("output", summed, zero);
            sd.setGraphExecutionMode(GraphExecutionMode.TRITON);

            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("input", inputData);

            for (int step = 0; step < 30; step++) {
                INDArray actual = sd.output(placeholders, "output").get("output");
                for (int row = 0; row < rows; row++) {
                    int expectedBits = Float.floatToRawIntBits(reference.getFloat(row));
                    int actualBits = Float.floatToRawIntBits(actual.getFloat(row));
                    assertEquals(expectedBits, actualBits,
                            String.format("TRITON step %d row %d: native=0x%08x triton=0x%08x",
                                    step, row, expectedBits, actualBits));
                }
            }

            DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                    "TRITON exact reduction parity after 30 steps");
            DspPlanAssertions.assertNoPhaseContractViolations(sd,
                    "TRITON exact reduction parity");
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 8: full Qwen GDN K-normalization chain must be bit-exact
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Numeric discriminator for the first current-binary divergence in fixed-buffer
     * reuse. The comparable warmups have identical input embeddings, layer-0 QKV,
     * reshaped K, and recurrent-state input, but differ after the production
     * {@code square -> reduce_sum(keepDims) -> add epsilon -> sqrt -> divide} chain.
     */
    @Test
    @DisplayName("Triton GDN K-normalization [1,1,16,128] matches native CUDA raw bits")
    public void testTritonGdnKNormalizationMatchesNativeExactly() {
        final int rows = 16;
        final int headDim = 128;
        float[] values = new float[rows * headDim];
        for (int row = 0; row < rows; row++) {
            for (int k = 0; k < headDim; k++) {
                float mantissa = 1.0f + (((k * 13) + (row * 7)) & 31) / 64.0f;
                float value = Math.scalb(mantissa, (((k * 5) + (row * 3)) & 15) - 8);
                values[row * headDim + k] = ((k + row) & 1) == 0 ? value : -value;
            }
        }
        INDArray inputData = Nd4j.createFromArray(values).reshape(1, 1, rows, headDim);

        INDArray reference;
        try (SameDiff ref = SameDiff.create()) {
            SDVariable input = ref.placeHolder("input", DataType.FLOAT, 1, 1, rows, headDim);
            SDVariable inputF32 = input.castTo("input_f32", DataType.FLOAT);
            SDVariable normSq = inputF32.mul(inputF32).sum("norm_sq", true, -1);
            SDVariable norm = ref.math.sqrt("norm", normSq.add(1e-6));
            input.div("output", norm.castTo("norm_cast", input.dataType()));
            ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("input", inputData);
            reference = ref.output(placeholders, "output").get("output").dup();
        }

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("REDUCTION,ELEMENTWISE");

            sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 1, rows, headDim);
            SDVariable inputF32 = input.castTo("input_f32", DataType.FLOAT);
            SDVariable normSq = inputF32.mul(inputF32).sum("norm_sq", true, -1);
            SDVariable norm = sd.math.sqrt("norm", normSq.add(1e-6));
            input.div("output", norm.castTo("norm_cast", input.dataType()));
            sd.setGraphExecutionMode(GraphExecutionMode.TRITON);

            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("input", inputData);

            for (int step = 0; step < 30; step++) {
                INDArray actual = sd.output(placeholders, "output").get("output");
                for (int i = 0; i < values.length; i++) {
                    int expectedBits = Float.floatToRawIntBits(reference.getFloat(i));
                    int actualBits = Float.floatToRawIntBits(actual.getFloat(i));
                    assertEquals(expectedBits, actualBits,
                            String.format("TRITON step %d element %d: native=0x%08x triton=0x%08x",
                                    step, i, expectedBits, actualBits));
                }
            }

            DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                    "TRITON exact GDN K-normalization parity after 30 steps");
            DspPlanAssertions.assertNoPhaseContractViolations(sd,
                    "TRITON exact GDN K-normalization parity");
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    @Test
    @DisplayName("Triton GDN K-normalization production prefill shape matches native CUDA raw bits")
    public void testTritonGdnKNormalizationMatchesNativeAtProductionPrefillShape() {
        final int sequence = 662;
        final int heads = 16;
        final int headDim = 128;
        final int length = sequence * heads * headDim;
        float[] values = new float[length];
        for (int row = 0; row < sequence * heads; row++) {
            for (int k = 0; k < headDim; k++) {
                float mantissa = 1.0f + (((k * 13) + (row * 7)) & 31) / 64.0f;
                float value = Math.scalb(mantissa, (((k * 5) + (row * 3)) & 15) - 8);
                values[row * headDim + k] = ((k + row) & 1) == 0 ? value : -value;
            }
        }

        INDArray inputData;
        try (INDArray floatInput = Nd4j.createFromArray(values)
                .reshape(1, sequence, heads, headDim)) {
            inputData = floatInput.castTo(DataType.HALF);
        }
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);

        INDArray reference;
        try (SameDiff ref = SameDiff.create()) {
            SDVariable input = ref.placeHolder(
                    "input", DataType.HALF, 1, sequence, heads, headDim);
            SDVariable inputF32 = input.castTo("input_f32", DataType.FLOAT);
            SDVariable normSq = inputF32.mul(inputF32).sum("norm_sq", true, -1);
            SDVariable norm = ref.math.sqrt("norm", normSq.add(1e-6));
            SDVariable normalized = input.div(
                    "normalized", norm.castTo("norm_cast", input.dataType()));
            normalized.castTo("output", DataType.FLOAT);
            ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            reference = ref.output(placeholders, "output").get("output").dup('c');
        }
        float[] expected = reference.data().asFloat();

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        boolean alwaysCompileBefore = environment.tritonAlwaysCompile();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonAlwaysCompile(true);
            environment.setTritonIncludeTypes("REDUCTION,ELEMENTWISE");

            sd = SameDiff.create();
            SDVariable input = sd.placeHolder(
                    "input", DataType.HALF, 1, sequence, heads, headDim);
            SDVariable inputF32 = input.castTo("input_f32", DataType.FLOAT);
            SDVariable normSq = inputF32.mul(inputF32).sum("norm_sq", true, -1);
            SDVariable norm = sd.math.sqrt("norm", normSq.add(1e-6));
            SDVariable normalized = input.div(
                    "normalized", norm.castTo("norm_cast", input.dataType()));
            normalized.castTo("output", DataType.FLOAT);
            sd.setGraphExecutionMode(GraphExecutionMode.TRITON);

            for (int step = 0; step < 4; step++) {
                INDArray actual = sd.output(placeholders, "output").get("output");
                float[] actualValues = actual.data().asFloat();
                assertEquals(expected.length, actualValues.length,
                        "Production-shaped normalization output length changed");
                for (int i = 0; i < expected.length; i++) {
                    int expectedBits = Float.floatToRawIntBits(expected[i]);
                    int actualBits = Float.floatToRawIntBits(actualValues[i]);
                    if (expectedBits != actualBits) {
                        int row = i / headDim;
                        int dimension = i % headDim;
                        int head = row % heads;
                        int token = row / heads;
                        fail(String.format(
                                "production K-normalization step %d token %d head %d dimension %d: "
                                        + "native=0x%08x triton=0x%08x",
                                step, token, head, dimension, expectedBits, actualBits));
                    }
                }
            }

            DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                    "TRITON production-shaped GDN K-normalization parity");
            DspPlanAssertions.assertNoPhaseContractViolations(sd,
                    "TRITON production-shaped GDN K-normalization parity");
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonAlwaysCompile(alwaysCompileBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            reference.close();
            inputData.close();
        }
    }

    @Test
    @DisplayName("Standalone Triton reduction rewrites compact outputs across the 4096 tile boundary")
    public void testTritonReductionOutputTileBoundaryFreshness() {
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        boolean alwaysCompileBefore = environment.tritonAlwaysCompile();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonAlwaysCompile(true);
            environment.setTritonIncludeTypes("REDUCTION,ELEMENTWISE");
            for (int rows : new int[]{4095, 4096, 4097, 8193}) {
                try (INDArray input = Nd4j.zeros(DataType.FLOAT, rows, 128);
                     SameDiff triton = SameDiff.create()) {
                    triton.placeHolder("input", DataType.FLOAT, rows, 128).sum("sum", true, -1);
                    triton.setGraphExecutionMode(GraphExecutionMode.TRITON);
                    Map<String, INDArray> placeholders = new LinkedHashMap<>();
                    placeholders.put("input", input);
                    for (int step = 0; step < 5; step++) {
                        // Capture with zeros, then change the actual producer input.
                        // Reading a copied output or retaining warmup data cannot pass.
                        float value = step < 3 ? 0.0f : step - 2.0f;
                        input.assign(value);
                        INDArray actual = triton.output(placeholders, "sum").get("sum");
                        assertEquals(DataType.FLOAT, actual.dataType());
                        assertArrayEquals(new long[]{rows, 1}, actual.shape());
                        float[] actualValues = actual.data().asFloat();
                        for (int row = 0; row < rows; row++) {
                            assertEquals(Float.floatToRawIntBits(value * 128),
                                    Float.floatToRawIntBits(actualValues[row]),
                                    "rows=" + rows + " step=" + step + " row=" + row);
                        }
                    }
                    DspPlanAssertions.assertPhaseReached(triton, PlanPhase.SHAPES_FROZEN,
                            "Standalone reduction tile boundary rows=" + rows);
                    DspPlanAssertions.assertNoPhaseContractViolations(triton,
                            "Standalone reduction tile boundary rows=" + rows);
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonAlwaysCompile(alwaysCompileBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    /** Diagnostic prefixes retain their requested output, not a dead aliased slot. */
    @Test
    @DisplayName("Attribute production HALF K-normalization by live requested prefixes")
    public void testTritonGdnKNormalizationProductionStageAttribution() {
        runProductionKNormalizationPrefixes(false);
    }

    @Test
    @DisplayName("Production K sum must overwrite every live output row during replay")
    public void testTritonGdnKNormalizationProductionReductionFullWrite() {
        runProductionKNormalizationPrefixes(true);
    }

    private void runProductionKNormalizationPrefixes(boolean zeroWarmup) {
        final int sequence = 662, heads = 16, headDim = 128;
        float[] values = new float[sequence * heads * headDim];
        for (int row = 0; row < sequence * heads; row++) {
            for (int k = 0; k < headDim; k++) {
                float mantissa = 1.0f + (((k * 13) + (row * 7)) & 31) / 64.0f;
                float value = Math.scalb(mantissa, (((k * 5) + (row * 3)) & 15) - 8);
                values[row * headDim + k] = ((k + row) & 1) == 0 ? value : -value;
            }
        }
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        boolean alwaysCompileBefore = environment.tritonAlwaysCompile();
        String includeTypesBefore = environment.tritonIncludeTypes();
        java.util.List<String> mismatches = new java.util.ArrayList<>();
        try (INDArray floatInput = Nd4j.createFromArray(values).reshape(1, sequence, heads, headDim);
             INDArray inputData = floatInput.castTo(DataType.HALF)) {
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("input", inputData);
            environment.setTritonCompileAll(true);
            environment.setTritonAlwaysCompile(true);
            environment.setTritonIncludeTypes("REDUCTION,ELEMENTWISE");
            String[] stages = zeroWarmup ? new String[]{"input_f32", "square", "norm_sq"}
                    : new String[]{"input_f32", "square", "norm_sq", "with_epsilon",
                            "norm", "norm_cast", "normalized", "output"};
            for (String stage : stages) {
                inputData.assign(floatInput);
                float[] expected;
                DataType expectedType;
                try (SameDiff ref = productionKNormalizationDiagnosticGraph(GraphExecutionMode.SLOT_BY_SLOT)) {
                    INDArray result = ref.output(placeholders, stage).get(stage);
                    expectedType = result.dataType();
                    expected = result.data().asFloat();
                }
                float[] warmupExpected = expected;
                if (zeroWarmup) {
                    inputData.assign(0);
                    try (SameDiff ref = productionKNormalizationDiagnosticGraph(GraphExecutionMode.SLOT_BY_SLOT)) {
                        warmupExpected = ref.output(placeholders, stage).get(stage).data().asFloat();
                    }
                }
                try (SameDiff triton = productionKNormalizationDiagnosticGraph(GraphExecutionMode.TRITON)) {
                    for (int step = 0; step < 4; step++) {
                        // Warm up AND capture with zeros; measured replay uses the exact
                        // original HALF production input and its native raw-bit oracle.
                        if (zeroWarmup && step == 3) {
                            inputData.assign(floatInput);
                            assertArrayEquals(values, inputData.data().asFloat(), "Exact production replay input");
                            System.out.println("KPROD_FRESHNESS restored exact production input before step 3 stage=" + stage);
                        }
                        float[] expectedThisStep = zeroWarmup && step < 3 ? warmupExpected : expected;
                        INDArray actual = triton.output(placeholders, stage).get(stage);
                        assertEquals(expectedType, actual.dataType(), stage + " dtype");
                        float[] actualValues = actual.data().asFloat();
                        assertEquals(expected.length, actualValues.length, stage + " length");
                        int first = -1, count = 0;
                        for (int i = 0; i < expected.length; i++) {
                            if (Float.floatToRawIntBits(expectedThisStep[i]) != Float.floatToRawIntBits(actualValues[i])) {
                                if (first < 0) first = i;
                                count++;
                            }
                        }
                        String report = String.format("KPROD_PREFIX stage=%s step=%d dtype=%s count=%d first=%d",
                                stage, step, actual.dataType(), count, first);
                        if (first >= 0) {
                            report += String.format(" native=0x%08x triton=0x%08x",
                                    Float.floatToRawIntBits(expectedThisStep[first]), Float.floatToRawIntBits(actualValues[first]));
                            mismatches.add(report);
                        }
                        System.out.println(report);
                    }
                }
            }
            assertTrue(mismatches.isEmpty(), String.join("; ", mismatches));
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonAlwaysCompile(alwaysCompileBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    private static SameDiff productionKNormalizationDiagnosticGraph(GraphExecutionMode mode) {
        SameDiff graph = SameDiff.create();
        SDVariable input = graph.placeHolder("input", DataType.HALF, 1, 662, 16, 128);
        SDVariable inputF32 = input.castTo("input_f32", DataType.FLOAT);
        SDVariable square = inputF32.mul("square", inputF32);
        SDVariable normSq = square.sum("norm_sq", true, -1);
        SDVariable withEpsilon = normSq.add("with_epsilon", 1e-6);
        SDVariable norm = graph.math.sqrt("norm", withEpsilon);
        SDVariable normalized = input.div("normalized", norm.castTo("norm_cast", input.dataType()));
        normalized.castTo("output", DataType.FLOAT);
        graph.setGraphExecutionMode(mode);
        return graph;
    }

    @Test
    @DisplayName("Attribute GDN K-normalization mismatch to its first arithmetic stage")
    public void testTritonGdnKNormalizationStageAttribution() {
        final int rows = 16;
        final int headDim = 128;
        float[] values = new float[rows * headDim];
        for (int row = 0; row < rows; row++) {
            for (int k = 0; k < headDim; k++) {
                float mantissa = 1.0f + (((k * 13) + (row * 7)) & 31) / 64.0f;
                float value = Math.scalb(mantissa, (((k * 5) + (row * 3)) & 15) - 8);
                values[row * headDim + k] = ((k + row) & 1) == 0 ? value : -value;
            }
        }
        INDArray inputData = Nd4j.createFromArray(values).reshape(1, 1, rows, headDim);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("REDUCTION,ELEMENTWISE");

            java.util.List<String> mismatches = new java.util.ArrayList<>();
            for (String stage : new String[]{"norm_sq", "with_epsilon", "norm", "output"}) {
                INDArray reference;
                try (SameDiff ref = SameDiff.create()) {
                    SDVariable input = ref.placeHolder("input", DataType.FLOAT, 1, 1, rows, headDim);
                    SDVariable inputF32 = input.castTo("input_f32", DataType.FLOAT);
                    SDVariable normSq = inputF32.mul(inputF32).sum("norm_sq", true, -1);
                    SDVariable withEpsilon = normSq.add("with_epsilon", 1e-6);
                    SDVariable norm = ref.math.sqrt("norm", withEpsilon);
                    input.div("output", norm.castTo("norm_cast", input.dataType()));
                    ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                    reference = ref.output(placeholders, stage).get(stage).dup();
                }

                try (SameDiff triton = SameDiff.create()) {
                    SDVariable input = triton.placeHolder("input", DataType.FLOAT, 1, 1, rows, headDim);
                    SDVariable inputF32 = input.castTo("input_f32", DataType.FLOAT);
                    SDVariable normSq = inputF32.mul(inputF32).sum("norm_sq", true, -1);
                    SDVariable withEpsilon = normSq.add("with_epsilon", 1e-6);
                    SDVariable norm = triton.math.sqrt("norm", withEpsilon);
                    input.div("output", norm.castTo("norm_cast", input.dataType()));
                    triton.setGraphExecutionMode(GraphExecutionMode.TRITON);

                    boolean found = false;
                    for (int step = 0; step < 4 && !found; step++) {
                        INDArray actual = triton.output(placeholders, stage).get(stage);
                        for (int i = 0; i < reference.length(); i++) {
                            int expectedBits = Float.floatToRawIntBits(reference.getFloat(i));
                            int actualBits = Float.floatToRawIntBits(actual.getFloat(i));
                            if (expectedBits != actualBits) {
                                mismatches.add(String.format(
                                        "%s step %d element %d: native=0x%08x triton=0x%08x",
                                        stage, step, i, expectedBits, actualBits));
                                found = true;
                                break;
                            }
                        }
                    }
                }
            }

            assertTrue(mismatches.isEmpty(),
                    "First mismatch per requested stage: " + String.join("; ", mismatches));
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    /**
     * Stage-attribution regression for Qwen's beta path. Production tracing proves
     * the projection is bit-exact while the immediately following sigmoid differs
     * between native slot-by-slot execution and Triton replay. Compare exp(-x)
     * separately so a failure identifies libdevice-exp versus final division.
     */
    @Test
    @DisplayName("Triton sigmoid and exp(-x) match native CUDA raw bits")
    public void testTritonSigmoidStagesMatchNativeExactly() {
        float[] projectionValues = new float[]{
                -10.0f, -8.0f, -4.0f, -2.0f, -1.92597f, -1.0f, -0.5f, -0.1f,
                0.0f, 0.1f, 0.5f, 1.0f, 1.92597f, 2.0f, 4.0f, 8.0f
        };
        INDArray inputData = Nd4j.createFromArray(projectionValues).reshape(1, projectionValues.length);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ELEMENTWISE");

            java.util.List<String> mismatches = new java.util.ArrayList<>();
            for (String stage : new String[]{"exp_neg", "sigmoid"}) {
                INDArray reference;
                try (SameDiff nativeGraph = SameDiff.create()) {
                    SDVariable input = nativeGraph.placeHolder(
                            "input", DataType.FLOAT, 1, projectionValues.length);
                    SDVariable negInput = input.neg("neg_input");
                    nativeGraph.math.exp("exp_neg", negInput);
                    nativeGraph.nn.sigmoid("sigmoid", input);
                    nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                    reference = nativeGraph.output(placeholders, stage).get(stage).dup();
                }

                try (SameDiff tritonGraph = SameDiff.create()) {
                    SDVariable input = tritonGraph.placeHolder(
                            "input", DataType.FLOAT, 1, projectionValues.length);
                    SDVariable negInput = input.neg("neg_input");
                    tritonGraph.math.exp("exp_neg", negInput);
                    tritonGraph.nn.sigmoid("sigmoid", input);
                    tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                    boolean found = false;
                    for (int step = 0; step < 4 && !found; step++) {
                        INDArray actual = tritonGraph.output(placeholders, stage).get(stage);
                        for (int i = 0; i < reference.length(); i++) {
                            int expectedBits = Float.floatToRawIntBits(reference.getFloat(i));
                            int actualBits = Float.floatToRawIntBits(actual.getFloat(i));
                            if (expectedBits != actualBits) {
                                mismatches.add(String.format(
                                        "%s step %d element %d: native=0x%08x triton=0x%08x",
                                        stage, step, i, expectedBits, actualBits));
                                found = true;
                                break;
                            }
                        }
                    }
                }
            }

            assertTrue(mismatches.isEmpty(),
                    "First sigmoid-stage mismatches: " + String.join("; ", mismatches));
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    /**
     * Qwen's recurrent beta path applies softplus before the gated-delta-rule
     * update. The first compiled Triton execution must reproduce native CUDA's
     * stable max + logf(1 + expf(-abs(x))) implementation exactly.
     */
    @Test
    @DisplayName("Triton softplus matches native CUDA raw bits")
    public void testTritonSoftplusMatchesNativeExactly() {
        float[] values = new float[]{
                -20.0f, -10.0f, -8.0f, -4.0f, -2.0f, -1.92597f, -1.0f, -0.5f,
                -0.1f, -0.01f, 0.0f, 0.01f, 0.1f, 0.5f, 1.0f, 1.92597f,
                2.0f, 4.0f, 8.0f, 10.0f, 20.0f
        };
        INDArray inputData = Nd4j.createFromArray(values).reshape(1, values.length);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);

        INDArray reference;
        try (SameDiff nativeGraph = SameDiff.create()) {
            SDVariable input = nativeGraph.placeHolder("input", DataType.FLOAT, 1, values.length);
            nativeGraph.nn.softplus("softplus", input);
            nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            reference = nativeGraph.output(placeholders, "softplus").get("softplus").dup();
        }

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ELEMENTWISE");

            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable input = tritonGraph.placeHolder("input", DataType.FLOAT, 1, values.length);
                tritonGraph.nn.softplus("softplus", input);
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < 4; step++) {
                    INDArray actual = tritonGraph.output(placeholders, "softplus").get("softplus");
                    for (int i = 0; i < reference.length(); i++) {
                        int expectedBits = Float.floatToRawIntBits(reference.getFloat(i));
                        int actualBits = Float.floatToRawIntBits(actual.getFloat(i));
                        assertEquals(expectedBits, actualBits,
                                String.format("softplus step %d element %d: native=0x%08x triton=0x%08x",
                                        step, i, expectedBits, actualBits));
                    }
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            reference.close();
            inputData.close();
        }
    }

    /**
     * Production beta projections are eligible for MATMUL_EPILOGUE fusion. K=1
     * makes the projection itself exact and leaves only the fused sigmoid math
     * under test.
     */
    @Test
    @DisplayName("Section-fused matmul sigmoid matches native CUDA raw bits")
    public void testTritonMatmulSigmoidEpilogueMatchesNativeExactly() {
        float[] projectionValues = new float[]{
                -10.0f, -8.0f, -4.0f, -2.0f, -1.92597f, -1.0f, -0.5f, -0.1f,
                0.0f, 0.1f, 0.5f, 1.0f, 1.92597f, 2.0f, 4.0f, 8.0f
        };
        INDArray inputData = Nd4j.ones(DataType.FLOAT, 1, 1);
        INDArray weightData = Nd4j.createFromArray(projectionValues)
                .reshape(1, projectionValues.length);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);
        placeholders.put("weights", weightData);

        INDArray reference;
        try (SameDiff nativeGraph = SameDiff.create()) {
            SDVariable input = nativeGraph.placeHolder("input", DataType.FLOAT, 1, 1);
            SDVariable weights = nativeGraph.placeHolder(
                    "weights", DataType.FLOAT, 1, projectionValues.length);
            SDVariable projection = nativeGraph.mmul("projection", input, weights);
            nativeGraph.nn.sigmoid("sigmoid", projection);
            nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            reference = nativeGraph.output(placeholders, "sigmoid").get("sigmoid").dup();
        }

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        boolean sectionFusionBefore = environment.tritonSectionFusion();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("MATMUL,ELEMENTWISE");
            environment.setTritonSectionFusion(true);

            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable input = tritonGraph.placeHolder("input", DataType.FLOAT, 1, 1);
                SDVariable weights = tritonGraph.placeHolder(
                        "weights", DataType.FLOAT, 1, projectionValues.length);
                SDVariable projection = tritonGraph.mmul("projection", input, weights);
                tritonGraph.nn.sigmoid("sigmoid", projection);
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < 4; step++) {
                    INDArray actual = tritonGraph.output(placeholders, "sigmoid").get("sigmoid");
                    for (int i = 0; i < reference.length(); i++) {
                        int expectedBits = Float.floatToRawIntBits(reference.getFloat(i));
                        int actualBits = Float.floatToRawIntBits(actual.getFloat(i));
                        assertEquals(expectedBits, actualBits,
                                String.format("fused step %d element %d: native=0x%08x triton=0x%08x",
                                        step, i, expectedBits, actualBits));
                    }
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            environment.setTritonSectionFusion(sectionFusionBefore);
        }
    }

    /**
     * The Qwen GDN gate uses standalone swish, whose native CUDA contract computes
     * {@code x * sigmoid(x)} as two rounded operations. Keep the compiled path raw-bit
     * identical so tiny gate differences do not amplify through recurrent layers.
     */
    @Test
    @DisplayName("Triton standalone swish matches native CUDA raw bits")
    public void testTritonStandaloneSwishMatchesNativeExactly() {
        assumeCuda();
        float[] inputValues = new float[]{
                -10.0f, -8.0f, -4.0f, -2.0f, -1.92597f, -1.0f, -0.5f, -0.1f,
                0.0f, 0.1f, 0.5f, 1.0f, 1.92597f, 2.0f, 4.0f, 8.0f
        };
        INDArray inputData = Nd4j.createFromArray(inputValues).reshape(1, inputValues.length);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);

        INDArray reference = null;
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            try (SameDiff nativeGraph = SameDiff.create()) {
                SDVariable input = nativeGraph.placeHolder("input", DataType.FLOAT, 1, inputValues.length);
                nativeGraph.nn.swish("swish", input);
                nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                reference = nativeGraph.output(placeholders, "swish").get("swish").dup();
            }

            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ELEMENTWISE");

            long totalMismatches = 0;
            StringBuilder differences = new StringBuilder();
            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable input = tritonGraph.placeHolder("input", DataType.FLOAT, 1, inputValues.length);
                tritonGraph.nn.swish("swish", input);
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                float[] expected = reference.toFloatVector();
                for (int step = 0; step < 4; step++) {
                    float[] actual = tritonGraph.output(placeholders, "swish")
                            .get("swish").toFloatVector();
                    long stepMismatches = 0;
                    for (int i = 0; i < expected.length; i++) {
                        int expectedBits = Float.floatToRawIntBits(expected[i]);
                        int actualBits = Float.floatToRawIntBits(actual[i]);
                        if (expectedBits != actualBits) {
                            stepMismatches++;
                            if (differences.length() < 512) {
                                differences.append(" step=").append(step)
                                        .append(" element=").append(i)
                                        .append(" native=0x").append(Integer.toHexString(expectedBits))
                                        .append(" triton=0x").append(Integer.toHexString(actualBits));
                            }
                        }
                    }
                    totalMismatches += stepMismatches;
                    log.info("STANDALONE_SWISH_EXACT step={} mismatches={}/{}",
                            step, stepMismatches, expected.length);
                }

                DspPlanAssertions.assertOpCompiled(
                        tritonGraph, "swish", "standalone swish exactness");
                DspPlanAssertions.assertAllSegmentsCompiledWith(
                        tritonGraph, "Triton GPU", "standalone swish exactness");
            }
            assertEquals(0L, totalMismatches,
                    "Standalone swish changed raw bits after Triton compilation:" + differences);
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            if (reference != null && !reference.wasClosed()) reference.close();
            inputData.close();
        }
    }

    /**
     * Qwen3.5 uses partial rotary embeddings on both full Q heads and GQA K heads.
     * Cover that pointer emitter and a full-head geometry that exercises the SSA
     * emitter. Both compiled paths must preserve native CUDA operation order and
     * raw bits across DSP warmup, compilation, and replay.
     */
    @Test
    @DisplayName("Triton fused RoPE pointer and SSA paths match native CUDA raw bits")
    public void testTritonFusedRoPEMatchesNativeExactly() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 64;
        final int qHeads = 8;
        final int kvHeads = 2;
        final int headDim = 256;
        final int rotaryDims = 64;
        final int fullHeads = 8;
        final int fullHeadDim = 64;
        final double frequencyBase = 10_000_000.0;

        INDArray qData = Nd4j.linspace(
                DataType.FLOAT, -2.0, 0.00003125, batch * sequence * qHeads * headDim)
                .reshape(batch, sequence, qHeads, headDim);
        INDArray kData = Nd4j.linspace(
                DataType.FLOAT, 1.5, -0.0000625, batch * sequence * kvHeads * headDim)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray fullData = Nd4j.linspace(
                DataType.FLOAT, -0.75, 0.000045, batch * sequence * fullHeads * fullHeadDim)
                .reshape(batch, sequence, fullHeads, fullHeadDim);
        INDArray positionData = Nd4j.scalar(DataType.INT64, 0L);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("q", qData);
        placeholders.put("k", kData);
        placeholders.put("full", fullData);
        placeholders.put("position", positionData);

        INDArray referenceQ = null;
        INDArray referenceK = null;
        INDArray referenceFull = null;
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            try (SameDiff nativeGraph = SameDiff.create()) {
                SDVariable q = nativeGraph.placeHolder(
                        "q", DataType.FLOAT, batch, sequence, qHeads, headDim);
                SDVariable k = nativeGraph.placeHolder(
                        "k", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable full = nativeGraph.placeHolder(
                        "full", DataType.FLOAT, batch, sequence, fullHeads, fullHeadDim);
                SDVariable position = nativeGraph.placeHolder("position", DataType.INT64);
                nativeGraph.nn().fusedRoPE(
                        "q_rope", q, position, 0, frequencyBase, 1.0, rotaryDims);
                nativeGraph.nn().fusedRoPE(
                        "k_rope", k, position, 0, frequencyBase, 1.0, rotaryDims);
                nativeGraph.nn().fusedRoPE(
                        "full_rope", full, position, 0, frequencyBase, 1.0, fullHeadDim);
                nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

                Map<String, INDArray> nativeOutputs =
                        nativeGraph.output(placeholders, "q_rope", "k_rope", "full_rope");
                referenceQ = nativeOutputs.get("q_rope").dup();
                referenceK = nativeOutputs.get("k_rope").dup();
                referenceFull = nativeOutputs.get("full_rope").dup();
            }

            // Partial RoPE is a fully-writing op: every unrotated tail element must
            // pass through unchanged before compiled-path parity is considered.
            long nativeTailMismatches = 0;
            INDArray[] inputs = new INDArray[]{qData, kData};
            INDArray[] references = new INDArray[]{referenceQ, referenceK};
            for (int geometry = 0; geometry < inputs.length; geometry++) {
                float[] inputValues = inputs[geometry].toFloatVector();
                float[] referenceValues = references[geometry].toFloatVector();
                for (int i = 0; i < inputValues.length; i++) {
                    if (i % headDim >= rotaryDims
                            && Float.floatToRawIntBits(inputValues[i])
                            != Float.floatToRawIntBits(referenceValues[i])) {
                        nativeTailMismatches++;
                    }
                }
            }
            assertEquals(0L, nativeTailMismatches,
                    "Native fused RoPE did not preserve the unrotated tail");

            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ELEMENTWISE");

            long totalMismatches = 0;
            StringBuilder differences = new StringBuilder();
            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable q = tritonGraph.placeHolder(
                        "q", DataType.FLOAT, batch, sequence, qHeads, headDim);
                SDVariable k = tritonGraph.placeHolder(
                        "k", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable full = tritonGraph.placeHolder(
                        "full", DataType.FLOAT, batch, sequence, fullHeads, fullHeadDim);
                SDVariable position = tritonGraph.placeHolder("position", DataType.INT64);
                tritonGraph.nn().fusedRoPE(
                        "q_rope", q, position, 0, frequencyBase, 1.0, rotaryDims);
                tritonGraph.nn().fusedRoPE(
                        "k_rope", k, position, 0, frequencyBase, 1.0, rotaryDims);
                tritonGraph.nn().fusedRoPE(
                        "full_rope", full, position, 0, frequencyBase, 1.0, fullHeadDim);
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                String[] names = new String[]{"q_rope", "k_rope", "full_rope"};
                float[][] expectedValues = new float[][]{
                        referenceQ.toFloatVector(), referenceK.toFloatVector(),
                        referenceFull.toFloatVector()
                };
                for (int step = 0; step < 4; step++) {
                    Map<String, INDArray> outputs = tritonGraph.output(placeholders, names);
                    for (int geometry = 0; geometry < names.length; geometry++) {
                        float[] expected = expectedValues[geometry];
                        float[] actual = outputs.get(names[geometry]).toFloatVector();
                        long mismatches = 0;
                        double maxAbsDiff = 0.0;
                        for (int i = 0; i < expected.length; i++) {
                            float expectedValue = expected[i];
                            float actualValue = actual[i];
                            int expectedBits = Float.floatToRawIntBits(expectedValue);
                            int actualBits = Float.floatToRawIntBits(actualValue);
                            maxAbsDiff = Math.max(
                                    maxAbsDiff, Math.abs((double) expectedValue - actualValue));
                            if (expectedBits != actualBits) {
                                mismatches++;
                                if (differences.length() < 768) {
                                    differences.append(" step=").append(step)
                                            .append(" output=").append(names[geometry])
                                            .append(" element=").append(i)
                                            .append(" native=0x").append(Integer.toHexString(expectedBits))
                                            .append(" triton=0x").append(Integer.toHexString(actualBits));
                                }
                            }
                        }
                        totalMismatches += mismatches;
                        log.info("FUSED_ROPE_EXACT step={} output={} mismatches={}/{} maxAbsDiff={}",
                                step, names[geometry], mismatches, expected.length, maxAbsDiff);
                    }
                }

                DspPlanAssertions.assertOpCompiled(
                        tritonGraph, "fused_rope", "Q/GQA fused RoPE exactness");
                DspPlanAssertions.assertAllSegmentsCompiledWith(
                        tritonGraph, "Triton GPU", "fused RoPE exactness");
            }
            assertEquals(0L, totalMismatches,
                    "Fused RoPE changed raw bits after Triton compilation:" + differences);
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            if (referenceQ != null && !referenceQ.wasClosed()) referenceQ.close();
            if (referenceK != null && !referenceK.wasClosed()) referenceK.close();
            if (referenceFull != null && !referenceFull.wasClosed()) referenceFull.close();
            qData.close();
            kData.close();
            fullData.close();
            positionData.close();
        }
    }

    /**
     * A partial position-offset RoPE must consume an unrequested RMSNorm
     * intermediate directly from SSA. Requesting only the final RoPE value keeps
     * this test sensitive to accidental intermediate materialization.
     */
    @Test
    @DisplayName("Triton RMSNorm to partial RoPE internal SSA handoff is exact")
    public void testTritonRmsNormToPartialRoPEInternalSsaHandoff() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 1;
        final int heads = 8;
        final int headDim = 256;
        final int rotaryDims = 64;
        final double frequencyBase = 10_000_000.0;

        INDArray inputData = Nd4j.linspace(
                DataType.FLOAT, -1.75, 0.00125, batch * sequence * heads * headDim)
                .reshape(batch, sequence, heads, headDim);
        INDArray gammaData = Nd4j.linspace(
                DataType.FLOAT, 0.5, 0.002, headDim);
        INDArray positionData = Nd4j.scalar(DataType.INT64, 7L);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);
        placeholders.put("gamma", gammaData);
        placeholders.put("position", positionData);

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        INDArray reference = null;
        try {
            try (SameDiff nativeGraph = SameDiff.create()) {
                SDVariable input = nativeGraph.placeHolder(
                        "input", DataType.FLOAT, batch, sequence, heads, headDim);
                SDVariable gamma =
                        nativeGraph.placeHolder("gamma", DataType.FLOAT, headDim);
                SDVariable position =
                        nativeGraph.placeHolder("position", DataType.INT64);
                SDVariable normalized =
                        nativeGraph.nn.rmsNorm("q_norm", input, gamma, 1e-6);
                nativeGraph.nn().fusedRoPE(
                        "q_rope", normalized, position,
                        0, frequencyBase, 1.0, rotaryDims);
                nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                reference =
                        nativeGraph.output(placeholders, "q_rope").get("q_rope").dup();
            }

            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes(
                    "NORMALIZATION,REDUCTION,ELEMENTWISE");

            long totalMismatches = 0;
            StringBuilder differences = new StringBuilder();
            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable input = tritonGraph.placeHolder(
                        "input", DataType.FLOAT, batch, sequence, heads, headDim);
                SDVariable gamma =
                        tritonGraph.placeHolder("gamma", DataType.FLOAT, headDim);
                SDVariable position =
                        tritonGraph.placeHolder("position", DataType.INT64);
                SDVariable normalized =
                        tritonGraph.nn.rmsNorm("q_norm", input, gamma, 1e-6);
                tritonGraph.nn().fusedRoPE(
                        "q_rope", normalized, position,
                        0, frequencyBase, 1.0, rotaryDims);
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                float[] expected = reference.toFloatVector();
                for (int step = 0; step < 4; step++) {
                    float[] actual = tritonGraph
                            .output(placeholders, "q_rope")
                            .get("q_rope").toFloatVector();
                    long stepMismatches = 0;
                    for (int i = 0; i < expected.length; i++) {
                        int expectedBits = Float.floatToRawIntBits(expected[i]);
                        int actualBits = Float.floatToRawIntBits(actual[i]);
                        if (expectedBits != actualBits) {
                            stepMismatches++;
                            if (differences.length() < 768) {
                                differences.append(" step=").append(step)
                                        .append(" element=").append(i)
                                        .append(" native=0x")
                                        .append(Integer.toHexString(expectedBits))
                                        .append(" triton=0x")
                                        .append(Integer.toHexString(actualBits));
                            }
                        }
                    }
                    totalMismatches += stepMismatches;
                    log.info("RMS_ROPE_INTERNAL_SSA_EXACT step={} mismatches={}/{}",
                            step, stepMismatches, expected.length);
                }

                DspPlanAssertions.assertOpCompiled(
                        tritonGraph, "rms_norm", "RMSNorm to partial RoPE SSA handoff");
                DspPlanAssertions.assertOpCompiled(
                        tritonGraph, "fused_rope", "RMSNorm to partial RoPE SSA handoff");
                DspPlanAssertions.assertAllSegmentsCompiledWith(
                        tritonGraph, "Triton GPU", "RMSNorm to partial RoPE SSA handoff");
                assertEquals(1, tritonGraph.dsp().numSegments(),
                        "RMSNorm and partial RoPE must remain in one compiled segment");
            }
            assertEquals(0L, totalMismatches,
                    "Internal RMSNorm to partial RoPE handoff changed raw bits:"
                            + differences);
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            if (reference != null && !reference.wasClosed()) reference.close();
            inputData.close();
            gammaData.close();
            positionData.close();
        }
    }

    /**
     * The fused SwiGLU custom op has its own emitter even though it shares the
     * standalone swish math contract. Pin that path independently so future
     * fusion changes cannot reintroduce a one-ULP recurrent-model drift.
     */
    @Test
    @DisplayName("Triton swish_mul matches native CUDA raw bits")
    public void testTritonSwishMulMatchesNativeExactly() {
        float[] inputValues = new float[]{
                -10.0f, -8.0f, -4.0f, -2.0f, -1.92597f, -1.0f, -0.5f, -0.1f,
                0.0f, 0.1f, 0.5f, 1.0f, 1.92597f, 2.0f, 4.0f, 8.0f
        };
        float[] gateValues = new float[]{
                0.5f, -0.75f, 1.25f, -1.5f, 2.0f, -2.25f, 0.125f, -0.25f,
                1.0f, -1.0f, 0.625f, -0.875f, 1.75f, -2.5f, 3.0f, -3.5f
        };
        INDArray inputData = Nd4j.createFromArray(inputValues).reshape(1, inputValues.length);
        INDArray gateData = Nd4j.createFromArray(gateValues).reshape(1, gateValues.length);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);
        placeholders.put("gate", gateData);

        INDArray reference;
        try (SameDiff nativeGraph = SameDiff.create()) {
            SDVariable input = nativeGraph.placeHolder("input", DataType.FLOAT, 1, inputValues.length);
            SDVariable gate = nativeGraph.placeHolder("gate", DataType.FLOAT, 1, gateValues.length);
            nativeGraph.nn.swishMul("swish_mul", input, gate);
            nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            reference = nativeGraph.output(placeholders, "swish_mul").get("swish_mul").dup();
        }

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ELEMENTWISE");

            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable input = tritonGraph.placeHolder("input", DataType.FLOAT, 1, inputValues.length);
                SDVariable gate = tritonGraph.placeHolder("gate", DataType.FLOAT, 1, gateValues.length);
                tritonGraph.nn.swishMul("swish_mul", input, gate);
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < 4; step++) {
                    INDArray actual = tritonGraph.output(placeholders, "swish_mul").get("swish_mul");
                    for (int i = 0; i < reference.length(); i++) {
                        int expectedBits = Float.floatToRawIntBits(reference.getFloat(i));
                        int actualBits = Float.floatToRawIntBits(actual.getFloat(i));
                        assertEquals(expectedBits, actualBits,
                                String.format("swish_mul step %d element %d: native=0x%08x triton=0x%08x",
                                        step, i, expectedBits, actualBits));
                    }
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    /**
     * Production fixed-buffer tracing isolates the first remaining warmup mismatch
     * to the per-head GDN RMSNorm output whose data input and gamma are identical.
     * Match its exact decode geometry: 16 independent rows with headDim=128.
     */
    @Test
    @DisplayName("Triton per-head GDN RMSNorm matches native CUDA raw bits")
    public void testTritonRmsNormMatchesNativeExactly() {
        final int rows = 16;
        final int headDim = 128;
        final int length = rows * headDim;
        float[] inputValues = new float[length];
        float[] gammaValues = new float[headDim];

        int state = 0x13579bdf;
        for (int i = 0; i < length; i++) {
            state = state * 1664525 + 1013904223;
            float mantissa = 0.5f + ((state >>> 8) & 2047) / 2048.0f;
            float value = Math.scalb(mantissa, ((state >>> 21) & 15) - 8);
            inputValues[i] = (state & 1) == 0 ? value : -value;
        }
        for (int i = 0; i < headDim; i++) {
            gammaValues[i] = 0.5f + ((i * 7) & 63) / 64.0f;
        }

        INDArray inputData = Nd4j.createFromArray(inputValues).reshape(1, 1, rows, headDim);
        INDArray gammaData = Nd4j.createFromArray(gammaValues);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);
        placeholders.put("gamma", gammaData);

        INDArray reference;
        try (SameDiff nativeGraph = SameDiff.create()) {
            SDVariable input = nativeGraph.placeHolder("input", DataType.FLOAT, 1, 1, rows, headDim);
            SDVariable gamma = nativeGraph.placeHolder("gamma", DataType.FLOAT, headDim);
            nativeGraph.nn.rmsNorm("rms_norm", input, gamma, 1e-6);
            nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            reference = nativeGraph.output(placeholders, "rms_norm").get("rms_norm").dup();
        }

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("NORMALIZATION,REDUCTION,ELEMENTWISE");

            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable input = tritonGraph.placeHolder("input", DataType.FLOAT, 1, 1, rows, headDim);
                SDVariable gamma = tritonGraph.placeHolder("gamma", DataType.FLOAT, headDim);
                tritonGraph.nn.rmsNorm("rms_norm", input, gamma, 1e-6);
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < 4; step++) {
                    INDArray actual = tritonGraph.output(placeholders, "rms_norm").get("rms_norm");
                    for (int i = 0; i < reference.length(); i++) {
                        int expectedBits = Float.floatToRawIntBits(reference.getFloat(i));
                        int actualBits = Float.floatToRawIntBits(actual.getFloat(i));
                        assertEquals(expectedBits, actualBits,
                                String.format("rms_norm step %d element %d: native=0x%08x triton=0x%08x",
                                        step, i, expectedBits, actualBits));
                    }
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    /**
     * Reproduces the fixed-buffer prefill RMSNorm island that first diverges
     * when a plan switches from native CUDA warmup to Triton execution.
     * The production tensor has 64 rows of width 1024 and rank three.
     */
    @Test
    @DisplayName("Triton fixed-buffer prefill RMSNorm matches native CUDA raw bits")
    public void testTritonPrefillRmsNorm1024MatchesNativeExactly() {
        final int rows = 64;
        final int headDim = 1024;
        final int length = rows * headDim;
        float[] inputValues = new float[length];
        float[] gammaValues = new float[headDim];

        int state = 0x6a09e667;
        for (int i = 0; i < length; i++) {
            state = state * 1664525 + 1013904223;
            float mantissa = 0.5f + ((state >>> 8) & 2047) / 2048.0f;
            float value = Math.scalb(mantissa, ((state >>> 21) & 15) - 10);
            inputValues[i] = (state & 1) == 0 ? value : -value;
        }
        for (int i = 0; i < headDim; i++) {
            gammaValues[i] = 0.5f + ((i * 11) & 127) / 128.0f;
        }

        INDArray inputData = Nd4j.createFromArray(inputValues).reshape(1, rows, headDim);
        INDArray gammaData = Nd4j.createFromArray(gammaValues);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", inputData);
        placeholders.put("gamma", gammaData);

        INDArray reference;
        try (SameDiff nativeGraph = SameDiff.create()) {
            SDVariable input = nativeGraph.placeHolder("input", DataType.FLOAT, 1, rows, headDim);
            SDVariable gamma = nativeGraph.placeHolder("gamma", DataType.FLOAT, headDim);
            nativeGraph.nn.rmsNorm("rms_norm", input, gamma, 1e-6);
            nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            reference = nativeGraph.output(placeholders, "rms_norm").get("rms_norm").dup();
        }
        float[] referenceValues = reference.toFloatVector();

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        boolean alwaysCompileBefore = environment.tritonAlwaysCompile();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonAlwaysCompile(true);
            environment.setTritonIncludeTypes("NORMALIZATION,REDUCTION,ELEMENTWISE");

            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable input = tritonGraph.placeHolder("input", DataType.FLOAT, 1, rows, headDim);
                SDVariable gamma = tritonGraph.placeHolder("gamma", DataType.FLOAT, headDim);
                tritonGraph.nn.rmsNorm("rms_norm", input, gamma, 1e-6);
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < 4; step++) {
                    INDArray actual = tritonGraph.output(placeholders, "rms_norm").get("rms_norm");
                    float[] actualValues = actual.toFloatVector();
                    // Census, not first-mismatch-stop: the full mismatch pattern
                    // (dense vs scattered vs single) discriminates a per-row
                    // scalar difference from a per-element tail difference.
                    List<Integer> mismatchPositions = new ArrayList<>();
                    for (int i = 0; i < referenceValues.length; i++) {
                        int expectedBits = Float.floatToRawIntBits(referenceValues[i]);
                        int actualBits = Float.floatToRawIntBits(actualValues[i]);
                        if (expectedBits != actualBits) mismatchPositions.add(i);
                    }
                    if (!mismatchPositions.isEmpty()) {
                        StringBuilder positions = new StringBuilder();
                        int shown = Math.min(8, mismatchPositions.size());
                        for (int p = 0; p < shown; p++) {
                            int i = mismatchPositions.get(p);
                            positions.append(String.format("[i=%d row=%d col=%d native=0x%08x triton=0x%08x]",
                                    i, i / headDim, i % headDim,
                                    Float.floatToRawIntBits(referenceValues[i]),
                                    Float.floatToRawIntBits(actualValues[i])));
                            if (p < shown - 1) positions.append(" ");
                        }
                        assertEquals(0, mismatchPositions.size(),
                                String.format("prefill rms_norm step %d: %d/%d elements differ (ulp deltas within ±1: %s) %s",
                                        step, mismatchPositions.size(), referenceValues.length,
                                        ulpDeltaSummary(referenceValues, actualValues, mismatchPositions),
                                        positions));
                    }
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonAlwaysCompile(alwaysCompileBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            reference.close();
            inputData.close();
            gammaData.close();
        }
    }

    /**
     * Summarizes signed ULP deltas (floatToRawIntBits distance) for the given
     * mismatch positions; keeps the assertion message bounded.
     */
    private static String ulpDeltaSummary(float[] expected, float[] actual,
                                          List<Integer> positions) {
        int maxDelta = 0;
        boolean beyondOne = false;
        for (int idx : positions) {
            int d = Math.abs(Float.floatToRawIntBits(expected[idx])
                    - Float.floatToRawIntBits(actual[idx]));
            if (d > maxDelta) maxDelta = d;
            if (d > 1) beyondOne = true;
        }
        return (beyondOne ? "YES" : "no") + " (maxDelta=" + maxDelta + ")";
    }

    /**
     * Reproduces the production decode boundary where a native CUDA
     * {@code gated_delta_rule} gap produces the input to a standalone Triton
     * RMSNorm island. Identical inputs must remain bit-identical while the plan
     * advances from warmup through merged CUDA-graph replay.
     */
    @Test
    @DisplayName("GDR-to-RMSNorm merged replay matches native CUDA raw bits")
    public void testGatedDeltaRuleToRmsNormReplayMatchesNativeExactly() {
        final int batch = 1;
        final int sequence = 1;
        final int heads = 16;
        final int headDim = 128;
        final int vectorLength = batch * sequence * heads * headDim;
        final int stateLength = batch * heads * headDim * headDim;

        float[] qValues = new float[vectorLength];
        float[] kValues = new float[vectorLength];
        float[] vValues = new float[vectorLength];
        float[] betaValues = new float[heads];
        float[] gateValues = new float[heads];
        float[] stateValues = new float[stateLength];
        float[] gammaValues = new float[headDim];

        int randomState = 0x2468ace1;
        for (int i = 0; i < vectorLength; i++) {
            randomState = randomState * 1664525 + 1013904223;
            qValues[i] = (((randomState >>> 8) & 2047) - 1024) * 0.00002f;
            randomState = randomState * 1664525 + 1013904223;
            kValues[i] = (((randomState >>> 8) & 2047) - 1024) * 0.00002f;
            randomState = randomState * 1664525 + 1013904223;
            vValues[i] = (((randomState >>> 8) & 2047) - 1024) * 0.0001f;
        }
        for (int i = 0; i < heads; i++) {
            randomState = randomState * 1664525 + 1013904223;
            betaValues[i] = 0.1f + ((randomState >>> 8) & 1023) * 0.0003f;
            randomState = randomState * 1664525 + 1013904223;
            gateValues[i] = -0.5f + ((randomState >>> 8) & 1023) * 0.0004f;
        }
        for (int i = 0; i < stateLength; i++) {
            randomState = randomState * 1664525 + 1013904223;
            stateValues[i] = (((randomState >>> 8) & 2047) - 1024) * 0.00001f;
        }
        for (int i = 0; i < headDim; i++) {
            gammaValues[i] = 0.75f + ((i * 11) & 63) / 128.0f;
        }

        INDArray qData = Nd4j.createFromArray(qValues).reshape(batch, sequence, heads, headDim);
        INDArray kData = Nd4j.createFromArray(kValues).reshape(batch, sequence, heads, headDim);
        INDArray vData = Nd4j.createFromArray(vValues).reshape(batch, sequence, heads, headDim);
        INDArray betaData = Nd4j.createFromArray(betaValues).reshape(batch, sequence, heads);
        INDArray gateData = Nd4j.createFromArray(gateValues).reshape(batch, sequence, heads);
        INDArray stateData = Nd4j.createFromArray(stateValues).reshape(batch, heads, headDim, headDim);
        INDArray actualLengthData = Nd4j.scalar(DataType.INT64, 1L);
        INDArray gammaData = Nd4j.createFromArray(gammaValues);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("q", qData);
        placeholders.put("k", kData);
        placeholders.put("v", vData);
        placeholders.put("beta", betaData);
        placeholders.put("gate", gateData);
        placeholders.put("state", stateData);
        placeholders.put("actual_length", actualLengthData);
        placeholders.put("gamma", gammaData);

        INDArray referenceGdr;
        INDArray referenceRms;
        try (SameDiff nativeGraph = SameDiff.create()) {
            SDVariable q = nativeGraph.placeHolder("q", DataType.FLOAT, batch, sequence, heads, headDim);
            SDVariable k = nativeGraph.placeHolder("k", DataType.FLOAT, batch, sequence, heads, headDim);
            SDVariable v = nativeGraph.placeHolder("v", DataType.FLOAT, batch, sequence, heads, headDim);
            SDVariable beta = nativeGraph.placeHolder("beta", DataType.FLOAT, batch, sequence, heads);
            SDVariable gate = nativeGraph.placeHolder("gate", DataType.FLOAT, batch, sequence, heads);
            SDVariable state = nativeGraph.placeHolder("state", DataType.FLOAT, batch, heads, headDim, headDim);
            SDVariable actualLength = nativeGraph.placeHolder("actual_length", DataType.INT64);
            SDVariable gamma = nativeGraph.placeHolder("gamma", DataType.FLOAT, headDim);
            SDVariable gdr = new GatedDeltaRule(nativeGraph, q, k, v, beta, gate, state, actualLength)
                    .outputVariables()[0];
            nativeGraph.updateVariableNameAndReference(gdr, "gdr_out");
            nativeGraph.nn.rmsNorm("gdr_rms", gdr, gamma, 1e-6);
            nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            Map<String, INDArray> nativeOutputs = nativeGraph.output(placeholders, "gdr_out", "gdr_rms");
            referenceGdr = nativeOutputs.get("gdr_out").dup();
            referenceRms = nativeOutputs.get("gdr_rms").dup();
        }

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        boolean alwaysCompileBefore = environment.tritonAlwaysCompile();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonAlwaysCompile(true);
            environment.setTritonIncludeTypes("NORMALIZATION,REDUCTION,ELEMENTWISE");

            try (SameDiff replayGraph = SameDiff.create()) {
                SDVariable q = replayGraph.placeHolder("q", DataType.FLOAT, batch, sequence, heads, headDim);
                SDVariable k = replayGraph.placeHolder("k", DataType.FLOAT, batch, sequence, heads, headDim);
                SDVariable v = replayGraph.placeHolder("v", DataType.FLOAT, batch, sequence, heads, headDim);
                SDVariable beta = replayGraph.placeHolder("beta", DataType.FLOAT, batch, sequence, heads);
                SDVariable gate = replayGraph.placeHolder("gate", DataType.FLOAT, batch, sequence, heads);
                SDVariable state = replayGraph.placeHolder("state", DataType.FLOAT, batch, heads, headDim, headDim);
                SDVariable actualLength = replayGraph.placeHolder("actual_length", DataType.INT64);
                SDVariable gamma = replayGraph.placeHolder("gamma", DataType.FLOAT, headDim);
                SDVariable gdr = new GatedDeltaRule(replayGraph, q, k, v, beta, gate, state, actualLength)
                        .outputVariables()[0];
                replayGraph.updateVariableNameAndReference(gdr, "gdr_out");
                replayGraph.nn.rmsNorm("gdr_rms", gdr, gamma, 1e-6);
                replayGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < 18; step++) {
                    Map<String, INDArray> actualOutputs = replayGraph.output(
                            placeholders, "gdr_out", "gdr_rms");
                    INDArray actualGdr = actualOutputs.get("gdr_out");
                    for (int i = 0; i < referenceGdr.length(); i++) {
                        int expectedBits = Float.floatToRawIntBits(referenceGdr.getFloat(i));
                        int actualBits = Float.floatToRawIntBits(actualGdr.getFloat(i));
                        assertEquals(expectedBits, actualBits,
                                String.format("GDR step %d element %d: native=0x%08x replay=0x%08x",
                                        step, i, expectedBits, actualBits));
                    }
                    INDArray actualRms = actualOutputs.get("gdr_rms");
                    for (int i = 0; i < referenceRms.length(); i++) {
                        int expectedBits = Float.floatToRawIntBits(referenceRms.getFloat(i));
                        int actualBits = Float.floatToRawIntBits(actualRms.getFloat(i));
                        assertEquals(expectedBits, actualBits,
                                String.format("RMS-after-GDR step %d element %d: native=0x%08x replay=0x%08x",
                                        step, i, expectedBits, actualBits));
                    }
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonAlwaysCompile(alwaysCompileBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            referenceGdr.close();
            referenceRms.close();
            for (INDArray input : placeholders.values()) {
                if (input != null && !input.wasClosed()) input.close();
            }
        }
    }

    @Test
    @DisplayName("Triton DPA-v2 honors explicit and automatic attention scales")
    public void testTritonDpaV2AttentionScale() {
        assumeCuda();
        final int headDim = 2;
        INDArray queryData = Nd4j.createFromArray(new float[] {1.0f, 0.0f})
                .reshape(1, 1, 1, headDim);
        INDArray keyData = Nd4j.createFromArray(new float[] {1.0f, 0.0f, 0.0f, 0.0f})
                .reshape(1, 2, 1, headDim);
        INDArray valueData = Nd4j.createFromArray(new float[] {1.0f, 1.0f, 0.0f, 0.0f})
                .reshape(1, 2, 1, headDim);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ATTENTION");

            for (double scale : new double[] {1.0, 0.5, 0.0}) {
                try (SameDiff graph = SameDiff.create()) {
                    SDVariable query = graph.placeHolder(
                            "query", DataType.FLOAT, 1, 1, 1, headDim);
                    SDVariable value = graph.placeHolder(
                            "value", DataType.FLOAT, 1, 2, 1, headDim);
                    SDVariable key = graph.placeHolder(
                            "key", DataType.FLOAT, 1, 2, 1, headDim);
                    SDVariable attention = new DotProductAttentionV2(
                            graph, query, value, key, null, null,
                            null, null, null, null, scale, 0.0, false, false).outputVariable();
                    graph.updateVariableNameAndReference(attention, "attention");
                    graph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                    double effectiveScale = scale <= 0.0 ? 1.0 / Math.sqrt(headDim) : scale;
                    float expected = (float) (1.0 / (1.0 + Math.exp(-effectiveScale)));
                    for (int step = 0; step < 4; step++) {
                        float[] actual = graph.output(placeholders, "attention")
                                .get("attention").toFloatVector();
                        assertEquals(2, actual.length,
                                "Attention output should contain one head vector at step " + step);
                        assertEquals(expected, actual[0], 1.0e-5f,
                                "First output value for scale " + scale + " at step " + step);
                        assertEquals(expected, actual[1], 1.0e-5f,
                                "Second output value for scale " + scale + " at step " + step);
                    }
                    DspPlanAssertions.assertOpCompiled(graph, "dot_product_attention_v2",
                            "DPA-v2 attention scale " + scale);
                    DspPlanAssertions.assertAllSegmentsCompiledWith(graph, "Triton GPU",
                            "DPA-v2 attention scale " + scale);
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            queryData.close();
            keyData.close();
            valueData.close();
        }
    }

    /**
     * DPA-v2 keeps optional input positions stable when KV-cache placeholders are present:
     * Q, V, K, query mask, value mask, key cache, value cache, cache position, then bias.
     * The compiled attention path must read the additive causal bias from input 8 instead of
     * mistaking the empty query-mask placeholder at input 3 for that bias.
     */
    @Test
    @DisplayName("Triton DPA-v2 cache-form attention bias matches native CUDA")
    public void testTritonDpaV2CacheFormAttentionBiasMatchesNative() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 8;
        final int qHeads = 8;
        final int kvHeads = 2;
        final int headDim = 16;

        INDArray queryData = Nd4j.zeros(DataType.FLOAT, batch, sequence, qHeads, headDim);
        INDArray keyData = Nd4j.zeros(DataType.FLOAT, batch, sequence, kvHeads, headDim);
        float[] valueValues = new float[batch * sequence * kvHeads * headDim];
        for (int s = 0; s < sequence; s++) {
            for (int h = 0; h < kvHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    int index = (s * kvHeads + h) * headDim + d;
                    valueValues[index] = (s + 1) * 0.125f + h * 0.03125f + d * 0.001953125f;
                }
            }
        }
        INDArray valueData = Nd4j.createFromArray(valueValues)
                .reshape(batch, sequence, kvHeads, headDim);
        float[] biasValues = new float[sequence * sequence];
        for (int q = 0; q < sequence; q++) {
            for (int k = q + 1; k < sequence; k++) {
                biasValues[q * sequence + k] = -1.0e9f;
            }
        }
        INDArray biasData = Nd4j.createFromArray(biasValues)
                .reshape(batch, 1, sequence, sequence);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);
        placeholders.put("attention_bias", biasData);

        INDArray reference = null;
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            try (SameDiff nativeGraph = SameDiff.create()) {
                SDVariable query = nativeGraph.placeHolder(
                        "query", DataType.FLOAT, batch, sequence, qHeads, headDim);
                SDVariable value = nativeGraph.placeHolder(
                        "value", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable key = nativeGraph.placeHolder(
                        "key", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable bias = nativeGraph.placeHolder(
                        "attention_bias", DataType.FLOAT, batch, 1, sequence, sequence);
                SDVariable emptyKeyCache = nativeGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueCache = nativeGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyCachePosition = nativeGraph.constant(Nd4j.empty(DataType.INT64));
                SDVariable attention = new DotProductAttentionV2(
                        nativeGraph, query, value, key, null, null,
                        emptyKeyCache, emptyValueCache, emptyCachePosition, bias,
                        0.0, 0.0, false, false).outputVariable();
                nativeGraph.updateVariableNameAndReference(attention, "attention");
                nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                reference = nativeGraph.output(placeholders, "attention").get("attention").dup();
            }

            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ATTENTION");

            long totalMismatches = 0;
            StringBuilder differences = new StringBuilder();
            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable query = tritonGraph.placeHolder(
                        "query", DataType.FLOAT, batch, sequence, qHeads, headDim);
                SDVariable value = tritonGraph.placeHolder(
                        "value", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable key = tritonGraph.placeHolder(
                        "key", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable bias = tritonGraph.placeHolder(
                        "attention_bias", DataType.FLOAT, batch, 1, sequence, sequence);
                SDVariable emptyKeyCache = tritonGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueCache = tritonGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyCachePosition = tritonGraph.constant(Nd4j.empty(DataType.INT64));
                SDVariable attention = new DotProductAttentionV2(
                        tritonGraph, query, value, key, null, null,
                        emptyKeyCache, emptyValueCache, emptyCachePosition, bias,
                        0.0, 0.0, false, false).outputVariable();
                tritonGraph.updateVariableNameAndReference(attention, "attention");
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                float[] expected = reference.toFloatVector();
                for (int step = 0; step < 4; step++) {
                    float[] actual = tritonGraph.output(placeholders, "attention")
                            .get("attention").toFloatVector();
                    long stepMismatches = 0;
                    double maxAbsDiff = 0.0;
                    for (int i = 0; i < expected.length; i++) {
                        double absDiff = Math.abs((double) expected[i] - actual[i]);
                        maxAbsDiff = Math.max(maxAbsDiff, absDiff);
                        if (absDiff > 1.0e-5) {
                            stepMismatches++;
                            if (differences.length() < 768) {
                                differences.append(" step=").append(step)
                                        .append(" element=").append(i)
                                        .append(" native=").append(expected[i])
                                        .append(" triton=").append(actual[i]);
                            }
                        }
                    }
                    totalMismatches += stepMismatches;
                    log.info("DPA_V2_CACHE_BIAS_PARITY step={} mismatches={}/{} maxAbsDiff={}",
                            step, stepMismatches, expected.length, maxAbsDiff);
                }

                DspPlanAssertions.assertOpCompiled(
                        tritonGraph, "dot_product_attention_v2", "DPA-v2 cache-form bias contract");
                DspPlanAssertions.assertAllSegmentsCompiledWith(
                        tritonGraph, "Triton GPU", "DPA-v2 cache-form bias contract");
            }
            assertEquals(0L, totalMismatches,
                    "DPA-v2 cache-form attention bias changed compiled output:" + differences);
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            if (reference != null && !reference.wasClosed()) reference.close();
            queryData.close();
            keyData.close();
            valueData.close();
            biasData.close();
        }
    }

    /**
     * Dense Q/K with one-hot V turns the first {@code sequence} output channels into
     * the attention probability vector itself. This isolates QK/softmax parity from
     * downstream value accumulation while exercising the production cache-form GQA
     * contract and its explicit additive bias.
     */
    @Test
    @DisplayName("Triton DPA-v2 GQA prefill probabilities match native CUDA")
    public void testTritonDpaV2GqaPrefillProbabilitiesMatchNative() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 64;
        final int qHeads = 8;
        final int kvHeads = 2;
        final int headDim = 256;

        float[] queryValues = new float[batch * sequence * qHeads * headDim];
        float[] keyValues = new float[batch * sequence * kvHeads * headDim];
        float[] valueValues = new float[batch * sequence * kvHeads * headDim];
        for (int s = 0; s < sequence; s++) {
            for (int h = 0; h < qHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    int index = (s * qHeads + h) * headDim + d;
                    queryValues[index] = (float) (1.5 * Math.sin(
                            (s + 1) * 0.173 + (h + 1) * 0.097 + (d + 1) * 0.013));
                }
            }
            for (int h = 0; h < kvHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    int index = (s * kvHeads + h) * headDim + d;
                    keyValues[index] = (float) (1.5 * Math.cos(
                            (s + 1) * 0.117 + (h + 1) * 0.071 + (d + 1) * 0.019));
                    valueValues[index] = d % sequence == s ? 1.0f : 0.0f;
                }
            }
        }

        float[] biasValues = new float[sequence * sequence];
        for (int q = 0; q < sequence; q++) {
            for (int k = q + 1; k < sequence; k++) {
                biasValues[q * sequence + k] = -1.0e9f;
            }
        }

        INDArray queryData = Nd4j.createFromArray(queryValues)
                .reshape(batch, sequence, qHeads, headDim);
        INDArray keyData = Nd4j.createFromArray(keyValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray valueData = Nd4j.createFromArray(valueValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray biasData = Nd4j.createFromArray(biasValues)
                .reshape(batch, 1, sequence, sequence);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);
        placeholders.put("attention_bias", biasData);

        INDArray reference = null;
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            try (SameDiff nativeGraph = SameDiff.create()) {
                SDVariable query = nativeGraph.placeHolder(
                        "query", DataType.FLOAT, batch, sequence, qHeads, headDim);
                SDVariable value = nativeGraph.placeHolder(
                        "value", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable key = nativeGraph.placeHolder(
                        "key", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable bias = nativeGraph.placeHolder(
                        "attention_bias", DataType.FLOAT, batch, 1, sequence, sequence);
                SDVariable emptyKeyCache = nativeGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueCache = nativeGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyCachePosition = nativeGraph.constant(Nd4j.empty(DataType.INT64));
                SDVariable attention = new DotProductAttentionV2(
                        nativeGraph, query, value, key, null, null,
                        emptyKeyCache, emptyValueCache, emptyCachePosition, bias,
                        0.0, 0.0, false, false).outputVariable();
                nativeGraph.updateVariableNameAndReference(attention, "attention");
                nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                reference = nativeGraph.output(placeholders, "attention").get("attention").dup();
            }

            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ATTENTION");

            long totalMismatches = 0;
            StringBuilder differences = new StringBuilder();
            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable query = tritonGraph.placeHolder(
                        "query", DataType.FLOAT, batch, sequence, qHeads, headDim);
                SDVariable value = tritonGraph.placeHolder(
                        "value", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable key = tritonGraph.placeHolder(
                        "key", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable bias = tritonGraph.placeHolder(
                        "attention_bias", DataType.FLOAT, batch, 1, sequence, sequence);
                SDVariable emptyKeyCache = tritonGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueCache = tritonGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyCachePosition = tritonGraph.constant(Nd4j.empty(DataType.INT64));
                SDVariable attention = new DotProductAttentionV2(
                        tritonGraph, query, value, key, null, null,
                        emptyKeyCache, emptyValueCache, emptyCachePosition, bias,
                        0.0, 0.0, false, false).outputVariable();
                tritonGraph.updateVariableNameAndReference(attention, "attention");
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                float[] expected = reference.toFloatVector();
                for (int step = 0; step < 4; step++) {
                    float[] actual = tritonGraph.output(placeholders, "attention")
                            .get("attention").toFloatVector();
                    long stepMismatches = 0;
                    double maxAbsDiff = 0.0;
                    int maxDiffIndex = -1;
                    for (int i = 0; i < expected.length; i++) {
                        double absDiff = Math.abs((double) expected[i] - actual[i]);
                        if (absDiff > maxAbsDiff) {
                            maxAbsDiff = absDiff;
                            maxDiffIndex = i;
                        }
                        if (absDiff > 1.0e-6) {
                            stepMismatches++;
                            if (differences.length() < 768) {
                                differences.append(" step=").append(step)
                                        .append(" element=").append(i)
                                        .append(" native=").append(expected[i])
                                        .append(" triton=").append(actual[i]);
                            }
                        }
                    }
                    totalMismatches += stepMismatches;
                    log.info("DPA_V2_GQA_PROBABILITY_PARITY step={} mismatches={}/{} "
                                    + "maxAbsDiff={} maxDiffIndex={} native={} triton={}",
                            step, stepMismatches, expected.length, maxAbsDiff, maxDiffIndex,
                            maxDiffIndex < 0 ? 0.0f : expected[maxDiffIndex],
                            maxDiffIndex < 0 ? 0.0f : actual[maxDiffIndex]);
                }

                DspPlanAssertions.assertOpCompiled(
                        tritonGraph, "dot_product_attention_v2", "DPA-v2 GQA probability parity");
                DspPlanAssertions.assertAllSegmentsCompiledWith(
                        tritonGraph, "Triton GPU", "DPA-v2 GQA probability parity");
            }
            assertEquals(0L, totalMismatches,
                    "DPA-v2 GQA prefill probabilities changed compiled output:" + differences);
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            if (reference != null && !reference.wasClosed()) reference.close();
            queryData.close();
            keyData.close();
            valueData.close();
            biasData.close();
        }
    }

    /**
     * Two keys and one non-zero QK product per key isolate the attention score
     * precision used by the production OPTIMAL profile. That profile enables
     * Triton TF32 globally, while native CUDA's direct GQA attention kernel keeps
     * its QK products in FP32. The selected logit delta produces a probability
     * near 0.5408, making an unintended TF32 attention dot numerically visible.
     */
    @Test
    @DisplayName("Triton DPA-v2 GQA two-key attention remains FP32 under the TF32 profile")
    public void testTritonDpaV2GqaTwoKeyTf32MatchesNativeCuda() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 2;
        final int qHeads = 8;
        final int kvHeads = 2;
        final int headDim = 256;
        final int qHeadsPerKvHead = qHeads / kvHeads;

        float[] queryValues = new float[batch * sequence * qHeads * headDim];
        float[] keyValues = new float[batch * sequence * kvHeads * headDim];
        float[] valueValues = new float[batch * sequence * kvHeads * headDim];

        for (int kvHead = 0; kvHead < kvHeads; kvHead++) {
            int firstDimension = kvHead * 31;
            int secondDimension = firstDimension + 17;
            keyValues[(kvHead * headDim) + firstDimension] = 16.0f;
            keyValues[((kvHeads + kvHead) * headDim) + secondDimension] = 16.0f;
            valueValues[(kvHead * headDim)] = 1.0f;
            valueValues[((kvHeads + kvHead) * headDim) + 1] = 1.0f;

            for (int qHead = kvHead * qHeadsPerKvHead;
                    qHead < (kvHead + 1) * qHeadsPerKvHead; qHead++) {
                int queryBase = ((qHeads + qHead) * headDim);
                queryValues[queryBase + firstDimension] = 0.1635f;
            }
        }

        INDArray queryData = Nd4j.createFromArray(queryValues)
                .reshape(batch, sequence, qHeads, headDim);
        INDArray keyData = Nd4j.createFromArray(keyValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray valueData = Nd4j.createFromArray(valueValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray biasData = Nd4j.createFromArray(0.0f, -1.0e9f, 0.0f, 0.0f)
                .reshape(batch, 1, sequence, sequence);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);
        placeholders.put("attention_bias", biasData);
        Environment environment = Nd4j.getEnvironment();
        boolean tritonTf32Before = environment.tritonTf32Enabled();
        try {
            environment.setTritonTf32Enabled(true);
            assertDpaV2GqaPrefillParity(
                    placeholders, batch, sequence, qHeads, kvHeads, headDim,
                    1.0e-6, "DPA_V2_GQA_TWO_KEY_TF32_PARITY");
        } finally {
            environment.setTritonTf32Enabled(tritonTf32Before);
            queryData.close();
            keyData.close();
            valueData.close();
            biasData.close();
        }
    }

    /**
     * Native CUDA computes invSum = 1 / sum once and then multiplies each
     * probability. A direct probability / sum produces a one-ULP difference for
     * this logit even though the expressions are mathematically equivalent. The
     * recurrent model amplifies that ULP enough to change a generated token.
     */
    @Test
    @DisplayName("Triton DPA-v2 GQA normalization preserves native reciprocal-then-multiply rounding")
    public void testTritonDpaV2GqaTwoKeyNormalizationOrderMatchesNativeCuda() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 2;
        final int qHeads = 8;
        final int kvHeads = 2;
        final int headDim = 256;
        final int qHeadsPerKvHead = qHeads / kvHeads;

        float[] queryValues = new float[batch * sequence * qHeads * headDim];
        float[] keyValues = new float[batch * sequence * kvHeads * headDim];
        float[] valueValues = new float[batch * sequence * kvHeads * headDim];
        for (int kvHead = 0; kvHead < kvHeads; kvHead++) {
            int firstDimension = kvHead * 31;
            int secondDimension = firstDimension + 17;
            keyValues[kvHead * headDim + firstDimension] = 16.0f;
            keyValues[(kvHeads + kvHead) * headDim + secondDimension] = 16.0f;
            valueValues[kvHead * headDim] = 1.0f;
            valueValues[(kvHeads + kvHead) * headDim + 1] = 1.0f;

            for (int qHead = kvHead * qHeadsPerKvHead;
                    qHead < (kvHead + 1) * qHeadsPerKvHead; qHead++) {
                int queryBase = (qHeads + qHead) * headDim;
                queryValues[queryBase + firstDimension] = -0.1396f;
            }
        }

        INDArray queryData = Nd4j.createFromArray(queryValues)
                .reshape(batch, sequence, qHeads, headDim);
        INDArray keyData = Nd4j.createFromArray(keyValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray valueData = Nd4j.createFromArray(valueValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray biasData = Nd4j.createFromArray(0.0f, -1.0e9f, 0.0f, 0.0f)
                .reshape(batch, 1, sequence, sequence);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);
        placeholders.put("attention_bias", biasData);
        Environment environment = Nd4j.getEnvironment();
        boolean tritonTf32Before = environment.tritonTf32Enabled();
        try {
            environment.setTritonTf32Enabled(true);
            assertDpaV2GqaPrefillParity(
                    placeholders, batch, sequence, qHeads, kvHeads, headDim,
                    0.0, "DPA_V2_GQA_NORMALIZATION_ORDER_PARITY");
        } finally {
            environment.setTritonTf32Enabled(tritonTf32Before);
            queryData.close();
            keyData.close();
            valueData.close();
            biasData.close();
        }
    }

    /**
     * With headDim 256 the production shared-memory budget previously collapsed
     * prefill to a 16-key tile. The seventeenth causal key therefore exercised a
     * second online-softmax tile even though a wider tile fits when blockM is
     * reduced jointly. One-hot V exposes every probability directly so this test
     * catches the resulting one-ULP boundary drift without downstream reduction.
     */
    @Test
    @DisplayName("Triton DPA-v2 GQA seventeen-key tile boundary remains exact")
    public void testTritonDpaV2GqaSeventeenKeyTileBoundaryMatchesNativeCuda() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 17;
        final int qHeads = 8;
        final int kvHeads = 2;
        final int headDim = 256;

        float[] queryValues = new float[batch * sequence * qHeads * headDim];
        float[] keyValues = new float[batch * sequence * kvHeads * headDim];
        float[] valueValues = new float[batch * sequence * kvHeads * headDim];
        for (int s = 0; s < sequence; s++) {
            for (int h = 0; h < qHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    int index = (s * qHeads + h) * headDim + d;
                    queryValues[index] = (float) (1.5 * Math.sin(
                            (s + 1) * 0.173 + (h + 1) * 0.097 + (d + 1) * 0.013));
                }
            }
            for (int h = 0; h < kvHeads; h++) {
                int keyDimension = (s * 17 + h * 31) % headDim;
                int keyIndex = (s * kvHeads + h) * headDim + keyDimension;
                keyValues[keyIndex] = 16.0f;
                int valueIndex = (s * kvHeads + h) * headDim + s;
                valueValues[valueIndex] = 1.0f;
            }
        }

        float[] biasValues = new float[sequence * sequence];
        for (int q = 0; q < sequence; q++) {
            for (int k = q + 1; k < sequence; k++) {
                biasValues[q * sequence + k] = -1.0e9f;
            }
        }

        INDArray queryData = Nd4j.createFromArray(queryValues)
                .reshape(batch, sequence, qHeads, headDim);
        INDArray keyData = Nd4j.createFromArray(keyValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray valueData = Nd4j.createFromArray(valueValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray biasData = Nd4j.createFromArray(biasValues)
                .reshape(batch, 1, sequence, sequence);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);
        placeholders.put("attention_bias", biasData);
        Environment environment = Nd4j.getEnvironment();
        boolean tritonTf32Before = environment.tritonTf32Enabled();
        try {
            environment.setTritonTf32Enabled(true);
            assertDpaV2GqaPrefillParity(
                    placeholders, batch, sequence, qHeads, kvHeads, headDim,
                    0.0, "DPA_V2_GQA_SEVENTEEN_KEY_TILE_BOUNDARY_PARITY");
        } finally {
            environment.setTritonTf32Enabled(tritonTf32Before);
            queryData.close();
            keyData.close();
            valueData.close();
            biasData.close();
        }
    }

    /**
     * Uses the same one-product Q/K logits and 17-key single tile as the exact
     * probability test, but makes V dense. Since the probabilities are already
     * proven bit-exact, any mismatch here is specifically the P-times-V reduction
     * and its normalization placement.
     */
    @Test
    @DisplayName("Triton DPA-v2 GQA seventeen-key dense-V accumulation remains exact")
    public void testTritonDpaV2GqaSeventeenKeyDenseValueMatchesNativeCuda() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 17;
        final int qHeads = 8;
        final int kvHeads = 2;
        final int headDim = 256;

        float[] queryValues = new float[batch * sequence * qHeads * headDim];
        float[] keyValues = new float[batch * sequence * kvHeads * headDim];
        float[] valueValues = new float[batch * sequence * kvHeads * headDim];
        for (int s = 0; s < sequence; s++) {
            for (int h = 0; h < qHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    int index = (s * qHeads + h) * headDim + d;
                    queryValues[index] = (float) (1.5 * Math.sin(
                            (s + 1) * 0.173 + (h + 1) * 0.097 + (d + 1) * 0.013));
                }
            }
            for (int h = 0; h < kvHeads; h++) {
                int keyDimension = (s * 17 + h * 31) % headDim;
                int keyIndex = (s * kvHeads + h) * headDim + keyDimension;
                keyValues[keyIndex] = 16.0f;
                for (int d = 0; d < headDim; d++) {
                    int valueIndex = (s * kvHeads + h) * headDim + d;
                    valueValues[valueIndex] = (float) (0.8 * Math.sin(
                            (s + 1) * 0.139 + (h + 1) * 0.083 + (d + 1) * 0.023));
                }
            }
        }

        float[] biasValues = new float[sequence * sequence];
        for (int q = 0; q < sequence; q++) {
            for (int k = q + 1; k < sequence; k++) {
                biasValues[q * sequence + k] = -1.0e9f;
            }
        }

        INDArray queryData = Nd4j.createFromArray(queryValues)
                .reshape(batch, sequence, qHeads, headDim);
        INDArray keyData = Nd4j.createFromArray(keyValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray valueData = Nd4j.createFromArray(valueValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray biasData = Nd4j.createFromArray(biasValues)
                .reshape(batch, 1, sequence, sequence);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);
        placeholders.put("attention_bias", biasData);
        Environment environment = Nd4j.getEnvironment();
        boolean tritonTf32Before = environment.tritonTf32Enabled();
        try {
            environment.setTritonTf32Enabled(true);
            assertDpaV2GqaPrefillParity(
                    placeholders, batch, sequence, qHeads, kvHeads, headDim,
                    0.0, "DPA_V2_GQA_SEVENTEEN_KEY_DENSE_VALUE_PARITY");
        } finally {
            environment.setTritonTf32Enabled(tritonTf32Before);
            queryData.close();
            keyData.close();
            valueData.close();
            biasData.close();
        }
    }

    /**
     * The probability-parity test above removes value reduction by making every
     * output channel depend on one key only. Dense V exercises the complementary
     * probability-times-value reduction with the same Q/K probabilities and layout.
     */
    @Test
    @DisplayName("Triton DPA-v2 GQA prefill dense-V accumulation matches native CUDA")
    public void testTritonDpaV2GqaPrefillDenseValueAccumulationMatchesNative() {
        assumeCuda();
        final int batch = 1;
        final int sequence = 64;
        final int qHeads = 8;
        final int kvHeads = 2;
        final int headDim = 256;

        float[] queryValues = new float[batch * sequence * qHeads * headDim];
        float[] keyValues = new float[batch * sequence * kvHeads * headDim];
        float[] valueValues = new float[batch * sequence * kvHeads * headDim];
        for (int s = 0; s < sequence; s++) {
            for (int h = 0; h < qHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    int index = (s * qHeads + h) * headDim + d;
                    queryValues[index] = (float) (1.5 * Math.sin(
                            (s + 1) * 0.173 + (h + 1) * 0.097 + (d + 1) * 0.013));
                }
            }
            for (int h = 0; h < kvHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    int index = (s * kvHeads + h) * headDim + d;
                    keyValues[index] = (float) (1.5 * Math.cos(
                            (s + 1) * 0.117 + (h + 1) * 0.071 + (d + 1) * 0.019));
                    valueValues[index] = (float) (0.8 * Math.sin(
                            (s + 1) * 0.139 + (h + 1) * 0.083 + (d + 1) * 0.023));
                }
            }
        }

        float[] biasValues = new float[sequence * sequence];
        for (int q = 0; q < sequence; q++) {
            for (int k = q + 1; k < sequence; k++) {
                biasValues[q * sequence + k] = -1.0e9f;
            }
        }

        INDArray queryData = Nd4j.createFromArray(queryValues)
                .reshape(batch, sequence, qHeads, headDim);
        INDArray keyData = Nd4j.createFromArray(keyValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray valueData = Nd4j.createFromArray(valueValues)
                .reshape(batch, sequence, kvHeads, headDim);
        INDArray biasData = Nd4j.createFromArray(biasValues)
                .reshape(batch, 1, sequence, sequence);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);
        placeholders.put("attention_bias", biasData);
        try {
            assertDpaV2GqaPrefillParity(
                    placeholders, batch, sequence, qHeads, kvHeads, headDim,
                    1.0e-6, "DPA_V2_GQA_DENSE_VALUE_PARITY");
        } finally {
            queryData.close();
            keyData.close();
            valueData.close();
            biasData.close();
        }
    }

    /**
     * The live-cache GQA path must not change arithmetic when native warmup becomes
     * compiled execution. HALF logits/probabilities expose rounding hidden by the
     * FLOAT-only prefill fixtures. Two keys and cancelling values amplify that
     * difference without a model, recurrent state, pointer replacement or a split call.
     */
    @ParameterizedTest(name = "live-cache GQA accumulator {0}")
    @EnumSource(value = DataType.class, names = {"HALF", "BFLOAT16"})
    public void testTritonHalfGqaLiveCacheMatchesNativeCuda(DataType dtype) {
        assumeCuda();
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        boolean captureBefore = environment.tritonGraphCapture();
        try (SameDiff nativeGraph = SameDiff.create();
             SameDiff tritonGraph = SameDiff.create();
             INDArray query = Nd4j.zeros(dtype, 1, 1, 8, 256);
             INDArray key = Nd4j.zeros(dtype, 1, 1, 2, 256);
             INDArray value = Nd4j.zeros(dtype, 1, 1, 2, 256);
             INDArray initialKeys = Nd4j.zeros(dtype, 1, 2, 2, 256);
             INDArray initialValues = Nd4j.zeros(dtype, 1, 2, 2, 256);
             INDArray nativeKeys = Nd4j.zeros(dtype, 1, 2, 2, 256);
             INDArray nativeValues = Nd4j.zeros(dtype, 1, 2, 2, 256);
             INDArray replayKeys = Nd4j.zeros(dtype, 1, 2, 2, 256);
             INDArray replayValues = Nd4j.zeros(dtype, 1, 2, 2, 256);
             INDArray position = Nd4j.scalar(DataType.INT64, 1);
             INDArray bias = Nd4j.zeros(dtype, 1, 1, 1, 2)) {
            for (int h = 0; h < 8; h++) {
                query.putScalar(new long[]{0, 0, h, 0}, 1.0);
            }
            for (int h = 0; h < 2; h++) {
                // HALF automatic scale gives logits [-0.693359375, 0].
                // The oracle below also accounts for BFLOAT16 input rounding.
                initialKeys.putScalar(new long[]{0, 0, h, 0}, -11.09375);
            }
            initialValues.assign(4096.0);
            value.assign(-2048.0);
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("query", query);
            inputs.put("key", key);
            inputs.put("value", value);
            inputs.put("position", position);
            inputs.put("bias", bias);
            inputs.put("keys", nativeKeys);
            inputs.put("values", nativeValues);
            nativeKeys.assign(initialKeys);
            nativeValues.assign(initialValues);
            addHalfGqaLiveCacheGraph(nativeGraph, dtype);
            // Explicit native CUDA-graph test arm: attention is opt-in for
            // Triton, so leave it native and verify actual replay below.
            environment.setTritonCompileAll(false);
            environment.setTritonIncludeTypes("");
            nativeGraph.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
            INDArray nativeOutput = nativeGraph.output(inputs, "attention").get("attention");
            assertEquals(dtype, nativeOutput.dataType());
            float[] expected = nativeOutput.data().asFloat();
            float[] expectedKeys = nativeKeys.data().asFloat();
            float[] expectedValues = nativeValues.data().asFloat();
            for (int step = 0; step < 6; step++) {
                nativeKeys.assign(initialKeys);
                nativeValues.assign(initialValues);
                INDArray repeated = nativeGraph.output(inputs, "attention").get("attention");
                assertArrayEquals(expected, repeated.data().asFloat(), 0.0f,
                        "native accumulator scratch replay at step " + step);
            }
            DspPlanAssertions.assertTotalGraphReplaysAtLeast(nativeGraph, 1,
                    "native GQA scratch must be exercised under capture/replay");
            DspPlanAssertions.assertNoCaptureFailures(nativeGraph, "native GQA scratch");
            assertNotEquals("Triton GPU", Nd4j.getNativeOps().getPlanSegmentCompiledBackend(
                    DspPlanAssertions.getPlanHandleForQuery(nativeGraph), 0),
                    "native scratch replay must not substitute the Triton emitter");

            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ATTENTION");
            environment.setTritonGraphCapture(true);
            addHalfGqaLiveCacheGraph(tritonGraph, dtype);
            tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);
            inputs.put("keys", replayKeys);
            inputs.put("values", replayValues);
            long mismatches = 0;
            for (int step = 0; step < 6; step++) {
                replayKeys.assign(initialKeys);
                replayValues.assign(initialValues);
                INDArray output = tritonGraph.output(inputs, "attention").get("attention");
                assertEquals(dtype, output.dataType());
                float[] actual = output.data().asFloat();
                long stepMismatches = 0;
                double maxAbsDiff = 0.0;
                for (int i = 0; i < expected.length; i++) {
                    if (Float.floatToIntBits(expected[i]) != Float.floatToIntBits(actual[i])) {
                        stepMismatches++;
                        maxAbsDiff = Math.max(maxAbsDiff, Math.abs((double) expected[i] - actual[i]));
                    }
                }
                assertArrayEquals(expectedKeys, replayKeys.data().asFloat(), 0.0f,
                        "complete key-cache writeback at step " + step);
                assertArrayEquals(expectedValues, replayValues.data().asFloat(), 0.0f,
                        "complete value-cache writeback at step " + step);
                log.info("HALF_GQA_LIVE_CACHE dtype={} step={} mismatches={}/{} maxAbsDiff={} native={} triton={}",
                        dtype, step, stepMismatches, expected.length, maxAbsDiff, expected[0], actual[0]);
                mismatches += stepMismatches;
            }
            DspPlanAssertions.assertOpCompiled(tritonGraph, "dot_product_attention_v2",
                    "HALF live-cache GQA numerical contract");
            DspPlanAssertions.assertAllSegmentsCompiledWith(tritonGraph, "Triton GPU",
                    "HALF live-cache GQA numerical contract");
            double unnormalized = Math.exp(initialKeys.getDouble(0, 0, 0, 0) / 16.0);
            double oracle = (4096.0 * unnormalized - 2048.0) / (1.0 + unnormalized);
            assertEquals(oracle, expected[0], dtype == DataType.HALF ? 0.0005 : 0.01,
                    "Native auxiliary-output rounding must not erase the attention result");
            assertEquals(0L, mismatches,
                    "HALF live-cache attention changed arithmetic after native warmup");
            if (dtype == DataType.BFLOAT16) {
                // Both values are finite BF16. Summing unnormalized P*V in
                // FLOAT32 overflows, although their weighted mean is finite.
                float largeValue = Math.scalb(1.0f, 127);
                query.assign(0);
                initialKeys.assign(0);
                initialValues.assign(largeValue);
                value.assign(largeValue);
                inputs.put("keys", nativeKeys);
                inputs.put("values", nativeValues);
                nativeKeys.assign(initialKeys);
                nativeValues.assign(initialValues);
                float[] largeNative = nativeGraph.output(inputs, "attention")
                        .get("attention").data().asFloat();
                inputs.put("keys", replayKeys);
                inputs.put("values", replayValues);
                replayKeys.assign(initialKeys);
                replayValues.assign(initialValues);
                float[] largeTriton = tritonGraph.output(inputs, "attention")
                        .get("attention").data().asFloat();
                for (int i = 0; i < largeNative.length; i++) {
                    assertEquals(largeValue, largeNative[i], "native BF16 finite weighted mean " + i);
                    assertEquals(largeValue, largeTriton[i], "Triton BF16 finite weighted mean " + i);
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            environment.setTritonGraphCapture(captureBefore);
        }
    }

    /**
     * A causal bias ending in headDim is not a BHSD past-key tensor. Qwen's
     * [1,1,W,256] bias used to make Triton infer one KV head for two-head BSHD
     * caches. Distinct head/row values expose both wrong reads and wrong scatter
     * strides; a 257-column control separates that shape collision from GQA math.
     */
    @ParameterizedTest(name = "GQA bias/cache roles: width={0}, liveCache={1}, capacity={2}")
    @CsvSource({"1,true,256", "2,true,256", "1,false,256", "2,false,256",
            "1,true,257", "2,true,257", "1,false,257", "2,false,257"})
    public void testTritonGqaBiasIsNotPastKey(int width, boolean liveCache, int capacity) {
        assumeCuda();
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        boolean captureBefore = environment.tritonGraphCapture();
        final int qHeads = 8, kvHeads = 2, headDim = 256;
        final int positionValue = liveCache ? 1 : 0;
        float[] keyData = new float[width * kvHeads * headDim];
        float[] valueData = new float[keyData.length];
        float[] initialKeyData = new float[capacity * kvHeads * headDim];
        float[] initialValueData = new float[initialKeyData.length];
        for (int row = 0; row < capacity; row++) {
            for (int head = 0; head < kvHeads; head++) {
                for (int dim = 0; dim < headDim; dim++) {
                    int index = (row * kvHeads + head) * headDim + dim;
                    initialKeyData[index] = row == 0 ? 100 + head : -7;
                    initialValueData[index] = row == 0 ? 2 + 8 * head : -9;
                }
            }
        }
        for (int row = 0; row < width; row++) {
            for (int head = 0; head < kvHeads; head++) {
                for (int dim = 0; dim < headDim; dim++) {
                    int index = (row * kvHeads + head) * headDim + dim;
                    keyData[index] = 10 + 2 * row + head;
                    valueData[index] = 4 + 2 * row + 8 * head;
                }
            }
        }
        float[] expectedKeys = initialKeyData.clone();
        float[] expectedValues = initialValueData.clone();
        System.arraycopy(keyData, 0, expectedKeys, positionValue * kvHeads * headDim, keyData.length);
        System.arraycopy(valueData, 0, expectedValues, positionValue * kvHeads * headDim, valueData.length);
        float[] biasData = new float[width * capacity];
        for (int row = 0; row < width; row++) {
            for (int col = positionValue + row + 1; col < capacity; col++) {
                biasData[row * capacity + col] = -1.0e9f;
            }
        }
        float[] expected = new float[width * qHeads * headDim];
        for (int row = 0; row < width; row++) {
            for (int head = 0; head < qHeads; head++) {
                // Q=0 gives uniform weights over the causal prefix. All these
                // means are exactly representable in HALF: 3/4 or 4/5, plus 8*h.
                float mean = (liveCache ? 3 : 4) + row + 8 * (head / (qHeads / kvHeads));
                for (int dim = 0; dim < headDim; dim++) {
                    expected[(row * qHeads + head) * headDim + dim] = mean;
                }
            }
        }
        try (SameDiff graph = SameDiff.create();
             INDArray query = Nd4j.zeros(DataType.HALF, 1, width, qHeads, headDim);
             INDArray key = Nd4j.create(keyData, new long[]{1, width, kvHeads, headDim},
                     new long[]{width * kvHeads * headDim, kvHeads * headDim, headDim, 1}, 'c', DataType.HALF);
             INDArray value = Nd4j.create(valueData, new long[]{1, width, kvHeads, headDim},
                     new long[]{width * kvHeads * headDim, kvHeads * headDim, headDim, 1}, 'c', DataType.HALF);
             INDArray initialKeys = Nd4j.create(initialKeyData, new long[]{1, capacity, kvHeads, headDim},
                     new long[]{capacity * kvHeads * headDim, kvHeads * headDim, headDim, 1}, 'c', DataType.HALF);
             INDArray initialValues = Nd4j.create(initialValueData, new long[]{1, capacity, kvHeads, headDim},
                     new long[]{capacity * kvHeads * headDim, kvHeads * headDim, headDim, 1}, 'c', DataType.HALF);
             INDArray keys = initialKeys.dup();
             INDArray values = initialValues.dup();
             INDArray position = Nd4j.scalar(DataType.INT64, positionValue);
             INDArray emptyKeys = Nd4j.empty(DataType.HALF);
             INDArray emptyValues = Nd4j.empty(DataType.HALF);
             INDArray emptyPosition = Nd4j.empty(DataType.INT64);
             INDArray bias = Nd4j.create(biasData, new long[]{1, 1, width, capacity}, 'c')) {
            SDVariable q = graph.placeHolder("query", query.dataType(), query.shape());
            SDVariable k = graph.placeHolder("key", key.dataType(), key.shape());
            SDVariable v = graph.placeHolder("value", value.dataType(), value.shape());
            SDVariable b = graph.placeHolder("bias", bias.dataType(), bias.shape());
            // Match GGUF prefill's nine-input ABI: empty caches preserve bias
            // at input 8, where the native op slices it to the current key length.
            SDVariable kc = liveCache ? graph.placeHolder("keys", keys.dataType(), keys.shape())
                    : graph.constant("empty_keys", emptyKeys);
            SDVariable vc = liveCache ? graph.placeHolder("values", values.dataType(), values.shape())
                    : graph.constant("empty_values", emptyValues);
            SDVariable cp = liveCache ? graph.placeHolder("position", DataType.INT64)
                    : graph.constant("empty_position", emptyPosition);
            SDVariable attention = new DotProductAttentionV2(graph, q, v, k, null, null,
                    kc, vc, cp, b, 0.0, 0.0, false, false).outputVariable();
            graph.updateVariableNameAndReference(attention, "attention");
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("query", query);
            inputs.put("key", key);
            inputs.put("value", value);
            inputs.put("bias", bias);
            if (liveCache) {
                inputs.put("keys", keys);
                inputs.put("values", values);
                inputs.put("position", position);
            }
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ATTENTION");
            environment.setTritonGraphCapture(true);
            graph.setGraphExecutionMode(GraphExecutionMode.TRITON);
            for (int step = 0; step < 6; step++) {
                keys.assign(initialKeys);
                values.assign(initialValues);
                INDArray output = graph.output(inputs, "attention").get("attention");
                assertArrayEquals(expected, output.data().asFloat(), 0.0f,
                        "causal grouped-head mean at step " + step);
                if (liveCache) {
                    assertArrayEquals(expectedKeys, keys.data().asFloat(), 0.0f,
                            "full key scatter and untouched sentinels at step " + step);
                    assertArrayEquals(expectedValues, values.data().asFloat(), 0.0f,
                            "full value scatter and untouched sentinels at step " + step);
                }
            }
            DspPlanAssertions.assertOpCompiled(graph, "dot_product_attention_v2", "bias/cache role collision");
            DspPlanAssertions.assertAllSegmentsCompiledWith(graph, "Triton GPU", "bias/cache role collision");
            DspPlanAssertions.assertTotalGraphReplaysAtLeast(graph, 1, "bias/cache role collision");
            DspPlanAssertions.assertNoCaptureFailures(graph, "bias/cache role collision");
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            environment.setTritonGraphCapture(captureBefore);
        }
    }

    private static void addHalfGqaLiveCacheGraph(SameDiff graph, DataType dtype) {
        SDVariable query = graph.placeHolder("query", dtype, 1, 1, 8, 256);
        SDVariable key = graph.placeHolder("key", dtype, 1, 1, 2, 256);
        SDVariable value = graph.placeHolder("value", dtype, 1, 1, 2, 256);
        SDVariable keys = graph.placeHolder("keys", dtype, 1, 2, 2, 256);
        SDVariable values = graph.placeHolder("values", dtype, 1, 2, 2, 256);
        SDVariable position = graph.placeHolder("position", DataType.INT64);
        SDVariable bias = graph.placeHolder("bias", dtype, 1, 1, 1, 2);
        SDVariable attention = new DotProductAttentionV2(graph, query, value, key, null, null,
                keys, values, position, bias, 0.0, 0.0, false, false).outputVariable();
        graph.updateVariableNameAndReference(attention, "attention");
    }

    /** Compiled attention mutates cache inputs, not only its ordinary outputs. */
    @ParameterizedTest(name = "KV publication {0}, Triton capture={1}")
    @CsvSource({"TRITON,true", "TRITON,false", "CUDA_GRAPHS,false"})
    public void testHostRestoredGqaCachePublication(GraphExecutionMode mode, boolean tritonCapture) {
        assumeCuda();
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        boolean captureBefore = environment.tritonGraphCapture();
        try (SameDiff graph = SameDiff.create();
             INDArray query = Nd4j.zeros(DataType.FLOAT, 1, 1, 8, 256);
             INDArray key = Nd4j.zeros(DataType.FLOAT, 1, 1, 2, 256);
             INDArray value = Nd4j.zeros(DataType.FLOAT, 1, 1, 2, 256);
             INDArray keys = Nd4j.zeros(DataType.FLOAT, 1, 2, 2, 256);
             INDArray values = Nd4j.zeros(DataType.FLOAT, 1, 2, 2, 256);
             INDArray position = Nd4j.scalar(DataType.INT64, 1);
             INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 2)) {
            environment.setTritonCompileAll(mode == GraphExecutionMode.TRITON);
            environment.setTritonIncludeTypes(mode == GraphExecutionMode.TRITON ? "ATTENTION" : "");
            environment.setTritonGraphCapture(tritonCapture);
            addHalfGqaLiveCacheGraph(graph, DataType.FLOAT);
            graph.setGraphExecutionMode(mode);
            Map<String, INDArray> inputs = Map.of("query", query, "key", key, "value", value,
                    "keys", keys, "values", values, "position", position, "bias", bias);
            long mismatches = 0;
            for (int step = 0; step < 6; step++) {
                key.assign(step + 2.0);
                value.assign(step + 3.0);
                // Public host writes mark primary storage newer than special.
                // No manual sync or device-actual tagging belongs in this test.
                for (long i = 0; i < keys.length(); i++) {
                    keys.data().put(i, 0.0f);
                    values.data().put(i, 1.0f);
                }
                float[] output = graph.output(inputs, "attention").get("attention").data().asFloat();
                for (float element : output) {
                    assertEquals((step + 4.0f) / 2, element, "uniform two-key attention at step " + step);
                }
                float[] deviceKeys = copyGqaDeviceValues(keys);
                float[] deviceValues = copyGqaDeviceValues(values);
                float[] observedKeys = keys.data().asFloat();
                float[] observedValues = values.data().asFloat();
                long stepMismatches = 0;
                for (int i = 0; i < observedKeys.length; i++) {
                    float expectedKey = i < 512 ? 0.0f : step + 2.0f;
                    float expectedValue = i < 512 ? 1.0f : step + 3.0f;
                    assertEquals(expectedKey, deviceKeys[i], "actual device key at step " + step + " index " + i);
                    assertEquals(expectedValue, deviceValues[i], "actual device value at step " + step + " index " + i);
                    if (observedKeys[i] != expectedKey || observedValues[i] != expectedValue) stepMismatches++;
                }
                log.info("GQA_HOST_CACHE_PUBLICATION step={} mismatches={} keyRow1={} valueRow1={} deviceKey={} deviceValue={}",
                        step, stepMismatches, observedKeys[512], observedValues[512], deviceKeys[512], deviceValues[512]);
                mismatches += stepMismatches;
            }
            if (mode == GraphExecutionMode.TRITON) {
                DspPlanAssertions.assertOpCompiled(graph, "dot_product_attention_v2", "host-restored KV publication");
                DspPlanAssertions.assertAllSegmentsCompiledWith(graph, "Triton GPU", "host-restored KV publication");
                if (tritonCapture) {
                    DspPlanAssertions.assertTotalGraphReplaysAtLeast(graph, 1, "Triton KV publication replay");
                }
            } else {
                DspPlanAssertions.assertTotalGraphReplaysAtLeast(graph, 1, "native KV publication");
                assertNotEquals("Triton GPU", Nd4j.getNativeOps().getPlanSegmentCompiledBackend(
                        DspPlanAssertions.getPlanHandleForQuery(graph), 0), "native CUDA-graph arm");
            }
            assertEquals(0L, mismatches, "Compiled attention must publish its caller-owned cache writes");
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            environment.setTritonGraphCapture(captureBefore);
        }
    }

    /** Cache write guards must follow native optional-input rank semantics. */
    @Test
    public void testAttentionBiasAndScalarCacheSentinelRemainReadOnly() {
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try (SameDiff graph = SameDiff.create();
             INDArray query = Nd4j.zeros(DataType.FLOAT, 1, 1, 2, 2);
             INDArray key = Nd4j.zeros(DataType.FLOAT, 1, 2, 1, 2);
             INDArray value = Nd4j.ones(DataType.FLOAT, 1, 2, 1, 2);
             INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 2);
             INDArray sentinel = Nd4j.scalar(DataType.FLOAT, 123.0);
             INDArray position = Nd4j.scalar(DataType.INT64, 0)) {
            environment.setTritonCompileAll(false);
            environment.setTritonIncludeTypes("");
            SDVariable q = graph.placeHolder("q", DataType.FLOAT, 1, 1, 2, 2);
            SDVariable k = graph.placeHolder("k", DataType.FLOAT, 1, 2, 1, 2);
            SDVariable v = graph.placeHolder("v", DataType.FLOAT, 1, 2, 1, 2);
            SDVariable b = graph.placeHolder("bias_slot5", DataType.FLOAT, 1, 2);
            SDVariable s = graph.placeHolder("scalar_slot6", DataType.FLOAT);
            SDVariable p = graph.placeHolder("position_slot7", DataType.INT64);
            // Native DPA treats rank-0 input 6 as absent: input 5 is bias, not KV.
            SDVariable attention = new DotProductAttentionV2(graph, q, v, k, null, null,
                    b, s, p, null, 0.0, 0.0, false, false).outputVariable();
            graph.updateVariableNameAndReference(attention, "attention");
            graph.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
            Map<String, INDArray> inputs = Map.of("q", query, "k", key, "v", value,
                    "bias_slot5", bias, "scalar_slot6", sentinel, "position_slot7", position);
            for (int step = 0; step < 7; step++) {
                sentinel.data().put(0, 123.0f + step);
                bias.data().put(0, 0.0f);
                bias.data().put(1, 0.0f);
                float[] output = graph.output(inputs, "attention").get("attention").data().asFloat();
                assertNotEquals("DEVICE", Nd4j.getAffinityManager().getActiveLocation(sentinel).name(),
                        "unwritten scalar sentinel must retain host actuality at step " + step);
                assertNotEquals("DEVICE", Nd4j.getAffinityManager().getActiveLocation(bias).name(),
                        "read-only bias must retain host actuality at step " + step);
                assertEquals(123.0f + step, sentinel.getFloat(0));
                for (float element : output) assertEquals(1.0f, element);
            }
            DspPlanAssertions.assertTotalGraphReplaysAtLeast(graph, 1, "read-only attention inputs");
            assertNotEquals("Triton GPU", Nd4j.getNativeOps().getPlanSegmentCompiledBackend(
                    DspPlanAssertions.getPlanHandleForQuery(graph), 0), "native CUDA-graph sentinel arm");
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    /** Independent device readback without changing the caller cache's actuality. */
    private static float[] copyGqaDeviceValues(INDArray array) {
        var ops = Nd4j.getNativeOps();
        var pointer = ops.dbSpecialBuffer(array.data().opaqueBuffer());
        assertNotNull(pointer);
        assertFalse(pointer.isNull());
        var borrowed = ops.dbCreateExternalDataBuffer(array.length(), array.dataType().toInt(), null, pointer);
        assertNotNull(borrowed);
        try (INDArray copy = Nd4j.createUninitialized(array.dataType(), array.shape(), 'c')) {
            ops.copyBuffer(copy.data().opaqueBuffer(), array.length(), borrowed, 0, 0);
            Nd4j.getExecutioner().commit();
            return copy.data().asFloat();
        } finally {
            ops.deleteDataBuffer(borrowed);
        }
    }

    private void assertDpaV2GqaPrefillParity(
            Map<String, INDArray> placeholders,
            int batch, int sequence, int qHeads, int kvHeads, int headDim,
            double tolerance, String diagnosticLabel) {
        INDArray reference = null;
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            try (SameDiff nativeGraph = SameDiff.create()) {
                SDVariable query = nativeGraph.placeHolder(
                        "query", DataType.FLOAT, batch, sequence, qHeads, headDim);
                SDVariable value = nativeGraph.placeHolder(
                        "value", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable key = nativeGraph.placeHolder(
                        "key", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable bias = nativeGraph.placeHolder(
                        "attention_bias", DataType.FLOAT, batch, 1, sequence, sequence);
                SDVariable emptyKeyCache = nativeGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueCache = nativeGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyCachePosition = nativeGraph.constant(Nd4j.empty(DataType.INT64));
                SDVariable attention = new DotProductAttentionV2(
                        nativeGraph, query, value, key, null, null,
                        emptyKeyCache, emptyValueCache, emptyCachePosition, bias,
                        0.0, 0.0, false, false).outputVariable();
                nativeGraph.updateVariableNameAndReference(attention, "attention");
                nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                reference = nativeGraph.output(placeholders, "attention").get("attention").dup();
            }

            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ATTENTION");

            long totalMismatches = 0;
            StringBuilder differences = new StringBuilder();
            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable query = tritonGraph.placeHolder(
                        "query", DataType.FLOAT, batch, sequence, qHeads, headDim);
                SDVariable value = tritonGraph.placeHolder(
                        "value", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable key = tritonGraph.placeHolder(
                        "key", DataType.FLOAT, batch, sequence, kvHeads, headDim);
                SDVariable bias = tritonGraph.placeHolder(
                        "attention_bias", DataType.FLOAT, batch, 1, sequence, sequence);
                SDVariable emptyKeyCache = tritonGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueCache = tritonGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyCachePosition = tritonGraph.constant(Nd4j.empty(DataType.INT64));
                SDVariable attention = new DotProductAttentionV2(
                        tritonGraph, query, value, key, null, null,
                        emptyKeyCache, emptyValueCache, emptyCachePosition, bias,
                        0.0, 0.0, false, false).outputVariable();
                tritonGraph.updateVariableNameAndReference(attention, "attention");
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                float[] expected = reference.toFloatVector();
                for (int step = 0; step < 4; step++) {
                    float[] actual = tritonGraph.output(placeholders, "attention")
                            .get("attention").toFloatVector();
                    long stepMismatches = 0;
                    double maxAbsDiff = 0.0;
                    int maxDiffIndex = -1;
                    for (int i = 0; i < expected.length; i++) {
                        double absDiff = Math.abs((double) expected[i] - actual[i]);
                        if (absDiff > maxAbsDiff) {
                            maxAbsDiff = absDiff;
                            maxDiffIndex = i;
                        }
                        if (absDiff > tolerance) {
                            stepMismatches++;
                            if (differences.length() < 768) {
                                differences.append(" step=").append(step)
                                        .append(" element=").append(i)
                                        .append(" native=").append(expected[i])
                                        .append(" triton=").append(actual[i]);
                            }
                        }
                    }
                    totalMismatches += stepMismatches;
                    log.info("{} step={} mismatches={}/{} maxAbsDiff={} maxDiffIndex={} "
                                    + "native={} triton={}",
                            diagnosticLabel, step, stepMismatches, expected.length,
                            maxAbsDiff, maxDiffIndex,
                            maxDiffIndex < 0 ? 0.0f : expected[maxDiffIndex],
                            maxDiffIndex < 0 ? 0.0f : actual[maxDiffIndex]);
                }

                DspPlanAssertions.assertOpCompiled(
                        tritonGraph, "dot_product_attention_v2", diagnosticLabel);
                DspPlanAssertions.assertAllSegmentsCompiledWith(
                        tritonGraph, "Triton GPU", diagnosticLabel);
            }
            assertEquals(0L, totalMismatches,
                    diagnosticLabel + " changed compiled output:" + differences);
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            if (reference != null && !reference.wasClosed()) reference.close();
        }
    }

    /**
     * A requested graph output can be produced in the middle of a Triton-fused
     * range while another requested output is terminal. The fused kernel must
     * materialize both values, including after a final-only kernel for the same
     * slot range and shape has already populated the process-wide cache.
     */
    @Test
    @DisplayName("Triton replay materializes requested fused intermediates across cache variants")
    public void testRequestedFusedIntermediateSurvivesReplayAndCacheReuse() {
        final int steps = 18;
        final int length = 256;
        INDArray[] inputs = new INDArray[steps];
        int[][] expectedIntermediateBits = new int[steps][length];
        float[][] expectedFinal = new float[steps][length];

        for (int step = 0; step < steps; step++) {
            float[] values = new float[length];
            for (int i = 0; i < length; i++) {
                values[i] = (step + 1) * 0.03125f + (i - 128) * 0.00390625f;
            }
            inputs[step] = Nd4j.createFromArray(values).reshape(1, length);
        }

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        try (SameDiff referenceGraph = SameDiff.create()) {
            SDVariable input = referenceGraph.placeHolder("input", DataType.FLOAT, 1, length);
            SDVariable intermediate = input.mul("requested_intermediate", 2.0);
            SDVariable shifted = intermediate.add("shifted", 0.25);
            referenceGraph.nn.sigmoid("final", shifted);
            referenceGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            for (int step = 0; step < steps; step++) {
                placeholders.put("input", inputs[step]);
                Map<String, INDArray> outputs = referenceGraph.output(
                        placeholders, "requested_intermediate", "final");
                INDArray expectedIntermediate = outputs.get("requested_intermediate");
                INDArray expectedTerminal = outputs.get("final");
                for (int i = 0; i < length; i++) {
                    expectedIntermediateBits[step][i] =
                            Float.floatToRawIntBits(expectedIntermediate.getFloat(i));
                    expectedFinal[step][i] = expectedTerminal.getFloat(i);
                }
            }
        }

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ELEMENTWISE");

            try (SameDiff cachePrimer = SameDiff.create();
                 SameDiff requestedGraph = SameDiff.create()) {
                SDVariable primerInput = cachePrimer.placeHolder("input", DataType.FLOAT, 1, length);
                SDVariable primerIntermediate = primerInput.mul("requested_intermediate", 2.0);
                SDVariable primerShifted = primerIntermediate.add("shifted", 0.25);
                cachePrimer.nn.sigmoid("final", primerShifted);
                cachePrimer.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < steps; step++) {
                    placeholders.put("input", inputs[step]);
                    INDArray terminal = cachePrimer.output(placeholders, "final").get("final");
                    assertFalse(terminal.isNaN().any(), "Cache primer produced NaN at step " + step);
                }
                DspPlanAssertions.assertPhaseReached(
                        cachePrimer, PlanPhase.SHAPES_FROZEN, "final-only Triton cache primer");

                SDVariable requestedInput = requestedGraph.placeHolder("input", DataType.FLOAT, 1, length);
                SDVariable requestedIntermediate = requestedInput.mul("requested_intermediate", 2.0);
                SDVariable requestedShifted = requestedIntermediate.add("shifted", 0.25);
                requestedGraph.nn.sigmoid("final", requestedShifted);
                requestedGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < steps; step++) {
                    placeholders.put("input", inputs[step]);
                    Map<String, INDArray> outputs = requestedGraph.output(
                            placeholders, "requested_intermediate", "final");
                    INDArray actualIntermediate = outputs.get("requested_intermediate");
                    INDArray actualFinal = outputs.get("final");
                    for (int i = 0; i < length; i++) {
                        int actualBits = Float.floatToRawIntBits(actualIntermediate.getFloat(i));
                        assertEquals(expectedIntermediateBits[step][i], actualBits,
                                String.format("requested intermediate step %d element %d", step, i));
                        assertEquals(expectedFinal[step][i], actualFinal.getFloat(i), 1e-6f,
                                String.format("terminal output step %d element %d", step, i));
                    }
                }
                DspPlanAssertions.assertPhaseReached(
                        requestedGraph, PlanPhase.SHAPES_FROZEN, "requested-output Triton graph");
                DspPlanAssertions.assertNoCaptureFailures(
                        requestedGraph, "requested fused intermediate replay");
                assertTrue(DspPlanAssertions.getTotalGraphReplays(requestedGraph) > 0,
                        "Requested-output graph never reached replay");
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            for (INDArray input : inputs) {
                if (input != null && !input.wasClosed()) input.close();
            }
        }
    }
}
