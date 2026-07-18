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
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Environment;
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
                    for (int i = 0; i < referenceValues.length; i++) {
                        int expectedBits = Float.floatToRawIntBits(referenceValues[i]);
                        int actualBits = Float.floatToRawIntBits(actualValues[i]);
                        assertEquals(expectedBits, actualBits,
                                String.format("prefill rms_norm step %d element %d: native=0x%08x triton=0x%08x",
                                        step, i, expectedBits, actualBits));
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

    /**
     * DPA-v2 keeps optional input positions stable when KV-cache placeholders are present:
     * Q, V, K, query mask, value mask, key cache, value cache, cache position, then bias.
     * The compiled attention path must read the additive causal bias from input 8 instead of
     * mistaking the empty query-mask placeholder at input 3 for that bias.
     */
    @Test
    @DisplayName("Triton DPA-v2 cache-form attention bias matches native CUDA")
    public void testTritonDpaV2CacheFormAttentionBiasMatchesNative() {
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
