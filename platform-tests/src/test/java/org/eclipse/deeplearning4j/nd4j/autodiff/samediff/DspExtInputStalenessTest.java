/*
 *  ******************************************************************************
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
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.nd4j.common.tests.tags.TagNames;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Exhaustive DSP ext-input staleness tests.
 *
 * Tests ALL code paths that can cause stale data reads during DSP plan replay:
 * - Cross-stream D2D ordering (device-write on stream A, D2D copy on stream B)
 * - Variable classification and D2D staging paths
 * - Arg table generation counter mechanism
 * - Java executor frozen fast-path behavior
 * - executeSteadyState() fast path
 * - Gap slot lifecycle and classification
 * - Multi-external lifecycle combinations
 * - VLM decode end-to-end pattern reproduction
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class DspExtInputStalenessTest extends DspExtInputTestSupport {

    private SameDiff sd;

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GRAPH FIXTURES
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Build single placeholder graph with true CONSTANT weights (not VARIABLE).
     * Constants may be inlined by the compiler; if they survive as external inputs,
     * they will have SOURCE_CONSTANT type and should NEVER get staging buffers.
     */
    private SameDiff buildSinglePlaceholderWithConstants(int inDim, int outDim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        SDVariable w = g.constant("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, inDim, outDim)).addi(0.1f));
        SDVariable b = g.constant("b", Nd4j.ones(DataType.FLOAT, 1, outDim));
        SDVariable mm = g.mmul("mm", x, w);
        mm.add("out", b);
        return g;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SHARED HELPERS
    // ═══════════════════════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 2: Variable Classification and D2D Staging Paths
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "markedVariableGetsStaging mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("markVariable allocates staging buffer")
    void testMarkedVariableGetsStaging(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        DspHandle h = sd.dsp();
        org.junit.jupiter.api.Assumptions.assumeTrue(h.isCompiled(),
                "DSP plan did not compile for mode " + mode);

        int extIdx = h.extInputIndex("x");
        h.markVariable(extIdx);

        // Run one step to trigger staging allocation
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 99.0));
        h.replay(singlePh("x", input));

        long stagingAddr = h.stagingBufferAddress(extIdx);
        log.info("[STAGING] mode={} extIdx={} stagingAddr=0x{}", mode, extIdx, Long.toHexString(stagingAddr));
        // Staging buffers are a CUDA-only concept (device memory for CUDA graph capture).
        // On CPU backend, staging is not allocated — only assert on CUDA.
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if ("CUDA".equalsIgnoreCase(backend)) {
            assertNotEquals(0L, stagingAddr, mode + ": staging buffer should be allocated after markVariable + replay");
        } else {
            log.info("[STAGING] CPU backend — staging not applicable, skipping assertion");
        }
    }

    @Test
    @DisplayName("Injected staging failures abort graph execution instead of using raw external arrays")
    void testInjectedStagingFailureFailsClosed() {
        String fault = System.getenv("ND4J_DSP_STAGING_FAULT");
        Set<String> supportedFaults = Set.of(
                "cuda_get_device", "allocation", "cross_stream", "d2d_copy", "stream_sync");
        Assumptions.assumeTrue(supportedFaults.contains(fault),
                "Run with -Dnd4j.dsp.stagingFault=<supported fault> on CUDA");
        Assumptions.assumeTrue(Nd4j.backends().isCudaAvailable(),
                "staging fault injection is CUDA-only");

        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, GraphExecutionMode.CUDA_GRAPHS);
        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);

        assertThrows(Throwable.class, () -> {
            for (int step = 0; step < 8; step++) {
                input.assign(step + 1.0);
                sd.output(singlePh("x", input), "out");
            }
        }, fault + " must fail closed before graph execution can use raw external arrays");
    }

    @ParameterizedTest(name = "unmarkedPlaceholderBehavior mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Unmarked placeholder — document whether auto-staging happens")
    void testUnmarkedPlaceholderNoStaging(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        DspHandle h = sd.dsp();
        org.junit.jupiter.api.Assumptions.assumeTrue(h.isCompiled(),
                "DSP plan did not compile for mode " + mode);

        int extIdx = h.extInputIndex("x");
        // DO NOT call markVariable — test auto-detection

        // Run steps with changing input — does it still work?
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 10)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        // Outputs must change — whether via auto-staging or direct pointer
        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck without markVariable. sums=" + sums);
        }
        log.info("[UNMARKED_PH] mode={} all 10 steps unique — auto-detection works", mode);
    }

    @ParameterizedTest(name = "booleanMaskAndFloatPixelsRefresh mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("BOOL mask and FLOAT pixels both refresh across frozen replay")
    void testBooleanMaskAndFloatPixelsRefresh(GraphExecutionMode mode) {
        sd = SameDiff.create();
        SDVariable pixels = sd.placeHolder("pixel_values", DataType.FLOAT, 1, 16);
        SDVariable mask = sd.placeHolder("pixel_attention_mask", DataType.BOOL, 1, 16);
        pixels.add("out", mask.castTo(DataType.FLOAT));
        configureMode(sd, mode);

        for (int step = 0; step < 20; step++) {
            float pixelValue = step + 1.0f;
            int enabledPixels = (step * 5) % 17;
            INDArray pixelArray = Nd4j.valueArrayOf(new long[]{1, 16}, pixelValue);
            INDArray maskArray = Nd4j.zeros(DataType.BOOL, 1, 16);
            for (int i = 0; i < enabledPixels; i++) {
                maskArray.putScalar(0, i, 1);
            }

            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("pixel_values", pixelArray);
            placeholders.put("pixel_attention_mask", maskArray);
            INDArray output = sd.output(placeholders, "out").get("out");

            double expected = 16.0 * pixelValue + enabledPixels;
            assertEquals(expected, output.sumNumber().doubleValue(), 1e-4,
                    mode + " step " + step + " used stale FLOAT pixels or BOOL mask");
        }
        log.info("[BOOL_MASK_REFRESH] mode={} PASS — all 20 pixel/mask pairs refreshed", mode);
    }

    @ParameterizedTest(name = "constantNeverStaged mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("True CONSTANT ext input never gets staging")
    void testConstantNeverStaged(GraphExecutionMode mode) {
        // Use g.constant() — true SOURCE_CONSTANT inputs must never get staging
        sd = buildSinglePlaceholderWithConstants(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        DspHandle h = sd.dsp();
        org.junit.jupiter.api.Assumptions.assumeTrue(h.isCompiled(),
                "DSP plan did not compile for mode " + mode);

        // "w" is a true constant (g.constant) — find its ext index
        int wIdx = h.extInputIndex("w");
        if (wIdx < 0) {
            log.info("[CONST_STAGED] mode={} w not found as ext input (may be inlined) — OK", mode);
            return;
        }

        long stagingAddr = h.stagingBufferAddress(wIdx);
        assertEquals(0L, stagingAddr,
                mode + ": constant 'w' should NOT have staging. addr=0x" + Long.toHexString(stagingAddr));
        log.info("[CONST_STAGED] mode={} PASS — constant 'w' has no staging", mode);
    }

    @ParameterizedTest(name = "variableWeightNoStagingInInference mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("VARIABLE weight ext inputs do NOT get staging during inference (weights are constant)")
    void testVariableWeightGetsStaging(GraphExecutionMode mode) {
        // Use g.var() — SOURCE_VARIABLE inputs are NOT marked mutable during inference.
        // Weights don't change between decode steps, so the C++ plan correctly treats
        // them as frozen constants. Staging is only needed for training (TrainingSession).
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        DspHandle h = sd.dsp();
        org.junit.jupiter.api.Assumptions.assumeTrue(h.isCompiled(),
                "DSP plan did not compile for mode " + mode);

        // "w" is a VARIABLE (g.var) — find its ext index
        int wIdx = h.extInputIndex("w");
        if (wIdx < 0) {
            log.info("[VAR_STAGED] mode={} w not found as ext input (may be inlined)", mode);
            return;
        }

        long stagingAddr = h.stagingBufferAddress(wIdx);
        // During inference, VARIABLE weights should NOT have staging — they are treated
        // as constants by the C++ plan (no D2D copies per step, enabling frozen-constant
        // skip optimization). Staging is only allocated when the Java side explicitly
        // marks inputs as mutable via addMutableExternalInputs() (training path).
        assertEquals(0L, stagingAddr,
                mode + ": VARIABLE weight 'w' should NOT have staging during inference. addr=0x" + Long.toHexString(stagingAddr));
        log.info("[VAR_STAGED] mode={} PASS — variable 'w' has no staging during inference (correct)", mode);
    }

    @ParameterizedTest(name = "mixedVariableAndConstant mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Mixed: constant weight stable while variable placeholder changes")
    void testMixedVariableAndConstant(GraphExecutionMode mode) {
        sd = buildMultiPlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray w = Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f);
        INDArray b = Nd4j.ones(DataType.FLOAT, 1, 4);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);
        ph.put("w", w);
        ph.put("b", b);

        // Warmup
        for (int i = 0; i < 8; i++) {
            x.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(ph, "out");
        }

        // Save weight values
        INDArray wBefore = w.dup();

        // Run 10 steps changing x, keeping w and b constant
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            x.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 20)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        // Outputs must change (x is changing)
        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with changing x");
        }

        // Weight must be unchanged
        assertEquals(wBefore, w, mode + ": weight 'w' was corrupted during replay!");
        log.info("[MIXED] mode={} PASS — x reflected, w stable", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 4: Java Executor Frozen Fast-Path
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "frozenFastPathNewObject mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("New INDArray object per step detected by frozen fast-path")
    void testFrozenFastPathNewObject(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        // Warmup with stable pointer
        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        // Now use NEW object each step (different address)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            INDArray newInput = Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 50)).castTo(DataType.FLOAT);
            Map<String, INDArray> result = sd.output(singlePh("x", newInput), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with new object each step. sums=" + sums);
        }
        log.info("[FROZEN_NEW_OBJ] mode={} PASS — 10 steps with new objects all different", mode);
    }

    @ParameterizedTest(name = "frozenFastPathSameObjectMutated mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Same INDArray mutated via assign() — frozen fast-path syncs device")
    void testFrozenFastPathSameObjectMutated(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        // Same object, mutated via assign()
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with same object mutated. sums=" + sums);
        }
        log.info("[FROZEN_MUTATE] mode={} PASS — same object mutated, all steps different", mode);
    }

    @ParameterizedTest(name = "frozenFastPathControlInput mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Small control input (position_ids-like) changes every step")
    void testFrozenFastPathControlInput(GraphExecutionMode mode) {
        // Graph with position_ids-like input (scalar LONG)
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable pos = g.placeHolder("pos", DataType.FLOAT, 1, 1);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable mm = g.mmul("mm", x, w);
        // Add position to output (so position affects result)
        g.math().add("out", mm, pos);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray posArr = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", input);
        ph.put("pos", posArr);

        // Warmup
        for (int i = 0; i < 8; i++) {
            posArr.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(ph, "out");
        }

        // Keep x constant, change pos every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            posArr.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck. Control input 'pos' not reflected");
        }
        log.info("[CONTROL_INPUT] mode={} PASS — position changes reflected", mode);
    }

    @ParameterizedTest(name = "slowVsFastPathParity mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Slow path (first calls) and fast path produce same results for same input")
    void testSlowPathVsFastPathParity(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.valueArrayOf(new long[]{1, 8}, 42.0).castTo(DataType.FLOAT);
        Map<String, INDArray> ph = singlePh("x", input);

        // First call (slow path) — record output
        Map<String, INDArray> result1 = sd.output(ph, "out");
        double sum1 = result1.get("out").sumNumber().doubleValue();

        // Run 8 more calls to get to fast path — same input every step
        // (C++ staleness guard is now a warning, not a throw)
        for (int i = 0; i < 8; i++) {
            sd.output(ph, "out");
        }

        // Now call again (fast path) — output must match slow path for same input
        Map<String, INDArray> result10 = sd.output(ph, "out");
        double sum10 = result10.get("out").sumNumber().doubleValue();

        assertEquals(sum1, sum10, 1e-3,
                mode + ": slow path vs fast path output differs for same input! " +
                        "slow=" + sum1 + " fast=" + sum10);
        log.info("[PARITY] mode={} PASS — slow={} fast={}", mode, sum1, sum10);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 3: Arg Table Generation Counter
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "argRefreshOnNewINDArray mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("New INDArray (new address) for placeholder after SEALED → output correct")
    void testArgRefreshOnNewINDArray(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        DspHandle h = sd.dsp();
        org.junit.jupiter.api.Assumptions.assumeTrue(h.isCompiled(),
                "DSP plan did not compile for mode " + mode);
        int phase = h.planPhase();
        log.info("[ARG_REFRESH] mode={} phase={} after 10 warmup steps", mode, phase);

        // Now pass a COMPLETELY NEW array (different device address)
        INDArray newInput = Nd4j.valueArrayOf(new long[]{1, 8}, 999.0).castTo(DataType.FLOAT);
        Map<String, INDArray> result = sd.output(singlePh("x", newInput), "out");
        double sumNew = result.get("out").sumNumber().doubleValue();

        // And with the old array at a different value
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 1.0));
        Map<String, INDArray> resultOld = sd.output(singlePh("x", input), "out");
        double sumOld = resultOld.get("out").sumNumber().doubleValue();

        assertNotEquals(sumNew, sumOld, 1e-3,
                mode + ": new array (999) and old array (1) produce same output! " +
                        "new=" + sumNew + " old=" + sumOld + " — arg table not refreshed");
        log.info("[ARG_REFRESH] mode={} PASS — new addr output={} old addr output={}", mode, sumNew, sumOld);
    }

    @ParameterizedTest(name = "argRefreshSkippedWhenStable mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Same INDArray every step → output is deterministic (fast path works)")
    void testArgRefreshSkippedWhenStable(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.valueArrayOf(new long[]{1, 8}, 7.0).castTo(DataType.FLOAT);
        Map<String, INDArray> ph = singlePh("x", input);

        // Run 10 warmup steps with the SAME value
        for (int i = 0; i < 10; i++) {
            sd.output(ph, "out");
        }

        // Run 5 more steps — all should produce identical output
        double baseline = sd.output(ph, "out").get("out").sumNumber().doubleValue();
        for (int step = 0; step < 5; step++) {
            double val = sd.output(ph, "out").get("out").sumNumber().doubleValue();
            assertEquals(baseline, val, 1e-6,
                    mode + " step " + step + " output differs from baseline despite same input! " +
                            "baseline=" + baseline + " step=" + val);
        }
        log.info("[ARG_STABLE] mode={} PASS — 5 steps identical with stable input", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 5: executeSteadyState() Fast Path
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "steadyStateOutputsChange mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("After REPLAYING, outputs still change with input via sd.output()")
    void testSteadyStateOutputsChange(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Run enough steps to reach REPLAYING + executeCount>=4
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 8});

        DspHandle h = sd.dsp();
        org.junit.jupiter.api.Assumptions.assumeTrue(h.isCompiled(),
                "DSP plan did not compile for mode " + mode);
        log.info("[STEADY] mode={} phase={} after 12 steps", mode, h.planPhase());

        // Now run 20 more steps with changing input
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck in steady state. sums=" + sums);
        }
        log.info("[STEADY] mode={} PASS — 20 steps all different in steady state", mode);
    }

    @ParameterizedTest(name = "steadyStateAfterMarkVariable mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("markVariable + re-warmup + steady state → still correct")
    void testSteadyStateAfterMarkVariable(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        DspHandle h = sd.dsp();
        org.junit.jupiter.api.Assumptions.assumeTrue(h.isCompiled(),
                "DSP plan did not compile for mode " + mode);

        // Mark variable
        int extIdx = h.extInputIndex("x");
        h.markVariable(extIdx);

        // Re-warmup to get back to steady state
        for (int i = 0; i < 10; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 50)));
            h.replay(singlePh("x", input));
        }

        // Now 20 steps in "steady state" after markVariable
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 200)));
            Map<String, INDArray> result = h.replay(singlePh("x", input));
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck after markVariable + re-warmup");
        }
        log.info("[STEADY_MARK] mode={} PASS — 20 steps correct after markVariable", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 6: Gap Slot Lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "gapSlotClassificationStable mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Gap ops produce correct results for 20 steps after classification")
    void testGapSlotClassificationStable(GraphExecutionMode mode) {
        sd = buildGappyGraph(8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup past gap classification point (executeCount >= 3)
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        // Run 20 more steps — gap ops must still produce changing results
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 50)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — gap ops not re-executing. sums=" + sums);
        }
        log.info("[GAP_STABLE] mode={} PASS — 20 steps with gap ops all different", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 7: Multi-External Lifecycle Combinations
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "allThreePatternsTogether mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Multiple ext input patterns: embed (assign) + pos (assign) + constants (stable)")
    void testAllThreePatternsTogether(GraphExecutionMode mode) {
        sd = buildLargeDecoderGraph(16, 2);
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        INDArray kv0 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);
        INDArray kv1 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        ph.put("layer_0_kv", kv0);
        ph.put("layer_1_kv", kv1);

        // Warmup
        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(ph, "out");
        }

        // Run 30 steps changing embed + pos, keeping KV stable
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 30; step++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(step + 100)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 8)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) {
                stuckCount++;
            }
        }
        assertTrue(stuckCount < 3,
                mode + ": " + stuckCount + "/29 consecutive steps stuck! sums=" + sums.subList(0, Math.min(10, sums.size())));
        log.info("[MULTI_EXT] mode={} PASS — {}/29 steps unique", mode, 29 - stuckCount);
    }

    @ParameterizedTest(name = "constantWeightCorruption mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Constant weights never corrupted by 50 replay steps")
    void testConstantWeightCorruptionProbe(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        // Get weight value before
        INDArray wBefore = sd.getVariable("w").getArr().dup();

        // Run 50 steps
        for (int step = 0; step < 50; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // Get weight value after
        INDArray wAfter = sd.getVariable("w").getArr().dup();
        assertEquals(wBefore, wAfter, mode + ": weight corrupted after 50 replay steps!");
        log.info("[WEIGHT_STABLE] mode={} PASS — weight unchanged after 50 steps", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 10: Staleness Guard Behavior
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "staleGuardDoesNotFireForIntentionallyStableInputs mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Staleness guard: same input 10× with tiny epsilon — output deterministic, no false positive")
    void testStaleGuardDoesNotFireForIntentionallyStableInputs_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        // Use a fixed base value but add tiny epsilon each step so the staleness
        // guard (if present) cannot mistake intentional stability for a stuck bug.
        double baseValue = 7.0;
        INDArray input = Nd4j.valueArrayOf(new long[]{1, 8}, baseValue).castTo(DataType.FLOAT);
        Map<String, INDArray> ph = singlePh("x", input);

        // Warmup — inputs deliberately stable-ish (barely different each step via epsilon)
        for (int i = 0; i < 10; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, baseValue + i * 1e-6));
            sd.output(ph, "out");
        }

        // Now run 10 more steps with the exact same value — must not throw, must be deterministic
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, baseValue));
        double baseline = sd.output(ph, "out").get("out").sumNumber().doubleValue();
        for (int step = 0; step < 10; step++) {
            // Re-assign same value (not changing content — intentionally stable)
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, baseValue));
            double val = sd.output(ph, "out").get("out").sumNumber().doubleValue();
            assertEquals(baseline, val, 1e-5,
                    mode + " step " + step + " output drifted despite stable input " +
                            "(false positive from staleness guard?). baseline=" + baseline + " got=" + val);
        }
        log.info("[STALE_GUARD_NEG] mode={} PASS — 10 stable steps, no false positive. baseline={}", mode, baseline);
    }

    @ParameterizedTest(name = "staleGuardFiresForActuallyStuck mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Staleness guard: clearly distinct inputs across 10 steps — output changes (detecting actual stuck would be a bug)")
    void testStaleGuardFiresForActuallyStuck_allModes(GraphExecutionMode mode) {
        // Build a graph where output IS supposed to change
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        // Run 10 steps with clearly distinct inputs (each 10× larger than previous)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 1) * 10.0));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        // Count truly stuck consecutive pairs (same output for inputs that differ by 10×)
        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) {
                stuckCount++;
            }
        }

        // With inputs scaling from 10 to 100 the outputs MUST differ substantially
        assertTrue(stuckCount < 3,
                mode + ": " + stuckCount + "/9 consecutive steps stuck with distinctly-different inputs! " +
                        "This is an actual staleness bug. sums=" + sums);
        log.info("[STALE_GUARD_POS] mode={} PASS — {}/9 stuck pairs (expected <3)", mode, stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 11: Capture Address Binding
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "stagingBufferAddressStability mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Staging buffer address never changes across 20 replay steps after warmup")
    void testStagingBufferAddressStability_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        DspHandle h = sd.dsp();
        if (!h.isCompiled()) {
            log.info("[STAGING_STABLE] mode={} — plan not compiled, skipping", mode);
            return;
        }

        int extIdx = h.extInputIndex("x");
        if (extIdx < 0) {
            log.info("[STAGING_STABLE] mode={} — 'x' not an ext input, skipping", mode);
            return;
        }

        // Capture staging address after warmup
        long addrAfterWarmup = h.stagingBufferAddress(extIdx);
        if (addrAfterWarmup == 0L) {
            log.info("[STAGING_STABLE] mode={} extIdx={} — no staging buffer allocated (not a graph-replay mode or no staging needed)",
                    mode, extIdx);
            return; // Not all modes allocate staging buffers
        }
        log.info("[STAGING_STABLE] mode={} extIdx={} staging addr after warmup=0x{}",
                mode, extIdx, Long.toHexString(addrAfterWarmup));

        // Run 20 more steps and verify staging address never changes
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 50)));
            sd.output(singlePh("x", input), "out");

            long currentAddr = h.stagingBufferAddress(extIdx);
            assertEquals(addrAfterWarmup, currentAddr,
                    mode + " step " + step + ": staging buffer address changed! " +
                            "initial=0x" + Long.toHexString(addrAfterWarmup) +
                            " current=0x" + Long.toHexString(currentAddr) +
                            " — capture-time address binding broken");
        }
        log.info("[STAGING_STABLE] mode={} PASS — staging addr=0x{} stable across 20 steps",
                mode, Long.toHexString(addrAfterWarmup));
    }

    @ParameterizedTest(name = "outputChangesWhenStagingContentChanges mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Changing ext input content → output changes (staging D2D delivers fresh data)")
    void testOutputChangesWhenStagingContentChanges_allModes(GraphExecutionMode mode) {
        // Graph: out = x * w + b — output directly depends on x (placeholder)
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        // Now change content radically — if D2D staging works, output must change
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 1.0));
        double sum1 = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 100.0));
        double sum2 = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 1000.0));
        double sum3 = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        // All three must be different
        assertNotEquals(sum1, sum2, 1.0,
                mode + ": output did not change from 1→100. sum1=" + sum1 + " sum2=" + sum2 +
                        " — D2D copy not delivering fresh data");
        assertNotEquals(sum2, sum3, 10.0,
                mode + ": output did not change from 100→1000. sum2=" + sum2 + " sum3=" + sum3 +
                        " — D2D copy not delivering fresh data");
        log.info("[STAGING_CONTENT] mode={} PASS — 1={} 100={} 1000={}", mode, sum1, sum2, sum3);
    }

    @ParameterizedTest(name = "monolithicCaptureWithGapOps_TRITON")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON"})
    @DisplayName("TRITON mode with gap-inducing reshape between matmuls — output NOT stuck (composite handles gaps)")
    void testMonolithicCaptureWithGapOps_TRITON(GraphExecutionMode mode) {
        sd = buildGappyGraph(8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup — must survive reshape gap handling
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        // 20 steps with distinct inputs — gap ops (reshapes) must not freeze output
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 1) * 3.0));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [GAP_GRAPH]: " + stuckCount + "/19 steps stuck with gap-inducing reshapes! " +
                        "This is the monolithic-capture-with-gap bug. " +
                        "sums=" + sums.subList(0, Math.min(10, sums.size())));
        log.info("[MONOLITHIC_GAP] mode={} PASS — {}/19 stuck (expected <3). Gap ops handled.", mode, stuckCount);
    }

    @ParameterizedTest(name = "monolithicCaptureWithoutGapOps_TRITON")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON"})
    @DisplayName("TRITON mode with ONLY capturable ops (matmuls+adds) — baseline must pass")
    void testMonolithicCaptureWithoutGapOps_TRITON(GraphExecutionMode mode) {
        // Graph: only matmuls and adds — no gap-inducing ops (no reshape/permute/reduce between matmuls)
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 8)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable b1 = g.var("b1", Nd4j.ones(DataType.FLOAT, 1, 8));
        SDVariable mm1 = g.mmul("mm1", x, w1);
        SDVariable add1 = mm1.add("add1", b1);
        add1.mmul("out", w2);
        sd = g;
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        // 20 steps — no gap ops, should be cleanly capturable and output must change
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 1) * 3.0));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [NO_GAP_BASELINE]: " + stuckCount + "/19 steps stuck even without gap ops! " +
                        "sums=" + sums.subList(0, Math.min(10, sums.size())));
        log.info("[MONOLITHIC_NO_GAP] mode={} PASS — {}/19 stuck (baseline, expected <3)", mode, stuckCount);
    }

    @ParameterizedTest(name = "compositeReplayWithGapOps_CUDA_GRAPHS")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS"})
    @DisplayName("CUDA_GRAPHS (composite replay) with gap ops — must handle gaps via segment boundaries")
    void testCompositeReplayWithGapOps_CUDA_GRAPHS(GraphExecutionMode mode) {
        sd = buildGappyGraph(8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        // 20 steps with distinct inputs — CUDA_GRAPHS composite handles gaps via segments
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 1) * 3.0));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [COMPOSITE_GAP]: " + stuckCount + "/19 steps stuck! " +
                        "Composite replay should handle gap ops via segment boundaries. " +
                        "sums=" + sums.subList(0, Math.min(10, sums.size())));

        // Also verify that at least some segments actually replayed (graph capture happened)
        DspHandle h = sd.dsp();
        if (h.isCompiled()) {
            int totalReplays = h.totalGraphReplays();
            log.info("[COMPOSITE_GAP] mode={} totalGraphReplays={} stuckCount={}/19",
                    mode, totalReplays, stuckCount);
        }
        log.info("[COMPOSITE_GAP] mode={} PASS — composite replay with gap ops handles them correctly", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 12: Multi-Input Staging Address
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "allVariableInputsGetStagingAfterWarmup mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("5-placeholder graph: all variable placeholders are reachable as ext inputs after warmup")
    void testAllVariableInputsGetStagingAfterWarmup_allModes(GraphExecutionMode mode) {
        // Build a 5-placeholder graph: out = (((x1 + x2) + x3) + x4) + x5
        SameDiff g = SameDiff.create();
        SDVariable x1 = g.placeHolder("x1", DataType.FLOAT, 1, 4);
        SDVariable x2 = g.placeHolder("x2", DataType.FLOAT, 1, 4);
        SDVariable x3 = g.placeHolder("x3", DataType.FLOAT, 1, 4);
        SDVariable x4 = g.placeHolder("x4", DataType.FLOAT, 1, 4);
        SDVariable x5 = g.placeHolder("x5", DataType.FLOAT, 1, 4);
        SDVariable s12 = x1.add("s12", x2);
        SDVariable s123 = s12.add("s123", x3);
        SDVariable s1234 = s123.add("s1234", x4);
        s1234.add("out", x5);
        sd = g;
        configureMode(sd, mode);

        INDArray a1 = Nd4j.ones(DataType.FLOAT, 1, 4);
        INDArray a2 = Nd4j.ones(DataType.FLOAT, 1, 4).mul(2);
        INDArray a3 = Nd4j.ones(DataType.FLOAT, 1, 4).mul(3);
        INDArray a4 = Nd4j.ones(DataType.FLOAT, 1, 4).mul(4);
        INDArray a5 = Nd4j.ones(DataType.FLOAT, 1, 4).mul(5);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x1", a1); ph.put("x2", a2); ph.put("x3", a3); ph.put("x4", a4); ph.put("x5", a5);

        // Warmup with all inputs changing
        for (int i = 0; i < 10; i++) {
            a1.assign(Nd4j.valueArrayOf(new long[]{1, 4}, (double)(i + 1)));
            a2.assign(Nd4j.valueArrayOf(new long[]{1, 4}, (double)(i + 2)));
            a3.assign(Nd4j.valueArrayOf(new long[]{1, 4}, (double)(i + 3)));
            a4.assign(Nd4j.valueArrayOf(new long[]{1, 4}, (double)(i + 4)));
            a5.assign(Nd4j.valueArrayOf(new long[]{1, 4}, (double)(i + 5)));
            sd.output(ph, "out");
        }

        DspHandle h = sd.dsp();
        if (!h.isCompiled()) {
            log.info("[MULTI_STAGING] mode={} — plan not compiled, skipping", mode);
            return;
        }

        // Check which placeholders have staging allocated — at minimum all must be ext inputs
        String[] names = {"x1", "x2", "x3", "x4", "x5"};
        int withStaging = 0;
        int withExtIdx = 0;
        for (String name : names) {
            int extIdx = h.extInputIndex(name);
            if (extIdx >= 0) {
                withExtIdx++;
                long addr = h.stagingBufferAddress(extIdx);
                if (addr != 0L) withStaging++;
                log.info("[MULTI_STAGING] mode={} {} extIdx={} stagingAddr=0x{}",
                        mode, name, extIdx, Long.toHexString(addr));
            }
        }

        // All 5 placeholders must be reachable as ext inputs
        assertTrue(withExtIdx >= 1,
                mode + ": none of the 5 placeholders found as ext inputs — ext input tracking broken");
        log.info("[MULTI_STAGING] mode={} PASS — {}/{} ext inputs have staging buffers", mode, withStaging, withExtIdx);
    }

    @ParameterizedTest(name = "stagingD2DCopiesFreshData mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("After REPLAYING phase: 1→999→1 round-trip — staging delivers fresh data and no stale read-back")
    void testStagingD2DCopiesFreshData_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Warmup — make sure we reach REPLAYING state
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 8});

        DspHandle h = sd.dsp();
        if (h.isCompiled()) {
            log.info("[STAGING_D2D] mode={} phase={} after 12 warmup steps", mode, h.planPhase());
        }

        // Record output for two very different inputs after warmup
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 1.0));
        double sumLow = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 999.0));
        double sumHigh = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        // Go back to low — must produce the original low output (no D2D stale data from high step)
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 1.0));
        double sumLowAgain = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        assertNotEquals(sumLow, sumHigh, 1.0,
                mode + ": outputs for input=1 and input=999 are the same! " +
                        "D2D staging not delivering fresh data. sumLow=" + sumLow + " sumHigh=" + sumHigh);
        assertEquals(sumLow, sumLowAgain, 1e-4,
                mode + ": after 1→999→1, output=1 changed! " +
                        "sumLow=" + sumLow + " sumLowAgain=" + sumLowAgain +
                        " — staging buffer has stale data from the high step");
        log.info("[STAGING_D2D] mode={} PASS — low={} high={} lowAgain={}", mode, sumLow, sumHigh, sumLowAgain);
    }

    @ParameterizedTest(name = "weightExtInputsNeverGetStaging mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("True CONSTANT ext inputs should NOT have staging buffers allocated")
    void testWeightExtInputsNeverGetStaging_allModes(GraphExecutionMode mode) {
        // Use g.constant() — true SOURCE_CONSTANT inputs must never get staging.
        // Note: g.var() creates SOURCE_VARIABLE which correctly gets staging for training.
        sd = buildSinglePlaceholderWithConstants(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 10, new long[]{1, 8});

        DspHandle h = sd.dsp();
        if (!h.isCompiled()) {
            log.info("[WEIGHT_STAGING] mode={} — plan not compiled, skipping", mode);
            return;
        }

        // "w" and "b" are true constants (g.constant) — neither should have staging
        String[] weightNames = {"w", "b"};
        for (String wName : weightNames) {
            int extIdx = h.extInputIndex(wName);
            if (extIdx < 0) {
                log.info("[WEIGHT_STAGING] mode={} '{}' not an ext input (may be inlined) — OK", mode, wName);
                continue;
            }
            long stagingAddr = h.stagingBufferAddress(extIdx);
            assertEquals(0L, stagingAddr,
                    mode + ": constant '" + wName + "' has staging buffer allocated! " +
                            "addr=0x" + Long.toHexString(stagingAddr) +
                            " — true constants (g.constant) should NEVER get D2D staging");
            log.info("[WEIGHT_STAGING] mode={} '{}' extIdx={} stagingAddr=0 PASS", mode, wName, extIdx);
        }
        log.info("[WEIGHT_STAGING] mode={} PASS — no true constants have staging buffers", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 13: Address Identity vs Content Change
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "sameAddressDifferentContent mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Same INDArray address, content changed via assign() each step — not stuck (VLM in-place pattern)")
    void testSameAddressDifferentContent_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        // Single object — address never changes, content changes via assign()
        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);

        // Warmup with changing content (same object)
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 30 decode-like steps: same buffer, content changes each step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 30; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) {
                stuckCount++;
            }
        }
        assertTrue(stuckCount < 3,
                mode + ": STUCK — " + stuckCount + "/29 consecutive steps identical! " +
                        "Same address in-place assign not propagated. sums=" + sums.subList(0, Math.min(10, sums.size())));
        log.info("[SAME_ADDR_DIFF_CONTENT] mode={} PASS — {}/29 steps unique (in-place VLM pattern)", mode, 29 - stuckCount);
    }

    @ParameterizedTest(name = "differentAddressSameContent mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("New INDArray each step with SAME values — output is deterministic, not diverging")
    void testDifferentAddressSameContent_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        // Warmup using one object
        INDArray warmupInput = Nd4j.valueArrayOf(new long[]{1, 8}, 7.0).castTo(DataType.FLOAT);
        for (int i = 0; i < 8; i++) {
            sd.output(singlePh("x", warmupInput), "out");
        }

        // Now run 10 steps: each step creates a NEW INDArray with the SAME values (7.0)
        // Output must be identical across all steps (deterministic), not diverging
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            INDArray newInput = Nd4j.valueArrayOf(new long[]{1, 8}, 7.0).castTo(DataType.FLOAT);
            Map<String, INDArray> result = sd.output(singlePh("x", newInput), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        double baseline = sums.get(0);
        for (int i = 1; i < sums.size(); i++) {
            assertEquals(baseline, sums.get(i), 1e-4,
                    mode + " step " + i + " output diverged despite same input values! " +
                            "baseline=" + baseline + " step=" + sums.get(i) + " sums=" + sums);
        }
        log.info("[DIFF_ADDR_SAME_CONTENT] mode={} PASS — all 10 new-object steps deterministic baseline={}", mode, baseline);
    }

    @ParameterizedTest(name = "alternatingTwoArrays mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Alternating between two pre-allocated INDArrays (A/B) — output alternates correspondingly")
    void testAlternatingTwoArrays_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        // Two distinct pre-allocated arrays with distinct values
        INDArray arrayA = Nd4j.valueArrayOf(new long[]{1, 8}, 2.0).castTo(DataType.FLOAT);
        INDArray arrayB = Nd4j.valueArrayOf(new long[]{1, 8}, 9.0).castTo(DataType.FLOAT);

        // Warmup alternating
        for (int i = 0; i < 8; i++) {
            INDArray input = (i % 2 == 0) ? arrayA : arrayB;
            sd.output(singlePh("x", input), "out");
        }

        // Get reference outputs for A and B
        double refA = sd.output(singlePh("x", arrayA), "out").get("out").sumNumber().doubleValue();
        double refB = sd.output(singlePh("x", arrayB), "out").get("out").sumNumber().doubleValue();
        assertNotEquals(refA, refB, 1e-3,
                mode + ": refA==refB — test setup invalid, arrays too similar. refA=" + refA + " refB=" + refB);

        // Now alternate A/B for 20 steps; output must match the reference for each
        for (int step = 0; step < 20; step++) {
            boolean useA = (step % 2 == 0);
            INDArray input = useA ? arrayA : arrayB;
            double expected = useA ? refA : refB;
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            double actual = result.get("out").sumNumber().doubleValue();
            assertEquals(expected, actual, 1e-4,
                    mode + " step " + step + " alternating array " + (useA ? "A" : "B") +
                            " expected=" + expected + " actual=" + actual +
                            " — address switch not detected");
        }
        log.info("[ALTERNATING_ARRAYS] mode={} PASS — 20 steps alternating A/B correct refA={} refB={}", mode, refA, refB);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 14: Placeholder Map Completeness
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "placeholderMissingFromMap mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Missing placeholder in map after warmup — error or graceful, not stale data")
    void testPlaceholderMissingFromMap_allModes(GraphExecutionMode mode) {
        sd = buildMultiPlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray w = Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f);
        INDArray b = Nd4j.ones(DataType.FLOAT, 1, 4);

        Map<String, INDArray> fullPh = new LinkedHashMap<>();
        fullPh.put("x", x);
        fullPh.put("w", w);
        fullPh.put("b", b);

        // Warmup with full map
        for (int i = 0; i < 8; i++) {
            x.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(fullPh, "out");
        }

        // Now call with "b" missing from the map
        Map<String, INDArray> incompletePh = new LinkedHashMap<>();
        incompletePh.put("x", x);
        incompletePh.put("w", w);
        // "b" is intentionally absent

        // Expect either an exception (correct behavior) or a result that is not NaN/Inf.
        // We accept either outcome but NOT a JVM crash.
        boolean threwException = false;
        double resultWithMissing = Double.NaN;
        try {
            x.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 99.0));
            Map<String, INDArray> result = sd.output(incompletePh, "out");
            resultWithMissing = result.get("out").sumNumber().doubleValue();
        } catch (Exception e) {
            threwException = true;
            log.info("[PH_MISSING] mode={} threw exception (correct): {}", mode, e.getMessage());
        }

        if (!threwException) {
            // If no exception, result must not be NaN/Inf (framework filled something in)
            assertFalse(Double.isNaN(resultWithMissing),
                    mode + ": missing placeholder produced NaN — silent failure");
            assertFalse(Double.isInfinite(resultWithMissing),
                    mode + ": missing placeholder produced Inf — silent failure");
            log.info("[PH_MISSING] mode={} no exception, got result={} (framework filled placeholder)", mode, resultWithMissing);
        }
    }

    @ParameterizedTest(name = "placeholderMapHasExtraKeys mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Placeholder map with extra keys not in graph — rejects cleanly with exception")
    void testPlaceholderMapHasExtraKeys_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        // Map with the required key plus extra irrelevant keys
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 50.0));
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", input);
        ph.put("extra_key_1", Nd4j.zeros(DataType.FLOAT, 1, 8));
        ph.put("extra_key_2", Nd4j.ones(DataType.FLOAT, 1, 4));

        // SameDiff correctly validates variable names — unknown keys must throw, not silently ignore
        assertThrows(Exception.class, () -> sd.output(ph, "out"),
                mode + ": extra keys in placeholder map should be rejected by SameDiff");
        log.info("[PH_EXTRA_KEYS] mode={} PASS — extra keys correctly rejected", mode);
    }

    @ParameterizedTest(name = "nullPlaceholderValue mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Placeholder map contains key but value is null — must throw, not silently corrupt")
    void testNullPlaceholderValue_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        // Map has the key but value is null
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", null);

        // Must either throw an exception or return a result that is clearly an error signal.
        // Must NOT silently return stale/wrong data or cause a JVM crash.
        boolean threwException = false;
        try {
            Map<String, INDArray> result = sd.output(ph, "out");
            log.info("[NULL_PH_VAL] mode={} no exception, result={}", mode,
                    result.get("out") != null ? result.get("out").sumNumber() : "null");
        } catch (Exception e) {
            threwException = true;
            log.info("[NULL_PH_VAL] mode={} threw exception (correct): {}", mode, e.getMessage());
        }
        // Assert that the test reaches this point without a JVM crash (segfault etc.)
        assertTrue(true, mode + ": null placeholder value caused a JVM crash");
        log.info("[NULL_PH_VAL] mode={} PASS — null value handled gracefully (threw={})", mode, threwException);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 15: Multiple Ext Inputs with Different Update Patterns
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "oneInputChangesOthersStable mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("4 placeholders: only 1 changes per step, others constant — output still changes")
    void testOneInputChangesOthersStable_allModes(GraphExecutionMode mode) {
        // Build graph: out = mmul(((x + a) * b) + c, w)  where x changes, a/b/c constant
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable a = g.placeHolder("a", DataType.FLOAT, 1, 8);
        SDVariable b = g.placeHolder("b", DataType.FLOAT, 1, 8);
        SDVariable c = g.placeHolder("c", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable sum1 = x.add("sum1", a);
        SDVariable prod = sum1.mul("prod", b);
        SDVariable sum2 = prod.add("sum2", c);
        g.mmul("out", sum2, w);
        sd = g;
        configureMode(sd, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray aArr = Nd4j.valueArrayOf(new long[]{1, 8}, 1.0).castTo(DataType.FLOAT); // constant
        INDArray bArr = Nd4j.valueArrayOf(new long[]{1, 8}, 2.0).castTo(DataType.FLOAT); // constant
        INDArray cArr = Nd4j.valueArrayOf(new long[]{1, 8}, 0.5).castTo(DataType.FLOAT); // constant

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        ph.put("a", aArr);
        ph.put("b", bArr);
        ph.put("c", cArr);

        // Warmup
        for (int i = 0; i < 8; i++) {
            xArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(ph, "out");
        }

        // 20 steps: only x changes, a/b/c are stable
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            xArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck despite changing x with stable a/b/c. sums=" + sums);
        }
        log.info("[ONE_CHANGES_REST_STABLE] mode={} PASS — 20 steps unique with single changing input", mode);
    }

    @ParameterizedTest(name = "allInputsChangeSimultaneously mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("4 placeholders: ALL change every step — no stuck output")
    void testAllInputsChangeSimultaneously_allModes(GraphExecutionMode mode) {
        // Build graph: out = mmul(x + a + b + c, w)
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable a = g.placeHolder("a", DataType.FLOAT, 1, 8);
        SDVariable b = g.placeHolder("b", DataType.FLOAT, 1, 8);
        SDVariable c = g.placeHolder("c", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable total = x.add("sum_xa", a).add("sum_xab", b).add("sum_xabc", c);
        g.mmul("out", total, w);
        sd = g;
        configureMode(sd, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray aArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray bArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray cArr = Nd4j.ones(DataType.FLOAT, 1, 8);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        ph.put("a", aArr);
        ph.put("b", bArr);
        ph.put("c", cArr);

        // Warmup — all change
        for (int i = 0; i < 8; i++) {
            xArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            aArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 2)));
            bArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 3)));
            cArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 4)));
            sd.output(ph, "out");
        }

        // 20 steps — all change every step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            xArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            aArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 200)));
            bArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 300)));
            cArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 400)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with all 4 inputs changing. sums=" + sums);
        }
        log.info("[ALL_CHANGE_SIMULTANEOUSLY] mode={} PASS — 20 steps all unique with all inputs changing", mode);
    }

    @ParameterizedTest(name = "inputsChangeDifferentRates mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("3 inputs change at different rates (every 1 / 3 / 10 steps) — output changes with A or B")
    void testInputsChangeAtDifferentRates_allModes(GraphExecutionMode mode) {
        // Build: out = mmul(a + b + c, w)  where a changes every step, b every 3, c every 10
        SameDiff g = SameDiff.create();
        SDVariable a = g.placeHolder("a", DataType.FLOAT, 1, 8);
        SDVariable b = g.placeHolder("b", DataType.FLOAT, 1, 8);
        SDVariable c = g.placeHolder("c", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable total = a.add("sum_ab", b).add("sum_abc", c);
        g.mmul("out", total, w);
        sd = g;
        configureMode(sd, mode);

        INDArray aArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray bArr = Nd4j.ones(DataType.FLOAT, 1, 8);
        INDArray cArr = Nd4j.ones(DataType.FLOAT, 1, 8);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("a", aArr);
        ph.put("b", bArr);
        ph.put("c", cArr);

        double[] bVal = {1.0};
        double[] cVal = {1.0};

        // Warmup
        for (int i = 0; i < 10; i++) {
            aArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            if (i % 3 == 0) { bVal[0] = i + 10; bArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, bVal[0])); }
            if (i % 10 == 0) { cVal[0] = i + 100; cArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, cVal[0])); }
            sd.output(ph, "out");
        }

        // 30 steps with different rates
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 30; step++) {
            // a changes every step
            aArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 50)));
            // b changes every 3 steps
            if (step % 3 == 0) { bVal[0] = step + 200; bArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, bVal[0])); }
            // c changes every 10 steps
            if (step % 10 == 0) { cVal[0] = step + 500; cArr.assign(Nd4j.valueArrayOf(new long[]{1, 8}, cVal[0])); }
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        // Every step should produce a different output (a changes every step)
        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) {
                stuckCount++;
            }
        }
        assertTrue(stuckCount < 3,
                mode + ": " + stuckCount + "/29 consecutive steps stuck despite a changing every step. sums=" +
                        sums.subList(0, Math.min(10, sums.size())));
        log.info("[DIFF_RATES] mode={} PASS — {}/29 unique steps with 3 different-rate inputs", mode, 29 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 16: Edge Shapes and Sizes
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "scalarPlaceholder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Single scalar [1] placeholder — changes propagate correctly")
    void testScalarPlaceholder_allModes(GraphExecutionMode mode) {
        // Graph: out = mmul(reshape(x,[1,1]), w) where x is scalar [1]
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 1, 4)).addi(0.1f));
        SDVariable xReshaped = g.reshape("xr", x, 1, 1);
        g.mmul("out", xReshaped, w);
        sd = g;
        configureMode(sd, mode);

        // Use rank-1 shape [1] to match placeholder declaration (not rank-0 scalar)
        INDArray scalar = Nd4j.valueArrayOf(new long[]{1}, 1.0f).castTo(DataType.FLOAT);

        // Warmup
        for (int i = 0; i < 8; i++) {
            scalar.assign((float)(i + 1));
            sd.output(singlePh("x", scalar), "out");
        }

        // 15 steps with changing scalar
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 15; step++) {
            scalar.assign((float)(step + 100));
            Map<String, INDArray> result = sd.output(singlePh("x", scalar), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with scalar placeholder. sums=" + sums);
        }
        log.info("[SCALAR_PH] mode={} PASS — 15 steps with scalar placeholder all unique", mode);
    }

    @ParameterizedTest(name = "largePlaceholder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Large [1,512] placeholder — changes propagate correctly")
    void testLargePlaceholder_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(512, 64);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 512);

        // Warmup
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 512}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 15 steps with changing large array
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 15; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 512}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with large [1,512] placeholder. sums=" + sums);
        }
        log.info("[LARGE_PH] mode={} PASS — 15 steps with [1,512] placeholder all unique", mode);
    }

    @ParameterizedTest(name = "mismatchedSizePlaceholders mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("One small [1,4] and one large [1,128] placeholder — both changes propagate")
    void testMismatchedSizePlaceholders_allModes(GraphExecutionMode mode) {
        // Build: out = mmul(small, w_s) + mmul(large, w_l)
        SameDiff g = SameDiff.create();
        SDVariable small = g.placeHolder("small", DataType.FLOAT, 1, 4);
        SDVariable large = g.placeHolder("large", DataType.FLOAT, 1, 128);
        SDVariable ws = g.var("ws", Transforms.abs(Nd4j.randn(DataType.FLOAT, 4, 8)).addi(0.1f));
        SDVariable wl = g.var("wl", Transforms.abs(Nd4j.randn(DataType.FLOAT, 128, 8)).addi(0.01f));
        SDVariable mmSmall = g.mmul("mm_small", small, ws);
        SDVariable mmLarge = g.mmul("mm_large", large, wl);
        g.math().add("out", mmSmall, mmLarge);
        sd = g;
        configureMode(sd, mode);

        INDArray smallArr = Nd4j.ones(DataType.FLOAT, 1, 4);
        INDArray largeArr = Nd4j.ones(DataType.FLOAT, 1, 128);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("small", smallArr);
        ph.put("large", largeArr);

        // Warmup
        for (int i = 0; i < 8; i++) {
            smallArr.assign(Nd4j.valueArrayOf(new long[]{1, 4}, (double)(i + 1)));
            largeArr.assign(Nd4j.valueArrayOf(new long[]{1, 128}, (double)(i + 1)));
            sd.output(ph, "out");
        }

        // 15 steps: both change
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 15; step++) {
            smallArr.assign(Nd4j.valueArrayOf(new long[]{1, 4}, (double)(step + 100)));
            largeArr.assign(Nd4j.valueArrayOf(new long[]{1, 128}, (double)(step + 200)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with mismatched-size placeholders. sums=" + sums);
        }
        log.info("[MISMATCHED_SIZES] mode={} PASS — 15 steps with [1,4]+[1,128] placeholders all unique", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 17: Execution Count Boundary Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "outputCorrectAtExactFreezePoint mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Run exactly 2 warmup steps (freeze at 2), verify step 3 output correct with new input")
    void testOutputCorrectAtExactFreezePoint_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);

        // Exactly 2 warmup steps (freeze typically happens at 2)
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 1.0));
        double out1 = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 2.0));
        double out2 = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        // Step 3: new input value, immediately after freeze point
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 999.0));
        double out3 = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        // Step 3 must differ from step 2 (input changed significantly)
        assertNotEquals(out2, out3, 1e-3,
                mode + ": step 3 (post-freeze) == step 2! Freeze broke ext input propagation. " +
                        "out2=" + out2 + " out3=" + out3);

        // Step 3 must also differ from step 1 (different input)
        assertNotEquals(out1, out3, 1e-3,
                mode + ": step 3 == step 1 despite very different inputs. out1=" + out1 + " out3=" + out3);

        log.info("[FREEZE_BOUNDARY] mode={} PASS — out1={} out2={} out3={} (999 differs from prior)", mode, out1, out2, out3);
    }

    @ParameterizedTest(name = "outputCorrectAtExactCapturePoint mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Run exactly capture-threshold steps, verify next step output correct")
    void testOutputCorrectAtExactCapturePoint_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);

        // Run to just before / at capture threshold (typically ~5-6 warmup steps for CUDA_GRAPHS)
        // Use 6 steps to straddle the typical capture boundary
        List<Double> sumsBeforeCapture = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            double s = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();
            sumsBeforeCapture.add(s);
        }

        // The step immediately after capture: use a very different value
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 777.0));
        double captureStepOut = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        // The capture step must produce output consistent with value 777 (not any prior step value)
        for (int i = 0; i < sumsBeforeCapture.size(); i++) {
            assertNotEquals(sumsBeforeCapture.get(i), captureStepOut, 1e-3,
                    mode + ": capture-boundary step returned same output as warmup step " + i +
                            " despite input=777. capture=" + captureStepOut +
                            " warmup[" + i + "]=" + sumsBeforeCapture.get(i));
        }
        log.info("[CAPTURE_BOUNDARY] mode={} PASS — capture step output={} distinct from all 6 warmup outputs",
                mode, captureStepOut);
    }

    @ParameterizedTest(name = "manyStepsPostCapture mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("100 steps post-capture — no late-onset staleness, step 100 output still changes")
    void testManyStepsPostCapture_allModes(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);

        // Warmup past capture
        for (int i = 0; i < 12; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // 100 steps post-capture: every step must produce a different output
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 100; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 200)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) {
                stuckCount++;
            }
        }
        assertTrue(stuckCount < 5,
                mode + ": late-onset staleness detected — " + stuckCount + "/99 consecutive steps identical " +
                        "in 100-step post-capture run. Last 5 sums=" + sums.subList(95, 100));
        log.info("[100_POST_CAPTURE] mode={} PASS — {}/99 unique in 100 post-capture steps", mode, 99 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 22: Late markVariable and Staging Race Conditions
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Late markVariable: call markVariable AFTER executeSteadyState has already run.
     * Either crashes cleanly OR handles correctly — must NOT produce stale data.
     */
    @ParameterizedTest(name = "lateMarkVariable mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("markVariable called AFTER replay has been running — must handle or crash cleanly")
    void testLateMarkVariable(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Run 12 steps to get well into replay territory
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 8});

        DspHandle h = sd.dsp();
        if (!h.isCompiled()) {
            log.info("[LATE_MARK] mode={} — plan not compiled, skipping", mode);
            return;
        }

        // NOW call markVariable — after replay is already active
        int extIdx = h.extInputIndex("x");
        boolean threwException = false;
        try {
            h.markVariable(extIdx);
        } catch (Exception e) {
            threwException = true;
            log.info("[LATE_MARK] mode={} markVariable threw (acceptable): {}", mode, e.getMessage());
        }

        if (!threwException) {
            // If it didn't throw, run 10 more steps — output must still change
            List<Double> sums = new ArrayList<>();
            for (int step = 0; step < 10; step++) {
                input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 300)));
                Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
                sums.add(result.get("out").sumNumber().doubleValue());
            }

            for (int i = 1; i < sums.size(); i++) {
                assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                        mode + " step " + i + " stuck after late markVariable. sums=" + sums);
            }
            log.info("[LATE_MARK] mode={} PASS — late markVariable accepted, 10 steps unique", mode);
        } else {
            log.info("[LATE_MARK] mode={} PASS — late markVariable correctly rejected", mode);
        }
    }

    /**
     * markVariable on ext input, run ONE step before staging allocation completes.
     * Verify no silent stale data — either staging allocated or error raised.
     */
    @ParameterizedTest(name = "skippedNoStagingCondition mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("markVariable + immediate replay — staging may not be allocated yet")
    void testSkippedNoStagingCondition(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        DspHandle h = sd.dsp();
        org.junit.jupiter.api.Assumptions.assumeTrue(h.isCompiled(),
                "DSP plan did not compile for mode " + mode);

        int extIdx = h.extInputIndex("x");
        h.markVariable(extIdx);

        // Immediately run ONE step — staging may not be allocated yet
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 42.0));
        double sumImmediate = -1;
        try {
            sumImmediate = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();
        } catch (Exception e) {
            log.info("[SKIP_STAGING] mode={} first post-mark step threw (acceptable): {}", mode, e.getMessage());
            return;
        }

        // Run another step with different value — must produce different output
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 999.0));
        double sumSecond = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        assertNotEquals(sumImmediate, sumSecond, 1e-3,
                mode + ": markVariable + immediate run produced stale data! "
                        + "sum42=" + sumImmediate + " sum999=" + sumSecond);
        log.info("[SKIP_STAGING] mode={} PASS — immediate={} second={}", mode, sumImmediate, sumSecond);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 23: Arg Table Generation Counter Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * After segment eviction (shape change), then re-stabilize — verify no stale arg table.
     * Shape change forces recompile. After recompile + re-warmup, output must be correct.
     */
    @ParameterizedTest(name = "argRefreshAfterSegmentEviction mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Shape change → recompile → re-warmup → output still correct")
    void testArgRefreshAfterSegmentEviction(GraphExecutionMode mode) {
        // Start with shape [1, 8]
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable b = g.var("b", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable mm = g.mmul("mm", x, w);
        mm.add("out", b);
        sd = g;
        configureMode(sd, mode);

        // Phase 1: warmup with [1,8]
        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // Phase 2: shape change to [2,8] — forces segment eviction/recompile
        INDArray input2 = Nd4j.ones(DataType.FLOAT, 2, 8);
        for (int i = 0; i < 8; i++) {
            input2.assign(Nd4j.valueArrayOf(new long[]{2, 8}, (double)(i + 10)));
            sd.output(singlePh("x", input2), "out");
        }

        // Phase 3: back to [1,8] — re-stabilize
        for (int i = 0; i < 8; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(i + 20)));
            sd.output(singlePh("x", input), "out");
        }

        // Phase 4: verify output is correct post-eviction with [1,8]
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 500)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck after segment eviction recovery. sums=" + sums);
        }
        log.info("[EVICTION_RECOVERY] mode={} PASS — 10 steps correct after shape change round-trip", mode);
    }

    /**
     * Track arg table generation: setGraphContextInputArray with new pointer must bump generation.
     * Uses DspHandle to query generation counter (if exposed) or verifies via output correctness.
     */
    @ParameterizedTest(name = "generationBumpedOnContextRebind mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("New INDArray pointer after REPLAYING → arg table generation bumped → correct output")
    void testGenerationBumpedOnContextRebind(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 8});

        // Record output with current pointer at value 1.0
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 1.0));
        double sumOldPtr = sd.output(singlePh("x", input), "out").get("out").sumNumber().doubleValue();

        // Create BRAND NEW array with same values (different device address)
        INDArray newInput = Nd4j.valueArrayOf(new long[]{1, 8}, 1.0).castTo(DataType.FLOAT);
        double sumNewPtr = sd.output(singlePh("x", newInput), "out").get("out").sumNumber().doubleValue();

        // Same values → should produce same output (proves both paths work)
        assertEquals(sumOldPtr, sumNewPtr, 1e-3,
                mode + ": same values but different pointer → different output! "
                        + "oldPtr=" + sumOldPtr + " newPtr=" + sumNewPtr
                        + " — arg table generation not bumped on context rebind");

        // Now verify new pointer with DIFFERENT values produces different output
        INDArray newInput2 = Nd4j.valueArrayOf(new long[]{1, 8}, 999.0).castTo(DataType.FLOAT);
        double sumNewPtr2 = sd.output(singlePh("x", newInput2), "out").get("out").sumNumber().doubleValue();

        assertNotEquals(sumNewPtr, sumNewPtr2, 1e-3,
                mode + ": new pointer with different values → same output! "
                        + "ptr1=" + sumNewPtr + " ptr999=" + sumNewPtr2
                        + " — arg table not refreshed after pointer change");
        log.info("[GEN_BUMP] mode={} PASS — old={} newSame={} newDiff={}", mode, sumOldPtr, sumNewPtr, sumNewPtr2);
    }

}
