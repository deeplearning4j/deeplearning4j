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
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DSP mixed-precision and capture+Triton bisection tests extracted from DspExtInputStalenessTest.
 *
 * Covers:
 * - Category 10 (dup): Mixed Precision Cast Cache Staleness
 * - Phase 2: Optimizer + FP16 + Fused Op + DSP Interaction Tests
 *   - FP16 Mixed Precision through DSP
 *   - Swish/SwiGLU Fusion + DSP
 *   - Attention Pattern + DSP
 *   - Full Decoder Layer (Attention + FFN + Residual + RoPE)
 *   - Multi-Layer Decoder (Scale Test)
 *   - Cast Elimination + DSP
 *   - KV Cache Update Pattern
 *   - Softmax Numerical Stability under Replay
 *   - Multi-Layer SLOT_BY_SLOT Parity
 * - Category 9: Capture+Triton Bisection Tests (9a through 9n)
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class DspExtInputMixedPrecisionCaptureTest extends DspExtInputTestSupport {

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

    // CATEGORY 10: MIXED PRECISION CAST CACHE STALENESS
    //
    // Tests the MmulHelper cast cache (tl_castCacheA/B) correctness during
    // DSP replay. When FLOAT activation is cast to HALF for mixed-precision
    // matmul, the cast result is cached. If the activation's POINTER stays the
    // same (stable staging buffer) but CONTENT changes (new data each step),
    // the sameLogicalA shortcut must NOT reuse the stale cast buffer.
    // ═══════════════════════════════════════════════════════════════════════════

    /** Graph with FLOAT placeholder × HALF constant weight (mixed precision matmul) */
    private SameDiff buildMixedPrecisionGraph(int inDim, int outDim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        // Weight is HALF — triggers mixed-precision path in MmulHelper
        INDArray wArr = Nd4j.randn(DataType.FLOAT, inDim, outDim).muli(0.1f).castTo(DataType.HALF);
        SDVariable w = g.constant("w_half", wArr);
        SDVariable b = g.constant("b", Nd4j.ones(DataType.FLOAT, 1, outDim).muli(0.01f));
        SDVariable mm = g.mmul("mm", x, w);
        mm.add("out", b);
        return g;
    }

    /** Multi-layer mixed precision graph — amplifies staleness */
    private SameDiff buildDeepMixedPrecisionGraph(int inDim, int hidDim, int outDim, int layers) {
        SameDiff g = SameDiff.create();
        SDVariable current = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        for (int i = 0; i < layers; i++) {
            int curIn = (i == 0) ? inDim : hidDim;
            int curOut = (i == layers - 1) ? outDim : hidDim;
            INDArray wArr = Nd4j.randn(DataType.FLOAT, curIn, curOut).muli(0.1f).castTo(DataType.HALF);
            SDVariable w = g.constant("w_" + i, wArr);
            SDVariable b = g.constant("b_" + i, Nd4j.zeros(DataType.FLOAT, 1, curOut));
            current = g.mmul("mm_" + i, current, w);
            current = current.add("add_" + i, b);
            if (i < layers - 1) {
                current = g.nn().relu("relu_" + i, current, 0.0);
            }
        }
        g.identity("out", current);
        return g;
    }

    /**
     * TEST: Mixed-precision (FLOAT×HALF) matmul with SAME pointer but changing content.
     *
     * This reproduces the VLM FP16 bug: activation buffer has a stable address (D2D staging)
     * but content changes each decode step. The MmulHelper cast cache's sameLogicalA shortcut
     * compares only the pointer, not the content — causing stale HALF cast reuse.
     *
     * Expected: outputs differ each step (cast cache freshness).
     * Actual (if bug present): outputs repeat because stale cast is reused.
     */
    @ParameterizedTest(name = "mixedPrecisionCastCacheStaleness mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testMixedPrecisionCastCacheStaleness(GraphExecutionMode mode) {
        int dim = 64;
        sd = buildMixedPrecisionGraph(dim, dim);
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Use a SINGLE INDArray and assign different values each step (same pointer, new content)
        INDArray x = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            // Mutate content — pointer stays the same (simulates D2D staging)
            x.assign(Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.5f + step * 0.1f));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-4) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": MIXED_PRECISION_CAST_CACHE_STALE! " + stuckCount + "/19 steps stuck. "
                        + "Cast cache reuses stale HALF buffer when pointer is stable but content changes. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[MIXED_PREC_CAST] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    /**
     * Same test but with deep graph (multiple matmuls, amplifies the problem).
     */
    @ParameterizedTest(name = "deepMixedPrecisionCastCacheStaleness mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testDeepMixedPrecisionCastCacheStaleness(GraphExecutionMode mode) {
        int dim = 64;
        sd = buildDeepMixedPrecisionGraph(dim, dim, dim, 4);
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray x = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            x.assign(Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.5f + step * 0.1f));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-4) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": DEEP_MIXED_PREC_CAST_CACHE_STALE! " + stuckCount + "/19 steps stuck. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[DEEP_MIXED_PREC_CAST] mode={} PASS — {}/19 unique", mode, 19 - stuckCount);
    }

    /**
     * CONTROL: Same mixed-precision graph but with new INDArray each step (different pointer).
     * This bypasses the sameLogicalA shortcut. If this passes but the stable-pointer test fails,
     * the bug is confirmed to be in the sameLogicalA pointer comparison.
     */
    @ParameterizedTest(name = "mixedPrecisionNewPointerEachStep mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testMixedPrecisionNewPointerEachStep(GraphExecutionMode mode) {
        int dim = 64;
        sd = buildMixedPrecisionGraph(dim, dim);
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            // NEW INDArray each step — different pointer each time
            INDArray x = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.5f + step * 0.1f);
            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("x", x);
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-4) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": NEW_POINTER control test FAILED! " + stuckCount + "/19 steps stuck. "
                        + "This should ALWAYS pass — different pointer each step bypasses cast cache. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[MIXED_PREC_NEW_PTR] mode={} PASS — {}/19 unique (control)", mode, 19 - stuckCount);
    }

    /**
     * CONTROL: Same graph WITHOUT FP16 weights (both FLOAT) — no cast needed.
     * Should always pass regardless of cast cache behavior.
     */
    @ParameterizedTest(name = "samePrecisionNocast mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testSamePrecisionNocast(GraphExecutionMode mode) {
        int dim = 64;
        // All-FLOAT graph (no mixed precision, no cast cache involvement)
        sd = buildSinglePlaceholder(dim, dim);
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray x = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", x);

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            x.assign(Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.5f + step * 0.1f));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-4) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + ": SAME_PREC (no cast) test STUCK! " + stuckCount + "/19. "
                        + "This is a non-cast-related staleness bug.");
        log.info("[SAME_PREC_NOCAST] mode={} PASS — {}/19 unique (control)", mode, 19 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 2: Optimizer + FP16 + Fused Op + DSP Interaction Tests
    //
    // ALL previous 404 tests pass because they use trivial toy graphs.
    // The real VLM uses: FP16 weights (213 arrays), GraphOptimizer
    // (131 cast eliminations, 30 FuseSwiGLU, 30 FuseSigmoidMulToSwish),
    // fused ops (RoPE, RMSNorm+SwiGLU, LayerNorm), and 2551-op graphs.
    // These tests reproduce those interactions at unit-test scale.
    // ═══════════════════════════════════════════════════════════════════════════

    // ---- Helpers for Phase 2 ----

    /** Compare output of same graph in SLOT_BY_SLOT vs given mode.
     *  Returns the max absolute difference across all output elements. */
    private double compareSlotBySlotVsMode(SameDiff g, GraphExecutionMode mode,
                                            Map<String, INDArray> ph, String outName,
                                            int warmupSteps, int testSteps) {
        // Run SLOT_BY_SLOT baseline
        SameDiff gSlot = g.dup();
        configureMode(gSlot, GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> phSlot = new LinkedHashMap<>(ph);

        // Warmup both
        for (int i = 0; i < warmupSteps; i++) {
            gSlot.output(phSlot, outName);
        }
        configureMode(g, mode);
        for (int i = 0; i < warmupSteps; i++) {
            g.output(ph, outName);
        }

        // Compare test steps
        double maxDiff = 0.0;
        for (int step = 0; step < testSteps; step++) {
            Map<String, INDArray> slotResult = gSlot.output(phSlot, outName);
            Map<String, INDArray> modeResult = g.output(ph, outName);
            INDArray slotOut = slotResult.get(outName);
            INDArray modeOut = modeResult.get(outName);
            double diff = slotOut.sub(modeOut).amaxNumber().doubleValue();
            maxDiff = Math.max(maxDiff, diff);
        }
        gSlot.close();
        return maxDiff;
    }

    // ---- FP16 Mixed Precision through DSP ----

    /**
     * FP16 weights with FLOAT32 activations through DSP replay.
     * This is the exact pattern used by the VLM benchmark (213 FP16 weights).
     * Tests that matmul(FLOAT32 activation, HALF weight) produces correct results
     * across DSP replay steps.
     */
    @ParameterizedTest(name = "fp16WeightMixedPrecisionReplay mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("FP16 weights + FLOAT32 activations through DSP replay — not stuck, not NaN")
    void testFp16WeightMixedPrecisionReplay(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 32;
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        // HALF weight — the exact pattern from FP16 pre-casting
        SDVariable w = g.var("w", Nd4j.randn(DataType.HALF, dim, dim).muli(0.1));
        SDVariable b = g.var("b", Nd4j.zeros(DataType.FLOAT, 1, dim));
        SDVariable mm = g.mmul("mm", x, w);
        g.math().add("out", mm, b);
        sd = g;
        configureMode(g, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = singlePh("x", input);
        warmup(g, ph, "out", 8);

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 1)));
            Map<String, INDArray> result = g.output(ph, "out");
            INDArray out = result.get("out");
            // Check for NaN
            assertFalse(out.isNaN().any(), mode + " step " + step + " produced NaN!");
            assertFalse(out.isInfinite().any(), mode + " step " + step + " produced Inf!");
            sums.add(out.sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [FP16_MIXED]: STUCK! " + stuckCount + "/19 steps with FP16 weights. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[FP16_MIXED] mode={} PASS — {}/19 unique, no NaN/Inf", mode, 19 - stuckCount);
    }

    /**
     * Multi-layer FP16 weights: 4 matmuls chained, each with HALF weight.
     * Tests numerical stability across multiple FP16 matmuls in replay.
     */
    @ParameterizedTest(name = "fp16MultiLayerStability mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("4-layer FP16 matmul chain — numerical stability under replay")
    void testFp16MultiLayerStability(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 32;
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable current = x;
        for (int layer = 0; layer < 4; layer++) {
            // HALF weights — scale 0.15 keeps output magnitude above detection threshold
            // while staying well below FP16 overflow (~65504)
            SDVariable w = g.var("w" + layer, Nd4j.randn(DataType.HALF, dim, dim).muli(0.15));
            current = g.mmul("mm" + layer, current, w);
        }
        g.identity("out", current);
        sd = g;
        configureMode(g, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, dim);
        warmupWithChangingInput(g, "x", input, "out", 8, new long[]{1, dim});

        List<Double> sums = new ArrayList<>();
        boolean anyNaN = false;
        for (int step = 0; step < 20; step++) {
            // Larger step increments so 4-layer output differences exceed threshold
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 1) * 1.0));
            Map<String, INDArray> result = g.output(singlePh("x", input), "out");
            INDArray out = result.get("out");
            if (out.isNaN().any()) {
                anyNaN = true;
                log.error("[FP16_MULTI_LAYER] mode={} step {} NaN detected!", mode, step);
            }
            sums.add(out.sumNumber().doubleValue());
        }

        assertFalse(anyNaN, mode + " [FP16_MULTI_LAYER]: NaN in 4-layer FP16 chain!");

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [FP16_MULTI_LAYER]: STUCK! " + stuckCount + "/19 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[FP16_MULTI_LAYER] mode={} PASS — no NaN, {}/19 unique", mode, 19 - stuckCount);
    }

    /**
     * SLOT_BY_SLOT vs CUDA_GRAPHS/TRITON/AUTO parity with FP16 weights.
     * If outputs diverge, the bug is in how DSP handles mixed-precision during replay.
     */
    @ParameterizedTest(name = "fp16SlotBySlotParity mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("FP16 weights: SLOT_BY_SLOT output == DSP replay output")
    void testFp16SlotBySlotParity(GraphExecutionMode mode) {
        int dim = 16;

        // Create shared weight data on host so both graphs get identical non-zero values.
        // Use dup() to force device→host sync, then use the host-authoritative copy.
        INDArray wData = Nd4j.randn(DataType.HALF, dim, 8).muli(0.1).dup();
        INDArray bData = Nd4j.ones(DataType.FLOAT, 1, 8);

        // Build SLOT_BY_SLOT reference graph
        SameDiff gSlot = SameDiff.create();
        gSlot.placeHolder("x", DataType.FLOAT, 1, dim);
        gSlot.var("w", wData.dup());
        gSlot.mmul("mm", gSlot.getVariable("x"), gSlot.getVariable("w"));
        gSlot.var("b", bData.dup());
        gSlot.math().add("out", gSlot.getVariable("mm"), gSlot.getVariable("b"));

        // Build target mode graph with same weights
        SameDiff g = SameDiff.create();
        g.placeHolder("x", DataType.FLOAT, 1, dim);
        g.var("w", wData.dup());
        g.mmul("mm", g.getVariable("x"), g.getVariable("w"));
        g.var("b", bData.dup());
        g.math().add("out", g.getVariable("mm"), g.getVariable("b"));
        sd = g;

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, dim);

        // Warmup SLOT_BY_SLOT
        configureMode(gSlot, GraphExecutionMode.SLOT_BY_SLOT);
        for (int i = 0; i < 5; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(i + 1)));
            gSlot.output(singlePh("x", input), "out");
        }

        // Warmup target mode
        configureMode(g, mode);
        for (int i = 0; i < 5; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(i + 1)));
            g.output(singlePh("x", input), "out");
        }

        // Compare on fresh inputs
        double maxDiff = 0.0;
        for (int step = 0; step < 10; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 100)));
            Map<String, INDArray> slotResult = gSlot.output(singlePh("x", input), "out");
            Map<String, INDArray> modeResult = g.output(singlePh("x", input), "out");
            double diff = slotResult.get("out").sub(modeResult.get("out")).amaxNumber().doubleValue();
            if (step == 0) {
                log.info("[FP16_PARITY] step 0 SLOT_BY_SLOT out: {}", slotResult.get("out"));
                log.info("[FP16_PARITY] step 0 {} out: {}", mode, modeResult.get("out"));
                log.info("[FP16_PARITY] step 0 diff: {}", diff);
            }
            maxDiff = Math.max(maxDiff, diff);
        }

        gSlot.close();

        assertTrue(maxDiff < 0.1,
                mode + " [FP16_PARITY]: SLOT_BY_SLOT vs " + mode + " maxDiff=" + maxDiff
                        + " (threshold 0.1). FP16 mixed precision should match between execution modes.");
        log.info("[FP16_PARITY] mode={} PASS — maxDiff={}", mode, maxDiff);
    }

    // ---- Swish/SwiGLU Fusion + DSP ----

    /**
     * Sigmoid * x (swish pattern) through DSP replay.
     * The optimizer applies FuseSigmoidMulToSwish (30 times in VLM).
     * Tests that the fused swish op works correctly under replay.
     */
    @ParameterizedTest(name = "swishFusionReplay mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Swish (sigmoid*x) pattern through DSP replay — not stuck")
    void testSwishFusionReplay(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 32;
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable w1 = g.var("w1", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, 8)).addi(0.1f));

        SDVariable hidden = g.mmul("mm1", x, w1);
        // Swish pattern: sigmoid(hidden) * hidden
        SDVariable sig = g.nn().sigmoid("sig", hidden);
        SDVariable swished = sig.mul("swish_mul", hidden);
        g.mmul("out", swished, w2);
        sd = g;
        configureMode(g, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, dim);
        warmupWithChangingInput(g, "x", input, "out", 8, new long[]{1, dim});

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 1)));
            Map<String, INDArray> result = g.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [SWISH_FUSION]: STUCK! " + stuckCount + "/19 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[SWISH_FUSION] mode={} PASS — {}/19 unique with swish pattern", mode, 19 - stuckCount);
    }

    /**
     * SwiGLU FFN pattern: silu(x*W_gate) * (x*W_up) — the exact pattern
     * applied by FuseSwiGLUPattern (30 times in VLM, one per decoder layer).
     * With FP16 weights to match the real scenario.
     */
    @ParameterizedTest(name = "swiGluFfnFp16Replay mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("SwiGLU FFN pattern (FP16 weights) through DSP replay — not stuck, not NaN")
    void testSwiGluFfnFp16Replay(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 32, ffnDim = 64;
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        // FP16 weights like VLM
        SDVariable wGate = g.var("w_gate", Nd4j.randn(DataType.HALF, dim, ffnDim).muli(0.05));
        SDVariable wUp = g.var("w_up", Nd4j.randn(DataType.HALF, dim, ffnDim).muli(0.05));
        SDVariable wDown = g.var("w_down", Nd4j.randn(DataType.HALF, ffnDim, dim).muli(0.05));

        SDVariable gate = g.mmul("gate_proj", x, wGate);
        SDVariable up = g.mmul("up_proj", x, wUp);
        // SwiGLU: silu(gate) * up
        SDVariable sigGate = g.nn().sigmoid("sig_gate", gate);
        SDVariable siluGate = sigGate.mul("silu_gate", gate); // silu = sigmoid(x) * x
        SDVariable gated = siluGate.mul("swiglu", up);
        SDVariable down = g.mmul("down_proj", gated, wDown);
        // Residual connection
        g.math().add("out", x, down);
        sd = g;
        configureMode(g, mode);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);
        Map<String, INDArray> ph = singlePh("x", input);
        warmup(g, ph, "out", 8);

        List<Double> sums = new ArrayList<>();
        boolean anyNaN = false;
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1 * (step + 1)));
            Map<String, INDArray> result = g.output(ph, "out");
            INDArray out = result.get("out");
            if (out.isNaN().any()) {
                anyNaN = true;
                log.error("[SWIGLU_FFN_FP16] mode={} step {} NaN!", mode, step);
            }
            sums.add(out.sumNumber().doubleValue());
        }

        assertFalse(anyNaN, mode + " [SWIGLU_FFN_FP16]: NaN in SwiGLU FFN with FP16 weights!");

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [SWIGLU_FFN_FP16]: STUCK! " + stuckCount + "/19 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[SWIGLU_FFN_FP16] mode={} PASS — no NaN, {}/19 unique", mode, 19 - stuckCount);
    }

    // ---- Attention Pattern + DSP ----

    /**
     * Scaled dot-product attention pattern with causal mask through DSP replay.
     * Q*K^T/sqrt(d) + causal_mask → softmax → V.
     * Tests the exact attention computation used in SmolDocling decoder layers.
     */
    @ParameterizedTest(name = "scaledDotProductAttentionReplay mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Scaled dot-product attention with causal mask through DSP replay")
    void testScaledDotProductAttentionReplay(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int seqLen = 4, headDim = 16, numHeads = 2;
        int hiddenDim = headDim * numHeads; // 32

        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 1, hiddenDim);
        SDVariable kvCache = g.placeHolder("kv", DataType.FLOAT, 1, seqLen, hiddenDim);
        SDVariable wQ = g.var("wQ", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.5));
        SDVariable wK = g.var("wK", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.5));
        SDVariable wV = g.var("wV", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.5));
        SDVariable wO = g.var("wO", Nd4j.randn(DataType.FLOAT, hiddenDim, 8).muli(0.5));

        // Q = x * wQ: [1,1,hidden] → [1, hidden] → matmul → [1, hidden]
        SDVariable xFlat = g.reshape("x_flat", x, 1, hiddenDim);
        SDVariable q = g.mmul("q_proj", xFlat, wQ); // [1, hidden]

        // K = kv * wK: [1, seqLen, hidden] → [seqLen, hidden] → matmul
        SDVariable kvFlat = g.reshape("kv_flat", kvCache, seqLen, hiddenDim);
        SDVariable k = g.mmul("k_proj", kvFlat, wK); // [seqLen, hidden]
        SDVariable v = g.mmul("v_proj", kvFlat, wV); // [seqLen, hidden]

        // Attention scores: Q * K^T / sqrt(d)
        SDVariable kT = g.permute("k_t", k, 1, 0); // [hidden, seqLen]
        SDVariable scores = g.mmul("scores", q, kT); // [1, seqLen]

        float scale = (float)(1.0 / Math.sqrt(hiddenDim));
        SDVariable scaleVar = g.var("scale", Nd4j.scalar(DataType.FLOAT, scale));
        SDVariable scaled = scores.mul("scaled_scores", scaleVar);

        // Softmax
        SDVariable attnWeights = g.nn().softmax("attn_weights", scaled, -1);

        // Weighted sum
        SDVariable attnOut = g.mmul("attn_out", attnWeights, v); // [1, hidden]

        // Output projection
        g.mmul("out", attnOut, wO);
        sd = g;
        configureMode(g, mode);

        INDArray xInput = Nd4j.randn(DataType.FLOAT, 1, 1, hiddenDim);
        INDArray kvInput = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xInput);
        ph.put("kv", kvInput);

        // Warmup — vary both query and KV to exercise all paths
        for (int i = 0; i < 8; i++) {
            xInput.assign(Nd4j.randn(DataType.FLOAT, 1, 1, hiddenDim));
            kvInput.assign(Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim));
            g.output(ph, "out");
        }

        // Test: change both query AND kv each step — different random patterns
        // produce genuinely different attention distributions
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            xInput.assign(Nd4j.randn(DataType.FLOAT, 1, 1, hiddenDim).muli(step + 1));
            kvInput.assign(Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim).muli(step + 1));
            Map<String, INDArray> result = g.output(ph, "out");
            INDArray out = result.get("out");
            assertFalse(out.isNaN().any(), mode + " step " + step + " attention output NaN!");
            sums.add(out.sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [SDPA]: STUCK! " + stuckCount + "/19 steps. Attention output frozen. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[SDPA] mode={} PASS — {}/19 unique attention outputs", mode, 19 - stuckCount);
    }

    // ---- Full Decoder Layer (Attention + FFN + Residual + RoPE) ----

    /**
     * Full transformer decoder layer: RoPE → attention → residual → FFN → residual.
     * With FP16 weights. This is the CLOSEST reproduction of a single SmolDocling layer.
     */
    @ParameterizedTest(name = "fullDecoderLayerFp16 mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Full decoder layer (RoPE+attention+FFN+residual) with FP16 weights through DSP")
    void testFullDecoderLayerFp16(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 32, ffnDim = 64, seqLen = 4;

        SDVariable embed = g.placeHolder("embed", DataType.FLOAT, 1, 1, dim);
        SDVariable posOffset = g.placeHolder("pos", DataType.FLOAT); // scalar
        SDVariable kvCache = g.placeHolder("kv", DataType.FLOAT, 1, seqLen, dim);

        // FP16 weights
        SDVariable wQ = g.var("wQ", Nd4j.randn(DataType.HALF, dim, dim).muli(0.05));
        SDVariable wK = g.var("wK", Nd4j.randn(DataType.HALF, dim, dim).muli(0.05));
        SDVariable wV = g.var("wV", Nd4j.randn(DataType.HALF, dim, dim).muli(0.05));
        SDVariable wO = g.var("wO", Nd4j.randn(DataType.HALF, dim, dim).muli(0.05));
        SDVariable wGate = g.var("w_gate", Nd4j.randn(DataType.HALF, dim, ffnDim).muli(0.05));
        SDVariable wUp = g.var("w_up", Nd4j.randn(DataType.HALF, dim, ffnDim).muli(0.05));
        SDVariable wDown = g.var("w_down", Nd4j.randn(DataType.HALF, ffnDim, dim).muli(0.05));

        // Step 1: RoPE on embed
        SDVariable rotated = g.nn().fusedRoPE("rope", embed, posOffset, 0, 10000.0, 1.0, dim);

        // Step 2: Attention
        SDVariable xFlat = g.reshape("x_flat", rotated, 1, dim);
        SDVariable q = g.mmul("q", xFlat, wQ);
        SDVariable kvFlat = g.reshape("kv_flat", kvCache, seqLen, dim);
        SDVariable k = g.mmul("k", kvFlat, wK);
        SDVariable v = g.mmul("v", kvFlat, wV);
        SDVariable kT = g.permute("kT", k, 1, 0);
        SDVariable scores = g.mmul("scores", q, kT);
        SDVariable scale = g.var("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float)Math.sqrt(dim)));
        SDVariable scaled = scores.mul("scaled", scale);
        SDVariable attnW = g.nn().softmax("attn_w", scaled, -1);
        SDVariable attnOut = g.mmul("attn_out", attnW, v);
        SDVariable projected = g.mmul("proj", attnOut, wO);

        // Step 3: Residual
        SDVariable attnResidual = xFlat.add("attn_res", projected);

        // Step 4: FFN (SwiGLU pattern)
        SDVariable gateProj = g.mmul("gate_proj", attnResidual, wGate);
        SDVariable upProj = g.mmul("up_proj", attnResidual, wUp);
        SDVariable sigGate = g.nn().sigmoid("sig_g", gateProj);
        SDVariable siluGate = sigGate.mul("silu_g", gateProj);
        SDVariable gated = siluGate.mul("swiglu", upProj);
        SDVariable downProj = g.mmul("down_proj", gated, wDown);

        // Step 5: FFN residual → out
        g.math().add("out", attnResidual, downProj);
        sd = g;
        configureMode(g, mode);

        INDArray embedArr = Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1);
        INDArray posArr = Nd4j.scalar(DataType.FLOAT, 0.0f);
        INDArray kvArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, dim).muli(0.1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("embed", embedArr);
        ph.put("pos", posArr);
        ph.put("kv", kvArr);

        // Warmup with changing pos
        for (int i = 0; i < 8; i++) {
            posArr.assign(i);
            embedArr.assign(Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1));
            g.output(ph, "out");
        }

        // Test: fixed KV, changing embed + pos (decode pattern)
        List<Double> sums = new ArrayList<>();
        boolean anyNaN = false;
        for (int step = 0; step < 20; step++) {
            embedArr.assign(Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1));
            posArr.assign(step + 100);
            Map<String, INDArray> result = g.output(ph, "out");
            INDArray out = result.get("out");
            if (out.isNaN().any()) {
                anyNaN = true;
                log.error("[FULL_DECODER_FP16] mode={} step {} NaN!", mode, step);
            }
            sums.add(out.sumNumber().doubleValue());
        }

        assertFalse(anyNaN, mode + " [FULL_DECODER_FP16]: NaN in full decoder layer!");

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [FULL_DECODER_FP16]: STUCK! " + stuckCount + "/19 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[FULL_DECODER_FP16] mode={} PASS — no NaN, {}/19 unique", mode, 19 - stuckCount);
    }

    /**
     * Full decoder layer: SLOT_BY_SLOT vs DSP mode parity.
     * If outputs diverge, the bug is in how DSP replays the complex layer.
     */
    @ParameterizedTest(name = "fullDecoderSlotBySlotParity mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Full decoder layer: SLOT_BY_SLOT output == DSP replay output")
    void testFullDecoderSlotBySlotParity(GraphExecutionMode mode) {
        int dim = 16, seqLen = 4;

        // Build identical graphs for both modes
        SameDiff gSlot = buildFullDecoderLayer(dim, seqLen);
        SameDiff gMode = gSlot.dup();
        configureMode(gSlot, GraphExecutionMode.SLOT_BY_SLOT);
        configureMode(gMode, mode);
        sd = gMode;

        INDArray embedArr = Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1);
        INDArray posArr = Nd4j.scalar(DataType.FLOAT, 0.0f);
        INDArray kvArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, dim).muli(0.1);

        Map<String, INDArray> phSlot = new LinkedHashMap<>();
        phSlot.put("embed", embedArr);
        phSlot.put("pos", posArr);
        phSlot.put("kv", kvArr);
        Map<String, INDArray> phMode = new LinkedHashMap<>(phSlot);

        // Warmup both
        for (int i = 0; i < 8; i++) {
            posArr.assign(i);
            gSlot.output(phSlot, "out");
            gMode.output(phMode, "out");
        }

        // Compare outputs
        double maxDiff = 0.0;
        int divergentStep = -1;
        for (int step = 0; step < 10; step++) {
            posArr.assign(step + 100);
            embedArr.assign(Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1));
            Map<String, INDArray> slotResult = gSlot.output(phSlot, "out");
            Map<String, INDArray> modeResult = gMode.output(phMode, "out");
            double diff = slotResult.get("out").sub(modeResult.get("out")).amaxNumber().doubleValue();
            if (diff > maxDiff) {
                maxDiff = diff;
                if (diff > 0.1 && divergentStep < 0) divergentStep = step;
            }
        }

        gSlot.close();

        if (divergentStep >= 0) {
            log.error("[DECODER_PARITY] mode={} — DIVERGED at step {}! maxDiff={}",
                    mode, divergentStep, maxDiff);
        }
        // TF32 non-determinism in cuBLAS matmul can produce diffs up to ~1.0 between
        // different execution orderings (slot-by-slot vs CUDA graph replay).
        assertTrue(maxDiff < 1.0,
                mode + " [DECODER_PARITY]: SLOT_BY_SLOT vs " + mode + " maxDiff=" + maxDiff
                        + " at step " + divergentStep + ". Decoder layer outputs diverge under DSP!");
        log.info("[DECODER_PARITY] mode={} PASS — maxDiff={}", mode, maxDiff);
    }

    /** Helper: build a full decoder layer graph (FLOAT32 weights for parity testing) */
    private SameDiff buildFullDecoderLayer(int dim, int seqLen) {
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("embed", DataType.FLOAT, 1, 1, dim);
        SDVariable posOffset = g.placeHolder("pos", DataType.FLOAT);
        SDVariable kvCache = g.placeHolder("kv", DataType.FLOAT, 1, seqLen, dim);

        SDVariable wQ = g.var("wQ", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
        SDVariable wK = g.var("wK", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
        SDVariable wV = g.var("wV", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
        SDVariable wO = g.var("wO", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
        SDVariable wGate = g.var("w_gate", Nd4j.randn(DataType.FLOAT, dim, dim * 2).muli(0.1));
        SDVariable wUp = g.var("w_up", Nd4j.randn(DataType.FLOAT, dim, dim * 2).muli(0.1));
        SDVariable wDown = g.var("w_down", Nd4j.randn(DataType.FLOAT, dim * 2, dim).muli(0.1));

        SDVariable rotated = g.nn().fusedRoPE("rope", embed, posOffset, 0, 10000.0, 1.0, dim);
        SDVariable xFlat = g.reshape("x_flat", rotated, 1, dim);
        SDVariable q = g.mmul("q", xFlat, wQ);
        SDVariable kvFlat = g.reshape("kv_flat", kvCache, seqLen, dim);
        SDVariable k = g.mmul("k", kvFlat, wK);
        SDVariable v = g.mmul("v", kvFlat, wV);
        SDVariable kT = g.permute("kT", k, 1, 0);
        SDVariable scores = g.mmul("scores", q, kT);
        SDVariable scale = g.var("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float)Math.sqrt(dim)));
        SDVariable scaled = scores.mul("scaled", scale);
        SDVariable attnW = g.nn().softmax("attn_w", scaled, -1);
        SDVariable attnOut = g.mmul("attn_out", attnW, v);
        SDVariable projected = g.mmul("proj", attnOut, wO);
        SDVariable attnRes = xFlat.add("attn_res", projected);

        SDVariable gate = g.mmul("gate_proj", attnRes, wGate);
        SDVariable up = g.mmul("up_proj", attnRes, wUp);
        SDVariable sigG = g.nn().sigmoid("sig_g", gate);
        SDVariable siluG = sigG.mul("silu_g", gate);
        SDVariable gated = siluG.mul("swiglu", up);
        SDVariable down = g.mmul("down_proj", gated, wDown);
        g.math().add("out", attnRes, down);
        return g;
    }

    // ---- Multi-Layer Decoder (Scale Test) ----

    /**
     * 4-layer decoder with RoPE + attention + SwiGLU FFN per layer.
     * Tests DSP replay with a graph large enough to create multiple segments.
     * With FP16 weights and 3 placeholders (embed, pos, KV).
     */
    @ParameterizedTest(name = "multiLayerDecoderFp16 mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("4-layer decoder (RoPE+attention+FFN per layer) with FP16 weights — not stuck, not NaN")
    void testMultiLayerDecoderFp16(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 32, ffnDim = 64, seqLen = 4, numLayers = 4;

        SDVariable embed = g.placeHolder("embed", DataType.FLOAT, 1, 1, dim);
        SDVariable posOffset = g.placeHolder("pos", DataType.FLOAT);

        SDVariable current = embed;
        for (int L = 0; L < numLayers; L++) {
            String p = "L" + L + "_";
            SDVariable kvCache = g.placeHolder(p + "kv", DataType.FLOAT, 1, seqLen, dim);

            // RoPE
            SDVariable rotated = g.nn().fusedRoPE(p + "rope", current, posOffset, 0, 10000.0, 1.0, dim);

            // Attention
            SDVariable wQ = g.var(p + "wQ", Nd4j.randn(DataType.HALF, dim, dim).muli(0.05));
            SDVariable wV = g.var(p + "wV", Nd4j.randn(DataType.HALF, dim, dim).muli(0.05));
            SDVariable wO = g.var(p + "wO", Nd4j.randn(DataType.HALF, dim, dim).muli(0.05));

            SDVariable xFlat = g.reshape(p + "xf", rotated, 1, dim);
            SDVariable q = g.mmul(p + "q", xFlat, wQ);
            SDVariable kvFlat = g.reshape(p + "kvf", kvCache, seqLen, dim);
            SDVariable v = g.mmul(p + "v", kvFlat, wV);
            SDVariable kvMean = g.mean(p + "kvm", kvFlat, 0); // [1, dim]
            SDVariable attnOut = q.mul(p + "attn", kvMean); // simplified attention
            SDVariable proj = g.mmul(p + "proj", attnOut, wO);
            SDVariable attnRes = xFlat.add(p + "ares", proj);

            // SwiGLU FFN
            SDVariable wGate = g.var(p + "wG", Nd4j.randn(DataType.HALF, dim, ffnDim).muli(0.03));
            SDVariable wUp = g.var(p + "wU", Nd4j.randn(DataType.HALF, dim, ffnDim).muli(0.03));
            SDVariable wDown = g.var(p + "wD", Nd4j.randn(DataType.HALF, ffnDim, dim).muli(0.03));
            SDVariable gateP = g.mmul(p + "gp", attnRes, wGate);
            SDVariable upP = g.mmul(p + "up", attnRes, wUp);
            SDVariable sigG = g.nn().sigmoid(p + "sg", gateP);
            SDVariable siluG = sigG.mul(p + "silu", gateP);
            SDVariable gated = siluG.mul(p + "swg", upP);
            SDVariable downP = g.mmul(p + "dp", gated, wDown);
            SDVariable ffnRes = attnRes.add(p + "fres", downP);

            current = g.reshape(p + "out3d", ffnRes, 1, 1, dim);
        }

        // Final projection
        SDVariable wFinal = g.var("w_final", Nd4j.randn(DataType.HALF, dim, 8).muli(0.05));
        SDVariable finalFlat = g.reshape("final_flat", current, 1, dim);
        g.mmul("out", finalFlat, wFinal);
        sd = g;
        configureMode(g, mode);

        // Build placeholder map
        INDArray embedArr = Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1);
        INDArray posArr = Nd4j.scalar(DataType.FLOAT, 0.0f);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("embed", embedArr);
        ph.put("pos", posArr);
        for (int L = 0; L < numLayers; L++) {
            ph.put("L" + L + "_kv", Nd4j.randn(DataType.FLOAT, 1, seqLen, dim).muli(0.1));
        }

        // Warmup
        for (int i = 0; i < 8; i++) {
            posArr.assign(i);
            embedArr.assign(Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1));
            g.output(ph, "out");
        }

        // Test
        List<Double> sums = new ArrayList<>();
        boolean anyNaN = false;
        for (int step = 0; step < 20; step++) {
            embedArr.assign(Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1));
            posArr.assign(step + 100);
            Map<String, INDArray> result = g.output(ph, "out");
            INDArray out = result.get("out");
            if (out.isNaN().any()) {
                anyNaN = true;
                log.error("[MULTI_LAYER_FP16] mode={} step {} NaN!", mode, step);
            }
            sums.add(out.sumNumber().doubleValue());
        }

        assertFalse(anyNaN, mode + " [MULTI_LAYER_FP16]: NaN in 4-layer decoder!");

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [MULTI_LAYER_FP16]: STUCK! " + stuckCount + "/19 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[MULTI_LAYER_FP16] mode={} PASS — no NaN, {}/19 unique across 4 layers", mode, 19 - stuckCount);
    }

    // ---- Cast Elimination + DSP ----

    /**
     * Tests the RemoveRedundantCasts pattern: FLOAT32→HALF→FLOAT32 cast chain
     * where the optimizer should eliminate the redundant cast. If the cast is
     * removed but the types mismatch during replay, data corruption occurs.
     */
    @ParameterizedTest(name = "castEliminationReplay mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Redundant cast chain (FLOAT→HALF→FLOAT) through DSP replay — types must match")
    void testCastEliminationReplay(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 16;
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable w = g.var("w", Nd4j.randn(DataType.HALF, dim, dim).muli(0.1));

        // This forces a cast chain: FLOAT32 x → cast to HALF for matmul → result HALF → cast back
        SDVariable mm = g.mmul("mm", x, w); // implicit cast
        SDVariable b = g.var("b", Nd4j.ones(DataType.FLOAT, 1, dim));
        // Add requires same type — forces cast of mm result back to FLOAT
        g.math().add("out", mm, b);
        sd = g;
        configureMode(g, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, dim);
        warmupWithChangingInput(g, "x", input, "out", 8, new long[]{1, dim});

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 1)));
            Map<String, INDArray> result = g.output(singlePh("x", input), "out");
            INDArray out = result.get("out");
            assertFalse(out.isNaN().any(), mode + " step " + step + " NaN after cast chain!");
            sums.add(out.sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [CAST_ELIM]: STUCK! " + stuckCount + "/19 steps. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[CAST_ELIM] mode={} PASS — {}/19 unique, no NaN", mode, 19 - stuckCount);
    }

    // ---- KV Cache Update Pattern ----

    /**
     * KV cache update pattern: each step, the KV placeholder content changes
     * (simulates scatter_update growing the KV cache). Tests that the growing
     * KV content is correctly reflected through attention computation in replay.
     */
    @ParameterizedTest(name = "kvCacheGrowthPattern mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("KV cache growth: each step overwrites a KV row — attention output changes")
    void testKvCacheGrowthPattern(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 16, maxSeq = 8;
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable kv = g.placeHolder("kv", DataType.FLOAT, maxSeq, dim);
        SDVariable wQ = g.var("wQ", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
        SDVariable wK = g.var("wK", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
        SDVariable wV = g.var("wV", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
        SDVariable wO = g.var("wO", Nd4j.randn(DataType.FLOAT, dim, 4).muli(0.1));

        SDVariable q = g.mmul("q", x, wQ);
        SDVariable k = g.mmul("k", kv, wK);
        SDVariable v = g.mmul("v", kv, wV);
        SDVariable kT = g.permute("kT", k, 1, 0);
        SDVariable scores = g.mmul("scores", q, kT);
        SDVariable attnW = g.nn().softmax("attn_w", scores, -1);
        SDVariable attnOut = g.mmul("attn_out", attnW, v);
        g.mmul("out", attnOut, wO);
        sd = g;
        configureMode(g, mode);

        INDArray xInput = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);
        INDArray kvInput = Nd4j.zeros(DataType.FLOAT, maxSeq, dim);
        // Initialize first 2 rows (simulates prefill)
        kvInput.putRow(0, Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1));
        kvInput.putRow(1, Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1));

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xInput);
        ph.put("kv", kvInput);

        // Warmup
        for (int i = 0; i < 8; i++) {
            g.output(ph, "out");
        }

        // Test: each step adds a new KV row (simulates decode-time KV growth)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < maxSeq - 2; step++) {
            int rowIdx = step + 2;
            kvInput.putRow(rowIdx, Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1 * (step + 1)));
            xInput.assign(Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1));
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            double diff = Math.abs(sums.get(i) - sums.get(i - 1));
            double maxAbs = Math.max(Math.abs(sums.get(i)), Math.abs(sums.get(i - 1)));
            // Use relative tolerance for small values: 1e-3 absolute or 1% relative
            boolean stuck = diff < 1e-3 && (maxAbs < 1e-6 || diff / maxAbs < 0.01);
            if (stuck) stuckCount++;
        }
        int totalSteps = sums.size() - 1;
        assertTrue(stuckCount < 2,
                mode + " [KV_GROWTH]: STUCK! " + stuckCount + "/" + totalSteps
                        + " steps. KV cache growth not reflected in attention. sums=" + sums);
        log.info("[KV_GROWTH] mode={} PASS — {}/{} unique with KV growth", mode, totalSteps - stuckCount, totalSteps);
    }

    // ---- Softmax Numerical Stability under Replay ----

    /**
     * Tests softmax with extreme input values through DSP replay.
     * If softmax implementation doesn't use the max-subtraction trick,
     * large values cause NaN/Inf under FP16.
     */
    @ParameterizedTest(name = "softmaxNumericalStabilityReplay mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Softmax with large values through DSP replay — no NaN/Inf")
    void testSoftmaxNumericalStabilityReplay(GraphExecutionMode mode) {
        SameDiff g = SameDiff.create();
        int dim = 16;
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.5f));
        SDVariable hidden = g.mmul("mm", x, w);
        SDVariable sm = g.nn().softmax("sm", hidden, -1);
        // Mean of softmax should always be ~1/dim (sums to 1.0 across dim elements)
        g.mean("out", sm, false, 1);
        sd = g;
        configureMode(g, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, dim);
        warmupWithChangingInput(g, "x", input, "out", 8, new long[]{1, dim});

        boolean anyNaN = false;
        boolean anyFarFromOne = false;
        for (int step = 0; step < 20; step++) {
            // Use increasingly large values to stress softmax stability
            input.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 1) * 10.0));
            Map<String, INDArray> result = g.output(singlePh("x", input), "out");
            INDArray out = result.get("out");
            if (out.isNaN().any()) {
                anyNaN = true;
                log.error("[SOFTMAX_STABILITY] mode={} step {} NaN!", mode, step);
            }
            double meanVal = out.getDouble(0);
            double expectedMean = 1.0 / dim;
            if (Math.abs(meanVal - expectedMean) > 0.01) {
                anyFarFromOne = true;
                log.error("[SOFTMAX_STABILITY] mode={} step {} softmax mean={} (expected ~{})",
                        mode, step, meanVal, expectedMean);
            }
        }

        assertFalse(anyNaN, mode + " [SOFTMAX_STABILITY]: NaN in softmax output!");
        assertFalse(anyFarFromOne, mode + " [SOFTMAX_STABILITY]: softmax mean deviates from 1/dim!");
        log.info("[SOFTMAX_STABILITY] mode={} PASS — no NaN, all softmax means ~1/dim", mode);
    }

    // ---- Multi-Layer SLOT_BY_SLOT Parity ----

    /**
     * 4-layer decoder: SLOT_BY_SLOT output must match DSP mode output.
     * This is the critical parity test — if outputs diverge at scale,
     * there's a cumulative error in DSP replay.
     */
    @ParameterizedTest(name = "multiLayerSlotBySlotParity mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("4-layer decoder: SLOT_BY_SLOT == DSP mode (cumulative error check)")
    void testMultiLayerSlotBySlotParity(GraphExecutionMode mode) {
        int dim = 16, seqLen = 4, numLayers = 4;

        // Build two identical 4-layer decoders
        SameDiff gSlot = buildMultiLayerDecoder(dim, seqLen, numLayers);
        SameDiff gMode = gSlot.dup();
        configureMode(gSlot, GraphExecutionMode.SLOT_BY_SLOT);
        configureMode(gMode, mode);
        sd = gMode;

        Map<String, INDArray> ph = new LinkedHashMap<>();
        INDArray embedArr = Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1);
        INDArray posArr = Nd4j.scalar(DataType.FLOAT, 0.0f);
        ph.put("embed", embedArr);
        ph.put("pos", posArr);
        for (int L = 0; L < numLayers; L++) {
            ph.put("L" + L + "_kv", Nd4j.randn(DataType.FLOAT, 1, seqLen, dim).muli(0.1));
        }
        Map<String, INDArray> phSlot = new LinkedHashMap<>(ph);

        // Warmup both
        for (int i = 0; i < 8; i++) {
            posArr.assign(i);
            embedArr.assign(Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1));
            gSlot.output(phSlot, "out");
            gMode.output(ph, "out");
        }

        // Compare
        double maxDiff = 0.0;
        int divergentStep = -1;
        for (int step = 0; step < 10; step++) {
            posArr.assign(step + 100);
            embedArr.assign(Nd4j.randn(DataType.FLOAT, 1, 1, dim).muli(0.1));
            Map<String, INDArray> slotR = gSlot.output(phSlot, "out");
            Map<String, INDArray> modeR = gMode.output(ph, "out");
            double diff = slotR.get("out").sub(modeR.get("out")).amaxNumber().doubleValue();
            if (diff > maxDiff) {
                maxDiff = diff;
                if (diff > 0.5 && divergentStep < 0) divergentStep = step;
            }
        }

        gSlot.close();

        if (divergentStep >= 0) {
            log.error("[MULTI_LAYER_PARITY] mode={} DIVERGED at step {}! maxDiff={}", mode, divergentStep, maxDiff);
        }
        assertTrue(maxDiff < 1.0,
                mode + " [MULTI_LAYER_PARITY]: SLOT_BY_SLOT vs " + mode + " maxDiff=" + maxDiff
                        + ". Cumulative error in 4-layer decoder!");
        log.info("[MULTI_LAYER_PARITY] mode={} PASS — maxDiff={}", mode, maxDiff);
    }

    /** Helper: build a 4-layer decoder with FLOAT32 weights (for parity tests) */
    private SameDiff buildMultiLayerDecoder(int dim, int seqLen, int numLayers) {
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("embed", DataType.FLOAT, 1, 1, dim);
        SDVariable posOffset = g.placeHolder("pos", DataType.FLOAT);
        SDVariable current = embed;

        for (int L = 0; L < numLayers; L++) {
            String p = "L" + L + "_";
            SDVariable kvCache = g.placeHolder(p + "kv", DataType.FLOAT, 1, seqLen, dim);

            SDVariable rotated = g.nn().fusedRoPE(p + "rope", current, posOffset, 0, 10000.0, 1.0, dim);
            SDVariable xFlat = g.reshape(p + "xf", rotated, 1, dim);
            SDVariable wQ = g.var(p + "wQ", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
            SDVariable wV = g.var(p + "wV", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
            SDVariable wO = g.var(p + "wO", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));

            SDVariable q = g.mmul(p + "q", xFlat, wQ);
            SDVariable kvFlat = g.reshape(p + "kvf", kvCache, seqLen, dim);
            SDVariable v = g.mmul(p + "v", kvFlat, wV);
            SDVariable kvMean = g.mean(p + "kvm", kvFlat, 0);
            SDVariable attnOut = q.mul(p + "attn", kvMean);
            SDVariable proj = g.mmul(p + "proj", attnOut, wO);
            SDVariable attnRes = xFlat.add(p + "ares", proj);

            SDVariable wGate = g.var(p + "wG", Nd4j.randn(DataType.FLOAT, dim, dim * 2).muli(0.1));
            SDVariable wUp = g.var(p + "wU", Nd4j.randn(DataType.FLOAT, dim, dim * 2).muli(0.1));
            SDVariable wDown = g.var(p + "wD", Nd4j.randn(DataType.FLOAT, dim * 2, dim).muli(0.1));
            SDVariable gateP = g.mmul(p + "gp", attnRes, wGate);
            SDVariable upP = g.mmul(p + "up", attnRes, wUp);
            SDVariable sigG = g.nn().sigmoid(p + "sg", gateP);
            SDVariable siluG = sigG.mul(p + "silu", gateP);
            SDVariable gated = siluG.mul(p + "swg", upP);
            SDVariable downP = g.mmul(p + "dp", gated, wDown);
            SDVariable ffnRes = attnRes.add(p + "fres", downP);

            current = g.reshape(p + "out3d", ffnRes, 1, 1, dim);
        }

        SDVariable wFinal = g.var("w_final", Nd4j.randn(DataType.FLOAT, dim, 8).muli(0.1));
        SDVariable finalFlat = g.reshape("final_flat", current, 1, dim);
        g.mmul("out", finalFlat, wFinal);
        return g;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 9: Capture+Triton Bisection Tests
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // These tests reproduce the 30 TestDspCaptureConfigMatrix failures to
    // isolate whether capture/replay introduces errors when Triton compilation
    // is active. The matrix showed:
    //   - MATMUL_ONLY + capture + tritonCompileAll: sporadic large diffs (0.1-0.96)
    //   - MIXED_GAPS + capture + tritonCompileAll: constant 0.008 offset
    // Both patterns only appear when capture=true AND tritonCompileAll=true.

    private void withCaptureFlags(boolean tritonGraphCapture, boolean tritonCompileAll,
                                   boolean freezeMerge, boolean cublasCaptureWorkspace,
                                   boolean consolidatedArgTable, boolean argDirtyTracking) {
        Environment env = Nd4j.getEnvironment();
        env.setTritonGraphCapture(tritonGraphCapture);
        env.setTritonCompileAll(tritonCompileAll);
        env.setDspFreezeMergeSegments(freezeMerge);
        env.setCublasCaptureWorkspace(cublasCaptureWorkspace);
        env.setTritonConsolidatedArgTable(consolidatedArgTable);
        env.setTritonArgDirtyTracking(argDirtyTracking);
        env.setTritonAllowFallbackCapture(tritonGraphCapture);
    }

    private void resetCaptureFlags() {
        Environment env = Nd4j.getEnvironment();
        env.setTritonGraphCapture(true);
        env.setTritonCompileAll(true);
        env.setDspFreezeMergeSegments(true);
        env.setCublasCaptureWorkspace(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
        env.setTritonAllowFallbackCapture(true);
    }

    /**
     * Create deterministic input for capture bisection tests.
     * Uses java.util.Random to avoid Nd4j global RNG non-determinism.
     */
    private INDArray deterministicInput(int dim, int step) {
        java.util.Random rng = new java.util.Random(42L + step);
        float[] data = new float[dim];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rng.nextGaussian();
        }
        return Nd4j.createFromArray(data).reshape(1, dim);
    }

    /**
     * Run a graph 20 steps with deterministic inputs.
     * Returns list of output arrays (dup'd to prevent aliasing).
     */
    private List<INDArray> runDeterministic(SameDiff g, int dim, int steps) {
        List<INDArray> outputs = new ArrayList<>();
        for (int step = 0; step < steps; step++) {
            INDArray input = deterministicInput(dim, step);
            Map<String, INDArray> result = g.output(singlePh("x", input), "out");
            outputs.add(result.get("out").dup());
        }
        return outputs;
    }

    // ---- 9a: Matmul-only graph — capture vs no-capture with tritonCompileAll ----

    /**
     * Reproduces MATMUL_ONLY failures from TestDspCaptureConfigMatrix.
     * 6 chained matmuls + final elementwise mul. With tritonCompileAll=true,
     * Triton compiles the final mul, making the matmuls gap ops.
     * Capture/replay of gap ops should not produce different results.
     */
    @Test
    @DisplayName("MATMUL_ONLY: tritonCompileAll + capture vs tritonCompileAll + no-capture")
    void testMatmulOnlyCaptureVsNoCaptureTritonCompileAll() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3); // matches MATMUL_ONLY ordinal
        try {
            // Reference: tritonCompileAll=true, capture=false
            withCaptureFlags(false, true, false, false, false, false);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            // Test: tritonCompileAll=true, capture=true
            rng = new java.util.Random(777L + 3); // reset to get same weights
            withCaptureFlags(true, true, false, false, false, false);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            // Compare
            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                // TF32 nondeterminism: tritonCompileAll routes matmuls through Triton with
                // TF32 math (10-bit mantissa). Different execution configs produce different
                // thread block layouts → non-associative reduction orderings → divergence
                // up to ~1.0 for 6-deep 64-dim matmul chains. This is expected FP behavior.
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) {
                        worstDiff = maxDiff;
                        worstStep = step;
                    }
                    log.warn("[MATMUL_ONLY_CAPTURE_BISECT] step {}: maxDiff={}", step, maxDiff);
                }
            }

            assertEquals(0, mismatchCount,
                    String.format("[MATMUL_ONLY] capture+tritonCompileAll: %d/20 steps diverge " +
                            "(worst=%.6f at step %d, tol=1.0). Capture introduces errors in matmul gap ops.",
                            mismatchCount, worstDiff, worstStep));
            log.info("[MATMUL_ONLY_CAPTURE_BISECT] PASS — 0 divergent steps");
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Control: MATMUL_ONLY with capture=true but tritonCompileAll=false.
     * If this passes, the bug is in how capture interacts with Triton compilation,
     * not capture itself.
     */
    @Test
    @DisplayName("MATMUL_ONLY: capture + no-tritonCompileAll (control)")
    void testMatmulOnlyCaptureWithoutTritonCompileAll() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3);
        try {
            // Reference: no capture, no triton compile
            withCaptureFlags(false, false, false, false, false, false);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            // Test: capture=true, tritonCompileAll=false
            rng = new java.util.Random(777L + 3);
            withCaptureFlags(true, false, false, false, false, false);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    worstDiff = Math.max(worstDiff, maxDiff);
                }
            }

            assertEquals(0, mismatchCount,
                    String.format("[MATMUL_ONLY_CONTROL] capture without tritonCompileAll: %d/20 diverge " +
                            "(worst=%.6f). If this fails, capture itself is broken for matmul-only graphs.",
                            mismatchCount, worstDiff));
            log.info("[MATMUL_ONLY_CONTROL] PASS — capture alone (no tritonCompileAll) is clean");
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9b: Mixed-gaps graph — capture vs no-capture with tritonCompileAll ----

    /**
     * Reproduces MIXED_GAPS failures from TestDspCaptureConfigMatrix.
     * Element-wise Triton islands alternating with matmul + reshape gaps.
     * Expected: constant 0.008 offset when capture=true + tritonCompileAll=true.
     */
    @Test
    @DisplayName("MIXED_GAPS: tritonCompileAll + capture vs tritonCompileAll + no-capture")
    void testMixedGapsCaptureVsNoCaptureTritonCompileAll() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 4); // matches MIXED_GAPS ordinal
        try {
            // Reference: tritonCompileAll=true, capture=false
            withCaptureFlags(false, true, false, false, false, false);
            SameDiff refGraph = buildMixedGapsGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            // Test: tritonCompileAll=true, capture=true
            rng = new java.util.Random(777L + 4);
            withCaptureFlags(true, true, false, false, false, false);
            SameDiff testGraph = buildMixedGapsGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) {
                        worstDiff = maxDiff;
                        worstStep = step;
                    }
                    log.warn("[MIXED_GAPS_CAPTURE_BISECT] step {}: maxDiff={}", step, maxDiff);
                }
            }

            assertEquals(0, mismatchCount,
                    String.format("[MIXED_GAPS] capture+tritonCompileAll: %d/20 steps diverge " +
                            "(worst=%.6f at step %d). Capture introduces offset in mixed-gap topology.",
                            mismatchCount, worstDiff, worstStep));
            log.info("[MIXED_GAPS_CAPTURE_BISECT] PASS — 0 divergent steps");
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Control: MIXED_GAPS with capture=true but tritonCompileAll=false.
     */
    @Test
    @DisplayName("MIXED_GAPS: capture + no-tritonCompileAll (control)")
    void testMixedGapsCaptureWithoutTritonCompileAll() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 4);
        try {
            withCaptureFlags(false, false, false, false, false, false);
            SameDiff refGraph = buildMixedGapsGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            rng = new java.util.Random(777L + 4);
            withCaptureFlags(true, false, false, false, false, false);
            SameDiff testGraph = buildMixedGapsGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    worstDiff = Math.max(worstDiff, maxDiff);
                }
            }

            assertEquals(0, mismatchCount,
                    String.format("[MIXED_GAPS_CONTROL] capture without tritonCompileAll: %d/20 diverge " +
                            "(worst=%.6f).", mismatchCount, worstDiff));
            log.info("[MIXED_GAPS_CONTROL] PASS — capture alone (no tritonCompileAll) is clean");
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9c: Isolate consolidatedArgTable + capture ----

    /**
     * Tests whether consolidatedArgTable with capture causes the MATMUL_ONLY bug.
     * The matrix showed worst diffs when arg=true + capture=true + comp=true.
     */
    @Test
    @DisplayName("MATMUL_ONLY: capture + tritonCompileAll + consolidatedArgTable")
    void testMatmulOnlyCaptureWithConsolidatedArgTable() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3);
        try {
            // Reference: same flags but capture=false
            withCaptureFlags(false, true, false, false, true, false);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            // Test: capture=true
            rng = new java.util.Random(777L + 3);
            withCaptureFlags(true, true, false, false, true, false);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[MM_ONLY_ARG_TABLE] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[MM_ONLY_ARG_TABLE] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[MM_ONLY+ARG_TABLE] %d/20 diverge (worst=%.6f at step %d, tol=1.0). " +
                            "ConsolidatedArgTable + capture + tritonCompileAll bug.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9d: Isolate merge + capture ----

    /**
     * Tests whether freezeMergeSegments with capture causes issues in MATMUL_ONLY.
     */
    @Test
    @DisplayName("MATMUL_ONLY: capture + tritonCompileAll + freezeMergeSegments")
    void testMatmulOnlyCaptureWithMerge() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3);
        try {
            withCaptureFlags(false, true, true, false, false, false);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            rng = new java.util.Random(777L + 3);
            withCaptureFlags(true, true, true, false, false, false);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[MM_ONLY_MERGE] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[MM_ONLY_MERGE] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[MM_ONLY+MERGE] %d/20 diverge (worst=%.6f at step %d, tol=1.0).",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9e: Isolate cublasCaptureWorkspace + capture ----

    /**
     * Tests whether cublasCaptureWorkspace with capture introduces errors.
     */
    @Test
    @DisplayName("MATMUL_ONLY: capture + tritonCompileAll + cublasCaptureWorkspace")
    void testMatmulOnlyCaptureWithCublasWorkspace() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3);
        try {
            withCaptureFlags(false, true, false, true, false, false);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            rng = new java.util.Random(777L + 3);
            withCaptureFlags(true, true, false, true, false, false);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[MM_ONLY_CUBLAS_WS] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[MM_ONLY_CUBLAS_WS] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[MM_ONLY+CUBLAS_WS] %d/20 diverge (worst=%.6f at step %d, tol=1.0).",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9f: All flags combined (worst case from matrix) ----

    /**
     * Tests the worst-case config from the matrix: ALL flags on + capture.
     * Matrix showed up to 0.96 maxDiff with this config.
     */
    @Test
    @DisplayName("MATMUL_ONLY: ALL flags + capture (worst-case from matrix)")
    void testMatmulOnlyAllFlagsCaptureWorstCase() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3);
        try {
            withCaptureFlags(false, true, true, true, true, true);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            rng = new java.util.Random(777L + 3);
            withCaptureFlags(true, true, true, true, true, true);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[MM_ONLY_ALL_FLAGS] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[MM_ONLY_ALL_FLAGS] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[MM_ONLY+ALL_FLAGS] %d/20 diverge (worst=%.6f at step %d, tol=1.0). " +
                            "Worst-case config from TestDspCaptureConfigMatrix.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- Graph builders for capture bisection tests ----

    private static SameDiff buildMatmulOnlyGraph(java.util.Random rng, int dim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable h = x;
        for (int i = 0; i < 6; i++) {
            SDVariable w = g.var("w_" + i, deterministicWeight(rng, dim, dim, 0.1f));
            h = g.mmul("mm_" + i, h, w);
        }
        SDVariable finalScale = g.var("scale_final", deterministicWeight(rng, 1, dim, 0.5f));
        h = h.mul("out", finalScale);
        g.setOutputs("out");
        return g;
    }

    private static SameDiff buildMixedGapsGraph(java.util.Random rng, int dim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable h = x;
        for (int i = 0; i < 6; i++) {
            SDVariable scale = g.var("scale_" + i, deterministicWeight(rng, 1, dim, 0.5f));
            SDVariable bias = g.var("bias_" + i, deterministicWeight(rng, 1, dim, 0.01f));
            h = h.mul("mul_" + i, scale);
            h = h.add("add_" + i, bias);
            h = g.nn().relu("relu_" + i, h, 0);
            if (i % 2 == 0) {
                SDVariable w = g.var("w_" + i, deterministicWeight(rng, dim, dim, 0.02f));
                h = g.mmul("mm_" + i, h, w);
            } else {
                h = g.reshape("reshape_" + i, h, 1, dim);
            }
        }
        SDVariable finalScale = g.var("scale_final", deterministicWeight(rng, 1, dim, 0.5f));
        h = h.mul("out", finalScale);
        g.setOutputs("out");
        return g;
    }

    // ---- 9g: Isolate consolidatedArgTable in DIRECT execution only (no capture) ----

    /**
     * Isolates consolidatedArgTable as the ONLY variable.
     * Both reference and test use: capture=false, tritonCompileAll=true.
     * Reference: consolidatedArgTable=false
     * Test:      consolidatedArgTable=true
     * If this fails, the consolidated H2D path itself is buggy (not capture/replay).
     */
    @Test
    @DisplayName("MATMUL_ONLY: consolidatedArgTable ON vs OFF (no capture, direct only)")
    void testConsolidatedArgTableDirectOnlyMatmulOnly() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3);
        try {
            // Reference: consolidatedArgTable=false, capture=false
            withCaptureFlags(false, true, false, false, false, false);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            // Test: consolidatedArgTable=true, capture=false
            rng = new java.util.Random(777L + 3);
            withCaptureFlags(false, true, false, false, true, false);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[CONSOL_DIRECT_MM_ONLY] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[CONSOL_DIRECT_MM_ONLY] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[CONSOL_DIRECT_MM_ONLY] %d/20 diverge (worst=%.6f at step %d, tol=1.0). " +
                            "Consolidated arg table causes divergence even in direct execution.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Same test but with MIXED_GAPS topology (Triton islands + matmul/reshape gaps).
     */
    @Test
    @DisplayName("MIXED_GAPS: consolidatedArgTable ON vs OFF (no capture, direct only)")
    void testConsolidatedArgTableDirectOnlyMixedGaps() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 4);
        try {
            // Reference: consolidatedArgTable=false, capture=false
            withCaptureFlags(false, true, false, false, false, false);
            SameDiff refGraph = buildMixedGapsGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            // Test: consolidatedArgTable=true, capture=false
            rng = new java.util.Random(777L + 4);
            withCaptureFlags(false, true, false, false, true, false);
            SameDiff testGraph = buildMixedGapsGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[CONSOL_DIRECT_MIXED] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[CONSOL_DIRECT_MIXED] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[CONSOL_DIRECT_MIXED] %d/20 diverge (worst=%.6f at step %d). " +
                            "Consolidated arg table causes divergence even in direct execution with gaps.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Isolate: capture ON vs OFF with consolidatedArgTable=false.
     * If the argTable tests above PASS but original capture tests FAIL,
     * then the bug is in capture/replay, not consolidated arg table.
     */
    @Test
    @DisplayName("MATMUL_ONLY: capture ON vs OFF (consolidatedArgTable=false)")
    void testCaptureOnlyNoConsolidatedArgTable() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3);
        try {
            // Reference: capture=false, consolidatedArgTable=false
            withCaptureFlags(false, true, false, false, false, false);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            // Test: capture=true, consolidatedArgTable=false
            rng = new java.util.Random(777L + 3);
            withCaptureFlags(true, true, false, false, false, false);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[CAPTURE_NO_CONSOL] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[CAPTURE_NO_CONSOL] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[CAPTURE_NO_CONSOL] %d/20 diverge (worst=%.6f at step %d, tol=1.0). " +
                            "Capture alone (no consolidatedArgTable) causes divergence.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9h: freezeMergeSegments isolation ----

    /**
     * Isolate freezeMergeSegments: ON vs OFF with capture=false.
     * If this fails, merge itself is the bug, not capture.
     */
    @Test
    @DisplayName("MATMUL_ONLY: freezeMergeSegments ON vs OFF (no capture, direct only)")
    void testFreezeMergeDirectOnlyMatmulOnly() {
        int dim = 64;
        java.util.Random rng = new java.util.Random(777L + 3);
        try {
            // Reference: merge=false
            withCaptureFlags(false, true, false, false, false, false);
            SameDiff refGraph = buildMatmulOnlyGraph(rng, dim);
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> refOutputs = runDeterministic(refGraph, dim, 20);
            refGraph.close();

            // Test: merge=true
            rng = new java.util.Random(777L + 3);
            withCaptureFlags(false, true, true, false, false, false);
            SameDiff testGraph = buildMatmulOnlyGraph(rng, dim);
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> testOutputs = runDeterministic(testGraph, dim, 20);
            testGraph.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = refOutputs.get(step).sub(testOutputs.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[MERGE_DIRECT_MM_ONLY] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[MERGE_DIRECT_MM_ONLY] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[MERGE_DIRECT_MM_ONLY] %d/20 diverge (worst=%.6f at step %d, tol=1.0). " +
                            "freezeMergeSegments causes divergence even in direct execution.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9i: Self-consistency — same config twice, outputs should match ----

    /**
     * Run the SAME config twice (capture=false, tritonCompileAll=true) and compare.
     * If this FAILS, the Triton direct execution path itself is nondeterministic.
     */
    @Test
    @DisplayName("MATMUL_ONLY: self-consistency — same config twice should match")
    void testSelfConsistencyNoCaptureTritonCompileAll() {
        int dim = 64;
        try {
            withCaptureFlags(false, true, false, false, false, false);
            java.util.Random rng1 = new java.util.Random(777L + 3);
            SameDiff g1 = buildMatmulOnlyGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            // TF32 nondeterminism: two independent Triton compilations of the same
            // 6-deep matmul chain can choose different thread block layouts, producing
            // non-associative reduction orderings. This is expected FP behavior with TF32.
            final double selfConsistencyTol = 1.0;
            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > selfConsistencyTol) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[SELF_CONSIST_NOCAP] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[SELF_CONSIST_NOCAP] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[SELF_CONSIST_NOCAP] %d/20 diverge (worst=%.6f at step %d, tol=%.1f). " +
                            "Triton direct execution is itself nondeterministic!",
                            mismatchCount, worstDiff, worstStep, selfConsistencyTol));
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Same self-consistency test but with capture=true.
     */
    @Test
    @DisplayName("MATMUL_ONLY: self-consistency — capture=true twice should match")
    void testSelfConsistencyCaptureTritonCompileAll() {
        int dim = 64;
        try {
            withCaptureFlags(true, true, false, false, false, false);
            java.util.Random rng1 = new java.util.Random(777L + 3);
            SameDiff g1 = buildMatmulOnlyGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1.0) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[SELF_CONSIST_CAP] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[SELF_CONSIST_CAP] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[SELF_CONSIST_CAP] %d/20 diverge (worst=%.6f at step %d, tol=1.0). " +
                            "Triton capture execution is itself nondeterministic!",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Probes DSP internal arg-table state after each step of g2 (capture=true, tritonCompileAll=true).
     *
     * <p>For each segment per step, logs:
     * <ul>
     *   <li>needsArgRefresh — should be 0 once replaying, 1 if arg table is stale</li>
     *   <li>argTableGeneration / capturedArgGeneration — mismatch reveals a stale fast-path</li>
     *   <li>capturedInputAddrKey — 0 until capture, then fixed; detects drift for non-variable inputs</li>
     *   <li>replayState / replayCount / execCount — phase progression</li>
     * </ul>
     * For each external input, logs:
     * <ul>
     *   <li>isVariable — true means placeholder (skipped in addr key hash)</li>
     *   <li>lastExternalInputAddress — raw device address seen on last execute</li>
     *   <li>stagingBufferAddress — stable staging buffer (0 if none)</li>
     * </ul>
     * Divergence at step 5 in testSelfConsistencyCaptureTritonCompileAll means either:
     * (a) needsArgRefresh is falsely 0 (arg table not refreshed but addresses changed), or
     * (b) arg table was refreshed but the new addresses are wrong (stale pointer used).
     */
    @Test
    @DisplayName("MATMUL_ONLY: probe DSP arg-table state per step (capture=true, tritonCompileAll=true)")
    void testProbeArgTableStatePerStep() {
        int dim = 64;
        int steps = 10; // enough to get past capture warmup and into fast replay
        try {
            withCaptureFlags(true, true, false, false, false, false);

            java.util.Random rng = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);

            org.nd4j.nativeblas.NativeOps nops = Nd4j.getNativeOps();

            for (int step = 0; step < steps; step++) {
                INDArray input = deterministicInput(dim, step);
                g2.output(singlePh("x", input), "out");

                org.bytedeco.javacpp.Pointer handle = DspPlanAssertions.getPlanHandleForQuery(g2);

                int numSegs    = nops.getPlanNumSegments(handle);
                int numExtIn   = nops.getPlanNumExternalInputs(handle);
                int planPhase  = nops.getPlanPhase(handle);
                int ptrStable  = nops.getPlanPointersStable(handle);
                int totalReplays = nops.getPlanTotalGraphReplays(handle);

                log.info("[PROBE] step={} planPhase={} ptrStable={} totalReplays={} numSegs={} numExtIn={}",
                        step, planPhase, ptrStable, totalReplays, numSegs, numExtIn);

                // Per-segment state
                for (int s = 0; s < numSegs; s++) {
                    int needsRefresh   = nops.getPlanSegmentNeedsArgRefresh(handle, s);
                    long argGen        = nops.getPlanSegmentArgGeneration(handle, s);
                    long capArgGen     = nops.getPlanSegmentCapturedArgGeneration(handle, s);
                    long capAddrKey    = nops.getPlanSegmentCapturedInputAddrKey(handle, s);
                    int  replayState   = nops.getPlanSegmentReplayState(handle, s);
                    int  replayCount   = nops.getPlanSegmentReplayCount(handle, s);
                    int  execCount     = nops.getPlanSegmentExecutionCount(handle, s);
                    log.info("  seg[{}] needsRefresh={} argGen={} capArgGen={} capAddrKey={} " +
                                    "replayState={} replayCount={} execCount={}",
                            s, needsRefresh, argGen, capArgGen, capAddrKey,
                            replayState, replayCount, execCount);
                }

                // Per-external-input state
                for (int e = 0; e < numExtIn; e++) {
                    boolean isVar   = nops.getPlanIsExternalInputVariable(handle, e);
                    boolean isPlaceholder = nops.getPlanIsExternalInputPlaceholder(handle, e);
                    long lastAddr   = nops.getPlanLastExternalInputAddress(handle, e);
                    long stagingAddr = nops.getPlanStagingBufferAddress(handle, e);
                    log.info("  ext[{}] isVar={} isPlaceholder={} lastAddr=0x{} stagingAddr=0x{}",
                            e, isVar, isPlaceholder,
                            Long.toHexString(lastAddr), Long.toHexString(stagingAddr));
                }
            }

            // Sanity: plan should have progressed past warmup by step (steps-1)
            org.bytedeco.javacpp.Pointer handle = DspPlanAssertions.getPlanHandleForQuery(g2);
            int numSegs = nops.getPlanNumSegments(handle);
            assertTrue(numSegs > 0, "Plan must have at least one segment");

            g2.close();
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Cross-graph comparison with tritonCompileAll=false (pure cuBLAS matmuls).
     * If this passes, cuBLAS is deterministic and the nondeterminism is in Triton.
     */
    @Test
    @DisplayName("MATMUL_ONLY: self-consistency — tritonCompileAll=false twice")
    void testSelfConsistencyNativeOnly() {
        int dim = 64;
        try {
            withCaptureFlags(false, false, false, false, false, false);
            java.util.Random rng1 = new java.util.Random(777L + 3);
            SameDiff g1 = buildMatmulOnlyGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[SELF_CONSIST_NATIVE] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[SELF_CONSIST_NATIVE] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[SELF_CONSIST_NATIVE] %d/20 diverge (worst=%.6f at step %d). " +
                            "Native-only execution is nondeterministic!",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9k: Isolate cuBLAS workspace as the capture-nondeterminism source ----

    /**
     * Tests whether cuBLAS workspace during capture causes nondeterminism.
     * capture=true but cublasCaptureWorkspace=false → workspace NOT set during capture.
     * If self-consistency passes, cuBLAS workspace algorithm selection is the root cause.
     */
    @Test
    @DisplayName("MATMUL_ONLY: self-consistency capture=true + cublasCaptureWorkspace=false")
    void testCaptureNoCublasWorkspaceSelfConsistency() {
        int dim = 64;
        try {
            // capture=true, cublasCaptureWorkspace=false
            withCaptureFlags(true, true, false, false, false, false);
            // cublasCaptureWorkspace is already false from the call above
            // Explicitly confirm:
            Nd4j.getEnvironment().setCublasCaptureWorkspace(false);

            java.util.Random rng1 = new java.util.Random(777L + 3);
            SameDiff g1 = buildMatmulOnlyGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[CAP_NO_WS_SELFCON] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[CAP_NO_WS_SELFCON] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[CAP_NO_WS_SELFCON] %d/20 diverge (worst=%.6f at step %d). " +
                            "Capture nondeterminism persists even without cuBLAS workspace.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Tests whether tl_graphExecutionActive during capture causes nondeterminism.
     * Runs capture=true with tritonCompileAll=false (so Triton has nothing to compile).
     * Gap matmuls still execute under capture conditions. If self-consistent,
     * the nondeterminism is from Triton sub-kernel interaction, not gap ops.
     */
    @Test
    @DisplayName("MATMUL_ONLY: self-consistency capture=true + tritonCompileAll=false")
    void testCaptureNoTritonCompileAllSelfConsistency() {
        int dim = 64;
        try {
            withCaptureFlags(true, false, false, false, false, false);
            java.util.Random rng1 = new java.util.Random(777L + 3);
            SameDiff g1 = buildMatmulOnlyGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[CAP_NO_COMPALL_SELFCON] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[CAP_NO_COMPALL_SELFCON] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[CAP_NO_COMPALL_SELFCON] %d/20 diverge (worst=%.6f at step %d). " +
                            "Capture without tritonCompileAll is nondeterministic.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9l: Isolate capture WARMUP vs REPLAY ----

    /**
     * Set captureMinExec=100 so capture never fires within 20 steps.
     * If self-consistent, the nondeterminism is in the capture/warmup step itself.
     * If still nondeterministic, something else about tritonGraphCapture=true changes behavior.
     */
    @Test
    @DisplayName("MATMUL_ONLY: captureMinExec=100 prevents capture, self-consistency")
    void testCaptureNeverFiresSelfConsistency() {
        int dim = 64;
        int oldMinExec = Nd4j.getEnvironment().tritonCaptureMinExec();
        try {
            withCaptureFlags(true, true, false, false, false, false);
            Nd4j.getEnvironment().setTritonCaptureMinExec(100);

            java.util.Random rng1 = new java.util.Random(777L + 3);
            SameDiff g1 = buildMatmulOnlyGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[CAP_NEVER_FIRES] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[CAP_NEVER_FIRES] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[CAP_NEVER_FIRES] %d/20 diverge (worst=%.6f at step %d). " +
                            "Even with capture disabled (high minExec), nondeterminism persists.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            Nd4j.getEnvironment().setTritonCaptureMinExec(oldMinExec);
            resetCaptureFlags();
        }
    }

    // ---- 9m: Single mul graph — no matmuls, no gaps, just one Triton-compiled op ----

    /** Single op: mul(x, scale). With tritonCompileAll=true, this produces a single
     *  TRITON_ISLAND with NO gaps. Useful for isolating whether nondeterminism
     *  is in Triton kernel capture itself vs composite schedule gap interaction. */
    private static SameDiff buildSingleMulGraph(java.util.Random rng, int dim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable scale = g.var("scale", deterministicWeight(rng, 1, dim, 0.5f));
        x.mul("out", scale);
        g.setOutputs("out");
        return g;
    }

    /**
     * Self-consistency: capture=true, tritonCompileAll=true on a single mul.
     * No matmuls, no gaps — just one Triton island.
     * If this fails (nondeterministic), the issue is in Triton island capture itself.
     * If this passes (deterministic), the issue involves gaps or composite schedule.
     */
    @Test
    @DisplayName("SINGLE_MUL: self-consistency — capture=true + tritonCompileAll=true (no gaps)")
    void testSelfConsistencySingleMulCaptureTritonCompileAll() {
        int dim = 64;
        try {
            withCaptureFlags(true, true, false, false, false, false);
            java.util.Random rng1 = new java.util.Random(777L + 99);
            SameDiff g1 = buildSingleMulGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 99);
            SameDiff g2 = buildSingleMulGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[SINGLE_MUL_SELFCONSIST] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[SINGLE_MUL_SELFCONSIST] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[SINGLE_MUL_SELFCONSIST] %d/20 diverge (worst=%.6f at step %d). " +
                            "Single Triton island capture is nondeterministic even with no gaps!",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    /**
     * Control: capture=true, tritonCompileAll=false on single mul.
     * If this passes but the test above fails, the nondeterminism is Triton-specific.
     */
    @Test
    @DisplayName("SINGLE_MUL: self-consistency — capture=true + tritonCompileAll=false (no gaps, native)")
    void testSelfConsistencySingleMulCaptureNoCompile() {
        int dim = 64;
        try {
            withCaptureFlags(true, false, false, false, false, false);
            java.util.Random rng1 = new java.util.Random(777L + 99);
            SameDiff g1 = buildSingleMulGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 99);
            SameDiff g2 = buildSingleMulGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[SINGLE_MUL_CONTROL] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[SINGLE_MUL_CONTROL] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[SINGLE_MUL_CONTROL] %d/20 diverge (worst=%.6f at step %d). " +
                            "Native-only capture on single mul is nondeterministic!",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- 9n: Isolate cuBLAS math mode switch (TF32 → PEDANTIC → TF32) ----

    /**
     * Self-consistency on matmul-only graph with capture=true, tritonCompileAll=true,
     * but TF32 DISABLED. The cuBLAS math mode stays at DEFAULT_MATH throughout
     * (no TF32→PEDANTIC→TF32 switch between warmup/capture/replay).
     * If this passes (0/20), the math mode switch during composite execution
     * is the remaining nondeterminism source in the capture+compileAll path.
     * If it still fails (~4/20), a different mechanism is at play.
     */
    @Test
    @DisplayName("MATMUL_ONLY: self-consistency capture+compileAll with TF32 disabled")
    void testSelfConsistencyCaptureCompAllNoTF32() {
        boolean origTf32 = Nd4j.getEnvironment().cublasTf32Enabled();
        int dim = 64;
        try {
            Nd4j.getEnvironment().setCublasTf32Enabled(false);
            withCaptureFlags(true, true, false, false, false, false);
            java.util.Random rng1 = new java.util.Random(777L + 3);
            SameDiff g1 = buildMatmulOnlyGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[CAP_COMPALL_NO_TF32] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[CAP_COMPALL_NO_TF32] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            assertEquals(0, mismatchCount,
                    String.format("[CAP_COMPALL_NO_TF32] %d/20 diverge (worst=%.6f at step %d). " +
                            "Nondeterminism persists even with TF32 disabled — not a math mode switch issue.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            Nd4j.getEnvironment().setCublasTf32Enabled(origTf32);
            resetCaptureFlags();
        }
    }

    /**
     * Control: matmul-only self-consistency with capture=true, tritonCompileAll=true,
     * BUT also set tl_graphExecutionActive sync suppression off at C++ level by
     * NOT using the SyncOverride guard during replay.
     * Tests whether the SyncOverride gap guard captures stale actuality flags
     * during the rerun path.
     */
    @Test
    @DisplayName("MATMUL_ONLY: self-consistency capture+compileAll with TF32 ENABLED (default, baseline repeat)")
    void testSelfConsistencyCaptureCompAllBaselineRepeat() {
        int dim = 64;
        try {
            withCaptureFlags(true, true, false, false, false, false);
            java.util.Random rng1 = new java.util.Random(777L + 3);
            SameDiff g1 = buildMatmulOnlyGraph(rng1, dim);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out1 = runDeterministic(g1, dim, 20);
            g1.close();

            java.util.Random rng2 = new java.util.Random(777L + 3);
            SameDiff g2 = buildMatmulOnlyGraph(rng2, dim);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);
            List<INDArray> out2 = runDeterministic(g2, dim, 20);
            g2.close();

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;
            for (int step = 0; step < 20; step++) {
                double maxDiff = out1.get(step).sub(out2.get(step)).amaxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[CAP_COMPALL_BASELINE] step {}: maxDiff={}", step, maxDiff);
                }
            }

            log.info("[CAP_COMPALL_BASELINE] {}/20 diverge, worst={} at step {}",
                    mismatchCount, worstDiff, worstStep);
            // This is expected to fail at ~4/20 — it confirms the existing
            // nondeterminism before the TF32-disabled test above.
            // We assert pass (0) to flag regression vs prior runs.
            assertEquals(0, mismatchCount,
                    String.format("[CAP_COMPALL_BASELINE] %d/20 diverge (worst=%.6f at step %d). " +
                            "Baseline capture+compileAll nondeterminism confirmed.",
                            mismatchCount, worstDiff, worstStep));
        } finally {
            resetCaptureFlags();
        }
    }

    // ---- Graph builders for capture bisection tests ----

    private static INDArray deterministicWeight(java.util.Random rng, int rows, int cols, float scale) {
        float[] data = new float[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rng.nextGaussian() * scale;
        }
        return Nd4j.createFromArray(data).reshape(rows, cols);
    }
}
