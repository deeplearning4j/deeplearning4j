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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone isolation tests for DSP/Triton configuration knobs that lack
 * dedicated test coverage.
 *
 * Each test builds a small SameDiff graph, runs it with standard execution
 * for a baseline, then enables a single knob, compiles DSP, and verifies
 * output matches the baseline.
 *
 * Knobs tested:
 * 1. Batch-zero (dspBatchZero / dspBatchZeroKernel)
 * 2. Cast sink matmul (dspCastSinkMatmul)
 * 3. Consolidated arg table (tritonConsolidatedArgTable) — requires Triton
 * 4. Arg dirty tracking (tritonArgDirtyTracking) — requires Triton
 * 5. Section fusion (tritonSectionFusion) — requires Triton
 */
public class DspKnobIsolationTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(DspKnobIsolationTest.class);
    private static final double TOL = 1e-3;

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void resetEnvironment() {
        Environment env = Nd4j.getEnvironment();
        env.setDspBatchZero(false);
        env.setDspBatchZeroKernel(false);
        env.setDspBatchedGemm(false);
        env.setDspCastSinkMatmul(false);
        env.setDspCastElimination(false);
        env.setTritonGraphCapture(false);
        env.setTritonSectionFusion(false);
        env.setTritonConsolidatedArgTable(false);
        env.setTritonArgDirtyTracking(false);
        env.setTritonCompileAll(false);
        env.setTritonIncludeTypes("");
        env.setTritonAllowFallbackCapture(false);
    }

    // ─── Helper: build a multi-op graph with matmuls and elementwise ops ─────

    /**
     * Builds a 2-layer MLP graph: x → (matmul + relu + matmul + add) × 2 → out
     * This exercises matmul zeroing, elementwise fusion, and arg table management.
     */
    private static SameDiff buildMlpGraph() {
        SameDiff sd = SameDiff.create();
        int hidden = 64;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, hidden);

        // Layer 1: up-project + relu + down-project + residual
        SDVariable w1u = sd.constant("w1u", Nd4j.randn(DataType.FLOAT, hidden, hidden * 2).mul(0.02));
        SDVariable w1d = sd.constant("w1d", Nd4j.randn(DataType.FLOAT, hidden * 2, hidden).mul(0.02));
        SDVariable l1up = sd.mmul("l1_up", x, w1u);
        SDVariable l1act = sd.nn.relu("l1_act", l1up, 0);
        SDVariable l1down = sd.mmul("l1_down", l1act, w1d);
        SDVariable l1out = x.add("l1_res", l1down);

        // Layer 2: same structure
        SDVariable w2u = sd.constant("w2u", Nd4j.randn(DataType.FLOAT, hidden, hidden * 2).mul(0.02));
        SDVariable w2d = sd.constant("w2d", Nd4j.randn(DataType.FLOAT, hidden * 2, hidden).mul(0.02));
        SDVariable l2up = sd.mmul("l2_up", l1out, w2u);
        SDVariable l2act = sd.nn.relu("l2_act", l2up, 0);
        SDVariable l2down = sd.mmul("l2_down", l2act, w2d);
        SDVariable out = l1out.add("out", l2down);

        return sd;
    }

    /**
     * Builds a graph with cast + matmul pattern for cast-sink testing.
     * x(FLOAT) → matmul(w1) → cast(FLOAT16) → cast(FLOAT) → matmul(w2) → out
     */
    private static SameDiff buildCastMatmulGraph() {
        SameDiff sd = SameDiff.create();
        int dim = 64;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, dim, dim).mul(0.02));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, dim, dim).mul(0.02));

        // matmul → cast to FP16 → cast back to FP32 → matmul
        SDVariable proj1 = sd.mmul("proj1", x, w1);
        SDVariable act1 = sd.nn.relu("act1", proj1, 0);
        SDVariable castDown = sd.castTo("cast_fp16", act1, DataType.HALF);
        SDVariable castUp = sd.castTo("cast_fp32", castDown, DataType.FLOAT);
        SDVariable proj2 = sd.mmul("proj2", castUp, w2);
        SDVariable out = proj2.add("out", x);  // residual

        return sd;
    }

    /**
     * Builds a graph with many elementwise ops for section fusion testing.
     * x → mul(scale) → add(bias) → relu → mul(scale2) → add(bias2) → sigmoid → out
     */
    private static SameDiff buildElementwiseChainGraph() {
        SameDiff sd = SameDiff.create();
        int dim = 128;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable scale1 = sd.constant("scale1", Nd4j.randn(DataType.FLOAT, 1, dim).mul(0.1).add(1.0));
        SDVariable bias1 = sd.constant("bias1", Nd4j.randn(DataType.FLOAT, 1, dim).mul(0.01));
        SDVariable scale2 = sd.constant("scale2", Nd4j.randn(DataType.FLOAT, 1, dim).mul(0.1).add(1.0));
        SDVariable bias2 = sd.constant("bias2", Nd4j.randn(DataType.FLOAT, 1, dim).mul(0.01));

        // Chain of elementwise ops — fuseable by section fusion
        SDVariable s1 = x.mul("scaled1", scale1);
        SDVariable b1 = s1.add("biased1", bias1);
        SDVariable a1 = sd.nn.relu("relu1", b1, 0);
        SDVariable s2 = a1.mul("scaled2", scale2);
        SDVariable b2 = s2.add("biased2", bias2);
        SDVariable out = sd.nn.sigmoid("out", b2);

        return sd;
    }

    // ─── Helper: compare standard vs DSP execution ──────────────────────────

    private void compareStandardVsDsp(SameDiff sd, String outputName, INDArray input, int numRuns, String knobName) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Environment env = Nd4j.getEnvironment();
        boolean isTriton = !env.tritonIncludeTypes().isEmpty();

        // Collect reference results using standard execution (clean session, no DSP).
        // Do this FIRST before any DSP session exists to avoid sd.output() accidentally
        // running through a DSP plan and corrupting executionCount state.
        INDArray[] refInputs = new INDArray[numRuns + 1];
        INDArray[] refOutputs = new INDArray[numRuns + 1];

        // Reference for warmup (same input)
        refInputs[0] = input;
        sd.resetSession();
        Map<String, INDArray> refResult0 = sd.output(Map.of("x", input), outputName);
        refOutputs[0] = refResult0.get(outputName).dup();
        log.info("[{}] Standard result shape: {} first values: [{}, {}, {}, {}]",
                knobName, Arrays.toString(refOutputs[0].shape()),
                refOutputs[0].getDouble(0), refOutputs[0].getDouble(1),
                refOutputs[0].getDouble(2), refOutputs[0].getDouble(3));

        // References for each run (different random inputs)
        for (int run = 0; run < numRuns; run++) {
            refInputs[run + 1] = Nd4j.randn(DataType.FLOAT, input.shape()).mul(0.5);
            sd.resetSession();
            Map<String, INDArray> refResult = sd.output(Map.of("x", refInputs[run + 1]), outputName);
            refOutputs[run + 1] = refResult.get(outputName).dup();
        }

        // Now set up DSP session
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        if (nativeOps.isTritonAvailable()) {
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();
        }

        GraphExecutionMode mode;
        if (isTriton) {
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);
            mode = sd.compileNativeDynamicShapePlan(
                    List.of(outputName), DspCompilationMode.MAX_AUTOTUNE);
            log.info("[{}] Compiled Triton plan, mode: {}", knobName, mode);
        } else {
            mode = sd.compileNativeDynamicShapePlan(
                    List.of(outputName), GraphExecutionMode.SLOT_BY_SLOT, true);
            log.info("[{}] Compiled DSP plan, mode: {}", knobName, mode);
        }

        // Warmup run (executionCount: 0 → 1, slot-by-slot)
        Map<String, INDArray> warmupResult = sd.outputDirect(Map.of("x", input), outputName);
        INDArray warmupActual = warmupResult.get(outputName).dup();
        double warmupDiff = refOutputs[0].sub(warmupActual).amaxNumber().doubleValue();
        log.info("[{}] Warmup: max diff = {}", knobName, warmupDiff);
        assertTrue(warmupDiff < TOL, knobName + " warmup: max diff " + warmupDiff + " exceeds tolerance " + TOL);

        // Freeze shapes to enable graph capture and other shape-dependent optimizations
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            log.info("[{}] Shapes frozen", knobName);
        }

        // Multiple runs on the SAME plan to exercise:
        // - Run 0: executionCount=1 → Triton compile + execute (or 2nd slot-by-slot for non-Triton)
        // - Run 1+: executionCount=2+ → compiled kernel replay / graph replay
        // Using different inputs each run to verify we're not reading cached outputs.
        for (int run = 0; run < numRuns; run++) {
            INDArray runInput = refInputs[run + 1];
            INDArray ref = refOutputs[run + 1];

            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", runInput), outputName);
            INDArray actual = dspResult.get(outputName).dup();

            double maxDiff = ref.sub(actual).amaxNumber().doubleValue();
            log.info("[{}] Run {}: max diff = {}", knobName, run, maxDiff);
            assertTrue(maxDiff < TOL,
                    knobName + " run " + run + ": max diff " + maxDiff + " exceeds tolerance " + TOL);
        }
    }

    // ─── Test 1: Batch-zero ─────────────────────────────────────────────────

    @Test
    @DisplayName("Batch-zero: batched memset-zero produces correct output")
    public void testBatchZero() {
        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Nd4j.getEnvironment().setDspBatchZero(true);
        Nd4j.getEnvironment().setDspBatchZeroKernel(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "batchZero");
        } finally {
            sd.close();
        }
    }

    // ─── Test 2: Cast sink matmul ───────────────────────────────────────────

    @Test
    @DisplayName("Cast sink matmul: sinking casts past matmul produces correct output")
    public void testCastSinkMatmul() {
        SameDiff sd = buildCastMatmulGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Nd4j.getEnvironment().setDspCastSinkMatmul(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "castSinkMatmul");
        } finally {
            sd.close();
        }
    }

    // ─── Test 3: Consolidated arg table (Triton) ────────────────────────────

    @Test
    @DisplayName("Consolidated arg table: single arg table produces correct output")
    public void testConsolidatedArgTable() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildElementwiseChainGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 128);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonConsolidatedArgTable(true);
        // Explicitly disable dirty tracking to isolate consolidated arg table
        env.setTritonArgDirtyTracking(false);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "consolidatedArgTable");
        } finally {
            sd.close();
        }
    }

    // ─── Test 4: Arg dirty tracking (Triton) ────────────────────────────────

    @Test
    @DisplayName("Arg dirty tracking: dirty-tracking arg refresh produces correct output")
    public void testArgDirtyTracking() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildElementwiseChainGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 128);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "argDirtyTracking");
        } finally {
            sd.close();
        }
    }

    // ─── Test 5: Section fusion (Triton) ────────────────────────────────────

    @Test
    @DisplayName("Section fusion: fused elementwise sections produce correct output")
    public void testSectionFusion() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildElementwiseChainGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 128);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        // Explicitly disable compileAll and graph capture to isolate section fusion
        env.setTritonCompileAll(false);
        env.setTritonGraphCapture(false);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "sectionFusion");
        } finally {
            sd.close();
        }
    }

    // ─── Test 6: All knobs combined (matches VLM optimal config) ────────────

    @Test
    @DisplayName("All knobs combined: full optimal config produces correct output")
    public void testAllKnobsCombined() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
        env.setDspBatchZero(true);
        env.setDspBatchZeroKernel(true);
        env.setDspBatchedGemm(true);
        env.setDspCastSinkMatmul(true);

        try {
            compareStandardVsDsp(sd, "out", input, 5, "allKnobsCombined");
        } finally {
            sd.close();
        }
    }

    // ─── Bisection tests: find the minimal breaking combination ─────────────

    @Test
    @DisplayName("Bisect: Triton + graphCapture only (no DSP batch knobs)")
    public void testBisect_TritonGraphCapture() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "bisect_TritonGC");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("Bisect: Triton + graphCapture + argTable + dirtyTracking")
    public void testBisect_TritonGC_ArgOpt() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "bisect_TritonGC_ArgOpt");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("Bisect: Triton + graphCapture + batchZero (no argOpt)")
    public void testBisect_TritonGC_BatchZero() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setDspBatchZero(true);
        env.setDspBatchZeroKernel(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "bisect_TritonGC_BatchZero");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("Bisect: Triton + graphCapture + batchedGemm (no argOpt, no batchZero)")
    public void testBisect_TritonGC_BatchedGemm() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setDspBatchedGemm(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "bisect_TritonGC_BatchedGemm");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("Bisect: Triton + graphCapture + argOpt + batchZero (no batchedGemm)")
    public void testBisect_TritonGC_ArgOpt_BatchZero() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
        env.setDspBatchZero(true);
        env.setDspBatchZeroKernel(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "bisect_TritonGC_ArgOpt_BatchZero");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("Bisect: Triton + graphCapture + argOpt + batchedGemm (no batchZero)")
    public void testBisect_TritonGC_ArgOpt_BatchedGemm() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
        env.setDspBatchedGemm(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "bisect_TritonGC_ArgOpt_BatchedGemm");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("Bisect: Triton + graphCapture + argOpt + batchZero + batchedGemm (no castSink)")
    public void testBisect_TritonGC_ArgOpt_BatchOps() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), "Triton unavailable — skipping");

        SameDiff sd = buildMlpGraph();
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
        env.setDspBatchZero(true);
        env.setDspBatchZeroKernel(true);
        env.setDspBatchedGemm(true);

        try {
            compareStandardVsDsp(sd, "out", input, 3, "bisect_TritonGC_ArgOpt_BatchOps");
        } finally {
            sd.close();
        }
    }
}
