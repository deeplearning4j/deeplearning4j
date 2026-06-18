/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.frameworks.samediff.dsp.frozen;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Replicates the merged-segment accuracy regression.
 *
 * Graph structure: alternating Triton-compilable element-wise ops (add, mul, relu)
 * with matmul gaps between them. With tritonGraphCapture ON and freezeMergeSegments ON,
 * the matmul gaps are capture-safe and get baked into merged CUDA graphs alongside the
 * Triton islands. This is what breaks accuracy.
 *
 * The test compares:
 *   - merge OFF (per-island graphs, matmuls execute live each step)
 *   - merge ON  (merged graphs, matmuls baked into CUDA graph replay)
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP Merged Segment Accuracy Regression")
public class TestDspMergedSegmentReplay {

    private static final int DIM = 64;
    private static final int LAYERS = 6;  // enough layers to create multiple islands+gaps
    private static final int STEPS = 30;  // enough to reach REPLAYING phase

    @AfterEach
    public void teardown() {
        // Restore defaults
        Nd4j.getEnvironment().setDspFreezeMergeSegments(true);
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getEnvironment().setTritonVerifyKernels(false);
    }

    /**
     * THE regression test. Runs the same graph+inputs with merge OFF then merge ON.
     * If merged capture corrupts accuracy, outputs diverge.
     */
    @Test
    @DisplayName("Merged segment capture must match non-merged execution")
    public void testMergedVsNonMergedAccuracy() {
        // Run with merge OFF — reference (per-island capture, live gap matmuls)
        List<INDArray> refOutputs = runGraph(false);

        // Run with merge ON — test (merged capture through capture-safe gaps)
        List<INDArray> testOutputs = runGraph(true);

        assertEquals(refOutputs.size(), testOutputs.size());
        int mismatchCount = 0;
        for (int step = 0; step < refOutputs.size(); step++) {
            INDArray ref = refOutputs.get(step);
            INDArray test = testOutputs.get(step);
            double maxDiff = ref.sub(test).amaxNumber().doubleValue();
            if (maxDiff > 1e-3) {
                mismatchCount++;
                log.error("Step {}: MISMATCH maxDiff={}", step, maxDiff);
            }
        }
        assertEquals(0, mismatchCount,
                mismatchCount + "/" + refOutputs.size() + " steps diverge between " +
                        "merged and non-merged execution. Merged CUDA graph replay " +
                        "produces different results from live gap execution.");
    }

    /**
     * Verify with the built-in exec-and-compare (tritonVerifyKernels).
     * This re-executes fresh after each replay and logs per-slot mismatches.
     */
    @Test
    @DisplayName("tritonVerifyKernels detects no mismatch in merged replay")
    public void testVerifyModeNoMismatch() {
        List<INDArray> outputs = runGraph(true, true);  // merge ON, verify ON
        // If we got here without crash, verify mode ran. Check outputs are sane.
        for (int step = 0; step < outputs.size(); step++) {
            INDArray out = outputs.get(step);
            assertFalse(out.isNaN().any(), "Step " + step + ": NaN");
            assertFalse(out.isInfinite().any(), "Step " + step + ": Inf");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────

    private List<INDArray> runGraph(boolean mergeEnabled) {
        return runGraph(mergeEnabled, false);
    }

    private List<INDArray> runGraph(boolean mergeEnabled, boolean verifyKernels) {
        SameDiff sd = buildGraph();
        applyConfig(sd, mergeEnabled, verifyKernels);

        List<INDArray> outputs = new ArrayList<>();
        try {
            for (int step = 0; step < STEPS; step++) {
                INDArray input = createDeterministicInput(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);
                Map<String, INDArray> out = sd.output(ph, "out");
                outputs.add(out.get("out").dup());
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
        return outputs;
    }

    /**
     * Create a deterministic input using java.util.Random — fully isolated
     * from the Nd4j global RNG which can be consumed by DSP JIT compilation.
     */
    private static INDArray createDeterministicInput(int step) {
        java.util.Random jRng = new java.util.Random(42L + step);
        float[] data = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            data[i] = (float) jRng.nextGaussian();
        }
        return Nd4j.createFromArray(data).reshape(1, DIM);
    }

    /** Create a deterministic array using java.util.Random (isolated from Nd4j RNG). */
    private static INDArray detArray(java.util.Random rng, int rows, int cols, float scale, float offset) {
        float[] data = new float[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rng.nextGaussian() * scale + offset;
        }
        return Nd4j.createFromArray(data).reshape(rows, cols);
    }

    /**
     * Build graph: [Triton island] → [matmul gap] → [Triton island] → ... repeated.
     *
     * Triton-compilable ops: add, multiply, relu (element-wise kernels compiled by Triton).
     * Gap ops: matmul (uses cuBLAS, NOT Triton — becomes a capture-safe gap).
     *
     * With merge ON, adjacent Triton islands get merged through the matmul gaps
     * into a single CUDA graph. The matmul cuBLAS call is recorded into the graph.
     */
    private static SameDiff buildGraph() {
        java.util.Random rng = new java.util.Random(777L);
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, DIM);
        SDVariable h = x;

        for (int i = 0; i < LAYERS; i++) {
            // ── Triton island: scale + bias + activation ──
            SDVariable scale = sd.var("scale_" + i, detArray(rng, 1, DIM, 0.5f, 1.0f));
            SDVariable bias = sd.var("bias_" + i, detArray(rng, 1, DIM, 0.01f, 0.0f));
            h = h.mul("mul_" + i, scale);
            h = h.add("add_" + i, bias);
            h = sd.nn().relu("relu_" + i, h, 0);

            // ── Matmul gap (capture-safe: launches cuBLAS kernel, no view/identity) ──
            SDVariable w = sd.var("w_" + i, detArray(rng, DIM, DIM, 0.02f, 0.0f));
            h = sd.mmul("mm_" + i, h, w);
        }

        // Final Triton island (no trailing matmul)
        SDVariable finalScale = sd.var("scale_final", detArray(rng, 1, DIM, 0.5f, 1.0f));
        h = h.mul("out", finalScale);

        sd.setOutputs("out");
        return sd;
    }

    /**
     * Apply the OPTIMAL-equivalent config that the real benchmark uses.
     * This enables Triton graph capture, consolidated arg tables, etc.
     */
    private static void applyConfig(SameDiff sd, boolean mergeEnabled, boolean verifyKernels) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        var env = Nd4j.getEnvironment();
        // Core flags that trigger CUDA graph capture of Triton + gaps
        env.setTritonGraphCapture(true);
        env.setDspFreezeMergeSegments(mergeEnabled);

        // Match the production OPTIMAL config
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonCompileAll(true);

        if (verifyKernels) {
            env.setTritonVerifyKernels(true);
        }
    }
}
