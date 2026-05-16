package org.eclipse.deeplearning4j.frameworks.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that view ops (reshape_no_copy, expand_dims, strided_slice) produce
 * correct results during DSP graph capture+replay with
 * tritonMergedCaptureThroughViews=true.
 *
 * View ops share their DataBuffer with the input. During merged capture, their
 * buffer addresses are baked into the CUDA graph. If the parent's DataBuffer
 * address changes between steps (reallocation), the captured view op reads
 * from stale addresses → wrong output.
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestDspViewCaptureCorrectness {

    private static final float TOL = 2.0f;
    private static final int STEPS = 20;
    private static final long SEED = 42;

    @AfterEach
    void cleanup() {
        Nd4j.getEnvironment().setTritonMergedCaptureThroughViews(true);
    }

    private static INDArray detWeight(java.util.Random rng, int... shape) {
        int size = shape[0];
        for (int i = 1; i < shape.length; i++) size *= shape[i];
        float[] data = new float[size];
        for (int i = 0; i < size; i++) data[i] = (float)rng.nextGaussian() * 0.1f;
        return Nd4j.createFromArray(data).reshape(shape);
    }

    private SameDiff buildGraph(java.util.Random rng, int dim, boolean hasReshape, boolean hasExpand) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, -1, dim);
        SDVariable w1 = g.var("w1", detWeight(rng, dim, dim));
        SDVariable h = g.mmul("mm", x, w1);
        if (hasReshape) {
            h = g.reshape("rs", h, 1, dim);
        }
        if (hasExpand) {
            h = g.expandDims("exp", h, 0);
        }
        SDVariable w2 = g.var("w2", detWeight(rng, dim, 8));
        g.mmul("out", h, w2);
        g.setOutputs("out");
        return g;
    }

    // ── Test: view ops with mergedCaptureThroughViews ─────────────────
    // Uses tritonCompileAll=false to isolate view address issue from
    // Triton FP differences.

    @Test
    @DisplayName("no-view graph: self-consistency for capture variants")
    void testNoViewSelfConsistency() {
        runViewSuite("NO_VIEW", false, false);
    }

    @Test
    @DisplayName("reshape view: self-consistency for capture variants")
    void testReshapeViewSelfConsistency() {
        runViewSuite("RESHAPE_VIEW", true, false);
    }

    @Test
    @DisplayName("expand_dims view: self-consistency for capture variants")
    void testExpandViewSelfConsistency() {
        runViewSuite("EXPAND_VIEW", false, true);
    }

    @Test
    @DisplayName("reshape+expand_dims: self-consistency for capture variants")
    void testReshapeExpandViewSelfConsistency() {
        runViewSuite("BOTH_VIEWS", true, true);
    }

    private void runViewSuite(String tag, boolean hasReshape, boolean hasExpand) {
        int dim = (hasReshape || hasExpand) ? 64 : 64;

        // Test 1: capture=true, no triton compilation, no merged views
        runViewSubtest(tag + "_nativeCap", dim, hasReshape, hasExpand,
                false, true, false, 0);

        // Test 2: capture=true + compileAll=true + mergedViews=false
        runViewSubtest(tag + "_tritonCap_noMerge", dim, hasReshape, hasExpand,
                true, true, false, 0);

        // Test 3: capture=true + compileAll=true + mergedViews=true (OPTIMAL)
        runViewSubtest(tag + "_tritonCap_mergedViews", dim, hasReshape, hasExpand,
                true, true, true, 0);
    }

    private void runViewSubtest(String label, int dim, boolean hasReshape, boolean hasExpand,
                                 boolean compileAll, boolean graphCapture,
                                 boolean mergedViews, int expectedMismatch) {
        try {
            Nd4j.getEnvironment().setTritonCompileAll(compileAll);
            Nd4j.getEnvironment().setTritonGraphCapture(graphCapture);
            Nd4j.getEnvironment().setTritonMergedCaptureThroughViews(mergedViews);
            Nd4j.getEnvironment().setDspFreezeMergeSegments(true);
            Nd4j.getEnvironment().setCublasCaptureWorkspace(true);

            // Run TWO identical graphs — same seed, same weights
            java.util.Random rng1 = new java.util.Random(SEED);
            SameDiff g1 = buildGraph(rng1, dim, hasReshape, hasExpand);
            g1.setDspAutoCompileEnabled(true);
            g1.setDspNativeAutoCompileEnabled(true);
            g1.setGraphExecutionMode(GraphExecutionMode.AUTO);

            java.util.Random rng2 = new java.util.Random(SEED);
            SameDiff g2 = buildGraph(rng2, dim, hasReshape, hasExpand);
            g2.setDspAutoCompileEnabled(true);
            g2.setDspNativeAutoCompileEnabled(true);
            g2.setGraphExecutionMode(GraphExecutionMode.AUTO);

            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;

            for (int step = 0; step < STEPS; step++) {
                INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);

                INDArray out1 = g1.output(ph, "out").get("out");
                INDArray out2 = g2.output(ph, "out").get("out");

                double maxDiff = Transforms.abs(out1.sub(out2)).maxNumber().doubleValue();
                if (maxDiff > 1e-3) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("[{}] step {}: maxDiff={}", label, step, maxDiff);
                }
            }

            g1.close();
            g2.close();

            String status = (mismatchCount <= expectedMismatch) ? "PASS" : "FAIL";
            log.info("[{}] {} — {}/{} diverge, worst={} at step {} (expected ≤{})",
                    label, status, mismatchCount, STEPS, worstDiff, worstStep, expectedMismatch);

            assertTrue(mismatchCount <= expectedMismatch,
                    String.format("[%s] %d/%d diverge (worst=%.6f). Expected ≤%d.",
                            label, mismatchCount, STEPS, worstDiff, expectedMismatch));

        } catch (Exception e) {
            log.error("[{}] Exception: {}", label, e.getMessage());
            if (expectedMismatch == 0) fail(label + " Unexpected exception: " + e.getMessage());
        } finally {
            Nd4j.getEnvironment().setTritonCompileAll(true);
            Nd4j.getEnvironment().setTritonGraphCapture(true);
            Nd4j.getEnvironment().setTritonMergedCaptureThroughViews(true);
            Nd4j.getEnvironment().setTritonSectionFusion(true);
        }
    }
}
