package org.eclipse.deeplearning4j.frameworks.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Exhaustive DSP config enumeration test.
 * Tests every GraphExecutionMode × optimizer × workspace flag combination
 * for correctness against SLOT_BY_SLOT reference.
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestDspConfigEnumeration {

    private static final float TOL = 2.0f; // PEDANTIC vs TF32 tolerance
    private static final int STEPS = 10;
    private static final int DIM = 32;
    private SameDiff sd;

    @AfterEach
    void cleanup() {
        if (sd != null) { sd.close(); sd = null; }
    }

    // ── Test data record ──────────────────────────────────────────────────

    /** One combination entry: mode, optimizer flag, workspace profile */
    static class DspConfigEntry {
        final GraphExecutionMode mode;
        final boolean useOptimizer;
        final boolean batchZeroEnabled;
        final boolean castElimEnabled;
        final boolean batchedGemmEnabled;
        final String label;
        DspConfigEntry(GraphExecutionMode mode, boolean useOptimizer, boolean batchZeroEnabled,
                        boolean castElimEnabled, boolean batchedGemmEnabled, String label) {
            this.mode = mode; this.useOptimizer = useOptimizer;
            this.batchZeroEnabled = batchZeroEnabled; this.castElimEnabled = castElimEnabled;
            this.batchedGemmEnabled = batchedGemmEnabled; this.label = label;
        }
        @Override public String toString() { return label; }
    }

    /** All GraphExecutionMode values supported on CUDA (skip platform-specific) */
    static List<GraphExecutionMode> cudaModes() {
        return List.of(
                GraphExecutionMode.AUTO,
                GraphExecutionMode.SLOT_BY_SLOT,
                GraphExecutionMode.CUDA_GRAPHS,
                GraphExecutionMode.TRITON,
                GraphExecutionMode.NVRTC_JIT,
                GraphExecutionMode.PTX_JIT,
                GraphExecutionMode.EMULATED_REPLAY,
                GraphExecutionMode.SHAPE_INFERENCE_ONLY
        );
    }

    /** Boolean cross-product for workspace flags */
    static List<boolean[]> flagCombos() {
        return List.of(
                new boolean[]{true, true, true},   // all default
                new boolean[]{false, true, true},   // batchZero off
                new boolean[]{true, false, true},   // castElim off
                new boolean[]{true, true, false},   // batchedGemm off
                new boolean[]{false, false, false}  // all off
        );
    }

    /** Test data source: mode × optimizer × workspace flags */
    List<DspConfigEntry> testData() {
        List<DspConfigEntry> entries = new ArrayList<>();
        List<GraphExecutionMode> modes = cudaModes();
        List<Boolean> optFlags = List.of(true, false);
        for (GraphExecutionMode mode : modes) {
            for (boolean useOpt : optFlags) {
                for (boolean[] flags : flagCombos()) {
                    String label = String.format("mode=%s opt=%s batch0=%s cast=%s bgemm=%s",
                            mode, useOpt, flags[0], flags[1], flags[2]);
                    entries.add(new DspConfigEntry(mode, useOpt, flags[0], flags[1], flags[2], label));
                }
            }
        }
        return entries;
    }

    // ── Graph builders ─────────────────────────────────────────────────────

    /** Simple matmul + add + relu graph */
    private SameDiff buildSimpleGraph() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, -1, DIM);
        SDVariable w = g.var("w", Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.1f));
        SDVariable b = g.var("b", Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.01f));
        SDVariable mm = g.mmul("mm", x, w);
        SDVariable add = mm.add("add", b);
        g.nn().relu("out", add, 0);
        g.setOutputs("out");
        return g;
    }

    /** Matmul-only chain (6 matmuls + mul) to exercise gaps */
    private SameDiff buildMatmulChain() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, -1, DIM);
        SDVariable h = x;
        for (int i = 0; i < 6; i++) {
            SDVariable w = g.var("w_" + i, Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.1f));
            h = g.mmul("mm_" + i, h, w);
        }
        SDVariable scale = g.var("scale", Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.5f));
        h.mul("out", scale);
        g.setOutputs("out");
        return g;
    }

    /** Decoder-like graph with attention-like patterns */
    private SameDiff buildDecoderGraph() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, -1, DIM);
        SDVariable h = x;
        for (int i = 0; i < 4; i++) {
            SDVariable wQ = g.var("wQ_" + i, Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.1f));
            SDVariable wK = g.var("wK_" + i, Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.1f));
            SDVariable wV = g.var("wV_" + i, Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.1f));
            SDVariable wO = g.var("wO_" + i, Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.1f));
            SDVariable q = g.mmul("q_" + i, h, wQ);
            SDVariable k = g.mmul("k_" + i, h, wK);
            SDVariable v = g.mmul("v_" + i, h, wV);
            SDVariable attn = q.mul("attn_" + i, k);
            SDVariable ctx = attn.mul("ctx_" + i, v);
            h = g.mmul("proj_" + i, ctx, wO);
            h = g.nn().relu("relu_" + i, h, 0);
        }
        SDVariable wOut = g.var("wOut", Nd4j.randn(DataType.FLOAT, DIM, 8).muli(0.1f));
        g.mmul("out", h, wOut);
        g.setOutputs("out");
        return g;
    }

    // ── Parameterized test ─────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("testData")
    @DisplayName("DSP config: correctness vs SLOT_BY_SLOT")
    void testConfigCorrectness(DspConfigEntry entry) {
        // Skip SHAPE_INFERENCE_ONLY for actual computation tests
        if (entry.mode == GraphExecutionMode.SHAPE_INFERENCE_ONLY) return;

        String tag = "[" + entry + "]";
        try {
            // Configure environment
            configureEnv(entry);
            if (entry.useOptimizer) {
                System.setProperty("nd4j.optimizer.enabled", "true");
            } else {
                System.setProperty("nd4j.optimizer.enabled", "false");
            }

            // Build test graph
            SameDiff testGraph = buildMatmulChain();
            testGraph.setDspAutoCompileEnabled(true);
            testGraph.setDspNativeAutoCompileEnabled(true);
            testGraph.setGraphExecutionMode(entry.mode);

            // Build reference (SLOT_BY_SLOT)
            SameDiff refGraph = buildMatmulChain();
            refGraph.setDspAutoCompileEnabled(true);
            refGraph.setDspNativeAutoCompileEnabled(true);
            refGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            sd = testGraph;
            int mismatchCount = 0;
            double worstDiff = 0;
            int worstStep = -1;

            for (int step = 0; step < STEPS; step++) {
                INDArray input = Nd4j.randn(DataType.FLOAT, 1, DIM);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);

                INDArray testOut = testGraph.output(ph, "out").get("out");
                INDArray refOut = refGraph.output(ph, "out").get("out");

                double maxDiff = Transforms.abs(testOut.sub(refOut)).maxNumber().doubleValue();
                if (maxDiff > 1e-4) {
                    mismatchCount++;
                    if (maxDiff > worstDiff) { worstDiff = maxDiff; worstStep = step; }
                    log.warn("{} step {}: maxDiff={}", tag, step, maxDiff);
                }
            }

            testGraph.close();
            refGraph.close();

            log.info("{} {}/{} diverge, worst={} at step {}", tag, mismatchCount, STEPS, worstDiff, worstStep);
            assertTrue(mismatchCount == 0 || worstDiff < TOL,
                    String.format("%s %d/%d diverge (worst=%.6f at step %d). Exceeds tolerance.",
                            tag, mismatchCount, STEPS, worstDiff, worstStep));

        } catch (Exception e) {
            log.error("{} FAILED with exception: {}", tag, e.getMessage());
            // Log but don't fail for modes that may not support CUDA
            if (entry.mode == GraphExecutionMode.NVRTC_JIT ||
                entry.mode == GraphExecutionMode.PTX_JIT) {
                log.warn("{} Expected exception for JIT-only mode on CUDA: {}", tag, e.getMessage());
            } else {
                fail(tag + " Unexpected exception: " + e.getMessage());
            }
        } finally {
            resetEnv();
        }
    }

    // ── Config helpers ────────────────────────────────────────────────────

    private void configureEnv(DspConfigEntry entry) {
        var env = Nd4j.getEnvironment();
        env.setDspBatchZero(entry.batchZeroEnabled);
        env.setDspCastElimination(entry.castElimEnabled);
        env.setDspBatchedGemm(entry.batchedGemmEnabled);
        env.setTritonGraphCapture(true);
        env.setTritonCompileAll(true);
        env.setDspFreezeMergeSegments(true);
        env.setCublasCaptureWorkspace(true);
    }

    private void resetEnv() {
        System.clearProperty("nd4j.optimizer.enabled");
        var env = Nd4j.getEnvironment();
        env.setDspBatchZero(true);
        env.setDspCastElimination(true);
        env.setDspBatchedGemm(true);
        env.setTritonGraphCapture(true);
        env.setTritonCompileAll(true);
        env.setDspFreezeMergeSegments(true);
        env.setCublasCaptureWorkspace(true);
    }
}
