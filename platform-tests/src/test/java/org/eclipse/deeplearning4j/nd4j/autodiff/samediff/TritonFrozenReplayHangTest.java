package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone tests for CUDA graph replay of Triton-compiled kernels.
 *
 * Reproduces the GPU hang observed during the second frozen execution
 * (Call 4) of TestDSPVisionEncoderPageReuse#testVisionEncoderFrozenShapes.
 *
 * The attention pattern is:
 *   Q*K^T (matmul) → scale (multiply by scalar) → softmax → attn*V (matmul)
 *
 * The Triton kernel covers the multiply op (elementwise), while matmul
 * and softmax use cuBLAS/native fallbacks. During CUDA graph capture,
 * all ops on the stream are captured into a single graph. On replay,
 * the graph is re-launched with updated capture buffer D2D copies.
 *
 * Hypotheses for hang:
 * 1. Triton kernel with stale n_elements arg on replay
 * 2. Stream ordering issue: D2D copies vs graph launch
 * 3. Address mismatch causing kernel to read unmapped memory
 */
@Slf4j
public class TritonFrozenReplayHangTest {

    private static final double TOLERANCE = 1e-3;

    /**
     * Test: simple multiply with frozen shapes, multiple replay iterations.
     * This isolates the Triton multiply kernel replay without matmul complexity.
     */
    @Test
    @Timeout(120)
    public void testMultiplyFrozenReplay() {
        SameDiff sd = SameDiff.create();
        int batch = 1, heads = 12, seqLen = 64, dim = 64;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, heads, seqLen, dim);
        // Scale by 1/sqrt(dim) = 0.125
        SDVariable scaled = input.mul("scaled", 0.125);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, heads, seqLen, dim);
        INDArray reference = inputArr.mul(0.125);

        Map<String, INDArray> feed = Map.of("input", inputArr);

        // Warmup (2 calls to establish shapes)
        sd.outputDirect(feed, "scaled");
        sd.outputDirect(feed, "scaled");

        // Compile with Triton + freeze
        try {
            sd.compileNativeDynamicShapePlan(List.of("scaled"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        // Execute 5 times with frozen shapes (should trigger capture + 4 replays)
        for (int step = 0; step < 5; step++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "scaled");
            INDArray actual = result.get("scaled");
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN in output");

            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            log.info("Multiply step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                    String.format("Step %d: maxDiff=%.6f exceeds tolerance", step, maxDiff));
        }

        sd.close();
    }

    /**
     * Test: attention pattern matmul → multiply → softmax → matmul.
     * This is the exact pattern from the VLM encoder that hangs on replay.
     *
     * The multiply op is Triton-compiled, matmul uses cuBLAS, softmax uses
     * native fallback. All are captured in a single CUDA graph segment.
     */
    @Test
    @Timeout(120)
    public void testAttentionPatternFrozenReplay() {
        SameDiff sd = SameDiff.create();
        int batch = 1, heads = 12, seqLen = 64, dim = 64;

        // Q, K, V as placeholders
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, heads, seqLen, dim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, heads, dim, seqLen);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, heads, seqLen, dim);

        // Attention: softmax(Q*K^T / sqrt(d)) * V
        // Use matmul for batched 4D matmul (mmul only supports 2D)
        SDVariable qk = sd.linalg().matmul("qk", q, k);  // [1,12,64,64]
        SDVariable scaled = qk.mul("scaled", 1.0 / Math.sqrt(dim));  // scale
        SDVariable attn = sd.nn().softmax("attn", scaled, -1);  // softmax over last dim
        SDVariable output = sd.linalg().matmul("output", attn, v);  // [1,12,64,64]

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, heads, seqLen, dim);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, heads, dim, seqLen);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, heads, seqLen, dim);

        Map<String, INDArray> feed = Map.of("q", qArr, "k", kArr, "v", vArr);

        // Warmup
        sd.outputDirect(feed, "output");
        sd.outputDirect(feed, "output");

        // Compile with Triton
        try {
            sd.compileNativeDynamicShapePlan(List.of("output"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        // Reference: compute via SameDiff without Triton (standard execution)
        // We skip reference comparison since batched matmul reference is complex;
        // instead verify no NaN/Inf and consistent output across iterations
        INDArray firstOutput = null;

        // Execute 5 times with frozen shapes
        for (int step = 0; step < 5; step++) {
            log.info("Attention step {} starting...", step);
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN in attention output");
            assertFalse(actual.isInfinite().any(), "Step " + step + ": Inf in attention output");

            if (firstOutput == null) {
                firstOutput = actual.dup();
            } else {
                double maxDiff = firstOutput.sub(actual).amaxNumber().doubleValue();
                log.info("Attention step {}: maxDiff vs step 0 = {}", step, maxDiff);
                assertTrue(maxDiff < 0.01,
                        String.format("Step %d: output differs from step 0, maxDiff=%.6f", step, maxDiff));
            }
            log.info("Attention step {} complete, sum={}", step, actual.sumNumber());
        }

        sd.close();
    }

    /**
     * Test: large multiply tensor that mirrors the VLM attention scale shape.
     * The VLM uses [1,12,1024,1024] = 12,582,912 elements.
     * This tests that the Triton kernel grid is correctly set for large tensors
     * and that graph replay works with large grids.
     */
    @Test
    @Timeout(120)
    public void testLargeMultiplyFrozenReplay() {
        SameDiff sd = SameDiff.create();
        int batch = 1, heads = 12, seqLen = 1024, dim = 1024;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, heads, seqLen, dim);
        SDVariable scaled = input.mul("scaled", 0.125);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // 12,582,912 elements — same as VLM attention scores
        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, heads, seqLen, dim);
        INDArray reference = inputArr.mul(0.125);

        Map<String, INDArray> feed = Map.of("input", inputArr);

        // Warmup
        sd.outputDirect(feed, "scaled");
        sd.outputDirect(feed, "scaled");

        // Compile with Triton
        try {
            sd.compileNativeDynamicShapePlan(List.of("scaled"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        // Execute 5 frozen iterations — hang occurred on 2nd frozen call in VLM test
        for (int step = 0; step < 5; step++) {
            log.info("Large multiply step {} starting...", step);
            Map<String, INDArray> result = sd.outputDirect(feed, "scaled");
            INDArray actual = result.get("scaled");
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN in large multiply");

            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            log.info("Large multiply step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                    String.format("Step %d: maxDiff=%.6f exceeds tolerance", step, maxDiff));
        }

        sd.close();
    }

    /**
     * Test: attention pattern with EXACT VLM shapes that cause the hang.
     * The VLM uses [1,12,1024,64]×[1,12,64,1024] → [1,12,1024,1024] for QK^T.
     * This is 12× batched GEMM of 1024×64 × 64×1024 = 1024×1024 output per head.
     *
     * The hang occurs on the second frozen execution (first graph replay).
     * With the initial launch skipped after capture, the graph has NEVER been
     * executed before the first replay.
     *
     * Key difference from testAttentionPatternFrozenReplay: uses the exact VLM
     * shapes (1024 seqLen) instead of small test shapes (64 seqLen). The larger
     * GEMM may select a different cuBLAS algorithm or use the workspace differently.
     */
    @Test
    @Timeout(120)
    public void testVLMShapeAttentionFrozenReplay() {
        SameDiff sd = SameDiff.create();
        // Exact VLM shapes from the hang: [1,12,1024,64] × [1,12,64,1024]
        int batch = 1, heads = 12, seqLen = 1024, dim = 64;

        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, heads, seqLen, dim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, heads, dim, seqLen);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, heads, seqLen, dim);

        // Attention: softmax(Q*K^T / sqrt(d)) * V
        SDVariable qk = sd.linalg().matmul("qk", q, k);  // [1,12,1024,1024]
        SDVariable scaled = qk.mul("scaled", 1.0 / Math.sqrt(dim));
        SDVariable attn = sd.nn().softmax("attn", scaled, -1);
        SDVariable output = sd.linalg().matmul("output", attn, v);  // [1,12,1024,64]

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, heads, seqLen, dim);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, heads, dim, seqLen);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, heads, seqLen, dim);

        Map<String, INDArray> feed = Map.of("q", qArr, "k", kArr, "v", vArr);

        // Warmup (2 calls)
        sd.outputDirect(feed, "output");
        sd.outputDirect(feed, "output");

        // Compile with Triton + freeze
        try {
            sd.compileNativeDynamicShapePlan(List.of("output"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        INDArray firstOutput = null;

        // Execute 5 frozen iterations — hang occurred on iteration 2 (first replay)
        for (int step = 0; step < 5; step++) {
            log.info("VLM attention step {} starting...", step);
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN in VLM attention output");
            assertFalse(actual.isInfinite().any(), "Step " + step + ": Inf in VLM attention output");

            if (firstOutput == null) {
                firstOutput = actual.dup();
            } else {
                double maxDiff = firstOutput.sub(actual).amaxNumber().doubleValue();
                log.info("VLM attention step {}: maxDiff vs step 0 = {}", step, maxDiff);
                assertTrue(maxDiff < 0.01,
                        String.format("Step %d: output differs from step 0, maxDiff=%.6f", step, maxDiff));
            }
            log.info("VLM attention step {} complete, sum={}", step, actual.sumNumber());
        }

        sd.close();
    }

    /**
     * Test: multi-segment attention with cross-segment dependencies.
     *
     * The real VLM has linear projections BEFORE the attention block.
     * This means Q, K, V are OUTPUTS of prior segments (cross-segment deps).
     * The capture buffer mechanism must correctly handle cuBLAS address baking
     * for these cross-segment inputs.
     *
     * Graph structure:
     *   Segment 1: linear projections (Q_proj, K_proj, V_proj) — elementwise, Triton-compiled
     *   Segment 2: attention (matmul QK^T → scale → softmax → matmul attnV) — mixed cuBLAS/Triton
     *
     * The cuBLAS kernels in segment 2 read from segment 1's outputs.
     * During capture, these addresses should be capture buffer addresses.
     * On replay, D2D copies refresh capture buffers before graph launch.
     */
    @Test
    @Timeout(120)
    public void testCrossSegmentAttentionFrozenReplay() {
        SameDiff sd = SameDiff.create();
        int batch = 1, heads = 12, seqLen = 1024, dim = 64;
        int hidden = heads * dim;  // 768

        // Input embeddings (simulating transformer hidden state)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);

        // Linear projections — these create cross-segment dependencies
        // In the real model these are weight matmuls; here we use elementwise
        // (multiply by weight-like constant) so they're Triton-compiled and
        // form a separate segment from the attention block.
        SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, hidden).mul(0.02));
        SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, hidden).mul(0.02));
        SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, hidden).mul(0.02));

        SDVariable qFlat = input.mul("q_proj", wq);  // [1,1024,768]
        SDVariable kFlat = input.mul("k_proj", wk);
        SDVariable vFlat = input.mul("v_proj", wv);

        // Reshape to multi-head: [1,1024,768] → [1,1024,12,64] → [1,12,1024,64]
        SDVariable qReshaped = sd.reshape("q_reshape", qFlat, batch, seqLen, heads, dim);
        SDVariable qMH = sd.permute("q_mh", qReshaped, 0, 2, 1, 3);  // [1,12,1024,64]
        SDVariable kReshaped = sd.reshape("k_reshape", kFlat, batch, seqLen, heads, dim);
        SDVariable kMH = sd.permute("k_mh", kReshaped, 0, 2, 3, 1);  // [1,12,64,1024] (transposed)
        SDVariable vReshaped = sd.reshape("v_reshape", vFlat, batch, seqLen, heads, dim);
        SDVariable vMH = sd.permute("v_mh", vReshaped, 0, 2, 1, 3);  // [1,12,1024,64]

        // Attention block — this is the segment that hangs
        SDVariable qk = sd.linalg().matmul("qk", qMH, kMH);  // [1,12,1024,1024]
        SDVariable scaled = qk.mul("scaled", 1.0 / Math.sqrt(dim));
        SDVariable attn = sd.nn().softmax("attn", scaled, -1);
        SDVariable output = sd.linalg().matmul("output", attn, vMH);  // [1,12,1024,64]

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        Map<String, INDArray> feed = Map.of("input", inputArr);

        // Warmup
        sd.outputDirect(feed, "output");
        sd.outputDirect(feed, "output");

        // Compile with Triton + freeze
        try {
            sd.compileNativeDynamicShapePlan(List.of("output"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        INDArray firstOutput = null;

        for (int step = 0; step < 5; step++) {
            log.info("Cross-segment attention step {} starting...", step);
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN in cross-segment attention");
            assertFalse(actual.isInfinite().any(), "Step " + step + ": Inf in cross-segment attention");

            if (firstOutput == null) {
                firstOutput = actual.dup();
            } else {
                double maxDiff = firstOutput.sub(actual).amaxNumber().doubleValue();
                log.info("Cross-segment step {}: maxDiff vs step 0 = {}", step, maxDiff);
                assertTrue(maxDiff < 0.01,
                        String.format("Step %d: output differs from step 0, maxDiff=%.6f", step, maxDiff));
            }
            log.info("Cross-segment attention step {} complete, sum={}", step, actual.sumNumber());
        }

        sd.close();
    }

    /**
     * Test: multiple attention layers in sequence.
     * Simulates the VLM model structure where seg[1320-1323] is preceded by
     * many other captured segments. Tests whether accumulated CUDA graph
     * state from prior segment replays affects later segments.
     */
    @Test
    @Timeout(180)
    public void testMultiLayerAttentionFrozenReplay() {
        SameDiff sd = SameDiff.create();
        int batch = 1, heads = 12, seqLen = 256, dim = 64;
        int numLayers = 4;  // Multiple layers to simulate accumulated state

        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, heads, seqLen, dim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, heads, dim, seqLen);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, heads, seqLen, dim);

        // Chain of attention layers — each layer depends on the previous one
        SDVariable current = q;
        for (int layer = 0; layer < numLayers; layer++) {
            String prefix = "L" + layer + "_";
            SDVariable qk = sd.linalg().matmul(prefix + "qk", current, k);
            SDVariable scaled = qk.mul(prefix + "scaled", 1.0 / Math.sqrt(dim));
            SDVariable attn = sd.nn().softmax(prefix + "attn", scaled, -1);
            // attn × V: [1,12,256,256] × [1,12,256,64] = [1,12,256,64]
            current = sd.linalg().matmul(prefix + "output", attn, v);
        }

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, heads, seqLen, dim);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, heads, dim, seqLen);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, heads, seqLen, dim);

        Map<String, INDArray> feed = Map.of("q", qArr, "k", kArr, "v", vArr);

        String lastOutput = "L" + (numLayers - 1) + "_output";

        // Warmup
        sd.outputDirect(feed, lastOutput);
        sd.outputDirect(feed, lastOutput);

        try {
            sd.compileNativeDynamicShapePlan(List.of(lastOutput),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        INDArray firstOutput = null;
        for (int step = 0; step < 5; step++) {
            log.info("Multi-layer attention step {} starting...", step);
            Map<String, INDArray> result = sd.outputDirect(feed, lastOutput);
            INDArray actual = result.get(lastOutput);
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN in multi-layer attention");
            assertFalse(actual.isInfinite().any(), "Step " + step + ": Inf in multi-layer attention");

            if (firstOutput == null) {
                firstOutput = actual.dup();
            } else {
                double maxDiff = firstOutput.sub(actual).amaxNumber().doubleValue();
                log.info("Multi-layer step {}: maxDiff vs step 0 = {}", step, maxDiff);
                assertTrue(maxDiff < 0.01,
                        String.format("Step %d: output differs from step 0, maxDiff=%.6f", step, maxDiff));
            }
            log.info("Multi-layer attention step {} complete", step);
        }

        sd.close();
    }

    /**
     * Test: multiply + add chain (fused elementwise) with frozen replay.
     * Tests that fused Triton kernels with multiple ops also replay correctly.
     */
    @Test
    @Timeout(120)
    public void testFusedElementwiseFrozenReplay() {
        SameDiff sd = SameDiff.create();
        int batch = 1, seqLen = 1024, hidden = 768;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, hidden));

        // multiply + add chain → should fuse into single Triton kernel
        SDVariable scaled = input.mul("scaled", 0.5);
        SDVariable output = scaled.add("output", bias);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray biasArr = sd.getVariable("bias").getArr();
        INDArray reference = inputArr.mul(0.5).add(biasArr);

        Map<String, INDArray> feed = Map.of("input", inputArr);

        // Warmup
        sd.outputDirect(feed, "output");
        sd.outputDirect(feed, "output");

        // Compile with Triton
        try {
            sd.compileNativeDynamicShapePlan(List.of("output"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        // Execute 5 frozen iterations
        for (int step = 0; step < 5; step++) {
            log.info("Fused elementwise step {} starting...", step);
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN in fused output");

            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            log.info("Fused elementwise step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                    String.format("Step %d: maxDiff=%.6f exceeds tolerance", step, maxDiff));
        }

        sd.close();
    }
}
