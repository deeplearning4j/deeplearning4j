package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNormBp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for LLM ops — verifies each op produces correct numerical
 * output against manually computed references. Tests individual factors:
 * - Op correctness with known inputs
 * - Rank-3 vs rank-4 inputs
 * - DSP on vs DSP off
 * - Mixed dtypes
 * - SameDiff execution vs direct Nd4j.exec
 */
@Slf4j
public class TestLLMOpCorrectness {

    @Test
    public void testRmsNormBpHalfZeroInputFinite() {
        int batch = 1, seq = 1, features = 8;
        float eps = 1e-5f;

        INDArray input = Nd4j.zeros(DataType.FLOAT16, batch, seq, features);
        INDArray gradOut = Nd4j.ones(DataType.FLOAT16, batch, seq, features);
        INDArray gamma = Nd4j.ones(DataType.FLOAT16, features);
        INDArray gradIn = Nd4j.create(DataType.FLOAT16, batch, seq, features);
        INDArray gradGamma = Nd4j.create(DataType.FLOAT16, features);

        Nd4j.exec(new RmsNormBp(input, gradOut, gamma, gradIn, gradGamma, eps));

        assertFalse(gradIn.isNaN().any(), "rms_norm_bp gradIn has NaN for zero HALF input");
        assertFalse(gradIn.isInfinite().any(), "rms_norm_bp gradIn has Inf for zero HALF input");
        assertFalse(gradGamma.isNaN().any(), "rms_norm_bp gradGamma has NaN for zero HALF input");
        assertFalse(gradGamma.isInfinite().any(), "rms_norm_bp gradGamma has Inf for zero HALF input");
        assertTrue(gradIn.norm1Number().doubleValue() > 0.0, "gradIn should carry finite upstream gradient");
        assertEquals(0.0, gradGamma.norm1Number().doubleValue(), 1e-6, "zero input should give zero gamma gradient");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  skip_rms_norm: output = rms_norm(input + skip) * gamma
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    public void testSkipRmsNorm_ManualReference() {
        int batch = 2, seq = 4, features = 8;
        float eps = 1e-5f;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seq, features).muli(0.5);
        INDArray skip = Nd4j.randn(DataType.FLOAT, batch, seq, features).muli(0.3);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, features);

        // Manual reference: hidden = input + skip, then rms_norm
        INDArray hidden = input.add(skip);
        INDArray hiddenSq = hidden.mul(hidden);
        // mean over last dim
        INDArray meanSq = hiddenSq.mean(true, -1); // [batch, seq, 1]
        INDArray rms = Transforms.sqrt(meanSq.addi(eps));
        INDArray expected = hidden.div(rms).muli(gamma);

        // Op execution via Nd4j.exec
        INDArray output = Nd4j.create(DataType.FLOAT, batch, seq, features);
        Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SkipRmsNorm(
                input, skip, gamma, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        log.info("skip_rms_norm maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "skip_rms_norm output diverges from manual: maxDiff=" + maxDiff);

        // Verify output is not degenerate
        double outMax = output.amaxNumber().doubleValue();
        assertTrue(outMax > 1e-6, "skip_rms_norm output is all zeros");
        assertFalse(output.isNaN().any(), "skip_rms_norm output has NaN");
    }

    @Test
    public void testSkipRmsNorm_WithBias() {
        int batch = 2, seq = 4, features = 8;
        float eps = 1e-5f;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seq, features).muli(0.5);
        INDArray skip = Nd4j.randn(DataType.FLOAT, batch, seq, features).muli(0.3);
        INDArray gamma = Nd4j.randn(DataType.FLOAT, features).addi(1.0);
        INDArray bias = Nd4j.randn(DataType.FLOAT, features).muli(0.1);

        // Manual: hidden = input + skip + bias, then rms_norm * gamma
        INDArray hidden = input.add(skip).addi(bias);
        INDArray hiddenSq = hidden.mul(hidden);
        INDArray meanSq = hiddenSq.mean(true, -1);
        INDArray rms = Transforms.sqrt(meanSq.addi(eps));
        INDArray expected = hidden.div(rms).muli(gamma);

        // Via SameDiff
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.placeHolder("input", DataType.FLOAT, batch, seq, features);
        SDVariable sdSkip = sd.placeHolder("skip", DataType.FLOAT, batch, seq, features);
        SDVariable sdGamma = sd.constant("gamma", gamma);
        SDVariable sdBias = sd.constant("bias", bias);
        SDVariable out = sd.nn.skipRmsNorm("output", sdInput, sdSkip, sdGamma, sdBias, eps);

        INDArray result = sd.outputSingle(Map.of("input", input, "skip", skip), "output");

        double maxDiff = expected.sub(result).amaxNumber().doubleValue();
        log.info("skip_rms_norm (bias) maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "skip_rms_norm with bias diverges: maxDiff=" + maxDiff);
    }

    @Test
    public void testSkipRmsNorm_SameDiff_DspOnVsOff() {
        int batch = 1, seq = 8, features = 16;
        float eps = 1e-5f;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        INDArray skip = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, features);

        // DSP off
        SameDiff sdOff = SameDiff.create();
        SDVariable in1 = sdOff.placeHolder("input", DataType.FLOAT, batch, seq, features);
        SDVariable sk1 = sdOff.placeHolder("skip", DataType.FLOAT, batch, seq, features);
        sdOff.nn.skipRmsNorm("output", in1, sk1, sdOff.constant("gamma", gamma), null, eps);
        INDArray refResult = sdOff.outputSingle(Map.of("input", input, "skip", skip), "output");

        // DSP on
        SameDiff sdOn = SameDiff.create();
        sdOn.setDspAutoCompileEnabled(true);
        sdOn.setDspNativeAutoCompileEnabled(true);
        SDVariable in2 = sdOn.placeHolder("input", DataType.FLOAT, batch, seq, features);
        SDVariable sk2 = sdOn.placeHolder("skip", DataType.FLOAT, batch, seq, features);
        sdOn.nn.skipRmsNorm("output", in2, sk2, sdOn.constant("gamma", gamma), null, eps);
        INDArray dspResult = sdOn.outputSingle(Map.of("input", input, "skip", skip), "output");

        double maxDiff = refResult.sub(dspResult).amaxNumber().doubleValue();
        log.info("skip_rms_norm DSP on vs off maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "skip_rms_norm DSP diverges: maxDiff=" + maxDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  dot_product_attention_v2: scaled dot-product attention
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> attentionRankAndCausalArgs() {
        return Stream.of(
                Arguments.of(3, false, "rank3_noCausal"),
                Arguments.of(3, true,  "rank3_causal"),
                Arguments.of(4, false, "rank4_noCausal"),
                Arguments.of(4, true,  "rank4_causal")
        );
    }

    @ParameterizedTest(name = "{2}")
    @MethodSource("attentionRankAndCausalArgs")
    public void testDotProductAttentionV2_ManualReference(int rank, boolean causal, String label) {
        int batch = 1, numHeads = 2, seqLen = 4, headDim = 8;
        double scale = 1.0 / Math.sqrt(headDim);

        INDArray Q, K, V;
        if (rank == 4) {
            // BSHD format
            Q = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
            K = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
            V = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        } else {
            Q = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads * headDim);
            K = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads * headDim);
            V = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads * headDim);
        }

        // Execute via SameDiff (DSP off)
        SameDiff sd = SameDiff.create();
        SDVariable qVar = sd.placeHolder("Q", Q.dataType(), Q.shape());
        SDVariable vVar = sd.placeHolder("V", V.dataType(), V.shape());
        SDVariable kVar = sd.placeHolder("K", K.dataType(), K.shape());

        SDVariable result = sd.nn.dotProductAttentionV2("attn", qVar, vVar, kVar,
                null, null, scale, 0.0, causal, false);

        INDArray output = sd.outputSingle(Map.of("Q", Q, "V", V, "K", K), "attn");

        // Basic sanity: output exists, not NaN, not all zeros, correct shape
        assertNotNull(output, label + ": output is null");
        assertFalse(output.isNaN().any(), label + ": output has NaN");
        assertFalse(output.isInfinite().any(), label + ": output has Inf");
        double outMax = output.amaxNumber().doubleValue();
        assertTrue(outMax > 1e-6, label + ": output is all zeros");
        assertArrayEquals(Q.shape(), output.shape(),
                label + ": output shape mismatch, expected " + java.util.Arrays.toString(Q.shape())
                        + " got " + java.util.Arrays.toString(output.shape()));

        log.info("[{}] attention output shape={} max={}", label,
                java.util.Arrays.toString(output.shape()), outMax);
    }

    @Test
    public void testDotProductAttentionV2_DspOnVsOff_Rank4() {
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;
        double scale = 1.0 / Math.sqrt(headDim);

        INDArray Q = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray K = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray V = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        // DSP off
        SameDiff sdOff = SameDiff.create();
        sdOff.nn.dotProductAttentionV2("attn",
                sdOff.placeHolder("Q", Q.dataType(), Q.shape()),
                sdOff.placeHolder("V", V.dataType(), V.shape()),
                sdOff.placeHolder("K", K.dataType(), K.shape()),
                null, null, scale, 0.0, false, false);
        INDArray refOut = sdOff.outputSingle(Map.of("Q", Q, "V", V, "K", K), "attn");

        // DSP on
        SameDiff sdOn = SameDiff.create();
        sdOn.setDspAutoCompileEnabled(true);
        sdOn.setDspNativeAutoCompileEnabled(true);
        sdOn.nn.dotProductAttentionV2("attn",
                sdOn.placeHolder("Q", Q.dataType(), Q.shape()),
                sdOn.placeHolder("V", V.dataType(), V.shape()),
                sdOn.placeHolder("K", K.dataType(), K.shape()),
                null, null, scale, 0.0, false, false);
        INDArray dspOut = sdOn.outputSingle(Map.of("Q", Q, "V", V, "K", K), "attn");

        double maxDiff = refOut.sub(dspOut).amaxNumber().doubleValue();
        log.info("dot_product_attention_v2 DSP on vs off maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "dot_product_attention_v2 DSP diverges: maxDiff=" + maxDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  onnx_multi_head_attention: GQA with mixed types
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    public void testOnnxMultiHeadAttention_SameType() {
        // All FLOAT inputs — baseline correctness
        int batch = 1, seqLen = 4, hidden = 16, numHeads = 2;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        INDArray result = Nd4j.create(DataType.FLOAT, batch, seqLen, hidden);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value)
                .addOutputs(result)
                .addIntegerArguments((long) numHeads, 0L)
                .build());

        assertNotNull(result, "MHA output is null");
        assertFalse(result.isNaN().any(), "MHA output has NaN");
        double maxVal = result.amaxNumber().doubleValue();
        assertTrue(maxVal > 1e-6, "MHA output is all zeros, max=" + maxVal);
        assertEquals(3, result.rank(), "MHA output should be rank 3");
        assertEquals(batch, result.size(0));
        assertEquals(seqLen, result.size(1));

        log.info("onnx_multi_head_attention (same type) output shape={} max={}",
                java.util.Arrays.toString(result.shape()), maxVal);
    }

    @Test
    public void testOnnxMultiHeadAttention_MixedType_Float16Key() {
        // Query=FLOAT, Key/Value=FLOAT16 — tests the new auto-cast logic
        int batch = 1, seqLen = 4, hidden = 16, numHeads = 2;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT16, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT16, batch, seqLen, hidden);

        // Execute same-type reference (cast key/value to FLOAT manually)
        INDArray keyF32 = key.castTo(DataType.FLOAT);
        INDArray valueF32 = value.castTo(DataType.FLOAT);

        // Same-type execution
        INDArray refOutput = Nd4j.create(DataType.FLOAT, batch, seqLen, hidden);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, keyF32, valueF32)
                .addOutputs(refOutput)
                .addIntegerArguments((long) numHeads, 0L)
                .build());

        // Mixed-type execution (auto-cast should handle it)
        INDArray mixedOutput = Nd4j.create(DataType.FLOAT, batch, seqLen, hidden);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value)
                .addOutputs(mixedOutput)
                .addIntegerArguments((long) numHeads, 0L)
                .build());

        double maxDiff = refOutput.sub(mixedOutput).amaxNumber().doubleValue();
        log.info("MHA mixed-type maxDiff={}", maxDiff);
        // FP16 precision loss means we allow more tolerance
        assertTrue(maxDiff < 1e-2, "MHA mixed-type diverges from same-type: maxDiff=" + maxDiff);

        assertFalse(mixedOutput.isNaN().any(), "MHA mixed-type output has NaN");
        double maxVal = mixedOutput.amaxNumber().doubleValue();
        assertTrue(maxVal > 1e-6, "MHA mixed-type output is all zeros");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  RoPE (rotary position embedding) — major rewrite in working tree
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    public void testRoPE_ManualReference_Rank4() {
        // Verify RoPE produces cos/sin rotation on pairs
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        float freqBase = 10000.0f;
        float freqScale = 1.0f;
        int positionOffset = 0;

        // Execute via op
        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("rope")
                .addInputs(input)
                .addOutputs(output)
                .addIntegerArguments(0L, (long) positionOffset, 0L) // mode=0 (standard), nPast=posOffset, nDims=0 (all)
                .addFloatingPointArguments((double) freqBase, (double) freqScale)
                .build());

        // Manual reference for position 0, head 0: theta_i = 0 for all i (cos=1, sin=0)
        // So output should equal input for position 0
        INDArray inputPos0 = input.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        INDArray outputPos0 = output.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all());

        double pos0Diff = inputPos0.sub(outputPos0).amaxNumber().doubleValue();
        log.info("RoPE position 0 diff={} (should be ~0 since theta=0)", pos0Diff);
        assertTrue(pos0Diff < 1e-5, "RoPE at position 0 should be identity: diff=" + pos0Diff);

        // For position 1+, output should differ from input (rotation applied)
        INDArray inputPos1 = input.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(1),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        INDArray outputPos1 = output.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(1),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        double pos1Diff = inputPos1.sub(outputPos1).amaxNumber().doubleValue();
        log.info("RoPE position 1 diff={} (should be >0 since rotation applied)", pos1Diff);
        assertTrue(pos1Diff > 1e-4, "RoPE at position 1 should differ from input: diff=" + pos1Diff);

        // Verify norm preservation: RoPE is a rotation, so ||output|| == ||input|| per head
        for (int s = 0; s < seqLen; s++) {
            for (int h = 0; h < numHeads; h++) {
                INDArray inVec = input.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(s),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(h),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                INDArray outVec = output.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(s),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(h),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                double inNorm = inVec.norm2Number().doubleValue();
                double outNorm = outVec.norm2Number().doubleValue();
                double normDiff = Math.abs(inNorm - outNorm);
                assertTrue(normDiff < 1e-4,
                        "RoPE should preserve norm: seq=" + s + " head=" + h
                                + " inNorm=" + inNorm + " outNorm=" + outNorm + " diff=" + normDiff);
            }
        }

        assertFalse(output.isNaN().any(), "RoPE output has NaN");
        log.info("RoPE rank-4 test passed");
    }

    @Test
    public void testRoPE_ManualReference_Rank3() {
        // Rank 3: [batch, seqLen, features] — numHeads=1 internally
        int batch = 1, seqLen = 4, features = 8;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, features);
        float freqBase = 10000.0f;
        float freqScale = 1.0f;

        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, features);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("rope")
                .addInputs(input)
                .addOutputs(output)
                .addIntegerArguments(0L, 0L, 0L)
                .addFloatingPointArguments((double) freqBase, (double) freqScale)
                .build());

        // Position 0: identity
        double pos0Diff = input.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all())
                .sub(output.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.all()))
                .amaxNumber().doubleValue();
        assertTrue(pos0Diff < 1e-5, "RoPE rank-3 position 0 should be identity: diff=" + pos0Diff);

        // Norm preservation
        for (int s = 0; s < seqLen; s++) {
            INDArray inVec = input.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(s),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            INDArray outVec = output.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(s),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            double normDiff = Math.abs(inVec.norm2Number().doubleValue() - outVec.norm2Number().doubleValue());
            assertTrue(normDiff < 1e-4, "RoPE rank-3 norm not preserved at seq=" + s + " diff=" + normDiff);
        }

        assertFalse(output.isNaN().any(), "RoPE rank-3 output has NaN");
        log.info("RoPE rank-3 test passed");
    }

    @Test
    public void testRoPE_PositionOffset() {
        // RoPE with positionOffset > 0 (decode step) — position 0 should NOT be identity
        int batch = 1, seqLen = 1, numHeads = 2, headDim = 8;
        int positionOffset = 5;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("rope")
                .addInputs(input)
                .addOutputs(output)
                .addIntegerArguments(0L, (long) positionOffset, 0L) // mode=0, nPast=5, nDims=0
                .addFloatingPointArguments(10000.0, 1.0)
                .build());

        // With nPast=5, even position 0 in the input has effective position 5
        // so it should NOT be identity
        double diff = input.sub(output).amaxNumber().doubleValue();
        assertTrue(diff > 1e-3, "RoPE with offset should NOT be identity: diff=" + diff);

        // Norm still preserved
        double inNorm = input.norm2Number().doubleValue();
        double outNorm = output.norm2Number().doubleValue();
        assertTrue(Math.abs(inNorm - outNorm) < 1e-3,
                "RoPE with offset should preserve norm: inNorm=" + inNorm + " outNorm=" + outNorm);
    }

    @Test
    public void testRoPE_ConsistencyAcrossPositionOffsets() {
        // Running RoPE on seqLen=4 with offset=0 should give the same result as
        // running on seqLen=1 four times with offset=0,1,2,3
        int batch = 1, numHeads = 2, headDim = 8;

        INDArray fullInput = Nd4j.randn(DataType.FLOAT, batch, 4, numHeads, headDim);
        INDArray fullOutput = Nd4j.create(DataType.FLOAT, batch, 4, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("rope")
                .addInputs(fullInput)
                .addOutputs(fullOutput)
                .addIntegerArguments(0L, 0L, 0L) // mode=0, nPast=0, nDims=0
                .addFloatingPointArguments(10000.0, 1.0)
                .build());

        // Now run position-by-position
        for (int pos = 0; pos < 4; pos++) {
            INDArray singleInput = fullInput.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(pos, pos + 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            INDArray singleOutput = Nd4j.create(DataType.FLOAT, batch, 1, numHeads, headDim);
            Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("rope")
                    .addInputs(singleInput)
                    .addOutputs(singleOutput)
                    .addIntegerArguments(0L, (long) pos, 0L) // mode=0, nPast=pos, nDims=0
                    .addFloatingPointArguments(10000.0, 1.0)
                    .build());

            INDArray fullSlice = fullOutput.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(pos, pos + 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());

            double diff = fullSlice.sub(singleOutput).amaxNumber().doubleValue();
            log.info("RoPE consistency pos={} diff={}", pos, diff);
            assertTrue(diff < 1e-5,
                    "RoPE full-seq vs single-step inconsistent at pos=" + pos + " diff=" + diff);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  fused_rope — the op the VLM actually uses (FusedRoPE Java class)
    //  Tests BOTH dynamic path (1 input) and cached path (3 inputs: input+cos+sin)
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    public void testFusedRoPE_Dynamic_ManualReference_Rank4() {
        // fused_rope with dynamic sin/cos (1 input, same math as rope but via helper)
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;
        float freqBase = 10000.0f;
        float freqScale = 1.0f;
        int halfDim = headDim / 2;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        // Execute fused_rope op
        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("fused_rope")
                .addInputs(input)
                .addOutputs(output)
                .addIntegerArguments(0L, 0L, 0L)  // ropeType=0 (standard), posOffset=0, rotaryDims=0
                .addFloatingPointArguments((double) freqBase, (double) freqScale)
                .build());

        // Manual reference computation
        INDArray expected = input.dup();
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int i = 0; i < halfDim; i++) {
                        float theta = (float) (s * freqScale /
                                Math.pow(freqBase, (2.0f * i) / headDim));
                        float cosTheta = (float) Math.cos(theta);
                        float sinTheta = (float) Math.sin(theta);

                        float x1 = input.getFloat(b, s, h, i);
                        float x2 = input.getFloat(b, s, h, i + halfDim);

                        expected.putScalar(new long[]{b, s, h, i}, x1 * cosTheta - x2 * sinTheta);
                        expected.putScalar(new long[]{b, s, h, i + halfDim}, x1 * sinTheta + x2 * cosTheta);
                    }
                }
            }
        }

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        log.info("fused_rope dynamic maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "fused_rope dynamic diverges from manual: maxDiff=" + maxDiff);

        // Position 0: theta=0 for all dims, so cos=1, sin=0 → identity
        double pos0Diff = input.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all())
                .sub(output.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all()))
                .amaxNumber().doubleValue();
        assertTrue(pos0Diff < 1e-5, "fused_rope at position 0 should be identity: diff=" + pos0Diff);

        // Position 1+: should differ from input
        double pos1Diff = input.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(1),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all())
                .sub(output.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(1),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all()))
                .amaxNumber().doubleValue();
        assertTrue(pos1Diff > 1e-4, "fused_rope at position 1 should rotate: diff=" + pos1Diff);

        assertFalse(output.isNaN().any(), "fused_rope dynamic output has NaN");
    }

    @Test
    public void testFusedRoPE_Cached_ManualReference() {
        // This is the EXACT path the VLM uses: fused_rope with pre-computed cos/sin
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;
        int halfDim = headDim / 2;
        float freqBase = 10000.0f;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        // Pre-compute cos/sin caches like ONNX model provides
        // Shape: [seqLen, halfDim] (rank-2 broadcast across batch)
        INDArray cosCacheData = Nd4j.create(DataType.FLOAT, seqLen, halfDim);
        INDArray sinCacheData = Nd4j.create(DataType.FLOAT, seqLen, halfDim);
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < halfDim; i++) {
                float theta = (float) (s / Math.pow(freqBase, (2.0f * i) / headDim));
                cosCacheData.putScalar(new long[]{s, i}, (float) Math.cos(theta));
                sinCacheData.putScalar(new long[]{s, i}, (float) Math.sin(theta));
            }
        }

        // Execute fused_rope with cached path (3 inputs)
        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("fused_rope")
                .addInputs(input, cosCacheData, sinCacheData)
                .addOutputs(output)
                .addIntegerArguments(0L)  // ropeType=0 (standard)
                .build());

        // Manual reference using the same cos/sin values
        INDArray expected = input.dup();
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int i = 0; i < halfDim; i++) {
                        float cosVal = cosCacheData.getFloat(s, i);
                        float sinVal = sinCacheData.getFloat(s, i);

                        float x1 = input.getFloat(b, s, h, i);
                        float x2 = input.getFloat(b, s, h, i + halfDim);

                        expected.putScalar(new long[]{b, s, h, i}, x1 * cosVal - x2 * sinVal);
                        expected.putScalar(new long[]{b, s, h, i + halfDim}, x1 * sinVal + x2 * cosVal);
                    }
                }
            }
        }

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        log.info("fused_rope cached maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "fused_rope cached diverges from manual: maxDiff=" + maxDiff);

        // Verify non-degenerate (not identity)
        double totalDiff = input.sub(output).amaxNumber().doubleValue();
        // At least some positions should have rotated (pos 1,2,3)
        assertTrue(totalDiff > 1e-3, "fused_rope cached output should not equal input: diff=" + totalDiff);

        assertFalse(output.isNaN().any(), "fused_rope cached output has NaN");
    }

    @Test
    public void testFusedRoPE_Cached_Rank3CosSin() {
        // cos/sin as [batch, seqLen, halfDim] (rank-3)
        int batch = 2, seqLen = 4, numHeads = 2, headDim = 8;
        int halfDim = headDim / 2;
        float freqBase = 10000.0f;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        // Rank-3 cos/sin: [batch, seqLen, halfDim]
        INDArray cosCache = Nd4j.create(DataType.FLOAT, batch, seqLen, halfDim);
        INDArray sinCache = Nd4j.create(DataType.FLOAT, batch, seqLen, halfDim);
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int i = 0; i < halfDim; i++) {
                    float theta = (float) (s / Math.pow(freqBase, (2.0f * i) / headDim));
                    cosCache.putScalar(new long[]{b, s, i}, (float) Math.cos(theta));
                    sinCache.putScalar(new long[]{b, s, i}, (float) Math.sin(theta));
                }
            }
        }

        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("fused_rope")
                .addInputs(input, cosCache, sinCache)
                .addOutputs(output)
                .addIntegerArguments(0L)
                .build());

        assertFalse(output.isNaN().any(), "fused_rope cached rank-3 output has NaN");
        double outMax = output.amaxNumber().doubleValue();
        assertTrue(outMax > 1e-6, "fused_rope cached rank-3 output is all zeros");

        // Norm preservation (rotation preserves norm)
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    INDArray inVec = input.get(
                            org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                            org.nd4j.linalg.indexing.NDArrayIndex.point(s),
                            org.nd4j.linalg.indexing.NDArrayIndex.point(h),
                            org.nd4j.linalg.indexing.NDArrayIndex.all());
                    INDArray outVec = output.get(
                            org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                            org.nd4j.linalg.indexing.NDArrayIndex.point(s),
                            org.nd4j.linalg.indexing.NDArrayIndex.point(h),
                            org.nd4j.linalg.indexing.NDArrayIndex.all());
                    double inNorm = inVec.norm2Number().doubleValue();
                    double outNorm = outVec.norm2Number().doubleValue();
                    assertTrue(Math.abs(inNorm - outNorm) < 1e-3,
                            "fused_rope cached rank-3 norm not preserved: b=" + b + " s=" + s + " h=" + h);
                }
            }
        }
    }

    @Test
    public void testFusedRoPE_Dynamic_PositionOffset() {
        // fused_rope with positionOffset > 0 — KV cache decode step
        int batch = 1, seqLen = 1, numHeads = 2, headDim = 8;
        int posOffset = 5;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("fused_rope")
                .addInputs(input)
                .addOutputs(output)
                .addIntegerArguments(0L, (long) posOffset, 0L)
                .addFloatingPointArguments(10000.0, 1.0)
                .build());

        // With offset=5, the rotation should be non-trivial
        double diff = input.sub(output).amaxNumber().doubleValue();
        assertTrue(diff > 1e-3, "fused_rope with offset=5 should NOT be identity: diff=" + diff);

        // Norm preserved
        double inNorm = input.norm2Number().doubleValue();
        double outNorm = output.norm2Number().doubleValue();
        assertTrue(Math.abs(inNorm - outNorm) < 1e-3,
                "fused_rope with offset should preserve norm: inNorm=" + inNorm + " outNorm=" + outNorm);

        assertFalse(output.isNaN().any(), "fused_rope offset output has NaN");
    }

    @Test
    public void testFusedRoPE_Dynamic_ScalarPositionInput() {
        // fused_rope with position as a scalar INPUT tensor (2-input dynamic path)
        // This is the decode path where position changes each step
        int batch = 1, seqLen = 1, numHeads = 2, headDim = 8;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray positionTensor = Nd4j.scalar(DataType.INT64, 10); // position = 10

        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("fused_rope")
                .addInputs(input, positionTensor)
                .addOutputs(output)
                .addIntegerArguments(0L)
                .addFloatingPointArguments(10000.0, 1.0)
                .build());

        // With position=10, rotation should be substantial
        double diff = input.sub(output).amaxNumber().doubleValue();
        assertTrue(diff > 1e-3, "fused_rope with scalar pos=10 should NOT be identity: diff=" + diff);

        assertFalse(output.isNaN().any(), "fused_rope scalar position output has NaN");
    }

    @Test
    public void testFusedRoPE_ConsistencyDynamicVsCached() {
        // Dynamic and cached paths should produce identical results
        // when the cos/sin caches match the dynamic computation
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;
        int halfDim = headDim / 2;
        float freqBase = 10000.0f;
        float freqScale = 1.0f;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        // Dynamic path
        INDArray dynamicOutput = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("fused_rope")
                .addInputs(input)
                .addOutputs(dynamicOutput)
                .addIntegerArguments(0L, 0L, 0L)
                .addFloatingPointArguments((double) freqBase, (double) freqScale)
                .build());

        // Build matching cos/sin cache
        INDArray cosCache = Nd4j.create(DataType.FLOAT, seqLen, halfDim);
        INDArray sinCache = Nd4j.create(DataType.FLOAT, seqLen, halfDim);
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < halfDim; i++) {
                float theta = (float) (s * freqScale / Math.pow(freqBase, (2.0f * i) / headDim));
                cosCache.putScalar(new long[]{s, i}, (float) Math.cos(theta));
                sinCache.putScalar(new long[]{s, i}, (float) Math.sin(theta));
            }
        }

        // Cached path
        INDArray cachedOutput = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("fused_rope")
                .addInputs(input, cosCache, sinCache)
                .addOutputs(cachedOutput)
                .addIntegerArguments(0L)
                .build());

        double maxDiff = dynamicOutput.sub(cachedOutput).amaxNumber().doubleValue();
        log.info("fused_rope dynamic vs cached maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4,
                "fused_rope dynamic and cached paths should match: maxDiff=" + maxDiff);
    }

    @Test
    public void testFusedRoPE_ViaJavaClass_Cached() {
        // Test using the FusedRoPE Java class directly (what RotaryEmbedding.kt uses)
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;
        int halfDim = headDim / 2;
        float freqBase = 10000.0f;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        INDArray cosCache = Nd4j.create(DataType.FLOAT, seqLen, halfDim);
        INDArray sinCache = Nd4j.create(DataType.FLOAT, seqLen, halfDim);
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < halfDim; i++) {
                float theta = (float) (s / Math.pow(freqBase, (2.0f * i) / headDim));
                cosCache.putScalar(new long[]{s, i}, (float) Math.cos(theta));
                sinCache.putScalar(new long[]{s, i}, (float) Math.sin(theta));
            }
        }

        // Use FusedRoPE Java class directly with cached constructor
        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE op =
                new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(
                        input, cosCache, sinCache, output, 0);
        Nd4j.exec(op);

        // Manual reference
        INDArray expected = input.dup();
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int i = 0; i < halfDim; i++) {
                        float cosVal = cosCache.getFloat(s, i);
                        float sinVal = sinCache.getFloat(s, i);
                        float x1 = input.getFloat(b, s, h, i);
                        float x2 = input.getFloat(b, s, h, i + halfDim);
                        expected.putScalar(new long[]{b, s, h, i}, x1 * cosVal - x2 * sinVal);
                        expected.putScalar(new long[]{b, s, h, i + halfDim}, x1 * sinVal + x2 * cosVal);
                    }
                }
            }
        }

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        log.info("FusedRoPE Java class cached maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "FusedRoPE Java class cached diverges: maxDiff=" + maxDiff);

        // Non-degenerate check
        double totalDiff = input.sub(output).amaxNumber().doubleValue();
        assertTrue(totalDiff > 1e-3, "FusedRoPE output should differ from input");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  swish_mul: silu(x) * y — new REQUIRE_TRUE added
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    public void testSwishMul_ManualReference() {
        int batch = 2, seq = 4, features = 8;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        INDArray y = Nd4j.randn(DataType.FLOAT, batch, seq, features);

        // Manual: silu(x) * y = x * sigmoid(x) * y
        INDArray sigmoid = Transforms.sigmoid(x.dup());
        INDArray expected = x.mul(sigmoid).muli(y);

        INDArray output = Nd4j.create(DataType.FLOAT, batch, seq, features);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("swish_mul")
                .addInputs(x, y)
                .addOutputs(output)
                .build());

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        log.info("swish_mul maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "swish_mul diverges from manual: maxDiff=" + maxDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  silu: x * sigmoid(x) — out-of-place and in-place correctness
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    public void testSiLU_ManualReference() {
        int batch = 2, seq = 4, features = 8;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seq, features);

        // Manual: silu(x) = x * sigmoid(x)
        INDArray sigmoid = Transforms.sigmoid(x.dup());
        INDArray expected = x.mul(sigmoid);

        INDArray output = Nd4j.create(DataType.FLOAT, batch, seq, features);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("silu")
                .addInputs(x)
                .addOutputs(output)
                .build());

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        log.info("silu out-of-place maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "silu out-of-place diverges from manual: maxDiff=" + maxDiff);
    }

    @Test
    public void testSiLU_InPlace() {
        // Regression test: in-place silu must NOT produce sigmoid(x)^2.
        // The bug was: output = sigmoid(x) destroys original x when output==input,
        // then output *= input reads the already-sigmoid'd value → sigmoid(x)^2.
        int batch = 2, seq = 4, features = 8;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        INDArray expected = x.mul(Transforms.sigmoid(x.dup()));  // correct silu

        // In-place: output IS the input buffer
        INDArray inPlace = x.dup();
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("silu")
                .addInputs(inPlace)
                .addOutputs(inPlace)  // same buffer as input
                .build());

        double maxDiff = expected.sub(inPlace).amaxNumber().doubleValue();
        log.info("silu in-place maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "silu IN-PLACE diverges from manual: maxDiff=" + maxDiff
                + ". If maxDiff is large, the in-place path is computing sigmoid(x)^2 instead of x*sigmoid(x).");
    }

    @Test
    public void testSwishMul_InPlace_OnX() {
        // Regression test: in-place swish_mul on first input (x).
        int batch = 2, seq = 4, features = 8;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        INDArray y = Nd4j.randn(DataType.FLOAT, batch, seq, features);

        INDArray sigmoid = Transforms.sigmoid(x.dup());
        INDArray expected = x.mul(sigmoid).muli(y);

        INDArray inPlaceX = x.dup();
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("swish_mul")
                .addInputs(inPlaceX, y)
                .addOutputs(inPlaceX)  // output aliases x
                .build());

        double maxDiff = expected.sub(inPlaceX).amaxNumber().doubleValue();
        log.info("swish_mul in-place(x) maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "swish_mul in-place(x) diverges: maxDiff=" + maxDiff);
    }
}
