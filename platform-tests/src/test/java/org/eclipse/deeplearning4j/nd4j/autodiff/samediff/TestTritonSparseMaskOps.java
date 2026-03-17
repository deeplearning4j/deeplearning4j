package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that Triton-compiled elementwise ops produce correct results with sparse
 * (mixed 0/1) input data — specifically the attn_mask_reformat pattern:
 *   attention_mask → expand → (1-mask) * MASK_FILL → bias
 *
 * Bug context: SLOT_BY_SLOT + padded mode = CORRECT, but Triton + padded = WRONG.
 * View-based (all-ones mask) works because the uniform values hide broadcast/indexing bugs.
 * Sparse masks (mix of 0s and 1s) expose the issue.
 */
@Slf4j
@DisplayName("Triton Sparse Mask Op Tests")
public class TestTritonSparseMaskOps {

    private static final double TOL = 1e-5;
    private static final float MASK_FILL = -3.4028235e+38f;

    /**
     * Test the core attn_mask_reformat pattern: (1 - mask) * MASK_FILL
     * with a sparse mask through DSP (outputDirect) vs standard (output).
     */
    @Test
    @DisplayName("Sparse mask: (1-mask)*FILL via DSP vs standard")
    void testSparseMaskInvertAndFill() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Sparse attention mask: [1, 16] — some 1s, some 0s
        SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, 1, -1);
        SDVariable ones = sd.onesLike("ones", mask);
        SDVariable inverted = ones.sub("inverted", mask);
        SDVariable fill = sd.constant("fill", Nd4j.scalar(MASK_FILL));
        SDVariable bias = inverted.mul("bias", fill);
        SDVariable result = sd.sum("result", bias);

        // Sparse mask: positions 0-4 = 1 (valid), 5-14 = 0 (padding), 15 = 1 (current token)
        INDArray maskArr = Nd4j.zeros(DataType.FLOAT, 1, 16);
        for (int i = 0; i < 5; i++) maskArr.putScalar(0, i, 1.0f);
        maskArr.putScalar(0, 15, 1.0f);

        Map<String, INDArray> inputs = Map.of("mask", maskArr);

        // Standard execution
        Map<String, INDArray> expected = sd.output(inputs, "bias", "result");
        INDArray expectedBias = expected.get("bias").dup();
        double expectedSum = expected.get("result").getDouble(0);

        // DSP execution
        Map<String, INDArray> dspResult = sd.outputDirect(inputs, "bias", "result");
        INDArray dspBias = dspResult.get("bias").dup();
        double dspSum = dspResult.get("result").getDouble(0);

        log.info("Expected bias: {}", expectedBias);
        log.info("DSP bias:      {}", dspBias);
        log.info("Expected sum: {}, DSP sum: {}", expectedSum, dspSum);

        // Verify bias values: positions with mask=1 should have bias=0, mask=0 should have bias=MASK_FILL
        double biasDiff = expectedBias.sub(dspBias).amaxNumber().doubleValue();
        log.info("Bias max diff: {}", biasDiff);
        assertTrue(biasDiff < TOL,
                "DSP bias should match standard execution. MaxDiff=" + biasDiff);

        sd.close();
    }

    /**
     * Test where/select op with sparse BOOL condition through DSP.
     * Where(condition, 0.0, MASK_FILL) — the exact pattern from attn_mask_reformat.
     */
    @Test
    @DisplayName("Sparse where: Where(sparse_bool, 0.0, MASK_FILL) via DSP vs standard")
    void testSparseWhereSelect() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable cond = sd.placeHolder("cond", DataType.BOOL, 1, -1);
        SDVariable trueVal = sd.constant("trueVal", Nd4j.scalar(DataType.FLOAT, 0.0));
        SDVariable falseVal = sd.constant("falseVal", Nd4j.scalar(DataType.FLOAT, MASK_FILL));
        SDVariable out = sd.where("out", trueVal, falseVal, cond);

        // Sparse condition: [true,true,true,true,true, false,false,...,false, true]
        INDArray condArr = Nd4j.zeros(DataType.BOOL, 1, 16);
        for (int i = 0; i < 5; i++) condArr.putScalar(0, i, 1);
        condArr.putScalar(0, 15, 1);

        Map<String, INDArray> inputs = Map.of("cond", condArr);

        Map<String, INDArray> expected = sd.output(inputs, "out");
        INDArray expectedOut = expected.get("out").dup();

        Map<String, INDArray> dspResult = sd.outputDirect(inputs, "out");
        INDArray dspOut = dspResult.get("out").dup();

        log.info("Expected: {}", expectedOut);
        log.info("DSP:      {}", dspOut);

        double maxDiff = expectedOut.sub(dspOut).amaxNumber().doubleValue();
        log.info("Where max diff: {}", maxDiff);
        assertTrue(maxDiff < TOL,
                "DSP where should match standard execution. MaxDiff=" + maxDiff);

        sd.close();
    }

    /**
     * Test less (comparison) with values that produce sparse BOOL output through DSP.
     * less(range, threshold) — the pattern from input_ids_subgraph.
     */
    @Test
    @DisplayName("Less comparison producing sparse BOOL output via DSP vs standard")
    void testLessProducingSparseBool() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, -1);
        SDVariable threshold = sd.constant("threshold", Nd4j.scalar(DataType.FLOAT, 5.0f));
        SDVariable lt = sd.lt("lt", x, threshold);
        SDVariable result = sd.castTo("result", lt, DataType.FLOAT);

        // Input: [0, 1, 2, ..., 15] → less than 5 → [1,1,1,1,1, 0,0,...,0]
        INDArray xArr = Nd4j.linspace(0, 15, 16, DataType.FLOAT).reshape(1, 16);

        Map<String, INDArray> inputs = Map.of("x", xArr);

        Map<String, INDArray> expected = sd.output(inputs, "result");
        INDArray expectedResult = expected.get("result").dup();

        Map<String, INDArray> dspResult = sd.outputDirect(inputs, "result");
        INDArray dspResultArr = dspResult.get("result").dup();

        log.info("Expected: {}", expectedResult);
        log.info("DSP:      {}", dspResultArr);

        double maxDiff = expectedResult.sub(dspResultArr).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "DSP less should match standard execution. MaxDiff=" + maxDiff);

        sd.close();
    }

    /**
     * Test the full chain: less → cast → where → mul with sparse data through DSP.
     * This mimics the complete attn_mask_reformat pipeline.
     */
    @Test
    @DisplayName("Full attn_mask_reformat chain via DSP vs standard")
    void testFullMaskReformatChain() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int seqLen = 16;

        // Sparse attention mask [1, seqLen] — FLOAT
        SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, 1, -1);

        // expandDims → [1,1,1,seqLen]
        SDVariable expanded = sd.expandDims("exp1", mask, 1);
        expanded = sd.expandDims("exp2", expanded, 1);

        // (1 - mask) * MASK_FILL
        SDVariable ones = sd.onesLike("ones", expanded);
        SDVariable inverted = ones.sub("inverted", expanded);
        SDVariable fillConst = sd.constant("fill", Nd4j.scalar(MASK_FILL));
        SDVariable bias = inverted.mul("bias", fillConst);

        // Sum to get scalar output (easy to compare)
        SDVariable result = sd.sum("result", bias);

        // Sparse mask
        INDArray maskArr = Nd4j.zeros(DataType.FLOAT, 1, seqLen);
        for (int i = 0; i < 5; i++) maskArr.putScalar(0, i, 1.0f);
        maskArr.putScalar(0, seqLen - 1, 1.0f);

        Map<String, INDArray> inputs = Map.of("mask", maskArr);

        Map<String, INDArray> expected = sd.output(inputs, "bias", "result");
        INDArray expectedBias = expected.get("bias").dup();
        double expectedSum = expected.get("result").getDouble(0);

        Map<String, INDArray> dspResult = sd.outputDirect(inputs, "bias", "result");
        INDArray dspBias = dspResult.get("bias").dup();
        double dspSum = dspResult.get("result").getDouble(0);

        log.info("Expected bias shape: {}", java.util.Arrays.toString(expectedBias.shape()));
        log.info("DSP bias shape: {}", java.util.Arrays.toString(dspBias.shape()));

        // Check each position
        for (int i = 0; i < seqLen; i++) {
            float expVal = expectedBias.getFloat(0, 0, 0, i);
            float dspVal = dspBias.getFloat(0, 0, 0, i);
            if (Math.abs(expVal - dspVal) > TOL) {
                log.error("Mismatch at position {}: expected={}, dsp={}, mask={}",
                        i, expVal, dspVal, maskArr.getFloat(0, i));
            }
        }

        double biasDiff = expectedBias.sub(dspBias).amaxNumber().doubleValue();
        assertTrue(biasDiff < TOL,
                "Full mask reformat chain: DSP bias should match. MaxDiff=" + biasDiff);

        sd.close();
    }

    /**
     * Test with multiple decode steps — verify sparse mask changes are tracked correctly.
     * This simulates what happens in the actual decode loop where the mask changes each step.
     */
    @Test
    @DisplayName("Multi-step sparse mask via DSP vs standard")
    void testMultiStepSparseMask() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int maxSeqLen = 20;

        SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, 1, maxSeqLen);
        SDVariable ones = sd.onesLike("ones", mask);
        SDVariable inverted = ones.sub("inverted", mask);
        SDVariable fill = sd.constant("fill", Nd4j.scalar(MASK_FILL));
        SDVariable bias = inverted.mul("bias", fill);

        // Simulate 5 decode steps with growing valid region
        for (int step = 0; step < 5; step++) {
            int validPositions = 5 + step;
            INDArray maskArr = Nd4j.zeros(DataType.FLOAT, 1, maxSeqLen);
            for (int i = 0; i < validPositions; i++) maskArr.putScalar(0, i, 1.0f);
            maskArr.putScalar(0, maxSeqLen - 1, 1.0f); // current token

            Map<String, INDArray> inputs = Map.of("mask", maskArr);

            Map<String, INDArray> expected = sd.output(inputs, "bias");
            INDArray expectedBias = expected.get("bias").dup();

            Map<String, INDArray> dspResult = sd.outputDirect(inputs, "bias");
            INDArray dspBias = dspResult.get("bias").dup();

            double maxDiff = expectedBias.sub(dspBias).amaxNumber().doubleValue();
            log.info("Step {}: validPositions={}, maxDiff={}", step, validPositions, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Step " + step + ": DSP bias should match standard. MaxDiff=" + maxDiff);
        }

        sd.close();
    }
}
