package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for onnx_multi_head_attention with empty past KV.
 *
 * Hypothesis: When pastKey/pastValue are empty [batch, heads, 0, dim], the C++ op
 * sets pastKey=nullptr and then skips writing present KV outputs (guarded by
 * ownKVFinal=false). This means the first prefill step never populates
 * present_key/present_value, breaking KV cache accumulation in autoregressive decode.
 *
 * The fix must ensure that even when pastKey is empty/null, the current K/V
 * (reshaped from input key/value) is still written to present output slots.
 */
@Slf4j
@org.junit.jupiter.api.Tag(TagNames.ONNX)
public class MhaEmptyPastKvIsolationTest {

    // ==================== Core Bug: Empty past KV → present KV not written ====================

    @Test
    @DisplayName("1: Empty past KV → present KV must contain current K/V")
    public void testEmptyPastKvProducesPresentKv() {
        int batch = 1, seqLen = 4, numHeads = 3, headDim = 64;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // Empty past KV: [1, 3, 0, 64] — the initial prefill state
        INDArray pastKey = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);
        INDArray pastValue = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);

        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, pastKey, pastValue,
                numHeads, 0.0, false));

        assertTrue(outputs.length >= 3, "Should have 3 outputs");

        // Present key must be [1, 3, seqLen, 64] — NOT empty
        INDArray presentKey = outputs[1];
        INDArray presentValue = outputs[2];

        log.info("Present key shape: {}", Arrays.toString(presentKey.shape()));
        log.info("Present value shape: {}", Arrays.toString(presentValue.shape()));

        assertFalse(presentKey.isEmpty(),
                "Present key must NOT be empty — got shape " + Arrays.toString(presentKey.shape()));
        assertFalse(presentValue.isEmpty(),
                "Present value must NOT be empty — got shape " + Arrays.toString(presentValue.shape()));

        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, presentKey.shape(),
                "Present key shape should be [1,3,4,64], got " + Arrays.toString(presentKey.shape()));
        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, presentValue.shape(),
                "Present value shape should be [1,3,4,64], got " + Arrays.toString(presentValue.shape()));

        // Present key/value must contain actual data (not zeros)
        double keyAbsMean = presentKey.ameanNumber().doubleValue();
        double valAbsMean = presentValue.ameanNumber().doubleValue();
        log.info("Present key abs mean: {}, present value abs mean: {}", keyAbsMean, valAbsMean);
        assertTrue(keyAbsMean > 1e-6,
                "Present key should contain data, not zeros. absMean=" + keyAbsMean);
        assertTrue(valAbsMean > 1e-6,
                "Present value should contain data, not zeros. absMean=" + valAbsMean);
    }

    @Test
    @DisplayName("2: No past KV at all → present KV must contain current K/V")
    public void testNoPastKvProducesPresentKv() {
        int batch = 1, seqLen = 4, numHeads = 3, headDim = 64;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // No past KV at all (null)
        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, false));

        assertTrue(outputs.length >= 3, "Should have 3 outputs");

        INDArray presentKey = outputs[1];
        INDArray presentValue = outputs[2];

        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, presentKey.shape(),
                "Present key shape should be [1,3,4,64]");
        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, presentValue.shape(),
                "Present value shape should be [1,3,4,64]");

        double keyAbsMean = presentKey.ameanNumber().doubleValue();
        assertTrue(keyAbsMean > 1e-6, "Present key should contain data. absMean=" + keyAbsMean);
    }

    // ==================== Consistency: empty past vs null past ====================

    @Test
    @DisplayName("3: Empty past KV and null past KV must produce identical results")
    public void testEmptyPastEqualsNullPast() {
        int batch = 1, seqLen = 8, numHeads = 3, headDim = 64;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // With null past
        INDArray[] nullPastOut = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, true));

        // With empty past [1,3,0,64]
        INDArray emptyPastK = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);
        INDArray emptyPastV = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);
        INDArray[] emptyPastOut = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, emptyPastK, emptyPastV,
                numHeads, 0.0, true));

        // Attention output must match
        double attnDiff = nullPastOut[0].sub(emptyPastOut[0]).amaxNumber().doubleValue();
        log.info("Attention output diff (null vs empty past): {}", attnDiff);
        assertTrue(attnDiff < 1e-4,
                "Attention output should be identical for null and empty past. diff=" + attnDiff);

        // Present KV must match
        if (nullPastOut.length >= 3 && emptyPastOut.length >= 3) {
            double keyDiff = nullPastOut[1].sub(emptyPastOut[1]).amaxNumber().doubleValue();
            double valDiff = nullPastOut[2].sub(emptyPastOut[2]).amaxNumber().doubleValue();
            log.info("Present key diff: {}, present value diff: {}", keyDiff, valDiff);
            assertTrue(keyDiff < 1e-4, "Present key should match. diff=" + keyDiff);
            assertTrue(valDiff < 1e-4, "Present value should match. diff=" + valDiff);
        }
    }

    // ==================== Multi-step decode: empty → accumulate ====================

    @Test
    @DisplayName("4: Multi-step decode with empty initial KV cache")
    public void testMultiStepDecodeFromEmptyKv() {
        int batch = 1, numHeads = 3, headDim = 64;
        int hidden = numHeads * headDim;
        int prefillLen = 10;

        // Step 0: Prefill with empty past KV
        INDArray query0 = Nd4j.randn(DataType.FLOAT, batch, prefillLen, hidden);
        INDArray key0 = Nd4j.randn(DataType.FLOAT, batch, prefillLen, hidden);
        INDArray value0 = Nd4j.randn(DataType.FLOAT, batch, prefillLen, hidden);
        INDArray pastK = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);
        INDArray pastV = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);

        INDArray[] out0 = Nd4j.exec(new OnnxMultiHeadAttention(
                query0, key0, value0,
                null, pastK, pastV,
                numHeads, 0.0, true));

        assertFalse(out0[1].isEmpty(),
                "After prefill, present key must not be empty. Shape: " +
                        Arrays.toString(out0[1].shape()));
        assertArrayEquals(new long[]{batch, numHeads, prefillLen, headDim}, out0[1].shape(),
                "Prefill present key should have seq=" + prefillLen);

        // Step 1: Decode single token using prefill's present KV
        INDArray query1 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key1 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray value1 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);

        INDArray[] out1 = Nd4j.exec(new OnnxMultiHeadAttention(
                query1, key1, value1,
                null, out0[1], out0[2],
                numHeads, 0.0, true));

        assertArrayEquals(new long[]{batch, numHeads, prefillLen + 1, headDim}, out1[1].shape(),
                "After decode step 1, present key should have seq=" + (prefillLen + 1));

        // Step 2: Another decode token
        INDArray query2 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key2 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray value2 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);

        INDArray[] out2 = Nd4j.exec(new OnnxMultiHeadAttention(
                query2, key2, value2,
                null, out1[1], out1[2],
                numHeads, 0.0, true));

        assertArrayEquals(new long[]{batch, numHeads, prefillLen + 2, headDim}, out2[1].shape(),
                "After decode step 2, present key should have seq=" + (prefillLen + 2));

        log.info("testMultiStepDecodeFromEmptyKv PASSED: 3 steps verified");
    }

    // ==================== Data correctness: present KV matches input K/V ====================

    @Test
    @DisplayName("5: Present KV data matches reshaped input K/V when past is empty")
    public void testPresentKvDataMatchesInput() {
        int batch = 1, seqLen = 4, numHeads = 2, headDim = 8;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // No past
        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, false));

        INDArray presentKey = outputs[1];

        // The present key should be the input key reshaped from [B,S,H*D] to [B,H,S,D]
        // key [1,4,16] → reshape [1,4,2,8] → permute [1,2,4,8]
        INDArray expectedKey = key.reshape(batch, seqLen, numHeads, headDim)
                .permute(0, 2, 1, 3).dup();

        double maxDiff = expectedKey.sub(presentKey).amaxNumber().doubleValue();
        log.info("Present key vs reshaped input key maxDiff: {}", maxDiff);
        assertTrue(maxDiff < 1e-4,
                "Present key should match reshaped input key. maxDiff=" + maxDiff);
    }

    // ==================== Attention output correctness with empty past ====================

    @Test
    @DisplayName("6: Attention output is non-zero and finite with empty past KV")
    public void testAttentionOutputWithEmptyPast() {
        int batch = 1, seqLen = 16, numHeads = 3, headDim = 64;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray pastK = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);
        INDArray pastV = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);

        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, pastK, pastV,
                numHeads, 0.0, true));

        INDArray attnOut = outputs[0];
        assertArrayEquals(new long[]{batch, seqLen, hidden}, attnOut.shape());
        assertFalse(attnOut.isNaN().any(), "Attention output should not contain NaN");
        assertFalse(attnOut.isInfinite().any(), "Attention output should not contain Inf");

        double absMean = attnOut.ameanNumber().doubleValue();
        assertTrue(absMean > 1e-6, "Attention output should not be all zeros. absMean=" + absMean);
    }

    // ==================== SmolDocling-like config ====================

    @Test
    @DisplayName("7: SmolDocling config (3 heads, 64 dim, 348 prefill tokens)")
    public void testSmolDoclingPrefillConfig() {
        // Matches the actual SmolDocling decoder configuration
        int batch = 1, seqLen = 348, numHeads = 3, headDim = 64;
        int hidden = numHeads * headDim;  // 192

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray pastK = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);
        INDArray pastV = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);

        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, pastK, pastV,
                numHeads, 0.0, true));

        // Present KV must be [1, 3, 348, 64]
        log.info("SmolDocling present key shape: {}", Arrays.toString(outputs[1].shape()));
        assertFalse(outputs[1].isEmpty(),
                "Present key must NOT be empty after 348-token prefill");
        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, outputs[1].shape(),
                "Present key should be [1,3,348,64]");

        // Verify data
        double keyAbsMean = outputs[1].ameanNumber().doubleValue();
        assertTrue(keyAbsMean > 1e-6, "Present key should have data. absMean=" + keyAbsMean);

        // Simulate one decode step
        INDArray q1 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray k1 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray v1 = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);

        INDArray[] out1 = Nd4j.exec(new OnnxMultiHeadAttention(
                q1, k1, v1,
                null, outputs[1], outputs[2],
                numHeads, 0.0, true));

        assertArrayEquals(new long[]{batch, numHeads, seqLen + 1, headDim}, out1[1].shape(),
                "After decode step, present key should be [1,3,349,64]");
        log.info("testSmolDoclingPrefillConfig PASSED");
    }

    // ==================== Incremental correctness: empty-past prefill == no-past prefill ====================

    @Test
    @DisplayName("8: Prefill with empty past matches full-sequence no-past execution")
    public void testPrefillEmptyPastMatchesNoPast() {
        int batch = 1, seqLen = 8, numHeads = 3, headDim = 64;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // Full sequence, no past
        INDArray[] fullOut = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, true));

        // Same sequence, empty past
        INDArray emptyK = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);
        INDArray emptyV = Nd4j.create(DataType.FLOAT, batch, numHeads, 0, headDim);
        INDArray[] emptyPastOut = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, emptyK, emptyV,
                numHeads, 0.0, true));

        // Attention outputs must match exactly
        double attnDiff = fullOut[0].sub(emptyPastOut[0]).amaxNumber().doubleValue();
        assertTrue(attnDiff < 1e-4,
                "Attention output with empty past should match no-past. diff=" + attnDiff);

        // Present KV must match (same shape and data)
        assertArrayEquals(fullOut[1].shape(), emptyPastOut[1].shape(),
                "Present key shapes must match");
        double keyDiff = fullOut[1].sub(emptyPastOut[1]).amaxNumber().doubleValue();
        assertTrue(keyDiff < 1e-4,
                "Present key data with empty past should match no-past. diff=" + keyDiff);
    }
}
