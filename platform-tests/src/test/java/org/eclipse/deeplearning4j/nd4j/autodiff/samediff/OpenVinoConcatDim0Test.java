package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduces the OpenVINO concat_shape_inference failure on CPU when KV cache
 * has dim=0 on the sequence axis.
 *
 * The SmolDocling decoder does:
 *   past_kv [1,3,0,64] concat new_kv [1,3,seq,64] → [1,3,seq,64]
 *
 * This works on CUDA but fails on CPU because OpenVINO's buildModel rejects
 * the dim=0 during concat shape inference.
 */
public class OpenVinoConcatDim0Test {

    /**
     * Minimal: concat a zero-length array with a non-zero array.
     * This is the raw op test — no SameDiff, just Nd4j.
     */
    @Test
    public void testConcatWithDim0Raw() {
        INDArray pastKv = Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64);
        INDArray newKv = Nd4j.ones(DataType.FLOAT, 1, 3, 1, 64);

        INDArray result = Nd4j.concat(2, pastKv, newKv);
        assertNotNull(result);
        assertArrayEquals(new long[]{1, 3, 1, 64}, result.shape());
        assertEquals(1.0f, result.getFloat(0, 0, 0, 0), 1e-5);
    }

    /**
     * SameDiff graph: concat past_kv (placeholder) with computed new_kv.
     * Step 0: past_kv is [1,3,0,64] (empty KV cache)
     * Step 1: past_kv is [1,3,5,64] (filled KV cache)
     */
    @Test
    public void testConcatDim0SameDiff() {
        SameDiff sd = SameDiff.create();

        SDVariable pastKv = sd.placeHolder("past_kv", DataType.FLOAT, -1, 3, -1, 64);
        SDVariable newKv = sd.placeHolder("new_kv", DataType.FLOAT, 1, 3, 1, 64);
        SDVariable result = sd.concat("result", 2, pastKv, newKv);

        // Step 0: empty KV cache
        INDArray pastKvArr = Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64);
        INDArray newKvArr = Nd4j.ones(DataType.FLOAT, 1, 3, 1, 64);

        Map<String, INDArray> out = sd.output(
                Map.of("past_kv", pastKvArr, "new_kv", newKvArr), "result");
        INDArray res = out.get("result");
        assertNotNull(res);
        assertArrayEquals(new long[]{1, 3, 1, 64}, res.shape());

        // Step 1: filled KV cache
        INDArray pastKvFilled = Nd4j.zeros(DataType.FLOAT, 1, 3, 5, 64);
        out = sd.output(
                Map.of("past_kv", pastKvFilled, "new_kv", newKvArr), "result");
        res = out.get("result");
        assertNotNull(res);
        assertArrayEquals(new long[]{1, 3, 6, 64}, res.shape());
    }

    /**
     * Full decoder-like pattern: matmul → concat(past_kv, new_kv) → matmul.
     * This forces the concat into a multi-op segment that backends try to compile.
     */
    @Test
    public void testDecoderKvConcatPattern() {
        SameDiff sd = SameDiff.create();

        // Simulated attention: query * key^T → concat with past_kv → output projection
        SDVariable query = sd.placeHolder("query", DataType.FLOAT, 1, 3, 1, 64);
        SDVariable keyWeight = sd.placeHolder("key_weight", DataType.FLOAT, 64, 64);
        SDVariable pastKv = sd.placeHolder("past_kv", DataType.FLOAT, -1, 3, -1, 64);

        // Compute new key from query
        // Reshape query [1,3,1,64] → [3,64] for matmul
        SDVariable qFlat = sd.reshape(query, 3, 64);
        SDVariable newKey = sd.mmul("new_key", qFlat, keyWeight);  // [3,64]
        SDVariable newKeyExpanded = sd.reshape(newKey, 1, 3, 1, 64);

        // Concat past + new
        SDVariable fullKey = sd.concat("full_key", 2, pastKv, newKeyExpanded);

        // Output: sum over last dim to produce a scalar per head
        SDVariable output = sd.sum("output", fullKey, 2, 3);

        // Step 0: empty past_kv
        INDArray queryArr = Nd4j.randn(DataType.FLOAT, 1, 3, 1, 64);
        INDArray keyWeightArr = Nd4j.randn(DataType.FLOAT, 64, 64);
        INDArray emptyPastKv = Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64);

        Map<String, INDArray> out = sd.output(
                Map.of("query", queryArr, "key_weight", keyWeightArr, "past_kv", emptyPastKv),
                "output");
        INDArray res = out.get("output");
        assertNotNull(res, "Step 0 failed");

        // Step 1: past_kv filled with previous result
        INDArray filledPastKv = Nd4j.randn(DataType.FLOAT, 1, 3, 5, 64);
        out = sd.output(
                Map.of("query", queryArr, "key_weight", keyWeightArr, "past_kv", filledPastKv),
                "output");
        res = out.get("output");
        assertNotNull(res, "Step 1 failed");
    }

    /**
     * Multi-step decode simulation: run 5 steps with growing KV cache.
     * Tests that DSP handles shape changes across steps.
     */
    @Test
    public void testMultiStepKvGrowth() {
        SameDiff sd = SameDiff.create();

        SDVariable pastKv = sd.placeHolder("past_kv", DataType.FLOAT, -1, 3, -1, 64);
        SDVariable newToken = sd.placeHolder("new_token", DataType.FLOAT, 1, 3, 1, 64);
        SDVariable concatted = sd.concat("concatted", 2, pastKv, newToken);
        // Add a compute op so the segment has multiple ops
        SDVariable scaled = concatted.mul("scaled", 2.0);
        SDVariable output = sd.sum("output", scaled, 3);  // sum over head_dim

        INDArray tokenArr = Nd4j.ones(DataType.FLOAT, 1, 3, 1, 64);

        // Simulate 5 decode steps with growing KV
        for (int step = 0; step < 5; step++) {
            INDArray kvArr = Nd4j.zeros(DataType.FLOAT, 1, 3, step, 64);
            Map<String, INDArray> out = sd.output(
                    Map.of("past_kv", kvArr, "new_token", tokenArr), "output");
            INDArray res = out.get("output");
            assertNotNull(res, "Step " + step + " failed");
            assertEquals(step + 1, res.size(2),
                    "Step " + step + ": expected seq_len=" + (step + 1) + " got " + res.size(2));
        }
    }

    /**
     * Test with DSP explicitly enabled to force backend compilation.
     */
    @Test
    public void testConcatDim0WithDSP() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable pastKv = sd.placeHolder("past_kv", DataType.FLOAT, -1, 3, -1, 64);
        SDVariable newKv = sd.placeHolder("new_kv", DataType.FLOAT, 1, 3, 1, 64);
        SDVariable result = sd.concat("result", 2, pastKv, newKv);
        SDVariable output = result.mul("output", 1.0);  // force non-trivial segment

        // Step 0: dim=0 KV cache
        INDArray emptyKv = Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64);
        INDArray newKvArr = Nd4j.ones(DataType.FLOAT, 1, 3, 1, 64);

        Map<String, INDArray> out = sd.output(
                Map.of("past_kv", emptyKv, "new_kv", newKvArr), "output");
        assertNotNull(out.get("output"), "DSP step 0 with dim=0 failed");
        assertArrayEquals(new long[]{1, 3, 1, 64}, out.get("output").shape());

        // Step 1: filled KV
        INDArray filledKv = Nd4j.randn(DataType.FLOAT, 1, 3, 10, 64);
        out = sd.output(
                Map.of("past_kv", filledKv, "new_kv", newKvArr), "output");
        assertNotNull(out.get("output"), "DSP step 1 with filled KV failed");
        assertArrayEquals(new long[]{1, 3, 11, 64}, out.get("output").shape());
    }
}
