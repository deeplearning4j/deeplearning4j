package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validate that permute operations produce correct strides and data
 * when executed through the DSP execution path. This catches regressions
 * in the per-op view descriptor and trait-driven view execution contract.
 */
@Slf4j
public class TestPermuteViewStrides {

    @Test
    @DisplayName("permute [0,2,1] on [1,3,4] should transpose last two dims")
    public void testPermute021() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 3, 4);
        SDVariable permuted = sd.permute("output", input, 0, 2, 1);

        // Input: [1,3,4] with sequential values
        INDArray inputArr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(1, 3, 4);
        Map<String, INDArray> result = sd.output(Map.of("input", inputArr), "output");
        INDArray output = result.get("output");

        log.info("Input shape: {} strides: {}", Arrays.toString(inputArr.shape()), Arrays.toString(inputArr.stride()));
        log.info("Output shape: {} strides: {}", Arrays.toString(output.shape()), Arrays.toString(output.stride()));

        // Output should be [1,4,3]
        assertArrayEquals(new long[]{1, 4, 3}, output.shape(), "Permuted shape should be [1,4,3]");

        // Verify actual values: output[0,j,i] == input[0,i,j]
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                float expected = inputArr.getFloat(0, i, j);
                float actual = output.getFloat(0, j, i);
                assertEquals(expected, actual, 1e-6,
                        String.format("output[0,%d,%d] should equal input[0,%d,%d]", j, i, i, j));
            }
        }
        log.info("permute [0,2,1]: PASS");
    }

    @Test
    @DisplayName("permute [0,2,1,3] on [1,heads,seq,dim] (attention-like)")
    public void testPermuteAttention() {
        SameDiff sd = SameDiff.create();
        // Simulate attention: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 3, 5, 4);
        SDVariable permuted = sd.permute("output", input, 0, 2, 1, 3);

        INDArray inputArr = Nd4j.linspace(1, 60, 60, DataType.FLOAT).reshape(1, 3, 5, 4);
        Map<String, INDArray> result = sd.output(Map.of("input", inputArr), "output");
        INDArray output = result.get("output");

        log.info("Attention permute: input shape={} -> output shape={}",
                Arrays.toString(inputArr.shape()), Arrays.toString(output.shape()));

        assertArrayEquals(new long[]{1, 5, 3, 4}, output.shape(),
                "Permuted attention shape should be [1,5,3,4]");

        // Verify: output[0,s,h,d] == input[0,h,s,d]
        for (int h = 0; h < 3; h++) {
            for (int s = 0; s < 5; s++) {
                for (int d = 0; d < 4; d++) {
                    float expected = inputArr.getFloat(0, h, s, d);
                    float actual = output.getFloat(0, s, h, d);
                    assertEquals(expected, actual, 1e-6,
                            String.format("Mismatch at [0,%d,%d,%d]", s, h, d));
                }
            }
        }
        log.info("permute attention [0,2,1,3]: PASS");
    }

    @Test
    @DisplayName("permute with negative indices [-1, -2] equivalent to [1, 0] on [3,4]")
    public void testPermuteNegativeIndices() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 3, 4);
        // -1 = last dim (1), -2 = second-to-last dim (0)
        // This should be equivalent to permute(1, 0) = transpose
        SDVariable permuted = sd.permute("output", input, -1, -2);

        INDArray inputArr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);
        Map<String, INDArray> result = sd.output(Map.of("input", inputArr), "output");
        INDArray output = result.get("output");

        assertArrayEquals(new long[]{4, 3}, output.shape(), "Permuted shape should be [4,3]");

        // Verify transpose: output[j,i] == input[i,j]
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                float expected = inputArr.getFloat(i, j);
                float actual = output.getFloat(j, i);
                assertEquals(expected, actual, 1e-6,
                        String.format("output[%d,%d] should equal input[%d,%d]", j, i, i, j));
            }
        }
        log.info("permute negative indices: PASS");
    }

    @Test
    @DisplayName("permute followed by matmul (common attention pattern)")
    public void testPermuteMatmul() {
        SameDiff sd = SameDiff.create();
        // Q: [1,heads,seq_q,dim], K: [1,heads,seq_k,dim]
        // After permuting K to [1,heads,dim,seq_k], matmul Q@K^T = [1,heads,seq_q,seq_k]
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 2, 3, 4);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, 2, 5, 4);
        SDVariable kT = sd.permute("kT", k, 0, 1, 3, 2);  // [1,2,4,5]
        SDVariable attn = sd.mmul("attn", q, kT);  // [1,2,3,5]

        INDArray qArr = Nd4j.rand(DataType.FLOAT, 1, 2, 3, 4);
        INDArray kArr = Nd4j.rand(DataType.FLOAT, 1, 2, 5, 4);

        Map<String, INDArray> result = sd.output(
                Map.of("q", qArr, "k", kArr), "attn", "kT");
        INDArray attnOut = result.get("attn");
        INDArray kTOut = result.get("kT");

        log.info("Q shape: {}", Arrays.toString(qArr.shape()));
        log.info("K shape: {}", Arrays.toString(kArr.shape()));
        log.info("K^T shape: {}", Arrays.toString(kTOut.shape()));
        log.info("Attention shape: {}", Arrays.toString(attnOut.shape()));

        assertArrayEquals(new long[]{1, 2, 4, 5}, kTOut.shape(), "K^T shape should be [1,2,4,5]");
        assertArrayEquals(new long[]{1, 2, 3, 5}, attnOut.shape(), "Attention shape should be [1,2,3,5]");
        assertFalse(attnOut.isNaN().any(), "Attention output should not contain NaN");
        assertFalse(attnOut.isInfinite().any(), "Attention output should not contain Inf");

        // Verify manually for head=0: attn[0,0] = q[0,0] @ k[0,0]^T
        INDArray q0 = qArr.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0));  // [3,4]
        INDArray k0 = kArr.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0));  // [5,4]
        INDArray expected = q0.mmul(k0.transpose());  // [3,5]
        INDArray actual = attnOut.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0));  // [3,5]

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Permute+matmul max diff from reference: {}", maxDiff);
        assertTrue(maxDiff < 1e-4, "Permute+matmul should match manual computation, maxDiff=" + maxDiff);

        log.info("permute+matmul attention pattern: PASS");
    }
}
