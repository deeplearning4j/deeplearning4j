/*
 * Test tensorMmul/tensorDot2 behavior for deconv2d use case
 */
package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for tensorMmul/tensorDot2 operations, specifically for deconv2d use case
 */
public class TensorDot2DeconvTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Test basic tensorMmul contraction - simple 2D case
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBasicTensorMmul2D(Nd4jBackend backend) {
        // Simple test: [2,3] x [3,4] contracting axis 1 of a with axis 0 of b
        // Expected: [2,4]
        INDArray a = Nd4j.ones(2, 3).castTo(DataType.DOUBLE);
        INDArray b = Nd4j.ones(3, 4).castTo(DataType.DOUBLE);
        
        // Contract axis 1 of a with axis 0 of b
        INDArray result = Nd4j.tensorMmul(a, b, new int[][]{{1}, {0}});
        
        assertArrayEquals(new long[]{2, 4}, result.shape());
        // Each element should be 3 (sum of 3 ones)
        assertEquals(3.0, result.getDouble(0, 0), 1e-5);
        assertEquals(3.0, result.getDouble(1, 3), 1e-5);
    }

    /**
     * Test tensorMmul with 4D arrays - deconv2d-like
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorMmul4D(Nd4jBackend backend) {
        // weights[kH,kW,oC,iC] x input[bS,iH,iW,iC]
        // Contract iC (axis 3 of weights, axis 3 of input)
        INDArray weights = Nd4j.ones(1, 1, 3, 8).castTo(DataType.DOUBLE);
        INDArray input = Nd4j.ones(1, 2, 2, 8).castTo(DataType.DOUBLE);
        
        // Contract axis 3 (iC) of weights with axis 3 (iC) of input
        // Expected output shape: [kH,kW,oC,bS,iH,iW] = [1,1,3,1,2,2]
        INDArray result = Nd4j.tensorMmul(weights, input, new int[][]{{3}, {3}});
        
        System.out.println("testTensorMmul4D result shape: " + java.util.Arrays.toString(result.shape()));
        assertArrayEquals(new long[]{1, 1, 3, 1, 2, 2}, result.shape());
        // Each element should be 8 (sum over iC=8)
        assertEquals(8.0, result.getDouble(0, 0, 0, 0, 0, 0), 1e-5);
        assertEquals(8.0, result.getDouble(0, 0, 2, 0, 1, 1), 1e-5);
    }

    /**
     * Test tensorMmul with wFormat=1: weights[oC,iC,kH,kW]
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorMmulWFormat1(Nd4jBackend backend) {
        // weights[oC,iC,kH,kW] x input[bS,iC,iH,iW] (NCHW)
        // Contract iC (axis 1 of weights, axis 1 of input)
        INDArray weights = Nd4j.ones(3, 8, 1, 1).castTo(DataType.DOUBLE);
        INDArray input = Nd4j.ones(1, 8, 2, 2).castTo(DataType.DOUBLE);
        
        // Contract axis 1 (iC) of weights with axis 1 (iC) of input
        // Expected output shape: [oC,kH,kW,bS,iH,iW] = [3,1,1,1,2,2]
        INDArray result = Nd4j.tensorMmul(weights, input, new int[][]{{1}, {1}});
        
        System.out.println("testTensorMmulWFormat1 result shape: " + java.util.Arrays.toString(result.shape()));
        assertArrayEquals(new long[]{3, 1, 1, 1, 2, 2}, result.shape());
        // Each element should be 8 (sum over iC=8)
        assertEquals(8.0, result.getDouble(0, 0, 0, 0, 0, 0), 1e-5);
        assertEquals(8.0, result.getDouble(2, 0, 0, 0, 1, 1), 1e-5);
    }

    /**
     * Test tensorMmul with wFormat=2: weights[oC,kH,kW,iC]
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorMmulWFormat2(Nd4jBackend backend) {
        // weights[oC,kH,kW,iC] x input[bS,iC,iH,iW] (NCHW)
        // Contract iC (axis 3 of weights, axis 1 of input)
        INDArray weights = Nd4j.ones(3, 1, 1, 8).castTo(DataType.DOUBLE);
        INDArray input = Nd4j.ones(1, 8, 2, 2).castTo(DataType.DOUBLE);
        
        // Contract axis 3 (iC) of weights with axis 1 (iC) of input
        // Expected output shape: [oC,kH,kW,bS,iH,iW] = [3,1,1,1,2,2]
        INDArray result = Nd4j.tensorMmul(weights, input, new int[][]{{3}, {1}});
        
        System.out.println("testTensorMmulWFormat2 result shape: " + java.util.Arrays.toString(result.shape()));
        assertArrayEquals(new long[]{3, 1, 1, 1, 2, 2}, result.shape());
        // Each element should be 8 (sum over iC=8)
        assertEquals(8.0, result.getDouble(0, 0, 0, 0, 0, 0), 1e-5);
        assertEquals(8.0, result.getDouble(2, 0, 0, 0, 1, 1), 1e-5);
    }

    /**
     * Test permutation from tensorMmul output to col2im input format
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermutationForCol2Im(Nd4jBackend backend) {
        // wFormat=0: weights[kH,kW,oC,iC], result after tensorMmul: [kH,kW,oC,bS,iH,iW]
        // Want: [bS,oC,kH,kW,iH,iW] for col2im
        INDArray weights = Nd4j.ones(1, 1, 3, 8).castTo(DataType.DOUBLE);
        INDArray input = Nd4j.ones(1, 2, 2, 8).castTo(DataType.DOUBLE);
        
        INDArray result = Nd4j.tensorMmul(weights, input, new int[][]{{3}, {3}});
        
        // Permute from [kH,kW,oC,bS,iH,iW] to [bS,oC,kH,kW,iH,iW]
        INDArray permuted = result.permute(3, 2, 0, 1, 4, 5);
        
        System.out.println("testPermutationForCol2Im permuted shape: " + java.util.Arrays.toString(permuted.shape()));
        assertArrayEquals(new long[]{1, 3, 1, 1, 2, 2}, permuted.shape());
        assertEquals(8.0, permuted.getDouble(0, 0, 0, 0, 0, 0), 1e-5);
    }

    /**
     * Test permutation for wFormat=1
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermutationForCol2ImWFormat1(Nd4jBackend backend) {
        // wFormat=1: weights[oC,iC,kH,kW], result after tensorMmul: [oC,kH,kW,bS,iH,iW]
        // Want: [bS,oC,kH,kW,iH,iW] for col2im
        INDArray weights = Nd4j.ones(3, 8, 1, 1).castTo(DataType.DOUBLE);
        INDArray input = Nd4j.ones(1, 8, 2, 2).castTo(DataType.DOUBLE);
        
        INDArray result = Nd4j.tensorMmul(weights, input, new int[][]{{1}, {1}});
        
        // Permute from [oC,kH,kW,bS,iH,iW] to [bS,oC,kH,kW,iH,iW]
        INDArray permuted = result.permute(3, 0, 1, 2, 4, 5);
        
        System.out.println("testPermutationForCol2ImWFormat1 permuted shape: " + java.util.Arrays.toString(permuted.shape()));
        assertArrayEquals(new long[]{1, 3, 1, 1, 2, 2}, permuted.shape());
        assertEquals(8.0, permuted.getDouble(0, 0, 0, 0, 0, 0), 1e-5);
    }

    /**
     * Test with actual deconv2d dimensions from failing test
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDeconvActualDimensions(Nd4jBackend backend) {
        // mb1_k1_sz8_s1_valid_d1_nchw: weights[1,1,3,8], input[1,8,8,8]
        INDArray weights = Nd4j.ones(1, 1, 3, 8).castTo(DataType.DOUBLE);
        INDArray input = Nd4j.ones(1, 8, 8, 8).castTo(DataType.DOUBLE);
        
        // Contract axis 3 (iC) of weights with axis 3 (iC) of input
        INDArray result = Nd4j.tensorMmul(weights, input, new int[][]{{3}, {3}});
        
        // Expected: [kH,kW,oC,bS,iH,iW] = [1,1,3,1,8,8]
        System.out.println("testDeconvActualDimensions result shape: " + java.util.Arrays.toString(result.shape()));
        assertArrayEquals(new long[]{1, 1, 3, 1, 8, 8}, result.shape());
        
        // Permute to [bS,oC,kH,kW,iH,iW]
        INDArray permuted = result.permute(3, 2, 0, 1, 4, 5);
        System.out.println("testDeconvActualDimensions permuted shape: " + java.util.Arrays.toString(permuted.shape()));
        assertArrayEquals(new long[]{1, 3, 1, 1, 8, 8}, permuted.shape());
        
        // Each element should be 8
        assertEquals(8.0, permuted.getDouble(0, 0, 0, 0, 0, 0), 1e-5);
        assertEquals(8.0, permuted.getDouble(0, 2, 0, 0, 7, 7), 1e-5);
    }
}
