package org.nd4lj.linalg;

import org.junit.Test;
import org.nd4lj.linalg.factory.Nd4j;

public class GPUMatMulTest {
    @Test
    public void testGPUMatMul() {
        System.out.println("Testing GPU matrix multiplication...");
        INDArray a = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray b = Nd4j.create(new float[]{5, 6, 7, 8}, new int[]{2, 2});
        INDArray c = a.mmul(b); // Matrix multiplication
        System.out.println("Result: " + c);
    }
}