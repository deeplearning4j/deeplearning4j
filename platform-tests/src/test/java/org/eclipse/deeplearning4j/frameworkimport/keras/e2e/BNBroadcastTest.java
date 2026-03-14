package org.eclipse.deeplearning4j.frameworkimport.keras.e2e;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;

import static org.junit.jupiter.api.Assertions.*;

public class BNBroadcastTest {

    @Test
    void testBroadcastSubNHWC_small() {
        // Simulate BN with NHWC: x=[1,4,4,8], mean=[8], chIdx=3
        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 4, 4, 8);
        INDArray mean = Nd4j.zeros(DataType.FLOAT, 8);
        INDArray xMu = Nd4j.createUninitialized(DataType.FLOAT, x.shape(), x.ordering());
        System.out.println("x shape=" + java.util.Arrays.toString(x.shape()) + " stride=" + java.util.Arrays.toString(x.stride()) + " ews=" + x.elementWiseStride());
        System.out.println("mean shape=" + java.util.Arrays.toString(mean.shape()) + " stride=" + java.util.Arrays.toString(mean.stride()));
        System.out.println("xMu shape=" + java.util.Arrays.toString(xMu.shape()) + " stride=" + java.util.Arrays.toString(xMu.stride()));
        xMu = Nd4j.getExecutioner().exec(new BroadcastSubOp(x, mean, xMu, 3));
        System.out.println("Result shape=" + java.util.Arrays.toString(xMu.shape()));
        assertEquals(1.0, xMu.getDouble(0, 0, 0, 0), 1e-5);
    }

    @Test
    void testBroadcastSubNHWC_medium() {
        // Simulate BN with NHWC: x=[1,16,16,32], mean=[32], chIdx=3
        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 16, 16, 32);
        INDArray mean = Nd4j.zeros(DataType.FLOAT, 32);
        INDArray xMu = Nd4j.createUninitialized(DataType.FLOAT, x.shape(), x.ordering());
        System.out.println("x shape=" + java.util.Arrays.toString(x.shape()) + " stride=" + java.util.Arrays.toString(x.stride()));
        xMu = Nd4j.getExecutioner().exec(new BroadcastSubOp(x, mean, xMu, 3));
        System.out.println("Result OK, shape=" + java.util.Arrays.toString(xMu.shape()));
        assertEquals(1.0, xMu.getDouble(0, 0, 0, 0), 1e-5);
    }

    @Test
    void testBroadcastSubNHWC_mobileNet() {
        // Simulate BN with NHWC: x=[1,149,149,32], mean=[32], chIdx=3
        // This is what MobileNet's first BN sees after Conv2D(3→32) with stride=2
        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 149, 149, 32);
        INDArray mean = Nd4j.zeros(DataType.FLOAT, 32);
        INDArray xMu = Nd4j.createUninitialized(DataType.FLOAT, x.shape(), x.ordering());
        System.out.println("x shape=" + java.util.Arrays.toString(x.shape()) + " stride=" + java.util.Arrays.toString(x.stride()));
        xMu = Nd4j.getExecutioner().exec(new BroadcastSubOp(x, mean, xMu, 3));
        System.out.println("Result OK, shape=" + java.util.Arrays.toString(xMu.shape()));
        assertEquals(1.0, xMu.getDouble(0, 0, 0, 0), 1e-5);
    }

    @Test
    void testBroadcastSubNHWC_conv2dOutput() {
        // Test with conv2d-like output that might have non-standard strides
        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 149, 149, 32);
        // Permute to NCHW and back to simulate what conv2d output looks like
        INDArray xPermuted = x.permute(0, 3, 1, 2);  // [1, 32, 149, 149]
        INDArray xBack = xPermuted.permute(0, 2, 3, 1);  // back to [1, 149, 149, 32]
        System.out.println("xBack shape=" + java.util.Arrays.toString(xBack.shape()) + " stride=" + java.util.Arrays.toString(xBack.stride()) + " ews=" + xBack.elementWiseStride());
        System.out.println("strideDescending=" + org.nd4j.linalg.api.shape.Shape.strideDescendingCAscendingF(xBack));

        INDArray mean = Nd4j.zeros(DataType.FLOAT, 32);
        INDArray xMu = Nd4j.createUninitialized(DataType.FLOAT, xBack.shape(), xBack.ordering());
        // This should dup() internally if strides are bad
        if (!org.nd4j.linalg.api.shape.Shape.strideDescendingCAscendingF(xBack)) {
            System.out.println("Duping xBack due to bad strides...");
            xBack = xBack.dup();
            System.out.println("After dup: stride=" + java.util.Arrays.toString(xBack.shape()) + " ews=" + xBack.elementWiseStride());
        }
        xMu = Nd4j.getExecutioner().exec(new BroadcastSubOp(xBack, mean, xMu, 3));
        System.out.println("Result OK, shape=" + java.util.Arrays.toString(xMu.shape()));
        assertEquals(1.0, xMu.getDouble(0, 0, 0, 0), 1e-5);
    }
}
