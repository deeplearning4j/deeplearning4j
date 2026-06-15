package org.eclipse.deeplearning4j.frameworkimport.keras.layers.normalization;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation test for BatchNorm broadcast subtract crash with Conv1D output shapes.
 */
@NativeTag
class KerasBatchNormIsolationTest extends BaseDL4JTest {

    @Test
    void testBroadcastSubWithReshapedRank3() {
        // Simulate what BatchNormalization.preOutput does for Conv1D output:
        // Conv1D output: [25, 24, 10] (batch=25, width=24, channels=10)
        INDArray conv1dOutput = Nd4j.ones(DataType.FLOAT, 25, 24, 10);

        // BatchNorm reshapes rank 3 -> rank 4 by prepending 1
        INDArray x = conv1dOutput.reshape(1, 25, 24, 10);
        System.out.println("x shape: " + java.util.Arrays.toString(x.shape()));
        System.out.println("x strides: " + java.util.Arrays.toString(x.stride()));
        System.out.println("x ordering: " + x.ordering());

        // Global mean for NHWC batch norm (chIdx=3)
        INDArray mean = Nd4j.zeros(DataType.FLOAT, 10);
        System.out.println("mean shape: " + java.util.Arrays.toString(mean.shape()));

        // This is the operation that crashes
        INDArray xMu = Nd4j.createUninitialized(DataType.FLOAT, x.shape(), x.ordering());
        System.out.println("xMu shape: " + java.util.Arrays.toString(xMu.shape()));

        int chIdx = 3; // NHWC
        xMu = Nd4j.getExecutioner().exec(new BroadcastSubOp(x, mean, xMu, chIdx));

        System.out.println("Broadcast subtract succeeded");
        assertArrayEquals(new long[]{1, 25, 24, 10}, xMu.shape());
    }

    @Test
    void testBroadcastSubWithDup() {
        // Same test but with dup() to ensure contiguous memory
        INDArray conv1dOutput = Nd4j.ones(DataType.FLOAT, 25, 24, 10);
        INDArray x = conv1dOutput.reshape(1, 25, 24, 10).dup();
        System.out.println("x shape: " + java.util.Arrays.toString(x.shape()));
        System.out.println("x strides: " + java.util.Arrays.toString(x.stride()));

        INDArray mean = Nd4j.zeros(DataType.FLOAT, 10);
        INDArray xMu = Nd4j.createUninitialized(DataType.FLOAT, x.shape(), x.ordering());
        int chIdx = 3;
        xMu = Nd4j.getExecutioner().exec(new BroadcastSubOp(x, mean, xMu, chIdx));

        System.out.println("Broadcast subtract with dup succeeded");
        assertArrayEquals(new long[]{1, 25, 24, 10}, xMu.shape());
    }

    @Test
    void testBroadcastSubWithFreshCreate() {
        // Skip reshape entirely - create fresh 4D array
        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 25, 24, 10);
        System.out.println("x shape: " + java.util.Arrays.toString(x.shape()));
        System.out.println("x strides: " + java.util.Arrays.toString(x.stride()));

        INDArray mean = Nd4j.zeros(DataType.FLOAT, 10);
        INDArray xMu = Nd4j.createUninitialized(DataType.FLOAT, x.shape(), x.ordering());
        int chIdx = 3;
        xMu = Nd4j.getExecutioner().exec(new BroadcastSubOp(x, mean, xMu, chIdx));

        System.out.println("Broadcast subtract with fresh create succeeded");
        assertArrayEquals(new long[]{1, 25, 24, 10}, xMu.shape());
    }

    @Test
    void testBroadcastSubWithAppend() {
        // Append 1 instead of prepend: [25, 24, 10] -> [25, 24, 10, 1]
        INDArray conv1dOutput = Nd4j.ones(DataType.FLOAT, 25, 24, 10);
        INDArray x = conv1dOutput.reshape(25, 24, 10, 1);
        System.out.println("x shape: " + java.util.Arrays.toString(x.shape()));
        System.out.println("x strides: " + java.util.Arrays.toString(x.stride()));

        INDArray mean = Nd4j.zeros(DataType.FLOAT, 1);
        INDArray xMu = Nd4j.createUninitialized(DataType.FLOAT, x.shape(), x.ordering());
        int chIdx = 3;
        xMu = Nd4j.getExecutioner().exec(new BroadcastSubOp(x, mean, xMu, chIdx));

        System.out.println("Broadcast subtract with append succeeded");
        assertArrayEquals(new long[]{25, 24, 10, 1}, xMu.shape());
    }
}
