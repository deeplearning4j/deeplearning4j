package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueNDArray;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that INT64 buffers have correct device data on CUDA.
 * Reproduces the issue where Reshape reads zeros from device buffer
 * despite host buffer having correct values.
 */
public class CudaInt64BufferSyncTest {

    @Test
    public void testInt64BufferDeviceSync() {
        long[] values = new long[]{-1, 3, 512, 512};

        // Create buffer via Nd4j.createBuffer(long[])
        DataBuffer buffer = Nd4j.createBuffer(values);
        INDArray arr = Nd4j.create(buffer, new long[]{4}, new long[]{1}, 0L, 'c');

        System.out.println("Created INT64 array: " + arr);
        System.out.println("  shape: " + java.util.Arrays.toString(arr.shape()));
        System.out.println("  dtype: " + arr.dataType());
        System.out.println("  host values: " + java.util.Arrays.toString(arr.data().asLong()));

        // Force device -> host sync via the C++ DataBuffer (simulates what getBufferAsVector does)
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbForceSyncToPrimary(arr.data().opaqueBuffer());

        long[] afterSync = arr.data().asLong();
        System.out.println("  after forced D2H sync: " + java.util.Arrays.toString(afterSync));

        assertArrayEquals(values, afterSync, "INT64 values should survive H2D->D2H round trip");
    }

    @Test
    public void testInt64ScalarDeviceSync() {
        // This is the path used for ONNX scalar constants (Nd4j.scalar(long))
        long[] testValues = new long[]{-1, 3, 512, 512};

        for (long val : testValues) {
            INDArray scalar = Nd4j.scalar(val);
            System.out.println("Scalar " + val + ": dtype=" + scalar.dataType() + ", shape=" +
                    java.util.Arrays.toString(scalar.shape()));

            // Force D2H sync
            NativeOpsHolder.getInstance().getDeviceNativeOps().dbForceSyncToPrimary(scalar.data().opaqueBuffer());

            long readBack = scalar.getLong(0);
            System.out.println("  after forced D2H sync: " + readBack);
            assertEquals(val, readBack, "Scalar value " + val + " should survive H2D->D2H round trip");
        }
    }

    @Test
    public void testConcatOfInt64Scalars() {
        // Reproduce the actual Concat execution path
        INDArray s1 = Nd4j.scalar(DataType.LONG, -1);
        INDArray s2 = Nd4j.scalar(DataType.LONG, 3);
        INDArray s3 = Nd4j.scalar(DataType.LONG, 512);
        INDArray s4 = Nd4j.scalar(DataType.LONG, 512);

        System.out.println("Before concat:");
        System.out.println("  s1=" + s1.getLong(0) + " s2=" + s2.getLong(0) +
                " s3=" + s3.getLong(0) + " s4=" + s4.getLong(0));

        // Reshape scalars to [1] before concat (Concat requires same rank)
        INDArray r1 = s1.reshape(1);
        INDArray r2 = s2.reshape(1);
        INDArray r3 = s3.reshape(1);
        INDArray r4 = s4.reshape(1);

        // Execute concat via Nd4j (which goes through C++ op)
        INDArray result = Nd4j.concat(0, r1, r2, r3, r4);

        System.out.println("Concat result: " + java.util.Arrays.toString(result.data().asLong()));
        System.out.println("  shape: " + java.util.Arrays.toString(result.shape()));

        // Now simulate what reshape does: read via forced D2H sync
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbForceSyncToPrimary(result.data().opaqueBuffer());

        long[] readBack = result.data().asLong();
        System.out.println("  after forced D2H: " + java.util.Arrays.toString(readBack));

        assertArrayEquals(new long[]{-1, 3, 512, 512}, readBack, "Concat should produce correct INT64 values");
    }

    @Test
    public void testInt64BufferOpaqueNDArraySync() {
        // Test the actual OpaqueNDArray.fromINDArray path
        long[] values = new long[]{-1, 3, 512, 512};
        INDArray arr = Nd4j.createFromArray(values);

        System.out.println("Before OpaqueNDArray: " + java.util.Arrays.toString(arr.data().asLong()));

        // This is what CudaExecutioner does before passing to C++ ops
        OpaqueNDArray opaque = OpaqueNDArray.fromINDArray(arr);

        // Now force D2H to check what the C++ side sees
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbForceSyncToPrimary(arr.data().opaqueBuffer());

        long[] afterOpaque = arr.data().asLong();
        System.out.println("After OpaqueNDArray sync: " + java.util.Arrays.toString(afterOpaque));

        assertArrayEquals(values, afterOpaque, "Values should be correct after OpaqueNDArray creation");
    }
}
