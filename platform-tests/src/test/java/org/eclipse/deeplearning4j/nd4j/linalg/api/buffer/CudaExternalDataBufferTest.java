/* ******************************************************************************
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.parallel.Isolated;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** External pointer shells must not allocate hidden scalar storage or take ownership of borrowed data. */
@Tag("cuda-only")
@Isolated("Changes native thread device and checks exact logical counters")
class CudaExternalDataBufferTest {
    @ParameterizedTest
    @CsvSource({"0,LONG,false", "0,LONG,true", "0,FLOAT,false", "0,FLOAT,true",
            "0,HALF,false", "0,HALF,true", "1,LONG,false", "1,LONG,true",
            "1,FLOAT,false", "1,FLOAT,true", "1,HALF,false", "1,HALF,true"})
    void emptyInteropShellDoesNotAllocate(int device, DataType type, boolean allocateBoth) {
        NativeOps ops = cudaOps();
        int original = ops.getDevice();
        OpaqueDataBuffer shell = null;
        try {
            assertEquals(1, ops.setDevice(device));
            long[] before = counters();
            shell = ops.dbAllocateDataBuffer(0, type.toInt(), allocateBoth);
            assertBuffer(shell);
            assertEquals(0, ops.dbBufferLength(shell));
            assertNullPointer(ops.dbPrimaryBuffer(shell));
            assertNullPointer(ops.dbSpecialBuffer(shell));
            assertArrayEquals(before, counters(), "empty shell charged device memory");
            ops.deleteDataBuffer(shell);
            shell = null;
            assertArrayEquals(before, counters(), "empty shell teardown changed device accounting");
        } finally {
            if (shell != null) ops.deleteDataBuffer(shell);
            ops.setDevice(original);
        }
    }

    @ParameterizedTest
    @CsvSource({"0,false,host", "0,false,device", "0,false,both", "0,true,host", "0,true,device", "0,true,both",
            "1,false,host", "1,false,device", "1,false,both", "1,true,host", "1,true,device", "1,true,both"})
    void externalWrappingPreservesCountersAndBorrowedStorage(int device, boolean constant, String layout) {
        NativeOps ops = cudaOps();
        int original = ops.getDevice();
        OpaqueDataBuffer source = null;
        OpaqueDataBuffer target = null;
        OpaqueDataBuffer wrapper = null;
        try {
            assertEquals(1, ops.setDevice(device));
            source = ops.allocateDataBuffer(4, DataType.LONG.toInt(), true);
            target = ops.allocateDataBuffer(4, DataType.LONG.toInt(), true);
            assertBuffer(source);
            assertBuffer(target);
            Pointer primary = ops.dbPrimaryBuffer(source);
            Pointer special = ops.dbSpecialBuffer(source);
            long[] expected = {7, -5, 12345678901L, 99};
            new LongPointer(primary).put(expected);
            ops.dbTickHostWrite(source);
            ops.dbSyncToSpecial(source);
            ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext()));
            long[] before = counters();
            for (int i = 0; i < 4; i++) {
                Pointer host = "device".equals(layout) ? null : primary;
                Pointer gpu = "host".equals(layout) ? null : special;
                wrapper = constant
                        ? ops.dbCreateConstantExternalDataBuffer(4, DataType.LONG.toInt(), host, gpu)
                        : ops.dbCreateExternalDataBuffer(4, DataType.LONG.toInt(), host, gpu);
                assertBuffer(wrapper);
                assertEquals(4, ops.dbBufferLength(wrapper));
                assertEquals(device, ops.dbDeviceId(wrapper));
                assertEquals(host == null ? 0 : host.address(), address(ops.dbPrimaryBuffer(wrapper)));
                assertEquals(gpu == null ? 0 : gpu.address(), address(ops.dbSpecialBuffer(wrapper)));
                assertArrayEquals(before, counters(), "wrapping borrowed pointers allocated scalar storage");

                // Also exercises the host-only/device-only actuality markers with no caller-side tick on the wrapper.
                ops.copyBuffer(target, 4, wrapper, 0, 0);
                ops.dbSyncToPrimary(target);
                long[] actual = new long[4];
                new LongPointer(ops.dbPrimaryBuffer(target)).get(actual);
                assertArrayEquals(expected, actual);
                if (constant) assertTrue(ops.dbSetConstant(wrapper, false));
                ops.deleteDataBuffer(wrapper);
                wrapper = null;
                assertArrayEquals(before, counters(), "wrapper deletion freed or leaked device storage");
            }
            assertEquals(primary.address(), ops.dbPrimaryBuffer(source).address());
            assertEquals(special.address(), ops.dbSpecialBuffer(source).address());
            // Caller storage remains writable and usable after every wrapper has been deleted.
            expected[0] = -101;
            new LongPointer(primary).put(expected);
            ops.dbTickHostWrite(source);
            ops.dbSyncToSpecial(source);
            ops.copyBuffer(target, 4, source, 0, 0);
            ops.dbSyncToPrimary(target);
            long[] actual = new long[4];
            new LongPointer(ops.dbPrimaryBuffer(target)).get(actual);
            assertArrayEquals(expected, actual);
        } finally {
            ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext()));
            if (wrapper != null) {
                if (constant) ops.dbSetConstant(wrapper, false);
                ops.deleteDataBuffer(wrapper);
            }
            if (target != null) ops.deleteDataBuffer(target);
            if (source != null) ops.deleteDataBuffer(source);
            ops.setDevice(original);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1})
    void oneElementStillAllocatesAndReleasesItsExactCharge(int device) {
        NativeOps ops = cudaOps();
        int original = ops.getDevice();
        OpaqueDataBuffer scalar = null;
        try {
            assertEquals(1, ops.setDevice(device));
            long[] before = counters();
            scalar = ops.allocateDataBuffer(1, DataType.LONG.toInt(), true);
            assertBuffer(scalar);
            assertEquals(1, ops.dbBufferLength(scalar));
            assertNotEquals(0, address(ops.dbPrimaryBuffer(scalar)));
            assertNotEquals(0, address(ops.dbSpecialBuffer(scalar)));
            long[] allocated = before.clone();
            allocated[device] += Long.BYTES;
            assertArrayEquals(allocated, counters());
            ops.deleteDataBuffer(scalar);
            scalar = null;
            assertArrayEquals(before, counters());
        } finally {
            if (scalar != null) ops.deleteDataBuffer(scalar);
            ops.setDevice(original);
        }
    }

    private static NativeOps cudaOps() {
        assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend")));
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        assumeTrue(ops.getAvailableDevices() >= 2, "Needs two CUDA devices");
        return ops;
    }

    private static long[] counters() {
        return new long[]{Nd4j.getEnvironment().getDeviceCounter(0), Nd4j.getEnvironment().getDeviceCounter(1)};
    }

    private static long address(Pointer pointer) {
        return pointer == null ? 0 : pointer.address();
    }

    private static void assertNullPointer(Pointer pointer) {
        assertEquals(0, address(pointer));
    }

    private static void assertBuffer(OpaqueDataBuffer buffer) {
        assertNotNull(buffer);
        assertFalse(buffer.isNull());
    }
}
