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
import org.junit.jupiter.api.Test;
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

    /**
     * Closing an external wrapper deletes its shell together with the device copy the shell
     * allocated for itself, and never the borrowed host storage. The shell used to be left
     * behind with that copy on every close.
     */
    @Test
    void closingHostOnlyWrapperReleasesItsDeviceCopy() {
        NativeOps ops = singleDeviceOps();
        int device = ops.getDevice();
        final int n = 512;
        LongPointer host = new LongPointer(n);
        OpaqueDataBuffer wrapper = null;
        try {
            for (int i = 0; i < n; i++) host.put(i, 3L * i - 7);
            long[] closesBefore = closeDiagnostics(ops);
            final int rounds = 64;
            for (int round = 0; round < rounds; round++) {
                long before = Nd4j.getEnvironment().getDeviceCounter(device);
                wrapper = ops.dbCreateExternalDataBuffer(n, DataType.LONG.toInt(), host, null);
                assertBuffer(wrapper);
                assertEquals(host.address(), address(ops.dbPrimaryBuffer(wrapper)));
                assertFalse(ops.dbIsOwner(wrapper), "a host-only wrapper owns no device memory yet");
                assertEquals(before, Nd4j.getEnvironment().getDeviceCounter(device));

                ops.dbSyncToSpecial(wrapper);
                assertNotEquals(0, address(ops.dbSpecialBuffer(wrapper)));
                assertTrue(ops.dbIsOwner(wrapper), "the device copy belongs to the wrapper");
                assertEquals(before + (long) n * Long.BYTES, Nd4j.getEnvironment().getDeviceCounter(device));

                ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext()));
                ops.deleteDataBuffer(wrapper);
                wrapper = null;
                assertEquals(before, Nd4j.getEnvironment().getDeviceCounter(device),
                        "closing the wrapper leaked its device copy");
            }
            long[] closesAfter = closeDiagnostics(ops);
            assertTrue(closesAfter[CLOSE_DELETED] - closesBefore[CLOSE_DELETED] >= rounds,
                    "external wrappers were closed without deleting their DataBuffer");

            // The borrowed host storage is still the caller's: intact and writable.
            for (int i = 0; i < n; i++) assertEquals(3L * i - 7, host.get(i));
            host.put(0, 42L);
            assertEquals(42L, host.get(0));
        } finally {
            if (wrapper != null) ops.deleteDataBuffer(wrapper);
            host.close();
        }
    }

    /**
     * {@code dbIsOwner} answers whether closing the wrapper frees the device memory
     * {@code dbSpecialBuffer} reports. DSP dedups device frees by that address, so a wrapper
     * that borrows another buffer's device memory must never claim it.
     */
    @Test
    void onlyWrappersThatFreeDeviceMemoryReportOwnership() {
        NativeOps ops = singleDeviceOps();
        final int n = 16;
        OpaqueDataBuffer source = null;
        OpaqueDataBuffer target = null;
        OpaqueDataBuffer deviceOnly = null;
        OpaqueDataBuffer constant = null;
        OpaqueDataBuffer view = null;
        try {
            source = ops.allocateDataBuffer(n, DataType.LONG.toInt(), true);
            target = ops.allocateDataBuffer(n, DataType.LONG.toInt(), true);
            assertBuffer(source);
            assertBuffer(target);
            long[] expected = new long[n];
            for (int i = 0; i < n; i++) expected[i] = 11L * i + 5;
            new LongPointer(ops.dbPrimaryBuffer(source)).put(expected);
            ops.dbTickHostWrite(source);
            ops.dbSyncToSpecial(source);
            ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext()));
            Pointer special = ops.dbSpecialBuffer(source);
            assertTrue(ops.dbIsOwner(source), "an allocated buffer owns its device memory");

            deviceOnly = ops.dbCreateExternalDataBuffer(n, DataType.LONG.toInt(), null, special);
            assertBuffer(deviceOnly);
            assertEquals(special.address(), address(ops.dbSpecialBuffer(deviceOnly)));
            assertFalse(ops.dbIsOwner(deviceOnly), "a device-only wrapper borrows the source's device memory");
            // Its own host copy does not make it the owner of the borrowed device address.
            ops.dbSyncToPrimary(deviceOnly);
            long[] actual = new long[n];
            new LongPointer(ops.dbPrimaryBuffer(deviceOnly)).get(actual);
            assertArrayEquals(expected, actual);
            assertFalse(ops.dbIsOwner(deviceOnly));

            constant = ops.dbCreateConstantExternalDataBuffer(n, DataType.LONG.toInt(), null, special);
            assertBuffer(constant);
            assertFalse(ops.dbIsOwner(constant), "constants are never freed by close");

            view = ops.dbCreateView(source, n);
            assertBuffer(view);
            assertFalse(ops.dbIsOwner(view), "a view never frees its parent's memory");

            ops.deleteDataBuffer(deviceOnly);
            deviceOnly = null;
            assertTrue(ops.dbSetConstant(constant, false));
            ops.deleteDataBuffer(constant);
            constant = null;
            ops.deleteDataBuffer(view);
            view = null;

            // Closing every borrower left the source's device memory in place and usable.
            assertEquals(special.address(), ops.dbSpecialBuffer(source).address());
            assertTrue(ops.dbIsOwner(source));
            ops.copyBuffer(target, n, source, 0, 0);
            ops.dbSyncToPrimary(target);
            new LongPointer(ops.dbPrimaryBuffer(target)).get(actual);
            assertArrayEquals(expected, actual);
        } finally {
            ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext()));
            if (view != null) ops.deleteDataBuffer(view);
            if (constant != null) {
                ops.dbSetConstant(constant, false);
                ops.deleteDataBuffer(constant);
            }
            if (deviceOnly != null) ops.deleteDataBuffer(deviceOnly);
            if (target != null) ops.deleteDataBuffer(target);
            if (source != null) ops.deleteDataBuffer(source);
        }
    }

    /** Index of the "deleted" counter in {@code dbCloseGetDiagnostics}. */
    private static final int CLOSE_DELETED = 7;

    private static long[] closeDiagnostics(NativeOps ops) {
        try (LongPointer stats = new LongPointer(9)) {
            ops.dbCloseGetDiagnostics(stats);
            long[] out = new long[9];
            stats.get(out);
            return out;
        }
    }

    private static NativeOps singleDeviceOps() {
        assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend")));
        return NativeOpsHolder.getInstance().getDeviceNativeOps();
    }

    private static NativeOps cudaOps() {
        NativeOps ops = singleDeviceOps();
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
