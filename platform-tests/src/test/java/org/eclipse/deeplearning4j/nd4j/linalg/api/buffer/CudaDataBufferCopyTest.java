/* ******************************************************************************
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import org.bytedeco.javacpp.FloatPointer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.parallel.Isolated;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Raw copies must preserve offsets, source actuality and destination device ownership. */
@Tag("cuda-only")
@Isolated("Changes native thread device")
public class CudaDataBufferCopyTest {
    @ParameterizedTest
    @CsvSource({"0,1,false", "1,0,false", "0,0,false", "1,1,false",
            "0,1,true", "1,0,true", "0,0,true", "1,1,true"})
    void specialCopyPreservesOffsetsAcrossDevices(int sourceDevice, int targetDevice, boolean callerOnSource) {
        copy(sourceDevice, targetDevice, false, callerOnSource);
    }

    @ParameterizedTest
    @CsvSource({"0,1,false", "1,0,false", "0,0,false", "1,1,false",
            "0,1,true", "1,0,true", "0,0,true", "1,1,true"})
    void primaryCopyUsesHostPointer(int sourceDevice, int targetDevice, boolean callerOnSource) {
        copy(sourceDevice, targetDevice, true, callerOnSource);
    }

    private void copy(int sourceDevice, int targetDevice, boolean primaryOnly, boolean callerOnSource) {
        assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner()
                .getEnvironmentInformation().getProperty("backend")));
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        assumeTrue(ops.getAvailableDevices() >= 2, "Needs two CUDA devices");
        int original = ops.getDevice();
        OpaqueDataBuffer source = null;
        OpaqueDataBuffer target = null;
        try {
            assertEquals(1, ops.setDevice(sourceDevice));
            source = ops.allocateDataBuffer(17, DataType.FLOAT.toInt(), true);
            assertNotNull(source);
            assertFalse(source.isNull());
            assertNotNull(ops.dbSpecialBuffer(source));
            assertFalse(ops.dbSpecialBuffer(source).isNull());
            float[] input = new float[17];
            for (int i = 0; i < input.length; i++) input[i] = i - 8.25f;
            new FloatPointer(ops.dbPrimaryBuffer(source)).put(input);
            ops.dbTickHostWrite(source);
            if (!primaryOnly) ops.dbSyncToSpecial(source);

            assertEquals(1, ops.setDevice(targetDevice));
            target = ops.allocateDataBuffer(19, DataType.FLOAT.toInt(), true);
            assertNotNull(target);
            assertFalse(target.isNull());
            assertFalse(ops.dbSpecialBuffer(target).isNull());
            float[] expected = new float[19];
            Arrays.fill(expected, -99f);
            new FloatPointer(ops.dbPrimaryBuffer(target)).put(expected);
            ops.dbTickHostWrite(target);
            // Host-input cases also exercise untouched primary-actual destination bytes.
            if (!primaryOnly) ops.dbSyncToSpecial(target);
            long address = ops.dbSpecialBuffer(target).address();
            int callerDevice = callerOnSource ? sourceDevice : targetDevice;
            assertEquals(1, ops.setDevice(callerDevice));

            // No host barrier between producer and copy: native write events must order them.
            ops.copyBuffer(target, 11, source, 2, 3);
            assertEquals(callerDevice, ops.getDevice());
            assertEquals(targetDevice, ops.dbDeviceId(target));
            assertEquals(sourceDevice, ops.dbDeviceId(source));
            assertEquals(address, ops.dbSpecialBuffer(target).address());
            ops.dbSyncToPrimary(target);
            System.arraycopy(input, 2, expected, 3, 11);
            float[] actual = new float[19];
            new FloatPointer(ops.dbPrimaryBuffer(target)).get(actual);
            assertArrayEquals(expected, actual, 0f);
        } finally {
            ops.clearLastError();
            if (target != null) {
                ops.setDevice(targetDevice);
                ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext()));
                ops.deleteDataBuffer(target);
            }
            if (source != null) {
                ops.setDevice(sourceDevice);
                ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext()));
                ops.deleteDataBuffer(source);
            }
            ops.setDevice(original);
        }
    }
}
