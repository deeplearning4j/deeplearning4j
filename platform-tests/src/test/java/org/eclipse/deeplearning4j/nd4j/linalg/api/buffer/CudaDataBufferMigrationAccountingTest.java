/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import org.bytedeco.javacpp.FloatPointer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Isolated;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Tiny, native-only allocation accounting checks; no NDArray/GC-owned ballast. */
@Tag("cuda-only")
@Isolated("Mutates process-wide native device and DEVICE-group limits")
public class CudaDataBufferMigrationAccountingTest {
    private static final int ELEMENTS = 17;
    private static final long BYTES = ELEMENTS * 4L;
    private static final int DEVICE_GROUP = 10; // memory::MemoryType::DEVICE

    @Test
    void migrationTransfersChargeAndRejectsInsufficientRemainingLimit() throws Exception {
        assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner()
                .getEnvironmentInformation().getProperty("backend")), "CUDA-only migration test");
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        assumeTrue(ops.getAvailableDevices() >= 2, "Needs two CUDA devices, with or without peer access");
        Environment env = Nd4j.getEnvironment();
        int source = ops.getDevice();
        int target = (source + 1) % ops.getAvailableDevices();
        long sourceLimit = env.getDeviceLimit(source);
        long targetLimit = env.getDeviceLimit(target);
        long groupLimit = env.getGroupLimit(DEVICE_GROUP);
        OpaqueDataBuffer buffer = null;
        float[] expected = new float[ELEMENTS];
        for (int i = 0; i < ELEMENTS; i++) expected[i] = i - 8.25f;

        try (FloatPointer input = new FloatPointer(expected);
             FloatPointer output = new FloatPointer(ELEMENTS)) {
            env.setDeviceLimit(source, 0);
            env.setDeviceLimit(target, 0);
            env.setGroupLimit(DEVICE_GROUP, 0);
            // Context/pool lazy initialization must not contaminate the measured deltas.
            warmDevice(ops, source);
            warmDevice(ops, target);
            assumeTrue(ops.getDeviceFreeMemory(source) > BYTES + 8 && ops.getDeviceFreeMemory(target) > BYTES + 8,
                    "Both devices need room for a tiny allocation");
            long baseSource = env.getDeviceCounter(source);
            long baseTarget = env.getDeviceCounter(target);
            long baseGroup = groupCounter();
            assertTrue(baseSource >= 0 && baseTarget >= 0 && baseGroup >= 0,
                    "Pre-existing negative accounting must not be hidden");

            assertEquals(1, ops.setDevice(source));
            buffer = allocate(ops);
            // allocateSpecial may allocate asynchronously on the context stream;
            // NativeOps.memcpySync uses the per-thread stream instead.
            assertEquals(1, ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext())));
            assertEquals(1, ops.memcpySync(ops.dbSpecialBuffer(buffer), input, BYTES, 1, null));
            ops.dbTickDeviceWrite(buffer);
            assertState(ops, env, buffer, source, source, target,
                    baseSource + BYTES, baseTarget, baseGroup + BYTES, output, expected);

            // Leave exactly N bytes, not padded N+8, on both devices; no spare
            // DEVICE-group capacity. All owned migrations must have group delta 0.
            env.setDeviceLimit(source, baseSource + BYTES);
            env.setDeviceLimit(target, baseTarget + BYTES);
            env.setGroupLimit(DEVICE_GROUP, baseGroup + BYTES);
            long originalPointer = ops.dbSpecialBuffer(buffer).address();
            ops.dbSetDeviceId(buffer, target); // stale logical ID, allocation still on source
            ops.dbMigrate(buffer);
            assertEquals(originalPointer, ops.dbSpecialBuffer(buffer).address(), "same-device is a true no-op");
            assertState(ops, env, buffer, source, source, target,
                    baseSource + BYTES, baseTarget, baseGroup + BYTES, output, expected);

            for (int round = 0; round < 3; round++) {
                assertEquals(1, ops.setDevice(target));
                ops.dbMigrate(buffer);
                assertEquals(target, ops.getDevice(), "migration must preserve caller affinity");
                assertState(ops, env, buffer, target, source, target,
                        baseSource, baseTarget + BYTES, baseGroup + BYTES, output, expected);
                long targetPointer = ops.dbSpecialBuffer(buffer).address();
                ops.dbMigrate(buffer);
                assertEquals(targetPointer, ops.dbSpecialBuffer(buffer).address(), "same-device at its cap must not allocate");
                assertEquals(1, ops.setDevice(source));
                ops.dbMigrate(buffer);
                assertState(ops, env, buffer, source, source, target,
                        baseSource + BYTES, baseTarget, baseGroup + BYTES, output, expected);
            }

            // The device has physical room, but only N-1 logical bytes remain.
            // Rejection must be transactional, not a successful failover elsewhere.
            env.setDeviceLimit(target, baseTarget + BYTES - 1);
            long beforeRejectedPointer = ops.dbSpecialBuffer(buffer).address();
            int beforeRejectedId = ops.dbDeviceId(buffer);
            assertEquals(1, ops.setDevice(target));
            OpaqueDataBuffer migrating = buffer;
            RuntimeException rejected = assertThrows(RuntimeException.class, () -> ops.dbMigrate(migrating));
            assertTrue(rejected.getMessage().contains("requested target"), rejected.getMessage());
            assertEquals(target, ops.getDevice(), "failure must restore caller affinity");
            assertEquals(beforeRejectedPointer, ops.dbSpecialBuffer(buffer).address());
            assertEquals(beforeRejectedId, ops.dbDeviceId(buffer));
            ops.clearLastError();
            assertState(ops, env, buffer, source, source, target,
                    baseSource + BYTES, baseTarget, baseGroup + BYTES, output, expected);

            // Raising by ONE byte is sufficient: padding must never be charged.
            env.setDeviceLimit(target, baseTarget + BYTES);
            ops.dbMigrate(buffer);
            assertState(ops, env, buffer, target, source, target,
                    baseSource, baseTarget + BYTES, baseGroup + BYTES, output, expected);
            OpaqueDataBuffer completed = buffer;
            buffer = null; // delete exactly once, even if a cleanup assertion fails
            ops.deleteDataBuffer(completed);
            assertEquals(baseSource, env.getDeviceCounter(source));
            assertEquals(baseTarget, env.getDeviceCounter(target));
            assertEquals(baseGroup, groupCounter());
            assertTrue(env.getDeviceCounter(source) >= 0 && env.getDeviceCounter(target) >= 0);
        } finally {
            try {
                if (buffer != null) ops.deleteDataBuffer(buffer);
            } finally {
                env.setDeviceLimit(source, sourceLimit);
                env.setDeviceLimit(target, targetLimit);
                env.setGroupLimit(DEVICE_GROUP, groupLimit);
                ops.setDevice(source);
            }
        }
    }

    private static OpaqueDataBuffer allocate(NativeOps ops) {
        OpaqueDataBuffer buffer = ops.allocateDataBuffer(ELEMENTS, DataType.FLOAT.toInt(), false);
        assertNotNull(buffer, ops.lastErrorMessage());
        assertFalse(buffer.isNull(), ops.lastErrorMessage());
        return buffer;
    }

    private static void warmDevice(NativeOps ops, int device) {
        assertEquals(1, ops.setDevice(device));
        assertEquals(1, ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext())));
        OpaqueDataBuffer warm = allocate(ops);
        try {
            assertEquals(device, ops.dbDeviceId(warm), "warmup must stay on its requested device");
        } finally {
            ops.deleteDataBuffer(warm);
        }
    }

    private static void assertState(NativeOps ops, Environment env, OpaqueDataBuffer buffer, int resident,
                                    int source, int target, long sourceBytes, long targetBytes, long groupBytes,
                                    FloatPointer output, float[] expected) throws Exception {
        assertEquals(resident, ops.dbDeviceId(buffer));
        assertEquals(sourceBytes, env.getDeviceCounter(source), "source device charge");
        assertEquals(targetBytes, env.getDeviceCounter(target), "target device charge");
        assertEquals(groupBytes, groupCounter(), "global DEVICE charge");
        assertEquals(1, ops.memcpySync(output, ops.dbSpecialBuffer(buffer), BYTES, 2, null));
        float[] actual = new float[ELEMENTS];
        output.get(actual);
        assertArrayEquals(expected, actual, 0.0f, "migration must preserve every element");
    }

    private static long groupCounter() throws Exception {
        // The public Java Environment interface exposes only per-device counters.
        // Use its existing native Environment binding, loaded only after the CUDA
        // assumption so this test still compiles/runs (skipped) on CPU classpaths.
        Class<?> type = Class.forName("org.nd4j.linalg.jcublas.bindings.Nd4jCuda$Environment");
        Object environment = type.getMethod("getInstance").invoke(null);
        return ((Number) type.getMethod("getGroupCounter", int.class).invoke(environment, DEVICE_GROUP)).longValue();
    }
}
