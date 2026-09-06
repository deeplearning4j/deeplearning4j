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

/** Native-only 68 -> 132 byte expansion checks; allocator padding is not a charge. */
@Tag("cuda-only")
@Isolated("Mutates process-wide native device and DEVICE-group limits")
public class CudaDataBufferExpansionAccountingTest {
    private static final int ELEMENTS = 17;
    private static final int EXPANDED_ELEMENTS = 33;
    private static final long BYTES = ELEMENTS * 4L;
    private static final long EXPANDED_BYTES = EXPANDED_ELEMENTS * 4L;
    private static final int DEVICE_GROUP = 10; // memory::MemoryType::DEVICE

    @Test
    void expansionChargesLogicalDeltaAndMigratesAtExactCap() throws Exception {
        NativeOps ops = cudaOps();
        Environment env = Nd4j.getEnvironment();
        int source = ops.getDevice();
        int target = (source + 1) % ops.getAvailableDevices();
        long sourceLimit = env.getDeviceLimit(source);
        long targetLimit = env.getDeviceLimit(target);
        long groupLimit = env.getGroupLimit(DEVICE_GROUP);
        OpaqueDataBuffer buffer = null;
        float[] expected = prefix();

        try (FloatPointer input = new FloatPointer(expected);
             FloatPointer output = new FloatPointer(ELEMENTS)) {
            env.setDeviceLimit(source, 0);
            env.setDeviceLimit(target, 0);
            env.setGroupLimit(DEVICE_GROUP, 0);
            warmDevice(ops, source);
            warmDevice(ops, target);
            long baseSource = env.getDeviceCounter(source);
            long baseTarget = env.getDeviceCounter(target);
            long baseGroup = groupCounter();
            assertTrue(baseSource >= 0 && baseTarget >= 0 && baseGroup >= 0,
                    "Pre-existing negative accounting must not be hidden");

            assertEquals(1, ops.setDevice(source));
            buffer = allocate(ops);
            initialize(ops, buffer, source, input);
            assertState(ops, env, buffer, ELEMENTS, source, source, target,
                    baseSource + BYTES, baseTarget, baseGroup + BYTES, output, expected);

            // Only the logical growth (64 bytes) remains, not a second allocation
            // or an additional 8 bytes of allocator padding.
            env.setDeviceLimit(source, baseSource + EXPANDED_BYTES);
            env.setGroupLimit(DEVICE_GROUP, baseGroup + EXPANDED_BYTES);
            ops.dbExpand(buffer, EXPANDED_ELEMENTS);
            assertState(ops, env, buffer, EXPANDED_ELEMENTS, source, source, target,
                    baseSource + EXPANDED_BYTES, baseTarget, baseGroup + EXPANDED_BYTES, output, expected);

            // Migration must transfer all 132 charged bytes, with no group growth.
            env.setDeviceLimit(target, baseTarget + EXPANDED_BYTES);
            assertEquals(1, ops.setDevice(target));
            ops.dbMigrate(buffer);
            assertState(ops, env, buffer, EXPANDED_ELEMENTS, target, source, target,
                    baseSource, baseTarget + EXPANDED_BYTES, baseGroup + EXPANDED_BYTES, output, expected);
            OpaqueDataBuffer completed = buffer;
            buffer = null; // delete exactly once even if a subsequent assertion fails
            ops.deleteDataBuffer(completed);
            assertEquals(baseSource, env.getDeviceCounter(source), "source after delete");
            assertEquals(baseTarget, env.getDeviceCounter(target), "target after delete");
            assertEquals(baseGroup, groupCounter(), "DEVICE group after delete");
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

    @Test
    void expansionRejectsOneByteShortGroupCapWithoutMutation() throws Exception {
        NativeOps ops = cudaOps();
        Environment env = Nd4j.getEnvironment();
        int source = ops.getDevice();
        int target = (source + 1) % ops.getAvailableDevices();
        long sourceLimit = env.getDeviceLimit(source);
        long targetLimit = env.getDeviceLimit(target);
        long groupLimit = env.getGroupLimit(DEVICE_GROUP);
        OpaqueDataBuffer buffer = null;
        float[] expected = prefix();

        try (FloatPointer input = new FloatPointer(expected);
             FloatPointer output = new FloatPointer(ELEMENTS)) {
            env.setDeviceLimit(source, 0);
            env.setDeviceLimit(target, 0);
            env.setGroupLimit(DEVICE_GROUP, 0);
            warmDevice(ops, source);
            warmDevice(ops, target);
            long baseSource = env.getDeviceCounter(source);
            long baseTarget = env.getDeviceCounter(target);
            long baseGroup = groupCounter();
            assertTrue(baseSource >= 0 && baseTarget >= 0 && baseGroup >= 0,
                    "Pre-existing negative accounting must not be hidden");

            assertEquals(1, ops.setDevice(source));
            buffer = allocate(ops);
            initialize(ops, buffer, source, input);
            assertState(ops, env, buffer, ELEMENTS, source, source, target,
                    baseSource + BYTES, baseTarget, baseGroup + BYTES, output, expected);
            long originalPointer = ops.dbSpecialBuffer(buffer).address();

            // Source has exactly enough logical capacity; the group is one byte
            // short, so moving the allocation to another GPU cannot satisfy it.
            env.setDeviceLimit(source, baseSource + EXPANDED_BYTES);
            env.setGroupLimit(DEVICE_GROUP, baseGroup + EXPANDED_BYTES - 1);
            OpaqueDataBuffer expanding = buffer;
            assertThrows(RuntimeException.class, () -> ops.dbExpand(expanding, EXPANDED_ELEMENTS));
            ops.clearLastError();
            assertEquals(originalPointer, ops.dbSpecialBuffer(buffer).address(), "rejected expansion pointer");
            assertState(ops, env, buffer, ELEMENTS, source, source, target,
                    baseSource + BYTES, baseTarget, baseGroup + BYTES, output, expected);

            // Raising ONLY the group cap by one byte must admit the same request.
            env.setGroupLimit(DEVICE_GROUP, baseGroup + EXPANDED_BYTES);
            ops.dbExpand(buffer, EXPANDED_ELEMENTS);
            assertState(ops, env, buffer, EXPANDED_ELEMENTS, source, source, target,
                    baseSource + EXPANDED_BYTES, baseTarget, baseGroup + EXPANDED_BYTES, output, expected);
            OpaqueDataBuffer completed = buffer;
            buffer = null;
            ops.deleteDataBuffer(completed);
            assertEquals(baseSource, env.getDeviceCounter(source), "source after delete");
            assertEquals(baseTarget, env.getDeviceCounter(target), "target after delete");
            assertEquals(baseGroup, groupCounter(), "DEVICE group after delete");
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

    private static NativeOps cudaOps() {
        assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner()
                .getEnvironmentInformation().getProperty("backend")), "CUDA-only expansion test");
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        assumeTrue(ops.getAvailableDevices() >= 2, "Needs two CUDA devices, with or without peer access");
        return ops;
    }

    private static float[] prefix() {
        float[] values = new float[ELEMENTS];
        for (int i = 0; i < ELEMENTS; i++) values[i] = i - 8.25f;
        return values;
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
            assertEquals(1, ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext())));
        } finally {
            ops.deleteDataBuffer(warm);
        }
    }

    private static void initialize(NativeOps ops, OpaqueDataBuffer buffer, int device, FloatPointer input) {
        assertEquals(device, ops.dbDeviceId(buffer));
        assertEquals(ELEMENTS, ops.dbBufferLength(buffer));
        // allocateSpecial may be async on the context stream; memcpySync uses
        // the per-thread stream. Finish allocation before the explicit copy.
        assertEquals(1, ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext())));
        assertEquals(1, ops.memcpySync(ops.dbSpecialBuffer(buffer), input, BYTES, 1, null));
        ops.dbTickDeviceWrite(buffer);
    }

    private static void assertState(NativeOps ops, Environment env, OpaqueDataBuffer buffer, int elements,
                                    int resident, int source, int target, long sourceBytes, long targetBytes,
                                    long groupBytes, FloatPointer output, float[] expected) throws Exception {
        assertEquals(resident, ops.getDevice(), "caller affinity");
        assertEquals(resident, ops.dbDeviceId(buffer), "allocation device");
        assertEquals(elements, ops.dbBufferLength(buffer), "logical element count");
        assertEquals(sourceBytes, env.getDeviceCounter(source), "source device charge");
        assertEquals(targetBytes, env.getDeviceCounter(target), "target device charge");
        assertEquals(groupBytes, groupCounter(), "global DEVICE charge");
        assertEquals(1, ops.streamSynchronize(ops.lcExecutionStream(ops.defaultLaunchContext())));
        // Only the initialized prefix has defined content after expansion.
        assertEquals(1, ops.memcpySync(output, ops.dbSpecialBuffer(buffer), BYTES, 2, null));
        float[] actual = new float[ELEMENTS];
        output.get(actual);
        assertArrayEquals(expected, actual, 0.0f, "expansion/migration must preserve the old prefix");
    }

    private static long groupCounter() throws Exception {
        // Keep CUDA bindings off the compile-time classpath, as in the migration test.
        Class<?> type = Class.forName("org.nd4j.linalg.jcublas.bindings.Nd4jCuda$Environment");
        Object environment = type.getMethod("getInstance").invoke(null);
        return ((Number) type.getMethod("getGroupCounter", int.class).invoke(environment, DEVICE_GROUP)).longValue();
    }
}
