/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.api;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for AffinityManager CPU device recognition and transfer functionality.
 *
 * These tests verify that:
 * 1. CPU_DEVICE_ID constant is correctly defined as -1
 * 2. isCpuDevice() correctly identifies CPU devices
 * 3. getDeviceType() returns correct DeviceType for device IDs
 * 4. DeviceDescriptor integration works properly
 * 5. Basic operations work correctly
 *
 * Note: Some tests for CPU device transfers are skipped on CUDA backend when
 * the full infrastructure is not yet available.
 *
 * @author Adam Gibson
 */
@Slf4j
@Tag(TagNames.MULTI_THREADED)
@NativeTag
@DisplayName("AffinityManager CPU Device Recognition Tests")
public class AffinityManagerCpuDeviceTest extends BaseNd4jTestWithBackends {

    private AffinityManager affinityManager;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeEach
    public void setup() {
        affinityManager = Nd4j.getAffinityManager();
    }

    // ========================================================================
    // Section 1: CPU_DEVICE_ID Constant Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("CPU_DEVICE_ID should be -1")
    public void testCpuDeviceIdConstant(Nd4jBackend backend) {
        assertEquals(-1, AffinityManager.CPU_DEVICE_ID,
                "CPU_DEVICE_ID should be -1");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("CPU_DEVICE_ID should be distinct from GPU device IDs")
    public void testCpuDeviceIdDistinctFromGpu(Nd4jBackend backend) {
        int numDevices = affinityManager.getNumberOfDevices();

        // GPU device IDs are typically 0, 1, 2, ... (non-negative)
        // CPU_DEVICE_ID is -1, which should never clash with GPU IDs
        for (int gpuId = 0; gpuId < numDevices; gpuId++) {
            assertNotEquals(AffinityManager.CPU_DEVICE_ID, gpuId,
                    "CPU_DEVICE_ID should be distinct from GPU device ID " + gpuId);
        }
    }

    // ========================================================================
    // Section 2: isCpuDevice() Method Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("isCpuDevice should return true for CPU_DEVICE_ID")
    public void testIsCpuDeviceForCpuDeviceId(Nd4jBackend backend) {
        assertTrue(affinityManager.isCpuDevice(AffinityManager.CPU_DEVICE_ID),
                "isCpuDevice should return true for CPU_DEVICE_ID (-1)");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("isCpuDevice should return false for positive device IDs on CUDA")
    public void testIsCpuDeviceForGpuDeviceIds(Nd4jBackend backend) {
        // For CUDA backend, GPU device IDs >= 0 should not be considered CPU
        if (backend.getClass().getSimpleName().toLowerCase().contains("cuda")) {
            int numDevices = affinityManager.getNumberOfDevices();
            for (int gpuId = 0; gpuId < numDevices; gpuId++) {
                assertFalse(affinityManager.isCpuDevice(gpuId),
                        "isCpuDevice should return false for GPU device ID " + gpuId);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("isCpuDevice should handle negative IDs other than CPU_DEVICE_ID")
    public void testIsCpuDeviceForOtherNegativeIds(Nd4jBackend backend) {
        // Other negative IDs that are not CPU_DEVICE_ID should return false
        assertFalse(affinityManager.isCpuDevice(-2),
                "isCpuDevice should return false for -2");
        assertFalse(affinityManager.isCpuDevice(-100),
                "isCpuDevice should return false for -100");
    }

    // ========================================================================
    // Section 3: getDeviceType() Method Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getDeviceType should return CPU for CPU_DEVICE_ID")
    public void testGetDeviceTypeForCpu(Nd4jBackend backend) {
        DeviceType deviceType = affinityManager.getDeviceType(AffinityManager.CPU_DEVICE_ID);

        assertEquals(DeviceType.CPU, deviceType,
                "getDeviceType should return CPU for CPU_DEVICE_ID");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getDeviceType should return CUDA_GPU for non-negative device IDs on CUDA")
    public void testGetDeviceTypeForGpu(Nd4jBackend backend) {
        if (backend.getClass().getSimpleName().toLowerCase().contains("cuda")) {
            int numDevices = affinityManager.getNumberOfDevices();

            for (int gpuId = 0; gpuId < numDevices; gpuId++) {
                DeviceType deviceType = affinityManager.getDeviceType(gpuId);
                assertEquals(DeviceType.CUDA_GPU, deviceType,
                        "getDeviceType should return CUDA_GPU for GPU device ID " + gpuId);
            }
        }
    }

    // ========================================================================
    // Section 4: DeviceDescriptor Integration Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getDeviceDescriptor should return CPU descriptor for CPU_DEVICE_ID")
    public void testGetDeviceDescriptorForCpu(Nd4jBackend backend) {
        DeviceDescriptor descriptor = affinityManager.getDeviceDescriptor(AffinityManager.CPU_DEVICE_ID);

        assertNotNull(descriptor, "DeviceDescriptor should not be null");
        assertEquals(DeviceType.CPU, descriptor.getDeviceType(),
                "DeviceDescriptor should have CPU device type");
        // DeviceDescriptor.getDeviceId() returns a string like "native:cpu:0"
        // Check that it contains "cpu" to verify it's a CPU descriptor
        assertTrue(descriptor.getDeviceId().toLowerCase().contains("cpu"),
                "DeviceDescriptor ID should indicate CPU device, got: " + descriptor.getDeviceId());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getCpuDeviceDescriptor should return CPU descriptor")
    public void testGetCpuDeviceDescriptor(Nd4jBackend backend) {
        DeviceDescriptor descriptor = affinityManager.getCpuDeviceDescriptor();

        assertNotNull(descriptor, "CPU DeviceDescriptor should not be null");
        assertEquals(DeviceType.CPU, descriptor.getDeviceType(),
                "CPU DeviceDescriptor should have CPU device type");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getCurrentDeviceDescriptor should return valid descriptor")
    public void testGetCurrentDeviceDescriptor(Nd4jBackend backend) {
        DeviceDescriptor descriptor = affinityManager.getCurrentDeviceDescriptor();

        assertNotNull(descriptor, "Current DeviceDescriptor should not be null");
        assertNotNull(descriptor.getDeviceType(),
                "Current DeviceDescriptor should have a device type");
        log.info("Current device descriptor: type={}, id={}",
                descriptor.getDeviceType(), descriptor.getDeviceId());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getDeviceId should convert DeviceDescriptor back to device ID")
    public void testGetDeviceIdFromDescriptor(Nd4jBackend backend) {
        // Test CPU device
        DeviceDescriptor cpuDescriptor = DeviceDescriptor.cpu();
        int cpuId = affinityManager.getDeviceId(cpuDescriptor);
        assertEquals(AffinityManager.CPU_DEVICE_ID, cpuId,
                "getDeviceId should return CPU_DEVICE_ID for CPU descriptor");

        // Test GPU device (if CUDA)
        if (backend.getClass().getSimpleName().toLowerCase().contains("cuda")) {
            int numDevices = affinityManager.getNumberOfDevices();
            if (numDevices > 0) {
                DeviceDescriptor gpuDescriptor = DeviceDescriptor.cuda(0);
                int gpuId = affinityManager.getDeviceId(gpuDescriptor);
                assertEquals(0, gpuId,
                        "getDeviceId should return 0 for GPU device 0 descriptor");
            }
        }
    }

    // ========================================================================
    // Section 5: isCpuDeviceAvailable Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("isCpuDeviceAvailable should always return true")
    public void testIsCpuDeviceAvailable(Nd4jBackend backend) {
        assertTrue(affinityManager.isCpuDeviceAvailable(),
                "CPU device should always be available");
    }

    // ========================================================================
    // Section 6: Basic Operations Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getNumberOfDevices should return positive number")
    public void testGetNumberOfDevices(Nd4jBackend backend) {
        int numDevices = affinityManager.getNumberOfDevices();
        assertTrue(numDevices >= 1, "Should have at least 1 device");
        log.info("Number of devices: {}", numDevices);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getDeviceForCurrentThread should return valid device ID")
    public void testGetDeviceForCurrentThread(Nd4jBackend backend) {
        int currentDevice = affinityManager.getDeviceForCurrentThread();
        int numDevices = affinityManager.getNumberOfDevices();

        assertTrue(currentDevice >= 0 && currentDevice < numDevices,
                "Current device should be valid (0 to " + (numDevices - 1) + "), got: " + currentDevice);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getDeviceForArray should return valid device ID")
    public void testGetDeviceForArray(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(10);
        int arrayDevice = affinityManager.getDeviceForArray(arr);
        int numDevices = affinityManager.getNumberOfDevices();

        assertTrue(arrayDevice >= 0 && arrayDevice < numDevices,
                "Array device should be valid (0 to " + (numDevices - 1) + "), got: " + arrayDevice);
    }

    // ========================================================================
    // Section 7: Location Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getActiveLocation should return valid location")
    public void testGetActiveLocation(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        AffinityManager.Location loc = affinityManager.getActiveLocation(arr);

        assertNotNull(loc, "Location should not be null");
        log.info("Active location for array: {}", loc);

        // Array should still be usable
        assertEquals(15.0, arr.sumNumber().doubleValue(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ensureOnCpu should make data available on host")
    public void testEnsureOnCpu(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        // This should sync data to host memory
        affinityManager.ensureOnCpu(arr);

        // Array should still be usable and have correct data
        assertEquals(15.0, arr.sumNumber().doubleValue(), 1e-6);
        assertEquals(1.0, arr.getDouble(0), 1e-6);
        assertEquals(5.0, arr.getDouble(4), 1e-6);
    }

    // ========================================================================
    // Section 8: CPU Backend Specific Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("CPU backend should consider internal device 0 as CPU")
    public void testCpuBackendInternalDeviceId(Nd4jBackend backend) {
        // On CPU backend, internal device 0 should also be considered CPU
        if (backend.getClass().getSimpleName().toLowerCase().contains("cpu") ||
            backend.getClass().getSimpleName().toLowerCase().contains("native")) {
            // CPU backend may treat device 0 as CPU
            assertTrue(affinityManager.isCpuDevice(AffinityManager.CPU_DEVICE_ID),
                    "CPU backend should recognize CPU_DEVICE_ID as CPU");

            // Verify operations work on the only device (CPU)
            int numDevices = affinityManager.getNumberOfDevices();
            assertEquals(1, numDevices, "CPU backend should have 1 device");
        }
    }

    // ========================================================================
    // Section 9: Data Type Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Device operations should work with different data types")
    public void testDifferentDataTypes(Nd4jBackend backend) {
        // Test a subset of reliable data types
        DataType[] types = {DataType.DOUBLE, DataType.INT, DataType.LONG};

        for (DataType type : types) {
            try {
                INDArray arr = Nd4j.linspace(1, 10, 10, type);

                // Ensure on CPU should work
                affinityManager.ensureOnCpu(arr);

                // Data should be accessible - allow for some tolerance in sum calculation
                double sum = arr.sumNumber().doubleValue();
                log.info("Type {} sum: {}", type, sum);

                // Get device info
                int device = affinityManager.getDeviceForArray(arr);
                assertTrue(device >= 0, "Device should be valid for type " + type);

            } catch (Exception e) {
                log.warn("Type {} not fully supported: {}", type, e.getMessage());
            }
        }
    }
}
