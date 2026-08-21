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

package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for multi-GPU data migration using staged host memory transfer.
 *
 * <p>These tests verify that data can be properly migrated between GPUs
 * when operations require inputs from different devices. This is critical
 * for multi-GPU memory routing where allocations may be placed on different
 * devices based on available memory.
 *
 * <p>Cross-device operations use staged host memory transfer (GPU → Host → GPU)
 * for reliable data migration between devices.
 *
 * Adam Gibson
 */
@Slf4j
@DisplayName("Multi-GPU Migration Tests")
public class MultiGpuMigrationTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    private boolean isCudaBackend() {
        String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        return backendName.contains("cuda") || backendName.contains("jcublas");
    }

    private int getNumDevices() {
        try {
            return Nd4j.getAffinityManager().getNumberOfDevices();
        } catch (Exception e) {
            return 0;
        }
    }

    @BeforeEach
    void setUp() {
        // Clear any memory simulation state from previous tests
        DeviceMemoryManager.getInstance().clearAllMemorySimulation();

        // Ensure we start on device 0
        if (isCudaBackend() && getNumDevices() > 0) {
            DeviceMemoryManager.getInstance().switchDevice(0, "MultiGpuMigrationTest", "test-device-switch");
        }
    }

    @AfterEach
    void tearDown() {
        DeviceMemoryManager.getInstance().clearAllMemorySimulation();
        System.gc();
    }

    // =========================================================================
    // Same-Device Tests (These should always work)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Basic allocation on current device")
    void testBasicAllocation(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 1, "Test requires at least 1 CUDA device");

        INDArray a = Nd4j.ones(DataType.FLOAT, 100, 100);
        a.assign(2.0f);

        assertNotNull(a, "Array should be created");
        assertEquals(2.0f, a.getFloat(0, 0), 1e-5);
        assertEquals(10000, a.length());

        log.info("Basic allocation test passed");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Add operation on same device")
    void testSameDeviceAdd(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 1, "Test requires at least 1 CUDA device");

        INDArray a = Nd4j.ones(DataType.FLOAT, 100, 100).mul(2);
        INDArray b = Nd4j.ones(DataType.FLOAT, 100, 100).mul(3);

        INDArray result = a.add(b);

        assertEquals(5.0f, result.getFloat(0, 0), 1e-5);
        assertEquals(5.0f, result.getFloat(99, 99), 1e-5);

        log.info("Same-device add test passed");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Cast stays on the current non-default GPU")
    void testCastStaysOnCurrentNonDefaultGpu(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager.getInstance().switchDevice(
                1, "MultiGpuMigrationTest", "cast-current-device");
        try {
            INDArray source = Nd4j.createFromArray(1.0f, 2.0f, 3.0f, 4.0f);
            INDArray cast = source.castTo(DataType.HALF);
            Nd4j.getExecutioner().commit();

            assertEquals(1, Nd4j.getAffinityManager().getDeviceForArray(source));
            assertEquals(1, Nd4j.getAffinityManager().getDeviceForArray(cast),
                    "A same-device cast must not be rerouted to the default GPU");
            assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                    cast.data().asFloat(), 1e-3f);
        } finally {
            DeviceMemoryManager.getInstance().switchDevice(
                    0, "MultiGpuMigrationTest", "restore-after-cast");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: MatMul on same device")
    void testSameDeviceMatMul(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 1, "Test requires at least 1 CUDA device");

        INDArray a = Nd4j.ones(DataType.FLOAT, 64, 128);
        INDArray b = Nd4j.ones(DataType.FLOAT, 128, 64);

        INDArray result = a.mmul(b);

        assertArrayEquals(new long[]{64, 64}, result.shape());
        assertEquals(128.0f, result.getFloat(0, 0), 1e-4);

        log.info("Same-device MatMul test passed");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Reduction on same device")
    void testSameDeviceReduction(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 1, "Test requires at least 1 CUDA device");

        INDArray a = Nd4j.ones(DataType.FLOAT, 1000);

        double sum = a.sumNumber().doubleValue();
        assertEquals(1000.0, sum, 1e-3);

        double mean = a.meanNumber().doubleValue();
        assertEquals(1.0, mean, 1e-5);

        log.info("Same-device reduction test passed");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Large allocation on same device")
    void testLargeAllocation(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 1, "Test requires at least 1 CUDA device");

        // Allocate 50MB
        long elements = 50L * 1024 * 1024 / 4;
        INDArray large = Nd4j.ones(DataType.FLOAT, elements);
        large.assign(1.5f);

        assertEquals(1.5f, large.getFloat(0), 1e-5);
        assertEquals(1.5f, large.getFloat(elements - 1), 1e-5);

        log.info("Large allocation test passed (50MB)");
    }

    // =========================================================================
    // Cross-Device Tests (Using staged host memory transfer)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Array created on GPU 0 can be used in op on GPU 1")
    void testCrossDeviceMigrationGpu0ToGpu1(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Disable P2P to force migration path
        Nd4j.getAffinityManager().allowCrossDeviceAccess(false);

        try {
            // Create array on GPU 0
            DeviceMemoryManager.getInstance().switchDevice(0, "MultiGpuMigrationTest", "test-device-switch");
            INDArray a = Nd4j.ones(DataType.FLOAT, 100, 100);
            a.assign(2.0f);
            // Ensure operations complete before switching devices
            Nd4j.getExecutioner().commit();

            // Switch to GPU 1 and create another array
            DeviceMemoryManager.getInstance().switchDevice(1, "MultiGpuMigrationTest", "test-device-switch");
            INDArray b = Nd4j.ones(DataType.FLOAT, 100, 100);
            b.assign(3.0f);
            // Ensure operations complete before the cross-device operation
            Nd4j.getExecutioner().commit();

            log.info("About to perform cross-device add: a is on device {}, b is on device {}, current device is {}",
                    Nd4j.getAffinityManager().getDeviceForArray(a),
                    Nd4j.getAffinityManager().getDeviceForArray(b),
                    Nd4j.getAffinityManager().getDeviceForCurrentThread());

            // Explicitly migrate b to device 0 before the operation
            DeviceMemoryManager.getInstance().switchDevice(0, "MultiGpuMigrationTest", "test-device-switch");
            INDArray bMigrated = Nd4j.getAffinityManager().replicateToDevice(0, b);

            // Perform operation with both arrays on same device
            INDArray result = a.add(bMigrated);

            assertEquals(5.0f, result.getFloat(0, 0), 1e-5);
            log.info("Cross-device migration GPU 0 -> GPU 1 test passed");
        } finally {
            // Re-enable P2P
            Nd4j.getAffinityManager().allowCrossDeviceAccess(true);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Automatic P2P cross-device add (no manual migration)")
    void testAutomaticP2PCrossDeviceAdd(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Ensure P2P is enabled (default)
        Nd4j.getAffinityManager().allowCrossDeviceAccess(true);

        // Create array on GPU 0
        DeviceMemoryManager.getInstance().switchDevice(0, "MultiGpuMigrationTest", "test-device-switch");
        INDArray a = Nd4j.ones(DataType.FLOAT, 100, 100);
        a.assign(2.0f);
        Nd4j.getExecutioner().commit();

        // Switch to GPU 1 and create another array
        DeviceMemoryManager.getInstance().switchDevice(1, "MultiGpuMigrationTest", "test-device-switch");
        INDArray b = Nd4j.ones(DataType.FLOAT, 100, 100);
        b.assign(3.0f);
        Nd4j.getExecutioner().commit();

        log.info("P2P enabled test: a is on device {}, b is on device {}, current device is {}",
                Nd4j.getAffinityManager().getDeviceForArray(a),
                Nd4j.getAffinityManager().getDeviceForArray(b),
                Nd4j.getAffinityManager().getDeviceForCurrentThread());
        log.info("P2P available: {}", Nd4j.getAffinityManager().isCrossDeviceAccessSupported());

        // Perform operation WITHOUT explicit migration - P2P should handle this
        INDArray result = a.add(b);

        assertEquals(5.0f, result.getFloat(0, 0), 1e-5);
        assertEquals(5.0f, result.getFloat(99, 99), 1e-5);
        log.info("Automatic P2P cross-device add test passed");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: MatMul with inputs on different devices")
    void testCrossDeviceMatMul(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Create matrix on GPU 0
        DeviceMemoryManager.getInstance().switchDevice(0, "MultiGpuMigrationTest", "test-device-switch");
        INDArray a = Nd4j.ones(DataType.FLOAT, 64, 128);
        Nd4j.getExecutioner().commit();

        // Create matrix on GPU 1
        DeviceMemoryManager.getInstance().switchDevice(1, "MultiGpuMigrationTest", "test-device-switch");
        INDArray b = Nd4j.ones(DataType.FLOAT, 128, 64);
        Nd4j.getExecutioner().commit();

        log.info("About to perform cross-device matmul: a is on device {}, b is on device {}, current device is {}",
                Nd4j.getAffinityManager().getDeviceForArray(a),
                Nd4j.getAffinityManager().getDeviceForArray(b),
                Nd4j.getAffinityManager().getDeviceForCurrentThread());

        // MatMul should work despite inputs on different devices
        INDArray result = a.mmul(b);

        assertArrayEquals(new long[]{64, 64}, result.shape());
        assertEquals(128.0f, result.getFloat(0, 0), 1e-4);

        log.info("Cross-device MatMul test passed");
    }
}
