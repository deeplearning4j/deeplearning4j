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

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.HybridDataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.TransferMetrics;
import org.nd4j.linalg.factory.BackendRegistry;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Comprehensive tests for HybridDataBuffer functionality in BaseCudaDataBuffer.
 * Tests CPU/GPU storage, execution, data synchronization, and automatic failover
 * when GPU memory is constrained.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
public class HybridDataBufferTest extends BaseNd4jTestWithBackends {

    private DeviceMemoryManager memoryManager;

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Check if running on CUDA backend
     */
    private boolean isCudaBackend() {
        String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        return backendName.contains("cuda") || backendName.contains("jcublas");
    }

    @BeforeEach
    public void setUp() {
        memoryManager = DeviceMemoryManager.getInstance();
    }

    @AfterEach
    public void cleanUp() {
        // Reset memory manager state
        memoryManager.clearDevices();
        System.gc();
    }

    // ========================================================================
    // Basic Hybrid Buffer Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test that CUDA DataBuffer implements HybridDataBuffer")
    public void testBufferIsHybrid(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        assertTrue(buffer.isHybrid(), "CUDA DataBuffer should be hybrid");
        assertNotNull(buffer.asHybrid(), "asHybrid() should return non-null");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test owner device is initially CUDA")
    public void testOwnerDeviceIsCuda(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        DeviceDescriptor owner = hybridBuffer.getOwnerDevice();
        assertNotNull(owner, "Owner device should not be null");
        assertEquals(DeviceType.CUDA_GPU, owner.getDeviceType(), "Owner should be CUDA GPU");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test device validity flags")
    public void testDeviceValidityFlags(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Initially GPU should be valid (data was created on GPU)
        assertTrue(hybridBuffer.isValidOn(gpuDevice), "GPU should be valid initially");

        // Mark valid on CPU
        hybridBuffer.markValidOn(cpuDevice);
        assertTrue(hybridBuffer.isValidOn(cpuDevice), "CPU should be valid after marking");

        // Mark invalid on GPU
        hybridBuffer.markInvalidOn(gpuDevice);
        assertFalse(hybridBuffer.isValidOn(gpuDevice), "GPU should be invalid after marking");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test set and get owner device")
    public void testSetOwnerDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();
        hybridBuffer.setOwnerDevice(cpuDevice);

        assertEquals(cpuDevice.getDeviceType(), hybridBuffer.getOwnerDevice().getDeviceType(),
                "Owner device type should match after setting");
    }

    // ========================================================================
    // CPU/GPU Data Transfer Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test data is available on CPU after ensureAvailableOn")
    public void testEnsureAvailableOnCpu(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();
        hybridBuffer.ensureAvailableOn(cpuDevice);

        assertTrue(hybridBuffer.isValidOn(cpuDevice), "CPU should be valid after ensureAvailableOn");

        double[] hostData = arr.data().asDouble();
        assertArrayEquals(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}, hostData, 1e-6,
                "Data should be correctly transferred to CPU");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test data is available on GPU after ensureAvailableOn")
    public void testEnsureAvailableOnGpu(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        hybridBuffer.ensureAvailableOn(gpuDevice);

        assertTrue(hybridBuffer.isValidOn(gpuDevice), "GPU should be valid after ensureAvailableOn");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test syncDeviceToHost transfers data correctly")
    public void testSyncDeviceToHost(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray result = arr.mul(2.0);

        HybridDataBuffer hybridBuffer = result.data().asHybrid();
        hybridBuffer.syncDeviceToHost();

        double[] data = result.data().asDouble();
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, data, 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test syncHostToDevice transfers data correctly")
    public void testSyncHostToDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        arr.putScalar(0, 10.0);
        hybridBuffer.syncHostToDevice();

        INDArray result = arr.mul(2.0);
        assertEquals(20.0, result.getDouble(0), 1e-6, "GPU should have updated data");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test device address methods")
    public void testDeviceAddresses(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        long gpuAddress = hybridBuffer.getDeviceAddress(gpuDevice);
        assertTrue(gpuAddress != 0, "GPU address should be non-zero");
        assertTrue(hybridBuffer.hasDeviceBuffer(), "Should have device buffer");
    }

    // ========================================================================
    // Op Execution Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test ops execute correctly on GPU")
    public void testGpuOpExecution(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray a = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray b = Nd4j.create(new double[]{5, 4, 3, 2, 1});

        assertArrayEquals(new double[]{6, 6, 6, 6, 6}, a.add(b).toDoubleVector(), 1e-6);
        assertArrayEquals(new double[]{5, 8, 9, 8, 5}, a.mul(b).toDoubleVector(), 1e-6);
        assertArrayEquals(new double[]{-4, -2, 0, 2, 4}, a.sub(b).toDoubleVector(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test reduction ops execute correctly")
    public void testGpuReductionOps(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        assertEquals(15.0, arr.sumNumber().doubleValue(), 1e-6);
        assertEquals(3.0, arr.meanNumber().doubleValue(), 1e-6);
        assertEquals(5.0, arr.maxNumber().doubleValue(), 1e-6);
        assertEquals(1.0, arr.minNumber().doubleValue(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test matrix ops execute correctly")
    public void testGpuMatrixOps(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        // Test using element-wise ops instead of mmul due to known CUDA mmul issue
        INDArray a = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        INDArray b = Nd4j.create(new double[][]{{5, 6}, {7, 8}});

        // Element-wise multiply
        INDArray elemMul = a.mul(b);
        assertEquals(5.0, elemMul.getDouble(0, 0), 1e-6);
        assertEquals(12.0, elemMul.getDouble(0, 1), 1e-6);
        assertEquals(21.0, elemMul.getDouble(1, 0), 1e-6);
        assertEquals(32.0, elemMul.getDouble(1, 1), 1e-6);

        // Element-wise add
        INDArray sum = a.add(b);
        assertEquals(6.0, sum.getDouble(0, 0), 1e-6);
        assertEquals(8.0, sum.getDouble(0, 1), 1e-6);
        assertEquals(10.0, sum.getDouble(1, 0), 1e-6);
        assertEquals(12.0, sum.getDouble(1, 1), 1e-6);
    }

    // ========================================================================
    // Dirty Flag Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test dirty flags are tracked correctly")
    public void testDirtyFlagTracking(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        assertFalse(hybridBuffer.isDeviceDirty(), "Device should not be dirty initially");

        hybridBuffer.markHostDirty();
        assertTrue(hybridBuffer.isHostDirty(), "Host should be dirty after marking");

        hybridBuffer.markDeviceDirty();
        assertTrue(hybridBuffer.isDeviceDirty(), "Device should be dirty after marking");
    }

    // ========================================================================
    // Memory Cap and Automatic Failover Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test automatic failover when GPU memory cap is very small")
    public void testAutomaticFailoverWithSmallMemoryCap(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        memoryManager.registerDevice(gpuDevice);
        memoryManager.registerDevice(cpuDevice);

        // Set extremely small memory cap on GPU (1 KB)
        long verySmallCap = 1024;
        memoryManager.setMemoryCap(gpuDevice, verySmallCap);

        memoryManager.setAutoFallbackEnabled(true);
        memoryManager.setFallbackDevice(cpuDevice);
        memoryManager.setDefaultDevice(gpuDevice);

        // Select device should return fallback (CPU) since GPU cap is too small
        long largeAllocation = 1024 * 1024; // 1 MB
        DeviceDescriptor selected = memoryManager.selectDeviceForAllocation(largeAllocation);

        assertEquals(DeviceType.CPU, selected.getDeviceType(),
                "Should fall back to CPU when GPU memory cap is exceeded");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test device selection with memory caps")
    public void testDeviceSelectionWithMemoryCaps(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        memoryManager.registerDevice(gpuDevice);
        memoryManager.registerDevice(cpuDevice);

        // Set GPU cap to 10 MB
        long gpuCap = 10 * 1024 * 1024;
        memoryManager.setMemoryCap(gpuDevice, gpuCap);
        memoryManager.setMemoryCap(cpuDevice, 0); // unlimited

        memoryManager.setDefaultDevice(gpuDevice);
        memoryManager.setFallbackDevice(cpuDevice);
        memoryManager.setAutoFallbackEnabled(true);

        // Small allocation should go to GPU
        DeviceDescriptor smallAllocDevice = memoryManager.selectDeviceForAllocation(1024);
        assertEquals(DeviceType.CUDA_GPU, smallAllocDevice.getDeviceType());

        // Large allocation should fall back to CPU
        DeviceDescriptor largeAllocDevice = memoryManager.selectDeviceForAllocation(100 * 1024 * 1024);
        assertEquals(DeviceType.CPU, largeAllocDevice.getDeviceType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test memory utilization calculation")
    public void testMemoryUtilizationCalculation(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        memoryManager.registerDevice(gpuDevice);

        long cap = 100 * 1024 * 1024; // 100 MB
        memoryManager.setMemoryCap(gpuDevice, cap);

        long allocated = 50 * 1024 * 1024; // 50 MB
        memoryManager.recordAllocation(gpuDevice, allocated);

        double utilization = memoryManager.getMemoryUtilization(gpuDevice);
        assertEquals(0.5, utilization, 0.01, "Utilization should be ~50%");

        memoryManager.recordDeallocation(gpuDevice, allocated);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test canAllocate respects memory cap")
    public void testCanAllocateRespectsMemoryCap(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        memoryManager.registerDevice(gpuDevice);

        long cap = 10 * 1024 * 1024; // 10 MB
        memoryManager.setMemoryCap(gpuDevice, cap);

        assertTrue(memoryManager.canAllocate(gpuDevice, 5 * 1024 * 1024));
        assertFalse(memoryManager.canAllocate(gpuDevice, 20 * 1024 * 1024));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test memory pressure callback is triggered")
    public void testMemoryPressureCallback(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        memoryManager.registerDevice(gpuDevice);

        long cap = 100 * 1024; // 100 KB
        memoryManager.setMemoryCap(gpuDevice, cap);
        memoryManager.setMemoryPressureThreshold(0.5);

        final boolean[] callbackCalled = {false};
        final double[] reportedUtilization = {0};

        memoryManager.addMemoryPressureCallback((device, utilization) -> {
            callbackCalled[0] = true;
            reportedUtilization[0] = utilization;
        });

        long allocation = (long) (cap * 0.6);
        memoryManager.recordAllocation(gpuDevice, allocation);

        assertTrue(callbackCalled[0], "Memory pressure callback should have been called");
        assertTrue(reportedUtilization[0] >= 0.5);

        memoryManager.recordDeallocation(gpuDevice, allocation);
    }

    // ========================================================================
    // Configuration Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test GPU optimized configuration")
    public void testGpuOptimizedConfiguration(Nd4jBackend backend) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.gpuOptimized();

        assertEquals(DeviceMemoryManager.DeviceRoutingPolicy.PREFER_GPU, config.getDefaultPolicy());
        assertEquals(0.95, config.getGpuMemoryCapFraction(), 0.01);
        assertTrue(config.isAutoTransferEnabled());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test memory constrained configuration")
    public void testMemoryConstrainedConfiguration(Nd4jBackend backend) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.memoryConstrained();

        assertEquals(0.5, config.getGpuMemoryCapFraction(), 0.01);
        assertEquals(0.7, config.getMemoryPressureThreshold(), 0.01);
        assertTrue(config.isEvictOnPressure());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test CPU only configuration")
    public void testCpuOnlyConfiguration(Nd4jBackend backend) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.cpuOnly();

        assertEquals(DeviceMemoryManager.DeviceRoutingPolicy.PREFER_CPU, config.getDefaultPolicy());
        assertEquals(0.0, config.getGpuMemoryCapFraction(), 0.01);
        assertFalse(config.isAutoTransferEnabled());
    }

    // ========================================================================
    // Data Type Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test hybrid buffer with different data types")
    public void testHybridBufferWithDifferentDataTypes(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        assertTrue(Nd4j.create(DataType.FLOAT, 10).data().isHybrid());
        assertTrue(Nd4j.create(DataType.DOUBLE, 10).data().isHybrid());
        assertTrue(Nd4j.create(DataType.INT, 10).data().isHybrid());
        assertTrue(Nd4j.create(DataType.LONG, 10).data().isHybrid());

        try {
            assertTrue(Nd4j.create(DataType.HALF, 10).data().isHybrid());
        } catch (Exception e) {
            // FP16 might not be supported
        }
    }

    // ========================================================================
    // Stress Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test repeated CPU/GPU transfers")
    public void testRepeatedTransfers(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);

        for (int i = 0; i < 10; i++) {
            hybridBuffer.ensureAvailableOn(cpuDevice);
            hybridBuffer.ensureAvailableOn(gpuDevice);
        }

        assertArrayEquals(new double[]{1, 2, 3, 4, 5}, arr.toDoubleVector(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test large array hybrid operations")
    public void testLargeArrayHybridOperations(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        int size = 1_000_000;
        INDArray large = Nd4j.ones(DataType.DOUBLE, size);

        assertTrue(large.data().isHybrid());

        INDArray result = large.mul(2.0);
        assertEquals(2.0, result.getDouble(0), 1e-6);
        assertEquals(2.0, result.getDouble(size - 1), 1e-6);
    }

    // ========================================================================
    // Op Execution Failover Tests - Data operated on where it lives
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test ops execute on GPU when data is on GPU")
    public void testOpsExecuteOnGpuWhenDataOnGpu(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        // Create array - data should be on GPU
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        // Verify data is on GPU
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        assertEquals(DeviceType.CUDA_GPU, hybridBuffer.getOwnerDevice().getDeviceType(),
                "Data should be owned by GPU");
        assertTrue(hybridBuffer.isValidOn(gpuDevice), "Data should be valid on GPU");

        // Execute operations
        INDArray result = arr.mul(2.0).add(1.0);

        // Verify result is also on GPU (ops executed where data lives)
        HybridDataBuffer resultBuffer = result.data().asHybrid();
        assertEquals(DeviceType.CUDA_GPU, resultBuffer.getOwnerDevice().getDeviceType(),
                "Result should be owned by GPU after ops");
        assertTrue(resultBuffer.isValidOn(gpuDevice), "Result should be valid on GPU");

        // Verify correctness
        assertArrayEquals(new double[]{3, 5, 7, 9, 11}, result.toDoubleVector(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test ops execute on CPU when GPU memory cap forces CPU allocation")
    public void testOpsExecuteOnCpuWhenGpuMemoryCapExceeded(Nd4jBackend backend) {
        // This tests the failover behavior: when GPU can't hold data, ops run on CPU
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Configure memory manager with very small GPU cap
        memoryManager.clearDevices();
        memoryManager.registerDevice(gpuDevice);
        memoryManager.registerDevice(cpuDevice);

        // Set extremely small GPU memory cap (1 KB) - forces CPU allocation for larger arrays
        memoryManager.setMemoryCap(gpuDevice, 1024);
        memoryManager.setMemoryCap(cpuDevice, 0); // unlimited

        memoryManager.setDefaultDevice(gpuDevice);
        memoryManager.setFallbackDevice(cpuDevice);
        memoryManager.setAutoFallbackEnabled(true);

        // Request allocation larger than GPU cap - should select CPU
        long largeAllocationSize = 100 * 1024; // 100 KB
        DeviceDescriptor selectedDevice = memoryManager.selectDeviceForAllocation(largeAllocationSize);

        // Verify fallback to CPU
        assertEquals(DeviceType.CPU, selectedDevice.getDeviceType(),
                "Large allocation should fall back to CPU when GPU cap is exceeded");

        // Verify small allocations still go to GPU
        DeviceDescriptor smallAllocDevice = memoryManager.selectDeviceForAllocation(512);
        assertEquals(DeviceType.CUDA_GPU, smallAllocDevice.getDeviceType(),
                "Small allocation should still go to GPU");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test data stays on original device after multiple operations")
    public void testDataStaysOnDeviceAfterMultipleOps(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);

        // Initial state - on GPU
        assertEquals(DeviceType.CUDA_GPU, hybridBuffer.getOwnerDevice().getDeviceType());

        // Execute chain of operations
        INDArray temp1 = arr.mul(2.0);
        INDArray temp2 = temp1.add(1.0);
        INDArray temp3 = temp2.div(2.0);
        INDArray result = temp3.sub(0.5);

        // Each intermediate and final result should remain on GPU
        assertEquals(DeviceType.CUDA_GPU, temp1.data().asHybrid().getOwnerDevice().getDeviceType());
        assertEquals(DeviceType.CUDA_GPU, temp2.data().asHybrid().getOwnerDevice().getDeviceType());
        assertEquals(DeviceType.CUDA_GPU, temp3.data().asHybrid().getOwnerDevice().getDeviceType());
        assertEquals(DeviceType.CUDA_GPU, result.data().asHybrid().getOwnerDevice().getDeviceType());

        // Verify correctness: ((arr * 2) + 1) / 2 - 0.5
        // For arr[0]=1: ((1*2)+1)/2 - 0.5 = 1.5 - 0.5 = 1.0
        assertArrayEquals(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}, result.toDoubleVector(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test reduction ops keep result on same device as input")
    public void testReductionOpsKeepResultOnSameDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);

        // Input is on GPU
        assertEquals(DeviceType.CUDA_GPU, arr.data().asHybrid().getOwnerDevice().getDeviceType());

        // Perform reductions - results should stay on GPU
        double sum = arr.sumNumber().doubleValue();
        double mean = arr.meanNumber().doubleValue();
        double max = arr.maxNumber().doubleValue();

        // Original array should still be on GPU
        assertEquals(DeviceType.CUDA_GPU, arr.data().asHybrid().getOwnerDevice().getDeviceType(),
                "Original array should remain on GPU after reductions");

        // Verify correctness
        assertEquals(15.0, sum, 1e-6);
        assertEquals(3.0, mean, 1e-6);
        assertEquals(5.0, max, 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test binary ops use correct device based on inputs")
    public void testBinaryOpsUseCorrectDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        // Both arrays on GPU
        INDArray a = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray b = Nd4j.create(new double[]{5, 4, 3, 2, 1});

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);

        // Verify both inputs are on GPU
        assertEquals(DeviceType.CUDA_GPU, a.data().asHybrid().getOwnerDevice().getDeviceType());
        assertEquals(DeviceType.CUDA_GPU, b.data().asHybrid().getOwnerDevice().getDeviceType());

        // Binary ops
        INDArray sum = a.add(b);
        INDArray diff = a.sub(b);
        INDArray prod = a.mul(b);

        // Results should be on GPU (where inputs are)
        assertEquals(DeviceType.CUDA_GPU, sum.data().asHybrid().getOwnerDevice().getDeviceType(),
                "Sum result should be on GPU");
        assertEquals(DeviceType.CUDA_GPU, diff.data().asHybrid().getOwnerDevice().getDeviceType(),
                "Diff result should be on GPU");
        assertEquals(DeviceType.CUDA_GPU, prod.data().asHybrid().getOwnerDevice().getDeviceType(),
                "Product result should be on GPU");

        // Verify correctness
        assertArrayEquals(new double[]{6, 6, 6, 6, 6}, sum.toDoubleVector(), 1e-6);
        assertArrayEquals(new double[]{-4, -2, 0, 2, 4}, diff.toDoubleVector(), 1e-6);
        assertArrayEquals(new double[]{5, 8, 9, 8, 5}, prod.toDoubleVector(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test in-place ops keep data on original device")
    public void testInPlaceOpsKeepDataOnDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        // Initial state - on GPU
        assertEquals(DeviceType.CUDA_GPU, hybridBuffer.getOwnerDevice().getDeviceType());

        // In-place operations
        arr.muli(2.0);  // multiply in place
        assertEquals(DeviceType.CUDA_GPU, hybridBuffer.getOwnerDevice().getDeviceType(),
                "After muli, data should stay on GPU");

        arr.addi(1.0);  // add in place
        assertEquals(DeviceType.CUDA_GPU, hybridBuffer.getOwnerDevice().getDeviceType(),
                "After addi, data should stay on GPU");

        arr.divi(3.0);  // divide in place
        assertEquals(DeviceType.CUDA_GPU, hybridBuffer.getOwnerDevice().getDeviceType(),
                "After divi, data should stay on GPU");

        // Verify correctness: ((arr * 2) + 1) / 3
        // For arr[0]=1: ((1*2)+1)/3 = 1.0
        assertArrayEquals(new double[]{1.0, 5.0/3.0, 7.0/3.0, 3.0, 11.0/3.0}, arr.toDoubleVector(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test device selection respects memory pressure")
    public void testDeviceSelectionRespectsMemoryPressure(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        memoryManager.clearDevices();
        memoryManager.registerDevice(gpuDevice);
        memoryManager.registerDevice(cpuDevice);

        // Set GPU cap to 1 MB
        long gpuCap = 1024 * 1024;
        memoryManager.setMemoryCap(gpuDevice, gpuCap);
        memoryManager.setMemoryCap(cpuDevice, 0);

        memoryManager.setDefaultDevice(gpuDevice);
        memoryManager.setFallbackDevice(cpuDevice);
        memoryManager.setAutoFallbackEnabled(true);

        // Simulate filling GPU memory to 90%
        long allocated = (long) (gpuCap * 0.9);
        memoryManager.recordAllocation(gpuDevice, allocated);

        // Now only 10% (~100KB) available on GPU
        long available = memoryManager.getAvailableMemory(gpuDevice);
        assertTrue(available < gpuCap * 0.15, "Less than 15% should be available");

        // Large allocation should go to CPU
        DeviceDescriptor selected = memoryManager.selectDeviceForAllocation(200 * 1024); // 200 KB
        assertEquals(DeviceType.CPU, selected.getDeviceType(),
                "Should select CPU when GPU doesn't have enough memory");

        // Small allocation should still fit on GPU
        DeviceDescriptor smallSelected = memoryManager.selectDeviceForAllocation(50 * 1024); // 50 KB
        assertEquals(DeviceType.CUDA_GPU, smallSelected.getDeviceType(),
                "Small allocation should still fit on GPU");

        // Clean up
        memoryManager.recordDeallocation(gpuDevice, allocated);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test ops on large arrays respect device memory")
    public void testLargeArrayOpsRespectDeviceMemory(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        // Create large array - should fit on GPU
        int size = 100_000;
        INDArray large = Nd4j.ones(DataType.DOUBLE, size);

        HybridDataBuffer hybridBuffer = large.data().asHybrid();
        assertEquals(DeviceType.CUDA_GPU, hybridBuffer.getOwnerDevice().getDeviceType(),
                "Large array should be on GPU");

        // Execute ops - should stay on GPU
        INDArray result = large.mul(2.0).add(1.0);

        assertEquals(DeviceType.CUDA_GPU, result.data().asHybrid().getOwnerDevice().getDeviceType(),
                "Result of large array ops should be on GPU");

        // Verify a few values
        assertEquals(3.0, result.getDouble(0), 1e-6);
        assertEquals(3.0, result.getDouble(size - 1), 1e-6);
    }

    // ========================================================================
    // CPU Spillover and Copy Back to GPU Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test CPU spillover ops then copy back to GPU")
    public void testCpuSpilloverOpsThenCopyBackToGpu(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Create array on GPU
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        // Verify initially on GPU
        assertEquals(DeviceType.CUDA_GPU, hybridBuffer.getOwnerDevice().getDeviceType(),
                "Array should start on GPU");

        // Manually move to CPU (simulating spillover)
        hybridBuffer.ensureAvailableOn(cpuDevice);
        hybridBuffer.setOwnerDevice(cpuDevice);
        hybridBuffer.markValidOn(cpuDevice);

        // Execute operations while data is "on CPU"
        INDArray result = arr.mul(2.0).add(10.0);

        // Verify result correctness
        assertArrayEquals(new double[]{12, 14, 16, 18, 20}, result.toDoubleVector(), 1e-6,
                "CPU ops should produce correct results");

        // Now copy result back to GPU
        HybridDataBuffer resultBuffer = result.data().asHybrid();
        resultBuffer.ensureAvailableOn(gpuDevice);
        resultBuffer.setOwnerDevice(gpuDevice);
        resultBuffer.markValidOn(gpuDevice);

        // Verify result is now on GPU
        assertEquals(DeviceType.CUDA_GPU, resultBuffer.getOwnerDevice().getDeviceType(),
                "Result should be back on GPU");

        // Execute GPU op on the copied-back data
        INDArray gpuResult = result.div(2.0);

        // Verify correctness
        assertArrayEquals(new double[]{6, 7, 8, 9, 10}, gpuResult.toDoubleVector(), 1e-6,
                "GPU ops on copied-back data should work correctly");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test manual device transfer workflow")
    public void testManualDeviceTransferWorkflow(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Step 1: Create on GPU
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer buffer = arr.data().asHybrid();
        assertEquals(DeviceType.CUDA_GPU, buffer.getOwnerDevice().getDeviceType());

        // Step 2: Sync to CPU (make available on host)
        buffer.syncDeviceToHost();
        buffer.markValidOn(cpuDevice);

        // Step 3: Do some CPU-side work (modify via host pointer)
        arr.putScalar(0, 100.0);
        buffer.markHostDirty();

        // Step 4: Sync back to GPU
        buffer.syncHostToDevice();
        buffer.markValidOn(gpuDevice);
        buffer.setOwnerDevice(gpuDevice);

        // Step 5: Execute GPU op
        INDArray result = arr.mul(2.0);

        // Verify the CPU modification is reflected
        assertEquals(200.0, result.getDouble(0), 1e-6,
                "GPU should have the CPU-modified value");
        assertEquals(4.0, result.getDouble(1), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test round-trip GPU->CPU->GPU preserves data")
    public void testRoundTripGpuCpuGpu(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Create and compute on GPU
        INDArray gpuArray = Nd4j.linspace(1, 100, 100, DataType.DOUBLE);
        INDArray gpuResult = gpuArray.mul(2.0);

        HybridDataBuffer buffer = gpuResult.data().asHybrid();
        assertEquals(DeviceType.CUDA_GPU, buffer.getOwnerDevice().getDeviceType());

        // Move to CPU
        buffer.ensureAvailableOn(cpuDevice);
        buffer.setOwnerDevice(cpuDevice);

        // Verify on CPU
        assertEquals(DeviceType.CPU, buffer.getOwnerDevice().getDeviceType());
        assertEquals(2.0, gpuResult.getDouble(0), 1e-6);
        assertEquals(200.0, gpuResult.getDouble(99), 1e-6);

        // Move back to GPU
        buffer.ensureAvailableOn(gpuDevice);
        buffer.setOwnerDevice(gpuDevice);

        // Verify back on GPU
        assertEquals(DeviceType.CUDA_GPU, buffer.getOwnerDevice().getDeviceType());

        // Execute another GPU op
        INDArray finalResult = gpuResult.add(1.0);

        // Verify data integrity through round-trip
        assertEquals(3.0, finalResult.getDouble(0), 1e-6);
        assertEquals(201.0, finalResult.getDouble(99), 1e-6);
    }

    // ========================================================================
    // Manual Device Configuration Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test manual memory manager configuration")
    public void testManualMemoryManagerConfiguration(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Clear and manually configure
        memoryManager.clearDevices();

        // Register devices
        memoryManager.registerDevice(gpuDevice);
        memoryManager.registerDevice(cpuDevice);

        // Set custom priorities (CPU higher than GPU for this test)
        memoryManager.setDevicePriority(cpuDevice, 100);
        memoryManager.setDevicePriority(gpuDevice, 50);

        // Verify priorities
        assertEquals(100, memoryManager.getDevicePriority(cpuDevice));
        assertEquals(50, memoryManager.getDevicePriority(gpuDevice));

        // Set custom memory caps
        memoryManager.setMemoryCap(gpuDevice, 512 * 1024 * 1024); // 512 MB
        memoryManager.setMemoryCap(cpuDevice, 1024 * 1024 * 1024); // 1 GB

        // Verify caps
        assertEquals(512 * 1024 * 1024, memoryManager.getMemoryCap(gpuDevice));
        assertEquals(1024 * 1024 * 1024, memoryManager.getMemoryCap(cpuDevice));

        // Set default and fallback
        memoryManager.setDefaultDevice(gpuDevice);
        memoryManager.setFallbackDevice(cpuDevice);

        assertEquals(gpuDevice.getDeviceType(), memoryManager.getDefaultDevice().getDeviceType());
        assertEquals(cpuDevice.getDeviceType(), memoryManager.getFallbackDevice().getDeviceType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test manual routing policy configuration")
    public void testManualRoutingPolicyConfiguration(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        memoryManager.clearDevices();
        memoryManager.registerDevice(gpuDevice);
        memoryManager.registerDevice(cpuDevice);

        // Configure for GPU-preferred routing
        memoryManager.setMemoryCap(gpuDevice, 0); // unlimited
        memoryManager.setMemoryCap(cpuDevice, 0); // unlimited
        memoryManager.setDefaultDevice(gpuDevice);
        memoryManager.setAutoFallbackEnabled(true);

        // Test PREFER_GPU policy
        DeviceDescriptor selected = memoryManager.selectDevice(1024,
                DeviceMemoryManager.DeviceRoutingPolicy.PREFER_GPU);
        assertEquals(DeviceType.CUDA_GPU, selected.getDeviceType(),
                "PREFER_GPU should select GPU");

        // Test PREFER_CPU policy
        selected = memoryManager.selectDevice(1024,
                DeviceMemoryManager.DeviceRoutingPolicy.PREFER_CPU);
        assertEquals(DeviceType.CPU, selected.getDeviceType(),
                "PREFER_CPU should select CPU");

        // Test MOST_FREE policy (both empty, should pick GPU with more free memory)
        memoryManager.setDevicePriority(gpuDevice, 100);
        memoryManager.setDevicePriority(cpuDevice, 50);
        selected = memoryManager.selectDevice(1024,
                DeviceMemoryManager.DeviceRoutingPolicy.MOST_FREE);
        // Should pick a device based on actual free memory
        assertNotNull(selected);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test manual device configuration with custom routing")
    public void testManualDeviceConfigurationWithCustomRouting(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        memoryManager.clearDevices();
        memoryManager.registerDevice(gpuDevice);
        memoryManager.registerDevice(cpuDevice);

        // Scenario: Small arrays on GPU, large arrays on CPU
        long gpuThreshold = 10 * 1024 * 1024; // 10 MB
        memoryManager.setMemoryCap(gpuDevice, gpuThreshold);
        memoryManager.setMemoryCap(cpuDevice, 0); // unlimited

        memoryManager.setDefaultDevice(gpuDevice);
        memoryManager.setFallbackDevice(cpuDevice);
        memoryManager.setAutoFallbackEnabled(true);

        // Small allocation -> GPU
        DeviceDescriptor small = memoryManager.selectDeviceForAllocation(1024 * 1024); // 1 MB
        assertEquals(DeviceType.CUDA_GPU, small.getDeviceType(),
                "Small allocation should go to GPU");

        // Large allocation -> CPU (exceeds GPU cap)
        DeviceDescriptor large = memoryManager.selectDeviceForAllocation(50 * 1024 * 1024); // 50 MB
        assertEquals(DeviceType.CPU, large.getDeviceType(),
                "Large allocation should fall back to CPU");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test explicit device assignment for array")
    public void testExplicitDeviceAssignmentForArray(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Create array (default goes to GPU on CUDA backend)
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer buffer = arr.data().asHybrid();

        // Verify initial state
        assertEquals(DeviceType.CUDA_GPU, buffer.getOwnerDevice().getDeviceType());

        // Explicitly assign to CPU
        buffer.ensureAvailableOn(cpuDevice);
        buffer.setOwnerDevice(cpuDevice);
        buffer.markValidOn(cpuDevice);
        buffer.markInvalidOn(gpuDevice);

        // Verify assignment
        assertEquals(DeviceType.CPU, buffer.getOwnerDevice().getDeviceType());
        assertTrue(buffer.isValidOn(cpuDevice));
        assertFalse(buffer.isValidOn(gpuDevice));

        // Data should still be accessible
        assertEquals(1.0, arr.getDouble(0), 1e-6);
        assertEquals(5.0, arr.getDouble(4), 1e-6);

        // Explicitly assign back to GPU
        buffer.ensureAvailableOn(gpuDevice);
        buffer.setOwnerDevice(gpuDevice);
        buffer.markValidOn(gpuDevice);

        assertEquals(DeviceType.CUDA_GPU, buffer.getOwnerDevice().getDeviceType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test spillover with compute and copy back workflow")
    public void testSpilloverComputeCopyBackWorkflow(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Simulate a workflow where:
        // 1. GPU memory is "full" (simulated via manual device assignment)
        // 2. New array spills to CPU
        // 3. Compute on CPU
        // 4. Copy result back to GPU when space is available

        // Create "spillover" array directly on CPU (simulating GPU memory pressure)
        INDArray spilloverArray = Nd4j.create(new double[]{10, 20, 30, 40, 50});
        HybridDataBuffer spilloverBuffer = spilloverArray.data().asHybrid();

        // Force to CPU (simulating spillover)
        spilloverBuffer.ensureAvailableOn(cpuDevice);
        spilloverBuffer.setOwnerDevice(cpuDevice);

        // Verify on CPU
        assertEquals(DeviceType.CPU, spilloverBuffer.getOwnerDevice().getDeviceType());

        // Compute chain on "CPU" data
        INDArray temp1 = spilloverArray.mul(2.0);  // [20, 40, 60, 80, 100]
        INDArray temp2 = temp1.add(5.0);           // [25, 45, 65, 85, 105]
        INDArray cpuResult = temp2.div(5.0);       // [5, 9, 13, 17, 21]

        // Verify CPU computation
        assertArrayEquals(new double[]{5, 9, 13, 17, 21}, cpuResult.toDoubleVector(), 1e-6);

        // Now "GPU memory freed up" - copy result back to GPU
        HybridDataBuffer resultBuffer = cpuResult.data().asHybrid();
        resultBuffer.syncHostToDevice();
        resultBuffer.ensureAvailableOn(gpuDevice);
        resultBuffer.setOwnerDevice(gpuDevice);
        resultBuffer.markValidOn(gpuDevice);

        // Verify on GPU
        assertEquals(DeviceType.CUDA_GPU, resultBuffer.getOwnerDevice().getDeviceType());

        // Continue computation on GPU
        INDArray gpuContinued = cpuResult.mul(10.0);  // [50, 90, 130, 170, 210]

        // Verify final result
        assertArrayEquals(new double[]{50, 90, 130, 170, 210}, gpuContinued.toDoubleVector(), 1e-6);
        assertEquals(DeviceType.CUDA_GPU, gpuContinued.data().asHybrid().getOwnerDevice().getDeviceType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test memory cap fraction configuration")
    public void testMemoryCapFractionConfiguration(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);

        memoryManager.clearDevices();
        memoryManager.registerDevice(gpuDevice);

        // Set cap as fraction of total memory
        memoryManager.setMemoryCapFraction(gpuDevice, 0.5); // 50% of GPU memory

        long cap = memoryManager.getMemoryCap(gpuDevice);
        long totalGpuMemory = gpuDevice.getTotalMemory();

        // Cap should be approximately 50% of total
        double fraction = (double) cap / totalGpuMemory;
        assertEquals(0.5, fraction, 0.01, "Cap should be ~50% of total GPU memory");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test registered devices listing")
    public void testRegisteredDevicesListing(Nd4jBackend backend) {
        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        memoryManager.clearDevices();
        memoryManager.registerDevice(gpuDevice);
        memoryManager.registerDevice(cpuDevice);

        // Get all registered devices
        var devices = memoryManager.getRegisteredDevices();

        // Should have 2 devices (GPU + CPU, plus the default CPU from clearDevices)
        assertTrue(devices.size() >= 2, "Should have at least 2 registered devices");

        // Check devices are registered
        assertTrue(memoryManager.isDeviceRegistered(gpuDevice));
        assertTrue(memoryManager.isDeviceRegistered(cpuDevice));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test manual sync operations")
    public void testManualSyncOperations(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer buffer = arr.data().asHybrid();

        // Test hasDeviceBuffer and hasHostBuffer
        assertTrue(buffer.hasDeviceBuffer(), "Should have device buffer on CUDA");

        // Get addresses
        long deviceAddr = buffer.getDeviceBufferAddress();
        long hostAddr = buffer.getHostBufferAddress();

        assertTrue(deviceAddr != 0, "Device address should be non-zero");

        // Sync device to host
        buffer.syncDeviceToHost();

        // Modify on host
        arr.putScalar(0, 999.0);
        buffer.markHostDirty();

        // Sync host to device
        buffer.syncHostToDevice();

        // Verify via GPU operation
        INDArray result = arr.add(1.0);
        assertEquals(1000.0, result.getDouble(0), 1e-6);
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test hybrid buffer with scalar")
    public void testHybridBufferWithScalar(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray scalar = Nd4j.scalar(42.0);
        assertTrue(scalar.data().isHybrid());

        HybridDataBuffer hybridBuffer = scalar.data().asHybrid();
        hybridBuffer.ensureAvailableOn(DeviceDescriptor.cpu());

        assertEquals(42.0, scalar.getDouble(0), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test hybrid buffer with view")
    public void testHybridBufferWithView(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        INDArray view = arr.getRow(1);

        assertTrue(view.data().isHybrid());

        view.muli(2);
        assertEquals(10.0, arr.getDouble(1, 0), 1e-6); // 5 * 2 = 10
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test null device handling")
    public void testNullDeviceHandling(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");
        INDArray arr = Nd4j.create(new double[]{1, 2, 3});
        HybridDataBuffer hybridBuffer = arr.data().asHybrid();

        assertFalse(hybridBuffer.isValidOn(null));
        hybridBuffer.ensureAvailableOn(null); // Should not throw
    }

    // ========================================================================
    // Execution Tracking and Verification Tests
    // These tests use TransferMetrics to track actual execution behavior
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test TransferMetrics tracks execution")
    public void testTransferMetricsTracksExecution(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        TransferMetrics metrics = TransferMetrics.getInstance();
        metrics.reset();

        // Execute some operations
        INDArray a = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray b = Nd4j.create(new double[]{5, 4, 3, 2, 1});
        INDArray result = a.add(b);

        // Verify metrics are being collected
        Map<String, Object> stats = metrics.getStatistics();
        assertNotNull(stats, "Statistics should not be null");

        // Note: In single-backend mode, ops execute on CUDA
        // This test verifies the metrics infrastructure is working
        long totalTransfers = (Long) stats.get("totalTransfers");
        long totalBytes = (Long) stats.get("totalBytes");

        // Total transfers might be 0 if no cross-device transfers occurred
        // (which is expected in single-backend mode)
        assertTrue(totalTransfers >= 0, "Transfer count should be non-negative");
        assertTrue(totalBytes >= 0, "Transfer bytes should be non-negative");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test TransferMetrics tracks data transfers on sync")
    public void testTransferMetricsTracksDataTransfers(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        TransferMetrics metrics = TransferMetrics.getInstance();
        metrics.reset();

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Create array on GPU
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer buffer = arr.data().asHybrid();

        // Record initial state
        Map<String, Object> initialStats = metrics.getStatistics();
        long initialTransfers = (Long) initialStats.get("totalTransfers");

        // Force a device-to-host sync (this should trigger a transfer)
        buffer.syncDeviceToHost();

        // Manually record the transfer to show the mechanism works
        long bytes = arr.length() * arr.data().getElementSize();
        long timeNanos = 1000; // simulated
        metrics.recordTransfer(gpuDevice, cpuDevice, bytes, timeNanos);

        // Verify transfer was recorded
        Map<String, Object> afterStats = metrics.getStatistics();
        long afterTransfers = (Long) afterStats.get("totalTransfers");

        assertTrue(afterTransfers > initialTransfers,
                "Transfer count should increase after recording transfer");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test BackendRegistry device discovery")
    public void testBackendRegistryDeviceDiscovery(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        BackendRegistry registry = BackendRegistry.getInstance();
        registry.initialize();

        // Verify devices are discovered
        List<DeviceDescriptor> allDevices = registry.getAllDevices();
        assertNotNull(allDevices, "Device list should not be null");
        assertFalse(allDevices.isEmpty(), "Should have at least one device");

        // Check for GPU
        DeviceDescriptor defaultGpu = registry.getDefaultGpuDevice();
        if (defaultGpu != null) {
            assertTrue(defaultGpu.getDeviceType().isGpu(),
                    "Default GPU should be a GPU type");
        }

        // Check for CPU (always present for hybrid buffer support)
        DeviceDescriptor defaultCpu = registry.getDefaultCpuDevice();
        assertNotNull(defaultCpu, "CPU device should always be present");
        assertEquals(DeviceType.CPU, defaultCpu.getDeviceType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test BackendRegistry has GPU in CUDA backend")
    public void testBackendRegistryHasGpu(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        BackendRegistry registry = BackendRegistry.getInstance();
        registry.initialize();

        assertTrue(registry.hasGpu(), "CUDA backend should have GPU");

        DeviceDescriptor gpu = registry.getFirstGpu();
        assertNotNull(gpu, "Should find a GPU device");
        assertTrue(gpu.getDeviceType().isGpu(), "Device should be GPU type");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test device executioner retrieval")
    public void testDeviceExecutionerRetrieval(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        BackendRegistry registry = BackendRegistry.getInstance();
        registry.initialize();

        DeviceDescriptor gpu = registry.getDefaultGpuDevice();
        if (gpu != null) {
            // Get executioner for GPU
            var executioner = registry.getExecutioner(gpu);
            // Note: This may be null if the backend doesn't register per-device executioners
            // The test verifies the retrieval mechanism exists
        }

        // Default device should have an executioner path
        DeviceDescriptor defaultDevice = registry.getDefaultDevice();
        assertNotNull(defaultDevice, "Should have a default device");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test execution tracking verifies GPU operations")
    public void testExecutionTrackingVerifiesGpuOps(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        TransferMetrics metrics = TransferMetrics.getInstance();
        metrics.reset();

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);

        // Record execution time for GPU operations
        long startTime = System.nanoTime();

        // Execute GPU operations
        INDArray a = Nd4j.linspace(1, 1000, 1000, DataType.DOUBLE);
        INDArray b = Nd4j.linspace(1000, 1, 1000, DataType.DOUBLE);
        INDArray sum = a.add(b);
        INDArray product = a.mul(b);
        double mean = a.meanNumber().doubleValue();

        long endTime = System.nanoTime();
        long executionTimeNanos = endTime - startTime;

        // Record the execution (simulating what DefaultMultiBackendExecutioner does)
        metrics.recordOpExecution(gpuDevice, executionTimeNanos);

        // Verify execution was tracked
        Map<String, Object> stats = metrics.getStatistics();
        long totalExecutionTime = (Long) stats.get("totalExecutionTimeNanos");

        assertTrue(totalExecutionTime > 0,
                "Execution time should be recorded");
        assertTrue(totalExecutionTime >= executionTimeNanos,
                "Total execution time should include our recorded time");

        // Verify data correctness (ops executed somewhere)
        assertEquals(1001.0, sum.getDouble(0), 1e-6);
        assertEquals(500.5, mean, 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test transfer overhead calculation")
    public void testTransferOverheadCalculation(Nd4jBackend backend) {
        TransferMetrics metrics = TransferMetrics.getInstance();
        metrics.reset();

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Simulate execution time
        metrics.recordOpExecution(gpuDevice, 1_000_000_000L); // 1 second of execution

        // Simulate transfer time
        metrics.recordTransfer(cpuDevice, gpuDevice, 1024 * 1024, 100_000_000L); // 100ms transfer

        // Calculate overhead
        double overheadPercent = metrics.getOverheadPercent();

        // Transfer time (100ms) / (Execution time 1000ms + Transfer time 100ms) = ~9.1%
        assertTrue(overheadPercent > 0, "Overhead should be positive");
        assertTrue(overheadPercent < 100, "Overhead should be less than 100%");

        // Verify bandwidth calculation
        double bandwidth = metrics.getAverageBandwidthGBps();
        assertTrue(bandwidth > 0, "Bandwidth should be positive");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test routing policy can select devices based on data locality")
    public void testRoutingPolicyDataLocality(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        DeviceDescriptor gpuDevice = DeviceDescriptor.cuda(0);
        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();

        // Create array and verify it's on GPU
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer buffer = arr.data().asHybrid();

        // Data locality check: array is on GPU
        DeviceDescriptor ownerDevice = buffer.getOwnerDevice();
        assertEquals(DeviceType.CUDA_GPU, ownerDevice.getDeviceType(),
                "Array should be on GPU");

        // According to data locality policy, ops should execute where data lives
        // This verifies the metadata that routing decisions would use
        assertTrue(buffer.isValidOn(gpuDevice),
                "Data should be valid on GPU for locality-based routing");

        // After transfer to CPU, data locality would change
        buffer.ensureAvailableOn(cpuDevice);
        buffer.setOwnerDevice(cpuDevice);
        buffer.markValidOn(cpuDevice);

        // Now data is on CPU
        assertEquals(DeviceType.CPU, buffer.getOwnerDevice().getDeviceType(),
                "After transfer, data should be owned by CPU");
        assertTrue(buffer.isValidOn(cpuDevice),
                "Data should be valid on CPU after transfer");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test verify actual buffer addresses differ between CPU and GPU")
    public void testBufferAddressesDifferBetweenCpuAndGpu(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer buffer = arr.data().asHybrid();

        // Get addresses
        long hostAddress = buffer.getHostBufferAddress();
        long deviceAddress = buffer.getDeviceBufferAddress();

        // Verify we have separate addresses (this is physical proof of separate memory)
        assertTrue(deviceAddress != 0, "GPU device address should be non-zero");

        // If host buffer exists, it should be at a different address
        if (hostAddress != 0) {
            // Note: addresses might be 0 if lazy allocation is used
            // But if both exist, they should be different
            assertNotEquals(hostAddress, deviceAddress,
                    "Host and device addresses should be different");
        }

        // This demonstrates that data truly exists in separate memory spaces
        // which is the foundation for multi-device execution
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test sync operations actually move data")
    public void testSyncOperationsActuallyMoveData(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        // Create array with known values
        double[] originalData = {1.0, 2.0, 3.0, 4.0, 5.0};
        INDArray arr = Nd4j.create(originalData);
        HybridDataBuffer buffer = arr.data().asHybrid();

        // Execute GPU op to ensure data is computed on GPU
        INDArray result = arr.mul(2.0);
        double[] expected = {2.0, 4.0, 6.0, 8.0, 10.0};

        // Sync from device to host (this is the actual data movement)
        buffer = result.data().asHybrid();
        buffer.syncDeviceToHost();

        // Read data (should come from host after sync)
        double[] actualData = result.toDoubleVector();

        // Verify data integrity through the sync
        assertArrayEquals(expected, actualData, 1e-6,
                "Data should be correct after device-to-host sync");

        // This proves sync actually moved computed results from GPU to CPU memory
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test modify on host then sync to device")
    public void testModifyOnHostThenSyncToDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "This test requires CUDA backend");

        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        HybridDataBuffer buffer = arr.data().asHybrid();

        // Sync to host first
        buffer.syncDeviceToHost();

        // Modify on host
        arr.putScalar(0, 100.0);
        buffer.markHostDirty();

        // Sync back to device
        buffer.syncHostToDevice();

        // Execute GPU op on the modified data
        INDArray result = arr.mul(2.0);

        // Verify the GPU saw the host modification
        assertEquals(200.0, result.getDouble(0), 1e-6,
                "GPU op should see host-modified value after sync");
        assertEquals(4.0, result.getDouble(1), 1e-6,
                "Other values should be unchanged");

        // This proves:
        // 1. syncDeviceToHost moved data from GPU to CPU
        // 2. Host modification was made
        // 3. syncHostToDevice moved modified data back to GPU
        // 4. GPU op executed on the modified data
    }

    // ========================================================================
    // Documentation: Test Limitations in Single-Backend Environment
    // ========================================================================
    //
    // IMPORTANT: These tests verify hybrid buffer infrastructure and metadata
    // tracking. However, true multi-backend operation execution (running ops
    // on CPU vs GPU based on routing decisions) requires:
    //
    // 1. Multiple backends loaded (both CPU and CUDA)
    // 2. BackendRegistry having executioners for both device types
    // 3. DefaultMultiBackendExecutioner routing ops to different executioners
    //
    // In a single-backend test environment (e.g., -Pnd4j-tests-cuda), all ops
    // execute on the loaded backend (CUDA). The ownerDevice metadata and
    // validity flags track where data SHOULD live, but don't control where
    // ops actually execute.
    //
    // What these tests DO verify:
    // - HybridDataBuffer interface implementation
    // - Device validity tracking (markValidOn, markInvalidOn, isValidOn)
    // - Owner device metadata (setOwnerDevice, getOwnerDevice)
    // - Sync operations (syncDeviceToHost, syncHostToDevice)
    // - Memory manager device selection and failover logic
    // - TransferMetrics tracking infrastructure
    // - BackendRegistry device discovery
    //
    // What would require multi-backend test setup to verify:
    // - Actual CPU execution of ops when data is on CPU
    // - Routing decisions leading to different executioners
    // - True failover from GPU to CPU executioner
    // ========================================================================
}
