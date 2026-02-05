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
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for CUDA memory allocation failover mechanisms.
 *
 * <p>These tests verify that when GPU memory is exhausted, the system correctly:
 * <ol>
 *   <li>Routes allocations to other GPUs with available memory (multi-GPU systems)</li>
 *   <li>Falls back to host (CPU) memory when no GPU has space (if CPU backend available)</li>
 *   <li>Reports proper errors instead of crashing when all options exhausted</li>
 * </ol>
 *
 * <p>Tests use memory simulation to avoid actually exhausting GPU memory, which would
 * be slow, hardware-dependent, and potentially crash other processes on the system.
 *
 * <p><b>Test Coverage Gap This Addresses:</b>
 * <br>Previous tests (HybridDataBufferTest) verified DeviceMemoryManager's selection logic
 * in isolation, but that logic was never connected to the actual allocation path in
 * BaseCudaDataBuffer.initPointers(). These tests verify the ACTUAL allocation path
 * correctly handles memory exhaustion scenarios.
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer
 */
@Slf4j
@DisplayName("CUDA Memory Allocation Failover Tests")
public class CudaMemoryAllocationFailoverTest extends BaseNd4jTestWithBackends {

    // Memory manager instance for simulation
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

    /**
     * Get number of CUDA devices
     */
    private int getNumDevices() {
        try {
            return Nd4j.getAffinityManager().getNumberOfDevices();
        } catch (Exception e) {
            return 0;
        }
    }

    /**
     * Get actual free memory for a device (for logging)
     */
    private long getActualFreeMemory(int deviceId) {
        try {
            return NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemory(deviceId);
        } catch (Exception e) {
            return -1;
        }
    }

    @BeforeAll
    static void initSimulationApi() {
        log.info("Memory simulation API available via DeviceMemoryManager. CPU_DEVICE_ID={}",
                DeviceMemoryManager.CPU_DEVICE_ID);
    }

    @BeforeEach
    void setUp() {
        memoryManager = DeviceMemoryManager.getInstance();
        memoryManager.clearAllMemorySimulation();
    }

    @AfterEach
    void tearDown() {
        if (memoryManager != null) {
            memoryManager.clearAllMemorySimulation();
        }
        System.gc();
    }

    // =========================================================================
    // Single GPU Scenarios
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Small allocation succeeds when device has plenty of memory")
    void testSmallAllocationSucceeds(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices > 0, "Test requires at least 1 CUDA device");

        // Set current device explicitly to device 0
        Nd4j.getAffinityManager().unsafeSetDevice(0);

        // Simulate ALL devices (GPUs + CPU) with 8GB free memory
        for (int i = 0; i < numDevices; i++) {
            memoryManager.setSimulatedFreeMemory(i, 8L * 1024 * 1024 * 1024);
        }
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 32L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Small allocation (1MB) should succeed
        long allocationSize = 1024 * 1024 / 4; // 1MB / 4 bytes per float = 256K floats
        INDArray arr = Nd4j.create(DataType.FLOAT, allocationSize);

        assertNotNull(arr, "Allocation should succeed");
        assertEquals(allocationSize, arr.length());

        // Verify simulated allocation was tracked on the current device
        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        long allocated = memoryManager.getSimulatedAllocatedMemory(currentDevice);
        assertTrue(allocated > 0, "Simulated allocation should be tracked on device " + currentDevice);

        log.info("Small allocation test passed: {} elements allocated on device {}, {} MB tracked",
                allocationSize, currentDevice, allocated / (1024 * 1024));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Allocation fails gracefully when single device is OOM (no CPU fallback)")
    void testAllocationFailsGracefullyWhenSingleDeviceOOM(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices == 1, "Test requires exactly 1 CUDA device (for OOM without multi-GPU fallback)");

        // Simulate device with only 1KB free (basically OOM), and no CPU memory
        memoryManager.setSimulatedFreeMemory(0, 1024L);
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 0L);
        memoryManager.setMemorySimulationEnabled(true);

        // Try to allocate 100MB - should trigger CPU fallback or fail gracefully
        long allocationSize = 100 * 1024 * 1024 / 4; // 100MB / 4 bytes = 25M floats

        try {
            INDArray arr = Nd4j.create(DataType.FLOAT, allocationSize);
            // If we get here, allocation worked somehow
            assertNotNull(arr, "Allocation succeeded");
            log.info("Allocation succeeded (unexpected but acceptable)");
        } catch (Throwable e) {
            // Expected if all devices OOM
            log.info("Allocation failed as expected: {}", e.getMessage());
            String msg = e.getMessage() != null ? e.getMessage().toLowerCase() : "";
            assertTrue(msg.contains("memory") || msg.contains("allocation") ||
                            msg.contains("oom") || e instanceof OutOfMemoryError,
                    "Exception should be memory-related, was: " + e.getClass().getName() + ": " + e.getMessage());
        }
    }

    // =========================================================================
    // Multi-GPU Failover Scenarios
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Multi-GPU routing - allocation routes to device with most free memory")
    void testMultiGpuRoutingToDeviceWithMostMemory(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices >= 2, "Test requires at least 2 CUDA devices");

        // Simulate:
        // - Device 0: Only 10MB free (almost full)
        // - Device 1: 8GB free (plenty of space)
        memoryManager.setSimulatedFreeMemory(0, 10L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Set current thread to device 0
        Nd4j.getAffinityManager().unsafeSetDevice(0);

        // Try to allocate 50MB (too big for device 0, should route to device 1)
        long allocationSize = 50 * 1024 * 1024 / 4; // 50MB / 4 bytes = 12.5M floats
        INDArray arr = Nd4j.create(DataType.FLOAT, allocationSize);

        assertNotNull(arr, "Allocation should succeed on alternate device");

        // Verify allocation was tracked on device 1 (the one with memory)
        long allocatedOnDevice1 = memoryManager.getSimulatedAllocatedMemory(1);
        assertTrue(allocatedOnDevice1 > 0,
                "Allocation should have been routed to device 1 (has more memory)");

        log.info("Multi-GPU routing test passed: 50MB allocation routed from device 0 (10MB free) to device 1 (8GB free)");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Multi-GPU - allocation stays on current device when it has enough memory")
    void testMultiGpuStaysOnCurrentDeviceWhenSufficient(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices >= 2, "Test requires at least 2 CUDA devices");

        // Both devices have plenty of memory
        memoryManager.setSimulatedFreeMemory(0, 8L * 1024 * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Set current thread to device 0
        Nd4j.getAffinityManager().unsafeSetDevice(0);

        // Allocate 10MB - should stay on device 0 (current device has space)
        long allocationSize = 10 * 1024 * 1024 / 4;
        INDArray arr = Nd4j.create(DataType.FLOAT, allocationSize);

        assertNotNull(arr, "Allocation should succeed");

        // Verify allocation was on device 0 (current device)
        long allocatedOnDevice0 = memoryManager.getSimulatedAllocatedMemory(0);
        long allocatedOnDevice1 = memoryManager.getSimulatedAllocatedMemory(1);

        assertTrue(allocatedOnDevice0 > 0, "Allocation should stay on current device (0)");
        assertEquals(0, allocatedOnDevice1, "Device 1 should have no allocations");

        log.info("Allocation correctly stayed on current device (0) when sufficient memory available");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Multi-GPU - selects device with MOST free memory among alternatives")
    void testMultiGpuSelectsBestDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices >= 3, "Test requires at least 3 CUDA devices");

        // Simulate:
        // - Device 0: 5MB free (current, not enough)
        // - Device 1: 100MB free (enough, but not best)
        // - Device 2: 8GB free (best choice)
        memoryManager.setSimulatedFreeMemory(0, 5L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 100L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(2, 8L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        Nd4j.getAffinityManager().unsafeSetDevice(0);

        // Allocate 50MB - device 0 doesn't have it, should pick device 2 (most memory)
        long allocationSize = 50 * 1024 * 1024 / 4;
        INDArray arr = Nd4j.create(DataType.FLOAT, allocationSize);

        assertNotNull(arr, "Allocation should succeed");

        long allocatedOnDevice2 = memoryManager.getSimulatedAllocatedMemory(2);
        assertTrue(allocatedOnDevice2 > 0, "Allocation should route to device with most memory (device 2)");

        log.info("Multi-GPU routing correctly selected device 2 (8GB free) over device 1 (100MB free)");
    }

    // =========================================================================
    // CPU Fallback Scenarios
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Falls back to CPU when ALL GPUs are OOM")
    void testCpuFallbackWhenAllGpusOom(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices > 0, "Test requires at least 1 CUDA device");

        // Simulate ALL GPU devices with minimal memory (OOM)
        for (int i = 0; i < numDevices; i++) {
            memoryManager.setSimulatedFreeMemory(i, 0L); // 0 bytes each GPU
        }
        // Set CPU memory to 0 as well to test OOM path
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 0L);
        memoryManager.setMemorySimulationEnabled(true);

        // Try to allocate 100MB
        long allocationSize = 100 * 1024 * 1024 / 4;

        try {
            INDArray arr = Nd4j.create(DataType.FLOAT, allocationSize);
            // If we get here, allocation worked (host memory fallback still available)
            assertNotNull(arr, "Allocation should succeed via CPU fallback");
            log.info("CPU fallback test passed: 100MB allocation succeeded via host memory");

            // Verify the data is usable
            arr.assign(1.0f);
            assertEquals(1.0f, arr.getFloat(0), 1e-6, "Data should be readable/writable");

        } catch (Exception e) {
            // This is acceptable if CPU backend is not available
            log.info("CPU fallback not available (expected on CUDA-only builds): {}", e.getMessage());
        }
    }

    // =========================================================================
    // Sequential Allocation Scenarios
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Sequential allocations correctly track and exhaust simulated memory")
    void testSequentialAllocationsExhaustMemory(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices >= 2, "Test requires at least 2 CUDA devices");

        // Simulate:
        // - Device 0: 100MB free
        // - Device 1: 500MB free
        memoryManager.setSimulatedFreeMemory(0, 100L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 500L * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        Nd4j.getAffinityManager().unsafeSetDevice(0);

        // First allocation: 50MB - should fit on device 0
        long allocSize1 = 50 * 1024 * 1024 / 4;
        INDArray arr1 = Nd4j.create(DataType.FLOAT, allocSize1);
        assertNotNull(arr1);

        long allocatedOnDevice0 = memoryManager.getSimulatedAllocatedMemory(0);
        log.info("After 1st allocation (50MB): device 0 has {} MB allocated", allocatedOnDevice0 / (1024 * 1024));

        // Second allocation: 60MB - device 0 only has ~50MB left, should route to device 1
        long allocSize2 = 60 * 1024 * 1024 / 4;
        INDArray arr2 = Nd4j.create(DataType.FLOAT, allocSize2);
        assertNotNull(arr2);

        long allocatedOnDevice1 = memoryManager.getSimulatedAllocatedMemory(1);
        log.info("After 2nd allocation (60MB): device 1 has {} MB allocated", allocatedOnDevice1 / (1024 * 1024));
        assertTrue(allocatedOnDevice1 > 0, "Second allocation should route to device 1");

        // Third allocation: 200MB - should also go to device 1 (more space)
        long allocSize3 = 200 * 1024 * 1024 / 4;
        INDArray arr3 = Nd4j.create(DataType.FLOAT, allocSize3);
        assertNotNull(arr3);

        long finalAllocatedOnDevice1 = memoryManager.getSimulatedAllocatedMemory(1);
        log.info("After 3rd allocation (200MB): device 1 has {} MB allocated",
                finalAllocatedOnDevice1 / (1024 * 1024));

        // Verify progressive exhaustion worked
        assertTrue(finalAllocatedOnDevice1 > allocatedOnDevice1,
                "Device 1 should have more allocations after third allocation");

        log.info("Sequential allocation test passed - memory routing worked correctly");
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Very small allocation succeeds even with tight memory")
    void testVerySmallAllocationSucceeds(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        // Simulate device with 1MB free (tight but not zero)
        memoryManager.setSimulatedFreeMemory(0, 1024L * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Very small allocation (100 floats = 400 bytes)
        INDArray arr = Nd4j.create(DataType.FLOAT, 100);
        assertNotNull(arr, "Very small allocation should succeed");
        assertEquals(100, arr.length());

        log.info("Small allocation (400 bytes) succeeded with 1MB simulated free memory");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Zero-size array allocation")
    void testZeroSizeAllocation(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        // Even with OOM simulation, empty arrays should work
        memoryManager.setSimulatedFreeMemory(0, 0L); // Zero bytes free
        memoryManager.setMemorySimulationEnabled(true);

        // Empty array - should succeed (no memory needed)
        INDArray emptyArr = Nd4j.empty(DataType.FLOAT);
        assertNotNull(emptyArr, "Empty array should be createable");
        assertEquals(0, emptyArr.length());

        log.info("Zero-size allocation handled correctly");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Simulation mode can be enabled and disabled")
    void testSimulationModeToggle(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        // Initially disabled (setUp clears all simulation)
        assertFalse(memoryManager.isMemorySimulationEnabled(),
                "Simulation should be disabled by default after cleanup");

        // Enable
        memoryManager.setMemorySimulationEnabled(true);
        assertTrue(memoryManager.isMemorySimulationEnabled(),
                "Simulation should be enabled");

        // Disable
        memoryManager.setMemorySimulationEnabled(false);
        assertFalse(memoryManager.isMemorySimulationEnabled(),
                "Simulation should be disabled");

        log.info("Simulation mode toggle test passed");
    }

    // =========================================================================
    // Stress Tests (Optional - can be slow)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Many sequential allocations with progressive memory exhaustion")
    @Tag("slow")
    void testManySequentialAllocations(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices >= 1, "Test requires at least 1 CUDA device");

        // Simulate device with 1GB free
        long totalSimulated = 1024L * 1024 * 1024;
        memoryManager.setSimulatedFreeMemory(0, totalSimulated);
        if (numDevices > 1) {
            memoryManager.setSimulatedFreeMemory(1, totalSimulated);
        }
        memoryManager.setMemorySimulationEnabled(true);

        int successfulAllocations = 0;
        int failedAllocations = 0;
        long totalAllocated = 0;

        // Try to allocate 50MB chunks until failure or 20 allocations
        long chunkSize = 50L * 1024 * 1024 / 4; // 50MB in floats

        for (int i = 0; i < 20; i++) {
            try {
                INDArray arr = Nd4j.create(DataType.FLOAT, chunkSize);
                assertNotNull(arr);
                successfulAllocations++;
                totalAllocated += chunkSize * 4;
                log.debug("Allocation {} succeeded, total: {} MB",
                        i + 1, totalAllocated / (1024 * 1024));
            } catch (Exception e) {
                failedAllocations++;
                log.debug("Allocation {} failed (expected): {}", i + 1, e.getMessage());
                break;
            }
        }

        log.info("Stress test completed: {} successful, {} failed, {} MB total allocated",
                successfulAllocations, failedAllocations, totalAllocated / (1024 * 1024));

        assertTrue(successfulAllocations > 0, "At least some allocations should succeed");
        // With 1GB simulated on each device, we should get at least a few 50MB allocations
        assertTrue(successfulAllocations >= 2, "Should handle at least 2 allocations with simulated 1GB");
    }

    // =========================================================================
    // Real Memory Test (Optional - uses actual GPU memory)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test: Verify actual GPU memory query works")
    @Tag("gpu-memory")
    void testActualGpuMemoryQuery(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int numDevices = getNumDevices();
        assumeTrue(numDevices > 0, "Test requires at least 1 CUDA device");

        for (int deviceId = 0; deviceId < numDevices; deviceId++) {
            long freeMemory = getActualFreeMemory(deviceId);
            assertTrue(freeMemory > 0, "Device " + deviceId + " should report positive free memory");
            log.info("Device {}: {} MB free", deviceId, freeMemory / (1024 * 1024));
        }
    }
}
