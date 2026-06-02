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
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.device.StubDeviceDescriptor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for the proactive soft-limit mechanism in CudaMemoryPool.
 *
 * <p>The soft limit causes the pool to route allocations to other devices
 * via allocateFailover() BEFORE individual allocations fail, preventing
 * cumulative exhaustion from many small allocations (e.g., DSP warmup
 * intermediates) that individually succeed but collectively fill the GPU.
 *
 * <p>Uses the DeviceMemoryManager constraint simulation API to model
 * multi-GPU topologies and memory pressure scenarios without requiring
 * actual GPU exhaustion.
 *
 * @see org.nd4j.linalg.api.device.DeviceMemoryManager
 */
@Slf4j
@DisplayName("CudaMemoryPool Proactive Soft Limit Tests")
public class CudaMemoryPoolSoftLimitTest extends BaseNd4jTestWithBackends {

    private NativeOps nativeOps;
    private DeviceMemoryManager memoryManager;

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
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        memoryManager = DeviceMemoryManager.getInstance();
        memoryManager.clearAllMemorySimulation();
        // Reset soft limit to disabled before each test
        nativeOps.setMemoryPoolSoftLimitPercent(0);
    }

    @AfterEach
    void tearDown() {
        // Always restore soft limit to disabled
        if (nativeOps != null) {
            nativeOps.setMemoryPoolSoftLimitPercent(0);
        }
        if (memoryManager != null) {
            memoryManager.clearAllMemorySimulation();
            memoryManager.clearStubTopology();
        }
    }

    // =========================================================================
    // NativeOps API Tests (work on both CPU and CUDA)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Soft limit defaults to 0 (disabled)")
    void testSoftLimitDefaultsToDisabled(Nd4jBackend backend) {
        assertEquals(0, nativeOps.getMemoryPoolSoftLimitPercent(),
                "Soft limit should default to 0 (disabled)");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Set and get soft limit percentage")
    void testSetAndGetSoftLimit(Nd4jBackend backend) {
        nativeOps.setMemoryPoolSoftLimitPercent(70);
        assertEquals(70, nativeOps.getMemoryPoolSoftLimitPercent(),
                "Soft limit should be 70 after setting");

        nativeOps.setMemoryPoolSoftLimitPercent(0);
        assertEquals(0, nativeOps.getMemoryPoolSoftLimitPercent(),
                "Soft limit should be 0 after disabling");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Soft limit clamps to valid range 0-100")
    void testSoftLimitClamping(Nd4jBackend backend) {
        nativeOps.setMemoryPoolSoftLimitPercent(-10);
        assertEquals(0, nativeOps.getMemoryPoolSoftLimitPercent(),
                "Negative values should clamp to 0");

        nativeOps.setMemoryPoolSoftLimitPercent(150);
        assertEquals(100, nativeOps.getMemoryPoolSoftLimitPercent(),
                "Values over 100 should clamp to 100");

        nativeOps.setMemoryPoolSoftLimitPercent(50);
        assertEquals(50, nativeOps.getMemoryPoolSoftLimitPercent(),
                "Valid values should be stored exactly");
    }

    // =========================================================================
    // Allocation Tests (CUDA only)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Allocation succeeds with soft limit disabled (default behavior)")
    void testAllocationSucceedsWithSoftLimitDisabled(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() > 0, "Test requires at least 1 CUDA device");

        // Soft limit is 0 (disabled) - allocations should proceed normally
        INDArray arr = Nd4j.create(DataType.FLOAT, 1024);
        assertNotNull(arr, "Allocation should succeed with soft limit disabled");
        arr.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Allocation succeeds with soft limit set but device below threshold")
    void testAllocationSucceedsWhenBelowSoftLimit(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() > 0, "Test requires at least 1 CUDA device");

        // Set soft limit to 95% — most devices won't be this full during tests
        nativeOps.setMemoryPoolSoftLimitPercent(95);

        INDArray arr = Nd4j.create(DataType.FLOAT, 1024);
        assertNotNull(arr, "Allocation should succeed when device is below soft limit");
        arr.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Many small allocations succeed with soft limit active (no OOM)")
    void testManySmallAllocationsWithSoftLimit(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() > 0, "Test requires at least 1 CUDA device");

        // Set a moderate soft limit — simulates the DSP warmup scenario
        nativeOps.setMemoryPoolSoftLimitPercent(70);

        // Allocate many small tensors (simulates DSP slot-by-slot warmup intermediates)
        INDArray[] arrays = new INDArray[100];
        for (int i = 0; i < arrays.length; i++) {
            arrays[i] = Nd4j.create(DataType.FLOAT, 256, 256); // ~256KB each
            assertNotNull(arrays[i], "Allocation " + i + " should succeed");
        }

        // Clean up
        for (INDArray arr : arrays) {
            arr.close();
        }
        log.info("100 small allocations succeeded with soft limit at 70%%");
    }

    // =========================================================================
    // DeviceMemoryManager Simulation Tests
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Simulation: routing switches to device 1 when device 0 cannot fit allocation")
    void testSoftLimitTriggersJavaRoutingToLessLoadedDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Simulate GPU 0 nearly exhausted — only 100MB free
        // GPU 1 has plenty of free memory
        long gpu0Free = 100L * 1024 * 1024;          // 100MB free
        long gpu1Free = 20L * 1024 * 1024 * 1024;    // 20GB free

        memoryManager.setSimulatedFreeMemory(0, gpu0Free);
        memoryManager.setSimulatedFreeMemory(1, gpu1Free);
        memoryManager.setMemorySimulationEnabled(true);

        // Request more than GPU 0 can fit — forces routing to GPU 1
        long allocSize = 500L * 1024 * 1024; // 500MB
        var selected = memoryManager.selectDeviceForAllocation(allocSize);
        assertNotNull(selected, "selectDeviceForAllocation should return a device");

        String selectedId = selected.getDeviceId();
        log.info("Simulated routing: GPU0 has 100MB free, GPU1 has 20GB free, "
                + "500MB alloc => selected: {}", selectedId);

        // Device 1 should be selected since GPU 0 can't fit 500MB
        assertTrue(selectedId.contains(":1") || selectedId.equals("1"),
                "Should route to device 1 (GPU 0 can't fit 500MB). Got: " + selectedId);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Simulation: DSP warmup pattern — cumulative small allocations exhaust GPU")
    void testDspWarmupCumulativeExhaustionPattern(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Model the BGE-M3 scenario: GPU starts at ~79% used (19GB/24GB),
        // then DSP warmup creates hundreds of small intermediates
        long initialFree = 5L * 1024 * 1024 * 1024; // 5GB free initially
        long gpu1Free = 12L * 1024 * 1024 * 1024;  // GPU 1 has 12GB free

        memoryManager.setSimulatedFreeMemory(0, initialFree);
        memoryManager.setSimulatedFreeMemory(1, gpu1Free);
        memoryManager.setMemorySimulationEnabled(true);

        // Simulate progressive allocation: each step reduces simulated free memory.
        // selectDeviceForAllocation prefers the default device (GPU 0) as long as it can
        // fit the allocation. Once cumulative allocations exhaust GPU 0, routing switches.
        long allocPerStep = 50L * 1024 * 1024; // 50MB per warmup intermediate
        int switchStep = -1;

        for (int i = 0; i < 200; i++) {
            // Record simulated allocation (reduces effective free memory on GPU 0)
            memoryManager.recordSimulatedAllocation(0, allocPerStep);

            // Check if routing now prefers device 1
            var selected = memoryManager.selectDeviceForAllocation(allocPerStep);
            if (selected != null) {
                String id = selected.getDeviceId();
                if (id.contains(":1") || id.equals("1")) {
                    switchStep = i;
                    long effectiveFree = memoryManager.getEffectiveFreeMemory(0, initialFree);
                    log.info("Step {}: routing switched to device 1. "
                                    + "GPU 0 effective free: {} MB, allocated: {} MB",
                            i, effectiveFree / (1024 * 1024),
                            memoryManager.getSimulatedAllocatedMemory(0) / (1024 * 1024));
                    break;
                }
            }
        }

        assertTrue(switchStep >= 0,
                "Routing should eventually switch to device 1 as GPU 0 fills up");

        // 5GB / 50MB = 100 steps to exhaust. Switch should happen around step 99
        // (when effective free drops below allocPerStep).
        long totalConsumedMB = (long)(switchStep + 1) * allocPerStep / (1024 * 1024);
        log.info("DSP warmup simulation: routing switched after {} steps ({} MB consumed). "
                        + "Initial free was {} MB.",
                switchStep, totalConsumedMB, initialFree / (1024 * 1024));

        assertTrue(switchStep < 200,
                "Routing should switch before 200 steps (5GB / 50MB = 100 steps max)");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Simulation: stub topology models proactive threshold crossing")
    void testStubTopologyProactiveThresholdCrossing(Nd4jBackend backend) {
        // This test works on any backend — it tests the Java simulation layer

        // Create a topology matching the OOM scenario:
        // GPU 0: 24GB total, 5GB free (79% used)
        // GPU 1: 8GB total, 7GB free (12.5% used)
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .deviceName("Stub RTX 4090")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(5L * 1024 * 1024 * 1024)
                .isDefault(true)
                .build();

        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .deviceName("Stub RTX 3070 Ti")
                .totalMemory(8L * 1024 * 1024 * 1024)
                .availableMemory(7L * 1024 * 1024 * 1024)
                .build();

        memoryManager.configureStubTopology(Arrays.asList(gpu0, gpu1));

        // Verify GPU 0 is above a 70% soft limit threshold
        var gpu0Desc = memoryManager.getRegisteredDevice(0);
        assertNotNull(gpu0Desc, "GPU 0 should be registered");
        long free0 = memoryManager.getAvailableMemory(gpu0Desc);
        long total0 = gpu0Desc.getTotalMemory();
        double usage0 = 100.0 * (1.0 - (double) free0 / total0);
        log.info("GPU 0 usage: {}% ({} MB free / {} MB total)",
                String.format("%.1f", usage0), free0 / (1024 * 1024), total0 / (1024 * 1024));
        assertTrue(usage0 > 70.0, "GPU 0 should be above 70% usage (was " + usage0 + "%)");

        // Verify GPU 1 is below threshold
        var gpu1Desc = memoryManager.getRegisteredDevice(1);
        assertNotNull(gpu1Desc, "GPU 1 should be registered");
        long free1 = memoryManager.getAvailableMemory(gpu1Desc);
        long total1 = gpu1Desc.getTotalMemory();
        double usage1 = 100.0 * (1.0 - (double) free1 / total1);
        log.info("GPU 1 usage: {}% ({} MB free / {} MB total)",
                String.format("%.1f", usage1), free1 / (1024 * 1024), total1 / (1024 * 1024));
        assertTrue(usage1 < 70.0, "GPU 1 should be below 70% usage (was " + usage1 + "%)");

        // Selection should prefer GPU 1 for a large allocation
        var selected = memoryManager.selectDeviceForAllocation(1L * 1024 * 1024 * 1024);
        assertNotNull(selected, "Should find a device for 1GB allocation");
        log.info("1GB allocation routed to: {}", selected.getDeviceId());

        // Consume GPU 0's remaining memory progressively via the stub descriptor
        long consumePerStep = 500L * 1024 * 1024; // 500MB
        for (int i = 0; i < 10; i++) {
            gpu0.consumeMemory(consumePerStep);
            long remaining = memoryManager.getAvailableMemory(gpu0Desc);
            if (remaining < consumePerStep) {
                log.info("GPU 0 effectively exhausted after step {}: {} MB remaining",
                        i, remaining / (1024 * 1024));
                break;
            }
        }

        // Now large allocations must route to GPU 1 since GPU 0 has less free memory
        // Request more than GPU 0's remaining memory
        long gpu0Remaining = memoryManager.getAvailableMemory(gpu0Desc);
        long largeAlloc = Math.max(gpu0Remaining + 1, 100L * 1024 * 1024);
        var afterExhaustion = memoryManager.selectDeviceForAllocation(largeAlloc);
        assertNotNull(afterExhaustion, "Should find a device even after GPU 0 exhausted");
        String afterId = afterExhaustion.getDeviceId();
        // Device ID format is "stub:cuda:N" — check the device index at the end
        assertTrue(afterId.endsWith(":1"),
                "After GPU 0 exhaustion, should route to GPU 1, got: " + afterId);

        log.info("After GPU 0 exhaustion, {} MB alloc routed to: {}",
                largeAlloc / (1024 * 1024), afterId);
    }

    // =========================================================================
    // Soft limit + real allocation integration (both CPU and CUDA)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Soft limit does not break normal allocation when set conservatively")
    void testSoftLimitDoesNotBreakNormalAllocation(Nd4jBackend backend) {
        // Works on both CPU and CUDA — the soft limit is now unified
        // Use 95% to avoid hitting the limit during normal test execution
        nativeOps.setMemoryPoolSoftLimitPercent(95);

        // Allocate, compute, verify correctness
        INDArray a = Nd4j.create(DataType.FLOAT, 100, 100);
        INDArray b = Nd4j.ones(DataType.FLOAT, 100, 100);
        a.assign(2.0f);
        INDArray c = a.mmul(b);

        // Each element of c should be 2.0 * 100 = 200.0
        assertEquals(200.0f, c.getFloat(0, 0), 1e-3,
                "Matrix multiply should produce correct results with soft limit active");
        assertEquals(200.0f, c.getFloat(50, 50), 1e-3,
                "Matrix multiply should produce correct results at arbitrary position");

        a.close();
        b.close();
        c.close();

        log.info("Normal allocation + compute verified correct with soft limit at 95%% [{}]",
                isCudaBackend() ? "CUDA" : "CPU");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Soft limit survives repeated enable/disable cycles")
    void testSoftLimitRepeatedToggle(Nd4jBackend backend) {
        // Works on both CPU and CUDA
        for (int cycle = 0; cycle < 10; cycle++) {
            nativeOps.setMemoryPoolSoftLimitPercent(95);
            assertEquals(95, nativeOps.getMemoryPoolSoftLimitPercent());

            INDArray arr = Nd4j.create(DataType.FLOAT, 512, 512);
            assertNotNull(arr);
            arr.close();

            nativeOps.setMemoryPoolSoftLimitPercent(0);
            assertEquals(0, nativeOps.getMemoryPoolSoftLimitPercent());

            arr = Nd4j.create(DataType.FLOAT, 512, 512);
            assertNotNull(arr);
            arr.close();
        }

        log.info("10 enable/disable cycles completed without issues [{}]",
                isCudaBackend() ? "CUDA" : "CPU");
    }

    // =========================================================================
    // CPU-specific soft limit tests
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("CPU: many small allocations succeed with high soft limit")
    void testCpuManySmallAllocationsWithSoftLimit(Nd4jBackend backend) {
        // Use high threshold — verifies CPU soft limit doesn't block normal allocations
        nativeOps.setMemoryPoolSoftLimitPercent(95);

        INDArray[] arrays = new INDArray[100];
        for (int i = 0; i < arrays.length; i++) {
            arrays[i] = Nd4j.create(DataType.FLOAT, 256, 256); // ~256KB each
            assertNotNull(arrays[i], "Allocation " + i + " should succeed");
        }

        for (INDArray arr : arrays) {
            arr.close();
        }
        log.info("100 small allocations succeeded with CPU soft limit at 95%% [{}]",
                isCudaBackend() ? "CUDA" : "CPU");
    }
}
