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

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceMemoryManager.DeviceRoutingPolicy;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ops.executioner.DefaultBackendRoutingStrategy;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for device routing, memory management, and allocation strategies.
 *
 * These tests validate the device routing infrastructure including:
 * - Device memory manager allocation strategies
 * - Device routing configuration
 * - Backend routing strategy
 * - Memory cap enforcement
 * - Device priority handling
 *
 * Adam Gibson
 */
@Tag("multi-backend")
@Tag("device-routing")
@Tag("memory-management")
@DisplayName("Device Routing and Memory Management Tests")
public class DeviceRoutingTest {

    private static final Logger log = LoggerFactory.getLogger(DeviceRoutingTest.class);

    private DeviceMemoryManager memoryManager;

    @BeforeAll
    static void setupClass() {
        MultiBackendNativeOpsHolder.enableMultiBackend();
    }

    @BeforeEach
    void setup() {
        memoryManager = DeviceMemoryManager.getInstance();
        // Reset to clean state for each test
        memoryManager.clearDevices();
    }

    // ========================================================================
    // Device Memory Manager - Basic Operations
    // ========================================================================

    @Test
    @DisplayName("DeviceMemoryManager should be singleton")
    void testMemoryManagerSingleton() {
        DeviceMemoryManager mgr1 = DeviceMemoryManager.getInstance();
        DeviceMemoryManager mgr2 = DeviceMemoryManager.getInstance();

        assertNotNull(mgr1);
        assertSame(mgr1, mgr2, "Should return same instance");
    }

    @Test
    @DisplayName("Should register CPU device")
    void testRegisterCpuDevice() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        memoryManager.registerDevice(cpu);

        assertTrue(memoryManager.isDeviceRegistered(cpu),
            "CPU device should be registered");
    }

    @Test
    @DisplayName("Should register multiple devices")
    void testRegisterMultipleDevices() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda0 = DeviceDescriptor.cuda(0);
        DeviceDescriptor cuda1 = DeviceDescriptor.cuda(1);

        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda0);
        memoryManager.registerDevice(cuda1);

        assertTrue(memoryManager.isDeviceRegistered(cpu));
        assertTrue(memoryManager.isDeviceRegistered(cuda0));
        assertTrue(memoryManager.isDeviceRegistered(cuda1));
        assertEquals(3, memoryManager.getRegisteredDeviceCount());
    }

    @Test
    @DisplayName("Should set default device")
    void testSetDefaultDevice() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        memoryManager.registerDevice(cpu);
        memoryManager.setDefaultDevice(cpu);

        DeviceDescriptor defaultDev = memoryManager.getDefaultDevice();
        assertEquals(cpu.getDeviceId(), defaultDev.getDeviceId(),
            "Default device should match set device");
    }

    // ========================================================================
    // Memory Cap Tests
    // ========================================================================

    @Test
    @DisplayName("Should set and get memory cap")
    void testMemoryCapSetGet() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        memoryManager.registerDevice(cpu);

        long cap = 4L * 1024 * 1024 * 1024; // 4GB
        memoryManager.setMemoryCap(cpu, cap);

        long retrievedCap = memoryManager.getMemoryCap(cpu);
        assertEquals(cap, retrievedCap, "Memory cap should match");
    }

    @Test
    @DisplayName("Should track allocation against cap")
    void testAllocationTracking() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        memoryManager.registerDevice(cpu);

        long cap = 1024 * 1024; // 1MB
        memoryManager.setMemoryCap(cpu, cap);

        long allocation = 512 * 1024; // 512KB
        memoryManager.recordAllocation(cpu, allocation);

        long available = memoryManager.getAvailableMemory(cpu);
        assertEquals(cap - allocation, available,
            "Available memory should decrease after allocation");
    }

    @Test
    @DisplayName("Should track deallocation")
    void testDeallocationTracking() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        memoryManager.registerDevice(cpu);

        long cap = 1024 * 1024; // 1MB
        memoryManager.setMemoryCap(cpu, cap);

        long allocation = 512 * 1024; // 512KB
        memoryManager.recordAllocation(cpu, allocation);
        memoryManager.recordDeallocation(cpu, allocation);

        long available = memoryManager.getAvailableMemory(cpu);
        assertEquals(cap, available,
            "Available memory should be restored after deallocation");
    }

    @Test
    @DisplayName("Should calculate memory utilization")
    void testMemoryUtilization() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        memoryManager.registerDevice(cpu);

        long cap = 1000;
        memoryManager.setMemoryCap(cpu, cap);
        memoryManager.recordAllocation(cpu, 500); // 50%

        double utilization = memoryManager.getMemoryUtilization(cpu);
        assertEquals(0.5, utilization, 0.001, "Utilization should be 50%");
    }

    // ========================================================================
    // Device Selection / Routing Policy Tests
    // ========================================================================

    @Test
    @DisplayName("Should select device with PREFER_GPU policy")
    void testPreferGpuPolicy() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda = DeviceDescriptor.cuda(0);

        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda);
        memoryManager.setMemoryCap(cpu, Long.MAX_VALUE);
        memoryManager.setMemoryCap(cuda, Long.MAX_VALUE);

        DeviceDescriptor selected = memoryManager.selectDevice(1024, DeviceRoutingPolicy.PREFER_GPU);

        // Should prefer GPU when available
        assertEquals(DeviceType.CUDA_GPU, selected.getDeviceType(),
            "PREFER_GPU should select GPU device");
    }

    @Test
    @DisplayName("Should select device with PREFER_CPU policy")
    void testPreferCpuPolicy() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda = DeviceDescriptor.cuda(0);

        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda);
        memoryManager.setMemoryCap(cpu, Long.MAX_VALUE);
        memoryManager.setMemoryCap(cuda, Long.MAX_VALUE);

        DeviceDescriptor selected = memoryManager.selectDevice(1024, DeviceRoutingPolicy.PREFER_CPU);

        assertEquals(DeviceType.CPU, selected.getDeviceType(),
            "PREFER_CPU should select CPU device");
    }

    @Test
    @DisplayName("Should select device with MOST_FREE policy")
    void testMostFreePolicy() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda = DeviceDescriptor.cuda(0);

        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda);

        // Set same caps
        memoryManager.setMemoryCap(cpu, 1000);
        memoryManager.setMemoryCap(cuda, 1000);

        // Load CPU more than CUDA
        memoryManager.recordAllocation(cpu, 800);  // 80% loaded
        memoryManager.recordAllocation(cuda, 200); // 20% loaded

        DeviceDescriptor selected = memoryManager.selectDevice(100, DeviceRoutingPolicy.MOST_FREE);

        assertEquals(cuda.getDeviceId(), selected.getDeviceId(),
            "MOST_FREE should select device with more free memory (CUDA)");
    }

    @Test
    @DisplayName("Should fallback when preferred device has insufficient memory")
    void testFallbackOnInsufficientMemory() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda = DeviceDescriptor.cuda(0);

        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda);

        // CUDA has small cap, CPU has large cap
        memoryManager.setMemoryCap(cuda, 100);
        memoryManager.setMemoryCap(cpu, 10000);

        // Try to allocate more than CUDA can handle
        DeviceDescriptor selected = memoryManager.selectDevice(1000, DeviceRoutingPolicy.PREFER_GPU);

        // Should fallback to CPU since CUDA doesn't have enough
        assertEquals(DeviceType.CPU, selected.getDeviceType(),
            "Should fallback to CPU when GPU has insufficient memory");
    }

    // ========================================================================
    // OOM Failover Tests (one GPU full -> use the other, else CPU)
    // ========================================================================

    @Test
    @DisplayName("Should fail an allocation over from a full GPU to the other GPU")
    void testFailoverToOtherGpuWhenOneFull() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda0 = DeviceDescriptor.cuda(0);
        DeviceDescriptor cuda1 = DeviceDescriptor.cuda(1);
        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda0);
        memoryManager.registerDevice(cuda1);

        // GPU0 nearly full (10MB free), GPU1 has 8GB, CPU has 32GB.
        memoryManager.setSimulatedFreeMemory(0, 10L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 32L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);
        try {
            // A 100MB allocation cannot fit on GPU0 -> must fail over to GPU1.
            DeviceDescriptor target = memoryManager.selectFailoverDevice(100L * 1024 * 1024, 0);
            assertNotNull(target, "Failover must find a device when another GPU has room");
            assertEquals(DeviceType.CUDA_GPU, target.getDeviceType());
            assertEquals(1, target.getDeviceIndex(),
                "Should fail over from full GPU0 to GPU1, not pin to the full device");
        } finally {
            memoryManager.clearAllMemorySimulation();
        }
    }

    @Test
    @DisplayName("Should fall back to CPU when every GPU is full")
    void testFailoverToCpuWhenAllGpusFull() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda0 = DeviceDescriptor.cuda(0);
        DeviceDescriptor cuda1 = DeviceDescriptor.cuda(1);
        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda0);
        memoryManager.registerDevice(cuda1);
        memoryManager.setFallbackDevice(cpu);
        memoryManager.setAutoFallbackEnabled(true);

        // Both GPUs full (10MB each), CPU has 32GB.
        memoryManager.setSimulatedFreeMemory(0, 10L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 10L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 32L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);
        try {
            DeviceDescriptor target = memoryManager.selectFailoverDevice(100L * 1024 * 1024, 0);
            assertNotNull(target, "Failover must reach the CPU fallback when no GPU fits");
            assertEquals(DeviceType.CPU, target.getDeviceType(),
                "Should fall back to CPU when every GPU is full");
        } finally {
            memoryManager.clearAllMemorySimulation();
        }
    }

    @Test
    @DisplayName("Should return null when no device — not even the fallback — can fit")
    void testFailoverReturnsNullWhenNothingFits() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda0 = DeviceDescriptor.cuda(0);
        DeviceDescriptor cuda1 = DeviceDescriptor.cuda(1);
        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda0);
        memoryManager.registerDevice(cuda1);
        memoryManager.setFallbackDevice(cpu);
        memoryManager.setAutoFallbackEnabled(true);

        // Everything tiny — a 100MB allocation fits nowhere.
        memoryManager.setSimulatedFreeMemory(0, 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);
        try {
            DeviceDescriptor target = memoryManager.selectFailoverDevice(100L * 1024 * 1024, 0);
            assertNull(target,
                "When nothing can fit, failover returns null so the caller surfaces a real OOM");
        } finally {
            memoryManager.clearAllMemorySimulation();
        }
    }

    // ========================================================================
    // Device Priority Tests
    // ========================================================================

    @Test
    @DisplayName("Should respect device priority in selection")
    void testDevicePrioritySelection() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        DeviceDescriptor cuda0 = DeviceDescriptor.cuda(0);
        DeviceDescriptor cuda1 = DeviceDescriptor.cuda(1);

        memoryManager.registerDevice(cpu);
        memoryManager.registerDevice(cuda0);
        memoryManager.registerDevice(cuda1);

        // Set memory caps
        memoryManager.setMemoryCap(cpu, Long.MAX_VALUE);
        memoryManager.setMemoryCap(cuda0, Long.MAX_VALUE);
        memoryManager.setMemoryCap(cuda1, Long.MAX_VALUE);

        // Set priorities - cuda1 highest
        memoryManager.setDevicePriority(cpu, 10);
        memoryManager.setDevicePriority(cuda0, 50);
        memoryManager.setDevicePriority(cuda1, 100);

        // Set cuda1 as default since MEMORY_PRIORITY respects default device first
        // when it has capacity. For true priority-based selection we'd need a different policy.
        memoryManager.setDefaultDevice(cuda1);

        DeviceDescriptor selected = memoryManager.selectDevice(1024, DeviceRoutingPolicy.MEMORY_PRIORITY);

        assertEquals(cuda1.getDeviceId(), selected.getDeviceId(),
            "Should select highest priority device (via default)");

        // Also verify that when default doesn't have capacity, it falls back to priority
        memoryManager.setDefaultDevice(cpu);
        memoryManager.setMemoryCap(cpu, 100); // Too small
        DeviceDescriptor selectedFallback = memoryManager.selectDevice(1024, DeviceRoutingPolicy.MEMORY_PRIORITY);

        // Should fallback to highest priority device with capacity (cuda1)
        assertEquals(cuda1.getDeviceId(), selectedFallback.getDeviceId(),
            "Should fallback to highest priority device when default lacks capacity");
    }

    // ========================================================================
    // Device Routing Configuration Tests
    // ========================================================================

    @Test
    @DisplayName("Should create default routing configuration")
    void testDefaultRoutingConfiguration() {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.defaultConfig();

        assertNotNull(config);
        assertTrue(config.isRouteOpsByInputLocation(),
            "Default should route by input location");
        assertFalse(config.isLogRoutingDecisions(),
            "Default should not log routing decisions");
    }

    @Test
    @DisplayName("Should build custom routing configuration")
    void testCustomRoutingConfiguration() {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.builder()
            .routeOpsByInputLocation(true)
            .logRoutingDecisions(true)
            .logDataTransfers(true)
            .preferredDeviceTypes(Arrays.asList(DeviceType.CUDA_GPU, DeviceType.CPU))
            .build();

        assertTrue(config.isRouteOpsByInputLocation());
        assertTrue(config.isLogRoutingDecisions());
        assertTrue(config.isLogDataTransfers());
        assertEquals(2, config.getPreferredDeviceTypes().size());
        assertEquals(DeviceType.CUDA_GPU, config.getPreferredDeviceTypes().get(0));
    }

    @Test
    @DisplayName("Should get and set current configuration")
    void testCurrentConfiguration() {
        DeviceRoutingConfiguration custom = DeviceRoutingConfiguration.builder()
            .logRoutingDecisions(true)
            .build();

        DeviceRoutingConfiguration.setCurrent(custom);
        DeviceRoutingConfiguration current = DeviceRoutingConfiguration.current();

        assertTrue(current.isLogRoutingDecisions(),
            "Current config should match set config");

        // Reset to default
        DeviceRoutingConfiguration.setCurrent(DeviceRoutingConfiguration.defaultConfig());
    }

    // ========================================================================
    // Backend Routing Strategy Tests
    // ========================================================================

    @Test
    @DisplayName("Should get backend routing strategy singleton")
    void testBackendRoutingStrategySingleton() {
        DefaultBackendRoutingStrategy strategy1 = DefaultBackendRoutingStrategy.getInstance();
        DefaultBackendRoutingStrategy strategy2 = DefaultBackendRoutingStrategy.getInstance();

        assertNotNull(strategy1);
        assertSame(strategy1, strategy2);
    }

    @Test
    @DisplayName("Should get default device from routing strategy")
    void testRoutingStrategyDefaultDevice() {
        DefaultBackendRoutingStrategy strategy = DefaultBackendRoutingStrategy.getInstance();

        DeviceDescriptor current = strategy.getCurrentDevice();

        assertNotNull(current, "Should have a current device");
        log.info("Current device from routing strategy: {}", current.getDeviceId());
    }

    @Test
    @DisplayName("Should set current device in routing strategy")
    void testRoutingStrategySetCurrentDevice() {
        DefaultBackendRoutingStrategy strategy = DefaultBackendRoutingStrategy.getInstance();

        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        strategy.setCurrentDevice(cpu);

        DeviceDescriptor current = strategy.getCurrentDevice();
        assertEquals(cpu.getDeviceId(), current.getDeviceId(),
            "Current device should match set device");
    }

    @Test
    @DisplayName("Should check backend availability")
    void testBackendAvailabilityCheck() {
        DefaultBackendRoutingStrategy strategy = DefaultBackendRoutingStrategy.getInstance();

        // At least one backend should be available
        boolean anyAvailable =
            strategy.isBackendAvailable(DeviceType.CPU) ||
            strategy.isBackendAvailable(DeviceType.CUDA_GPU);

        assertTrue(anyAvailable, "At least one backend should be available");
    }

    @Test
    @DisplayName("Should get routing statistics")
    void testRoutingStatistics() {
        DefaultBackendRoutingStrategy strategy = DefaultBackendRoutingStrategy.getInstance();
        strategy.resetStats();

        String stats = strategy.getRoutingStats();

        assertNotNull(stats);
        assertTrue(stats.contains("Backend Routing Stats"),
            "Stats should contain header");
        assertTrue(stats.contains("Total routing decisions"),
            "Stats should contain routing count");

        log.info("Routing stats:\n{}", stats);
    }

    // ========================================================================
    // Multi-Backend Routing Tests
    // ========================================================================

    @Test
    @DisplayName("Should route to correct backend based on device type")
    @EnabledIf("multiBackendAvailable")
    void testRoutingToCorrectBackend() {
        DefaultBackendRoutingStrategy strategy = DefaultBackendRoutingStrategy.getInstance();

        // Get ops for each available backend
        var cpuOps = strategy.getNativeOpsForDeviceType(DeviceType.CPU);
        var cudaOps = strategy.getNativeOpsForDeviceType(DeviceType.CUDA_GPU);

        if (MultiBackendNativeOpsHolder.getInstance().isBackendAvailable(DeviceType.CPU)) {
            assertNotNull(cpuOps, "Should get CPU ops when available");
        }

        if (MultiBackendNativeOpsHolder.getInstance().isBackendAvailable(DeviceType.CUDA_GPU)) {
            assertNotNull(cudaOps, "Should get CUDA ops when available");
        }
    }

    // ========================================================================
    // Test Condition Methods
    // ========================================================================

    static boolean multiBackendAvailable() {
        MultiBackendNativeOpsHolder.enableMultiBackend();
        return MultiBackendNativeOpsHolder.isMultiBackendEnabled();
    }
}
