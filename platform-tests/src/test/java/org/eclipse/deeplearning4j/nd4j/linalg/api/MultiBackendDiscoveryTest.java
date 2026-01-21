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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.nativeblas.NativeOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for multi-backend discovery, device management, and routing infrastructure.
 *
 * These tests validate the backend discovery and device routing systems work correctly,
 * even when only a single backend is available. When multiple backends are present
 * (e.g., when running with -Pmulti-backend-dual), additional cross-backend tests are enabled.
 *
 * Run with different profiles:
 * - Default (single backend): mvn test -Dtest=MultiBackendDiscoveryTest
 * - Dual backend: mvn test -Pmulti-backend-dual -Dtest=MultiBackendDiscoveryTest
 * - All backends: mvn test -Pmulti-backend-all -Dtest=MultiBackendDiscoveryTest
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Tag("multi-backend")
@Tag("backend-discovery")
@DisplayName("Multi-Backend Discovery and Routing Tests")
public class MultiBackendDiscoveryTest {

    private static final Logger log = LoggerFactory.getLogger(MultiBackendDiscoveryTest.class);

    @BeforeAll
    static void setup() {
        // Enable multi-backend mode
        MultiBackendNativeOpsHolder.enableMultiBackend();
        log.info("Multi-backend test setup complete");
        log.info(MultiBackendNativeOpsHolder.getInstance().getBackendInfo());
    }

    // ========================================================================
    // Backend Discovery Tests
    // ========================================================================

    @Test
    @DisplayName("Should discover at least one backend")
    void testAtLeastOneBackendAvailable() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        Set<DeviceType> available = holder.getAvailableBackendTypes();

        assertNotNull(available, "Available backends should not be null");
        assertFalse(available.isEmpty(), "At least one backend should be available");

        log.info("Discovered {} backend(s): {}", available.size(), available);
    }

    @Test
    @DisplayName("Should have a primary backend set")
    void testPrimaryBackendSet() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        NativeOps primaryOps = holder.getPrimaryOps();
        DeviceType primaryType = holder.getPrimaryBackendType();

        assertNotNull(primaryOps, "Primary NativeOps should not be null");
        assertNotNull(primaryType, "Primary backend type should not be null");

        log.info("Primary backend: {}", primaryType);
    }

    @Test
    @DisplayName("Should return correct NativeOps for available device types")
    void testGetOpsForAvailableDeviceTypes() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        for (DeviceType type : holder.getAvailableBackendTypes()) {
            NativeOps ops = holder.getOpsForDeviceType(type);

            assertNotNull(ops, "NativeOps for " + type + " should not be null");
            assertTrue(holder.isBackendAvailable(type), type + " should be marked as available");

            log.info("Backend {} -> NativeOps class: {}", type, ops.getClass().getSimpleName());
        }
    }

    @Test
    @DisplayName("Should return primary ops for unavailable device types")
    void testGetOpsForUnavailableDeviceTypes() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();
        NativeOps primaryOps = holder.getPrimaryOps();

        // Find a device type that isn't available
        for (DeviceType type : DeviceType.values()) {
            if (!holder.isBackendAvailable(type)) {
                NativeOps ops = holder.getOpsForDeviceType(type);

                // Should fallback to primary
                assertSame(primaryOps, ops,
                    "Unavailable device type " + type + " should fallback to primary ops");

                log.info("Unavailable backend {} falls back to primary", type);
                return; // Found one, test passes
            }
        }

        // If all device types are available, that's fine too
        log.info("All device types are available - fallback test skipped");
    }

    @Test
    @DisplayName("Should handle null device type gracefully")
    void testNullDeviceTypeHandling() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        NativeOps ops = holder.getOpsForDeviceType(null);

        assertNotNull(ops, "Should return primary ops for null device type");
        assertSame(holder.getPrimaryOps(), ops, "Null device type should return primary ops");
    }

    @Test
    @DisplayName("Should report backend info correctly")
    void testBackendInfoReporting() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        String info = holder.getBackendInfo();

        assertNotNull(info, "Backend info should not be null");
        assertFalse(info.isEmpty(), "Backend info should not be empty");
        assertTrue(info.contains("Multi-Backend"), "Should contain header");
        assertTrue(info.contains("Initialized: true"), "Should show initialized");

        log.info("Backend info:\n{}", info);
    }

    // ========================================================================
    // Device Descriptor Tests
    // ========================================================================

    @Test
    @DisplayName("Should create CPU device descriptor")
    void testCpuDeviceDescriptor() {
        DeviceDescriptor cpu = DeviceDescriptor.cpu();

        assertNotNull(cpu, "CPU descriptor should not be null");
        assertEquals(DeviceType.CPU, cpu.getDeviceType(), "Should be CPU type");
        assertNotNull(cpu.getDeviceId(), "Device ID should not be null");

        log.info("CPU device: {}", cpu.getDeviceId());
    }

    @Test
    @DisplayName("Should create indexed CPU device descriptors")
    void testIndexedCpuDeviceDescriptor() {
        DeviceDescriptor cpu0 = DeviceDescriptor.cpu(0);
        DeviceDescriptor cpu1 = DeviceDescriptor.cpu(1);

        assertNotNull(cpu0, "CPU:0 descriptor should not be null");
        assertNotNull(cpu1, "CPU:1 descriptor should not be null");
        assertEquals(DeviceType.CPU, cpu0.getDeviceType());
        assertEquals(DeviceType.CPU, cpu1.getDeviceType());
        assertNotEquals(cpu0.getDeviceId(), cpu1.getDeviceId(),
            "Different indices should have different IDs");
    }

    @Test
    @DisplayName("Should create CUDA device descriptor")
    void testCudaDeviceDescriptor() {
        DeviceDescriptor cuda = DeviceDescriptor.cuda(0);

        assertNotNull(cuda, "CUDA descriptor should not be null");
        assertEquals(DeviceType.CUDA_GPU, cuda.getDeviceType(), "Should be CUDA type");
        assertNotNull(cuda.getDeviceId(), "Device ID should not be null");

        log.info("CUDA device: {}", cuda.getDeviceId());
    }

    @Test
    @DisplayName("Should get ops for device descriptor")
    void testGetOpsForDeviceDescriptor() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        NativeOps opsForCpu = holder.getOpsForDevice(cpu);

        assertNotNull(opsForCpu, "Should return ops for CPU device");

        // If CPU backend is available, should return CPU ops
        if (holder.isBackendAvailable(DeviceType.CPU)) {
            assertSame(holder.getCpuOps(), opsForCpu,
                "CPU device should return CPU ops when available");
        }
    }

    @Test
    @DisplayName("Should handle null device descriptor")
    void testNullDeviceDescriptor() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        NativeOps ops = holder.getOpsForDevice(null);

        assertNotNull(ops, "Should return primary ops for null device");
        assertSame(holder.getPrimaryOps(), ops);
    }

    // ========================================================================
    // Device Memory Manager Tests
    // ========================================================================

    @Test
    @DisplayName("Should get DeviceMemoryManager singleton")
    void testDeviceMemoryManagerSingleton() {
        DeviceMemoryManager mgr1 = DeviceMemoryManager.getInstance();
        DeviceMemoryManager mgr2 = DeviceMemoryManager.getInstance();

        assertNotNull(mgr1, "DeviceMemoryManager should not be null");
        assertSame(mgr1, mgr2, "Should return same singleton instance");
    }

    @Test
    @DisplayName("Should register and track devices")
    void testDeviceRegistration() {
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();

        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        mgr.registerDevice(cpu);

        // Should not throw
        assertDoesNotThrow(() -> mgr.getAvailableMemory(cpu));
    }

    @Test
    @DisplayName("Should set and respect memory caps")
    void testMemoryCaps() {
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        DeviceDescriptor cpu = DeviceDescriptor.cpu();

        long cap = 1024L * 1024 * 1024; // 1GB
        mgr.setMemoryCap(cpu, cap);

        // The cap should be respected in allocation decisions
        // (Full verification would require actual allocation)
        log.info("Set memory cap for {} to {} bytes", cpu.getDeviceId(), cap);
    }

    @Test
    @DisplayName("Should select device for allocation")
    void testDeviceSelectionForAllocation() {
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();

        // Register at least CPU
        DeviceDescriptor cpu = DeviceDescriptor.cpu();
        mgr.registerDevice(cpu);

        long allocationSize = 1024 * 1024; // 1MB
        DeviceDescriptor selected = mgr.selectDeviceForAllocation(allocationSize);

        assertNotNull(selected, "Should select a device for allocation");
        log.info("Selected device for {}B allocation: {}", allocationSize, selected.getDeviceId());
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    @Test
    @DisplayName("Should track op execution statistics")
    void testOpExecutionStatistics() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        // Reset and record some ops
        holder.resetStatistics();
        assertEquals(0, holder.getTotalOpCount(), "Should start at 0 after reset");

        holder.recordOpExecution(DeviceType.CPU);
        holder.recordOpExecution(DeviceType.CPU);
        holder.recordOpExecution(DeviceType.CUDA_GPU);

        assertEquals(2, holder.getOpCount(DeviceType.CPU), "Should count CPU ops");
        assertEquals(1, holder.getOpCount(DeviceType.CUDA_GPU), "Should count CUDA ops");
        assertEquals(3, holder.getTotalOpCount(), "Should count total ops");

        // Log statistics
        holder.logStatistics();
    }

    @Test
    @DisplayName("Should provide legacy op count accessors")
    void testLegacyOpCountAccessors() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        holder.resetStatistics();
        holder.recordOpExecution(DeviceType.CPU);
        holder.recordOpExecution(DeviceType.CUDA_GPU);

        assertEquals(1, holder.getCpuOpCount(), "Legacy CPU count should work");
        assertEquals(1, holder.getCudaOpCount(), "Legacy CUDA count should work");
    }

    // ========================================================================
    // Multi-Backend Specific Tests (only run when multiple backends available)
    // ========================================================================

    @Test
    @DisplayName("Should enable multi-backend mode when multiple backends present")
    @EnabledIf("multiBackendAvailable")
    void testMultiBackendModeEnabled() {
        assertTrue(MultiBackendNativeOpsHolder.isMultiBackendEnabled(),
            "Multi-backend mode should be enabled when multiple backends present");

        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();
        assertTrue(holder.getLoadedBackendCount() > 1,
            "Should have more than one backend loaded");

        log.info("Multi-backend mode enabled with {} backends", holder.getLoadedBackendCount());
    }

    @Test
    @DisplayName("Should have both CPU and CUDA ops when dual backend")
    @EnabledIf("cpuAndCudaAvailable")
    void testDualBackendOpsAvailable() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        assertNotNull(holder.getCpuOps(), "CPU ops should be available");
        assertNotNull(holder.getCudaOps(), "CUDA ops should be available");
        assertNotSame(holder.getCpuOps(), holder.getCudaOps(),
            "CPU and CUDA ops should be different instances");

        log.info("Dual backend: CPU={}, CUDA={}",
            holder.getCpuOps().getClass().getSimpleName(),
            holder.getCudaOps().getClass().getSimpleName());
    }

    @Test
    @DisplayName("Should return correct ops for each device type in multi-backend mode")
    @EnabledIf("cpuAndCudaAvailable")
    void testCorrectOpsPerDeviceType() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        NativeOps cpuOps = holder.getOpsForDeviceType(DeviceType.CPU);
        NativeOps cudaOps = holder.getOpsForDeviceType(DeviceType.CUDA_GPU);

        assertNotNull(cpuOps);
        assertNotNull(cudaOps);
        assertNotSame(cpuOps, cudaOps, "Different device types should return different ops");

        assertSame(holder.getCpuOps(), cpuOps, "CPU device type should return CPU ops");
        assertSame(holder.getCudaOps(), cudaOps, "CUDA device type should return CUDA ops");
    }

    @Test
    @DisplayName("Should have registered devices for each backend")
    @EnabledIf("multiBackendAvailable")
    void testRegisteredDevicesPerBackend() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();

        Set<String> deviceIds = holder.getRegisteredDeviceIds();

        assertNotNull(deviceIds);
        assertFalse(deviceIds.isEmpty(), "Should have registered devices");

        log.info("Registered devices: {}", deviceIds);

        // Should have at least one device per loaded backend
        assertTrue(deviceIds.size() >= holder.getLoadedBackendCount(),
            "Should have at least one device per backend");
    }

    // ========================================================================
    // Test Condition Methods
    // ========================================================================

    static boolean multiBackendAvailable() {
        MultiBackendNativeOpsHolder.enableMultiBackend();
        return MultiBackendNativeOpsHolder.isMultiBackendEnabled();
    }

    static boolean cpuAndCudaAvailable() {
        MultiBackendNativeOpsHolder.enableMultiBackend();
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();
        return holder.isBackendAvailable(DeviceType.CPU) &&
               holder.isBackendAvailable(DeviceType.CUDA_GPU);
    }
}
