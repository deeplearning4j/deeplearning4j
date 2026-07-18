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

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for automatic multi-backend discovery and routing.
 * This test verifies that multiple backends can be discovered and used
 * when they are on the classpath.
 *
 * Run with: mvn test -pl platform-tests -Ptest-device-routing -Dtest=MultiBackendRoutingTest
 */
public class MultiBackendRoutingTest {

    @Test
    public void testBackendDiscovery() {
        // Get all available backends
        List<Map.Entry<Nd4jBackend, DeviceType>> backends = Nd4j.getAvailableBackends();

        // Should have at least one backend
        assertFalse(backends.isEmpty(), "Should have at least one backend available");

        // Print discovered backends
        System.out.println("Discovered " + backends.size() + " backend(s):");
        for (Map.Entry<Nd4jBackend, DeviceType> entry : backends) {
            System.out.println("  - " + entry.getKey().getClass().getSimpleName() +
                             " (DeviceType: " + entry.getValue() +
                             ", Priority: " + entry.getKey().getPriority() + ")");
        }
    }

    @Test
    public void testPrimaryBackendDeviceType() {
        // Get the device type of the primary backend
        DeviceType deviceType = Nd4j.getBackendDeviceType();

        assertNotNull(deviceType, "Primary backend device type should not be null");
        System.out.println("Primary backend device type: " + deviceType);

        // Verify it's a valid device type
        assertTrue(deviceType == DeviceType.CPU || deviceType.isAccelerator(),
                   "Device type should be CPU or a recognized accelerator");
    }

    @Test
    public void testBackendAvailabilityCheck() {
        // Check which device types are available
        boolean hasCpu = Nd4j.isBackendAvailable(DeviceType.CPU);
        boolean hasCudaGpu = Nd4j.isBackendAvailable(DeviceType.CUDA_GPU);
        boolean hasRocmGpu = Nd4j.isBackendAvailable(DeviceType.ROCM_GPU);
        boolean hasVulkanGpu = Nd4j.isBackendAvailable(DeviceType.VULKAN_GPU);
        boolean hasMetalGpu = Nd4j.isBackendAvailable(DeviceType.METAL_GPU);
        boolean hasTpu = Nd4j.isBackendAvailable(DeviceType.TPU);

        System.out.println("Backend availability:");
        System.out.println("  CPU: " + hasCpu);
        System.out.println("  CUDA_GPU: " + hasCudaGpu);
        System.out.println("  ROCM_GPU: " + hasRocmGpu);
        System.out.println("  VULKAN_GPU: " + hasVulkanGpu);
        System.out.println("  METAL_GPU: " + hasMetalGpu);
        System.out.println("  TPU: " + hasTpu);

        // At least one should be available
        assertTrue(hasCpu || hasCudaGpu || hasRocmGpu || hasVulkanGpu || hasMetalGpu || hasTpu,
                   "At least one backend device type should be available");
    }

    @Test
    public void testMultiBackendWhenMultipleAvailable() {
        List<Map.Entry<Nd4jBackend, DeviceType>> backends = Nd4j.getAvailableBackends();

        if (backends.size() >= 2) {
            System.out.println("Multiple backends detected - multi-backend mode should be active");

            // When multiple backends are available, multi-backend should be automatically enabled
            // The highest priority backend becomes the primary
            DeviceType primaryType = Nd4j.getBackendDeviceType();
            DeviceType firstBackendType = backends.get(0).getValue();

            assertEquals(firstBackendType, primaryType,
                        "Primary backend should match highest priority backend");

            // Verify we can check for other backend types
            for (Map.Entry<Nd4jBackend, DeviceType> entry : backends) {
                assertTrue(Nd4j.isBackendAvailable(entry.getValue()),
                          "Backend " + entry.getValue() + " should report as available");
            }
        } else {
            System.out.println("Only one backend detected - single backend mode");
        }
    }
}
