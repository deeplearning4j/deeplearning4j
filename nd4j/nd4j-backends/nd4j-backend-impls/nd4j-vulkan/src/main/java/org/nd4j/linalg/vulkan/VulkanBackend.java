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

package org.nd4j.linalg.vulkan;

import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.MemoryManager;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Vulkan backend for ND4J.
 *
 * <p>Provides GPU-accelerated tensor operations via the Vulkan compute API,
 * enabling cross-vendor GPU execution on hardware from AMD, Intel, NVIDIA,
 * Qualcomm, ARM Mali, and other Vulkan-capable devices.</p>
 *
 * <p>Key properties:</p>
 * <ul>
 *   <li>Cross-platform: negotiates the loader/device Vulkan API and required compute capabilities,
 *       including Vulkan 1.0 devices that expose promoted functionality through KHR extensions</li>
 *   <li>Compute shaders compiled from SPIR-V via the vulkan-preset JavaCPP bindings</li>
 *   <li>VulkanReplayHandle enables command-buffer capture and replay analogous to
 *       CUDA graph replay for low-overhead inference</li>
 *   <li>Higher scheduling priority than CPU (GPU base priority + 20)</li>
 * </ul>
 *
 * <h3>Runtime discovery</h3>
 * <p>{@link #canRun()} and {@link #discoverDevices()} use the native Vulkan bindings
 * as the device authority. The backend is runnable only when the Vulkan runtime
 * reports at least one usable device.</p>
 */
public class VulkanBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(VulkanBackend.class);

    private static final String LINALG_PROPS = "/nd4j-vulkan.properties";

    static {
        // Vulkan devices own ICD worker threads and must be destroyed before
        // JavaCPP/native-library teardown unloads the loader and driver DSOs.
        Runtime.getRuntime().addShutdownHook(
                new Thread(VulkanBackend::shutdownNativeRuntime, "ND4J-Vulkan-Shutdown"));
    }

    private static void shutdownNativeRuntime() {
        try {
            VulkanRuntime.shutdownIfInitialized();
        } catch (LinkageError | RuntimeException e) {
            log.warn("Vulkan native shutdown failed", e);
        }
    }

    @Override
    public boolean isAvailable() {
        return canRun();
    }

    @Override
    public boolean canRun() {
        return getAvailableDeviceCount() > 0;
    }

    @Override
    public List<DeviceDescriptor> discoverDevices() {
        int deviceCount = getAvailableDeviceCount();
        List<DeviceDescriptor> devices = new ArrayList<>(deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            devices.add(DeviceDescriptor.accelerator(getBackendId(), DeviceType.VULKAN_GPU, i));
        }
        return devices;
    }

    private int getAvailableDeviceCount() {
        return VulkanRuntime.getInstance().deviceCount();
    }

    @Override
    public void initialize() {
        VulkanRuntime.getInstance();
    }

    @Override
    public OpExecutioner createExecutioner() {
        return VulkanRuntime.getInstance().executioner();
    }

    @Override
    public MemoryManager createMemoryManager() {
        return VulkanRuntime.getInstance().memoryManager();
    }

    @Override
    public boolean allowsOrder() {
        return false;
    }

    @Override
    public int getPriority() {
        // Above TPU (+10) and Metal (+15) for systems where Vulkan is the primary
        // GPU compute path, still relative to the configurable GPU base priority.
        return BACKEND_PRIORITY_GPU + 20;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, VulkanBackend.class.getClassLoader());
    }

    @Override
    public Class<?> getNDArrayClass() {
        return VulkanNDArray.class;
    }

    @Override
    public Environment getEnvironment() {
        return VulkanEnvironment.getInstance();
    }

    @Override
    public String buildInfo() {
        return "ND4J Vulkan Backend (cross-vendor, runtime-negotiated Vulkan compute)";
    }

    @Override
    public void logBackendInit() {
        log.info("ND4J Vulkan backend initialized");
        log.info("  API:    runtime-negotiated Vulkan core/KHR capability path");
        log.info("  Target: cross-vendor (AMD / Intel / NVIDIA / Qualcomm / ARM Mali)");
    }

    @Override
    public String getBackendId() {
        return "vulkan";
    }
}
