/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.vulkan;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;
import org.nd4j.linalg.vulkan.ops.executioner.VulkanExecutioner;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.nativeblas.NativeBufferOwner;
import org.nd4j.nativeblas.NativeOps;

/**
 * Backend-scoped owner for Vulkan's native authority and Java services.
 *
 * <p>The native instance is the exact Vulkan binding registered by
 * {@link MultiBackendNativeOpsHolder}. It is never inferred from ND4J's primary
 * backend and no second binding instance is constructed.</p>
 */
public final class VulkanRuntime implements NativeBufferOwner {
    private static volatile VulkanRuntime instance;

    private final Nd4jVulkan nativeOps;
    private final VulkanAffinityManager affinityManager;
    private final VulkanDataBufferFactory dataBufferFactory;
    private final VulkanExecutioner executioner;
    private final VulkanMemoryManager memoryManager;

    private VulkanRuntime() {
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();
        NativeOps exactOps = holder.getOpsForDeviceType(DeviceType.VULKAN_GPU);
        if (!(exactOps instanceof Nd4jVulkan)) {
            throw new IllegalStateException(
                    "Vulkan registry returned " + exactOps.getClass().getName()
                            + " instead of " + Nd4jVulkan.class.getName());
        }

        nativeOps = (Nd4jVulkan) exactOps;
        affinityManager = new VulkanAffinityManager(nativeOps);
        dataBufferFactory = new VulkanDataBufferFactory();
        executioner = new VulkanExecutioner(nativeOps, affinityManager, dataBufferFactory);
        memoryManager = new VulkanMemoryManager(nativeOps, affinityManager, executioner);
        holder.registerBackendOwner(DeviceType.VULKAN_GPU, this);
    }

    public static VulkanRuntime getInstance() {
        VulkanRuntime current = instance;
        if (current == null) {
            synchronized (VulkanRuntime.class) {
                current = instance;
                if (current == null) {
                    current = new VulkanRuntime();
                    instance = current;
                }
            }
        }
        return current;
    }

    /**
     * Resolves the backend owner for an exact Vulkan binding selected from the
     * multi-backend registry. Identity is checked so a context or buffer can
     * never silently migrate to another binding instance.
     */
    public static VulkanRuntime forNativeOps(NativeOps selectedNativeOps) {
        VulkanRuntime runtime = getInstance();
        if (selectedNativeOps != runtime.nativeOps) {
            throw new IllegalArgumentException(
                    "Selected NativeOps is not the registered Vulkan binding instance");
        }
        return runtime;
    }

    static void shutdownIfInitialized() {
        VulkanRuntime current = instance;
        if (current != null) {
            current.nativeOps.vulkanShutdown();
        }
    }

    @Override
    public Nd4jVulkan nativeOps() {
        return nativeOps;
    }

    public VulkanAffinityManager affinityManager() {
        return affinityManager;
    }

    public VulkanDataBufferFactory dataBufferFactory() {
        return dataBufferFactory;
    }

    public VulkanExecutioner executioner() {
        return executioner;
    }

    public VulkanMemoryManager memoryManager() {
        return memoryManager;
    }

    @Override
    public DeallocatorService deallocatorService() {
        return DeallocatorService.getInstance();
    }

    @Override
    public int currentDevice() {
        return affinityManager.getDeviceForCurrentThread();
    }

    @Override
    public int deviceCount() {
        int count = nativeOps.getAvailableDevices();
        if (count < 0) {
            throw new IllegalStateException(
                    "Vulkan native discovery returned an invalid device count: " + count);
        }
        return count;
    }

    @Override
    public void setDevice(int deviceId) {
        affinityManager.setDeviceForCurrentThread(deviceId);
    }

    @Override
    public void commit() {
        executioner.commit();
    }

    @Override
    public DeviceDescriptor deviceDescriptor(int deviceId) {
        return affinityManager.getDeviceDescriptor(deviceId);
    }

    @Override
    public void recordAllocation(DeviceDescriptor device, long bytes) {
        memoryManager.recordBufferAllocation(requireVulkanDevice(device), bytes);
    }

    @Override
    public void recordDeallocation(DeviceDescriptor device, long bytes) {
        memoryManager.recordBufferDeallocation(requireVulkanDevice(device), bytes);
    }

    private int requireVulkanDevice(DeviceDescriptor device) {
        if (device == null || device.getDeviceType() != DeviceType.VULKAN_GPU) {
            throw new IllegalArgumentException("Expected a Vulkan allocation device, got " + device);
        }
        int deviceId = device.getDeviceIndex();
        int count = deviceCount();
        if (deviceId < 0 || deviceId >= count) {
            throw new IllegalArgumentException(
                    "Invalid Vulkan allocation device " + deviceId + " for " + count + " devices");
        }
        return deviceId;
    }
}
