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
package org.nd4j.linalg.vulkan.cache;

import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.vulkan.VulkanDataBuffer;
import org.nd4j.linalg.vulkan.VulkanRuntime;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Vulkan constant-buffer service backed by libnd4j's native constant store.
 */
public final class VulkanConstantHandler extends BasicConstantHandler {
    private final Map<Integer, Map<ArrayDescriptor, DataBuffer>> buffersCache = new ConcurrentHashMap<>();

    @Override
    public DataBuffer relocateConstantSpace(DataBuffer dataBuffer) {
        if (!(dataBuffer instanceof VulkanDataBuffer)) {
            throw new IllegalArgumentException(
                    "Vulkan constants require VulkanDataBuffer, received "
                            + dataBuffer.getClass().getName());
        }

        VulkanDataBuffer vulkanBuffer = (VulkanDataBuffer) dataBuffer;
        vulkanBuffer.syncToSpecial();
        vulkanBuffer.setConstant(true);
        return vulkanBuffer;
    }

    @Override
    public DataBuffer getConstantBuffer(boolean[] values, DataType dataType) {
        return getConstantBuffer(ArrayUtil.toLongs(values), dataType);
    }

    @Override
    public DataBuffer getConstantBuffer(int[] values, DataType dataType) {
        return cachedConstant(new ArrayDescriptor(values, dataType),
                () -> VulkanRuntime.getInstance().executioner().createConstantBuffer(values, dataType));
    }

    @Override
    public DataBuffer getConstantBuffer(long[] values, DataType dataType) {
        return cachedConstant(new ArrayDescriptor(values, dataType),
                () -> VulkanRuntime.getInstance().executioner().createConstantBuffer(values, dataType));
    }

    @Override
    public DataBuffer getConstantBuffer(double[] values, DataType dataType) {
        return cachedConstant(new ArrayDescriptor(values, dataType),
                () -> VulkanRuntime.getInstance().executioner().createConstantBuffer(values, dataType));
    }

    @Override
    public DataBuffer getConstantBuffer(float[] values, DataType dataType) {
        return cachedConstant(new ArrayDescriptor(values, dataType),
                () -> VulkanRuntime.getInstance().executioner().createConstantBuffer(values, dataType));
    }

    /**
     * Returns the constant buffer for this content and type on the current device, creating it once.
     * The native constant cache keeps each content for the life of the process, but every Java buffer
     * wrapping it is constant and never released, so wrapping it again on each call leaked one buffer
     * per call.
     *
     * @param descriptor descriptor over the caller's array; a new entry is keyed by a copy of it
     * @param create     creates the buffer on the first request for this content
     */
    private DataBuffer cachedConstant(ArrayDescriptor descriptor, Supplier<DataBuffer> create) {
        Map<ArrayDescriptor, DataBuffer> cache = buffersCache.computeIfAbsent(
                VulkanRuntime.getInstance().currentDevice(), device -> new ConcurrentHashMap<>());
        DataBuffer cached = cache.get(descriptor);
        return cached != null ? cached : cache.computeIfAbsent(descriptor.copy(), key -> create.get());
    }

    @Override
    public void purgeConstants() {
        // Native constant buffers are owned and reference-managed by libnd4j, which keeps each for the
        // life of the process. The buffers cached here wrap them and are never released, so dropping
        // them would leak every wrapper and create it again on the next request.
    }

    @Override
    public long getCachedBytes() {
        Nd4jVulkan nativeOps = VulkanRuntime.getInstance().nativeOps();
        long cachedBytes = 0L;
        for (int deviceId = 0; deviceId < nativeOps.getAvailableDevices(); deviceId++) {
            cachedBytes += nativeOps.getConstantCacheBytes(deviceId);
        }
        return cachedBytes;
    }
}
