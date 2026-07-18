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
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.vulkan.VulkanDataBuffer;
import org.nd4j.linalg.vulkan.VulkanRuntime;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;

/**
 * Vulkan constant-buffer service backed by libnd4j's native constant store.
 */
public final class VulkanConstantHandler extends BasicConstantHandler {
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
        return VulkanRuntime.getInstance().executioner().createConstantBuffer(values, dataType);
    }

    @Override
    public DataBuffer getConstantBuffer(long[] values, DataType dataType) {
        return VulkanRuntime.getInstance().executioner().createConstantBuffer(values, dataType);
    }

    @Override
    public DataBuffer getConstantBuffer(double[] values, DataType dataType) {
        return VulkanRuntime.getInstance().executioner().createConstantBuffer(values, dataType);
    }

    @Override
    public DataBuffer getConstantBuffer(float[] values, DataType dataType) {
        return VulkanRuntime.getInstance().executioner().createConstantBuffer(values, dataType);
    }

    @Override
    public void purgeConstants() {
        // Native constant buffers are owned and reference-managed by libnd4j.
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
