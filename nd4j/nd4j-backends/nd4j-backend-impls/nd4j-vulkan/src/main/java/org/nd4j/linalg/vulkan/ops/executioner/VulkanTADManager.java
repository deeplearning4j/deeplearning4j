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
package org.nd4j.linalg.vulkan.ops.executioner;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.vulkan.VulkanRuntime;

import java.util.Arrays;

/**
 * Vulkan TAD metadata manager backed by libnd4j's native shape service.
 */
public final class VulkanTADManager implements TADManager {
    private final VulkanExecutioner executioner;

    public VulkanTADManager() {
        this(VulkanRuntime.getInstance().executioner());
    }

    public VulkanTADManager(VulkanExecutioner executioner) {
        if (executioner == null) {
            throw new IllegalArgumentException("Vulkan executioner must not be null");
        }
        this.executioner = executioner;
    }

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, long... dimensions) {
        long[] normalized = dimensions;
        if (normalized == null || normalized.length == 0) {
            normalized = new long[] {-1};
        } else if (normalized.length > 1) {
            normalized = Arrays.copyOf(normalized, normalized.length);
            Arrays.sort(normalized);
        }

        org.nd4j.linalg.api.shape.TadPack pack =
                executioner.tadShapeInfoAndOffsets(array, normalized);
        return Pair.makePair(pack.getTadShapeInfo(), pack.getTadOffsets());
    }

    @Override
    public void purgeBuffers() {
        executioner.purgeTadCache();
    }

    @Override
    public long getCachedBytes() {
        return executioner.tadCachedBytes();
    }
}
