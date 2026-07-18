/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.vulkan;

import org.bytedeco.javacpp.LongPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ndarray.INDArrayStatisticsProvider;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;
import org.nd4j.nativeblas.NativeOps;

/** Vulkan implementation of {@link INDArrayStatisticsProvider}. */
public class VulkanStatisticsProvider implements INDArrayStatisticsProvider {

    private final NativeOps loop = VulkanRuntime.getInstance().nativeOps();

    @Override
    public INDArrayStatistics inspectArray(INDArray arr) {
        Nd4jVulkan.DebugInfo debugInfo = new Nd4jVulkan.DebugInfo();

        loop.inspectArray(null, arr.data().addressPointer(),
                (LongPointer) arr.shapeInfoDataBuffer().addressPointer(), null, null, debugInfo);

        if (loop.lastErrorCode() != 0) {
            throw new RuntimeException(loop.lastErrorMessage());
        }

        return INDArrayStatistics.builder()
                .minValue(debugInfo._minValue())
                .maxValue(debugInfo._maxValue())
                .meanValue(debugInfo._meanValue())
                .stdDevValue(debugInfo._stdDevValue())
                .countInf(debugInfo._infCount())
                .countNaN(debugInfo._nanCount())
                .countNegative(debugInfo._negativeCount())
                .countPositive(debugInfo._positiveCount())
                .countZero(debugInfo._zeroCount())
                .build();
    }
}
