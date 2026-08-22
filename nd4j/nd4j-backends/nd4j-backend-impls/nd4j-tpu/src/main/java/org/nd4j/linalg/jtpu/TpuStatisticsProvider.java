/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  * ****************************************************************************
 */

package org.nd4j.linalg.jtpu;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ndarray.INDArrayStatisticsProvider;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/** Native array statistics using the active TPU NativeOps binding. */
public final class TpuStatisticsProvider implements INDArrayStatisticsProvider {

    private static final String DEBUG_INFO_CLASS =
            "org.nd4j.linalg.jtpu.bindings.Nd4jTpu$DebugInfo";

    @Override
    public INDArrayStatistics inspectArray(INDArray array) {
        Pointer debugInfo = null;
        try {
            debugInfo = (Pointer) Class.forName(
                    DEBUG_INFO_CLASS, true, TpuStatisticsProvider.class.getClassLoader())
                    .getDeclaredConstructor().newInstance();
            NativeOps nativeOps = Nd4j.getNativeOps();
            nativeOps.inspectArray(null, array.data().addressPointer(),
                    (LongPointer) array.shapeInfoDataBuffer().addressPointer(),
                    null, null, debugInfo);
            if (nativeOps.lastErrorCode() != 0) {
                throw new IllegalStateException(nativeOps.lastErrorMessage());
            }

            return INDArrayStatistics.builder()
                    .minValue(number(debugInfo, "_minValue").doubleValue())
                    .maxValue(number(debugInfo, "_maxValue").doubleValue())
                    .meanValue(number(debugInfo, "_meanValue").doubleValue())
                    .stdDevValue(number(debugInfo, "_stdDevValue").doubleValue())
                    .countInf(number(debugInfo, "_infCount").longValue())
                    .countNaN(number(debugInfo, "_nanCount").longValue())
                    .countNegative(number(debugInfo, "_negativeCount").longValue())
                    .countPositive(number(debugInfo, "_positiveCount").longValue())
                    .countZero(number(debugInfo, "_zeroCount").longValue())
                    .build();
        } catch (ReflectiveOperationException failure) {
            throw new IllegalStateException("Unable to inspect an array with the TPU binding", failure);
        } finally {
            if (debugInfo != null) {
                debugInfo.close();
            }
        }
    }

    private static Number number(Pointer debugInfo, String methodName)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Method method = debugInfo.getClass().getMethod(methodName);
        return (Number) method.invoke(debugInfo);
    }
}
