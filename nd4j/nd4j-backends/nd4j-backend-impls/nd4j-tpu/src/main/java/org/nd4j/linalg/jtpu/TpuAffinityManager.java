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

package org.nd4j.linalg.jtpu;

import lombok.NonNull;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.CpuAffinityManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Thread-to-TPU selection for the host-native/PJRT backend. */
public final class TpuAffinityManager extends CpuAffinityManager {

    private final Map<Long, Integer> affinity = new ConcurrentHashMap<>();

    private NativeOps nativeOps() {
        return Nd4j.getNativeOps();
    }

    @Override
    public boolean isCpuDevice(int deviceId) {
        return deviceId == CPU_DEVICE_ID;
    }

    @Override
    public DeviceType getDeviceType(int deviceId) {
        return deviceId == CPU_DEVICE_ID ? DeviceType.CPU : DeviceType.TPU;
    }

    @Override
    public Integer getDeviceForCurrentThread() {
        long threadId = Thread.currentThread().getId();
        return affinity.computeIfAbsent(threadId, ignored -> nativeOps().getDevice());
    }

    @Override
    public Integer getDeviceForThread(long threadId) {
        if (threadId == Thread.currentThread().getId()) {
            return getDeviceForCurrentThread();
        }
        return affinity.getOrDefault(threadId, 0);
    }

    @Override
    public Integer getDeviceForArray(@NonNull INDArray array) {
        // Public NDArrays are host-backed; the current thread selects the PJRT
        // device used when a compiled segment uploads this value.
        return getDeviceForCurrentThread();
    }

    @Override
    public void setDeviceForCurrentThread(int deviceId) {
        int count = getNumberOfDevices();
        if (deviceId < 0 || deviceId >= count) {
            throw new IllegalArgumentException(
                    "TPU device index " + deviceId + " is outside [0," + count + ")");
        }
        int status = nativeOps().setDevice(deviceId);
        if (status != 0) {
            throw new IllegalStateException("Native PJRT runtime rejected TPU device " + deviceId);
        }
        affinity.put(Thread.currentThread().getId(), deviceId);
    }

    @Override
    public int getNumberOfDevices() {
        return nativeOps().getAvailableDevices();
    }

    @Override
    public boolean isCrossDeviceAccessSupported() {
        // Arrays remain in host memory and can be uploaded to any addressable TPU.
        return true;
    }
}
