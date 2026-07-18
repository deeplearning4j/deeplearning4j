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
package org.nd4j.nativeblas;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Backend owner for an exact NativeOps instance selected from the multi-backend
 * registry. It uses only that binding for device selection, synchronization,
 * allocation accounting, and native object cleanup.
 */
public final class NativeOpsBufferOwner implements NativeBufferOwner {
    private final NativeOps nativeOps;
    private final DeviceType deviceType;
    private final DeallocatorService deallocatorService;
    private final Map<Integer, DeviceDescriptor> descriptors = new ConcurrentHashMap<>();
    private final Map<Integer, AtomicLong> allocatedBytes = new ConcurrentHashMap<>();

    NativeOpsBufferOwner(NativeOps nativeOps, DeviceType deviceType) {
        if (nativeOps == null) {
            throw new IllegalArgumentException("NativeOps cannot be null");
        }
        if (deviceType == null) {
            throw new IllegalArgumentException("DeviceType cannot be null");
        }
        this.nativeOps = nativeOps;
        this.deviceType = deviceType;
        this.deallocatorService = DeallocatorService.getInstance();
    }

    @Override
    public NativeOps nativeOps() {
        return nativeOps;
    }

    public DeviceType deviceType() {
        return deviceType;
    }

    @Override
    public DeallocatorService deallocatorService() {
        return deallocatorService;
    }

    @Override
    public int currentDevice() {
        return nativeOps.getDevice();
    }

    @Override
    public int deviceCount() {
        int count = nativeOps.getAvailableDevices();
        if (count < 0) {
            throw new IllegalStateException(
                    "Backend " + deviceType + " returned an invalid device count: " + count);
        }
        return count;
    }

    @Override
    public void setDevice(int deviceId) {
        int count = deviceCount();
        if (deviceId < 0 || deviceId >= count) {
            throw new IllegalArgumentException(
                    "Invalid " + deviceType + " device " + deviceId + " for " + count + " devices");
        }
        nativeOps.clearLastError();
        nativeOps.setDevice(deviceId);
        checkNativeError("select device " + deviceId);
    }

    @Override
    public void commit() {
        OpaqueLaunchContext launchContext = nativeOps.defaultLaunchContext();
        if (launchContext == null || launchContext.isNull()) {
            return;
        }
        Pointer stream = nativeOps.lcExecutionStream(launchContext);
        if (stream == null || stream.isNull()) {
            return;
        }
        nativeOps.clearLastError();
        nativeOps.streamSynchronize(stream);
        checkNativeError("synchronize the backend execution stream");
    }

    @Override
    public DeviceDescriptor deviceDescriptor(int deviceId) {
        int count = deviceCount();
        if (deviceId < 0 || deviceId >= count) {
            throw new IllegalArgumentException(
                    "Invalid " + deviceType + " device " + deviceId + " for " + count + " devices");
        }
        return descriptors.computeIfAbsent(deviceId, this::createDescriptor);
    }

    @Override
    public void recordAllocation(DeviceDescriptor device, long bytes) {
        validateDescriptor(device);
        if (bytes > 0) {
            allocatedBytes.computeIfAbsent(device.getDeviceIndex(), ignored -> new AtomicLong())
                    .addAndGet(bytes);
        }
    }

    @Override
    public void recordDeallocation(DeviceDescriptor device, long bytes) {
        validateDescriptor(device);
        if (bytes <= 0) {
            return;
        }
        AtomicLong allocated = allocatedBytes.get(device.getDeviceIndex());
        if (allocated == null) {
            throw new IllegalStateException(
                    "No allocation accounting exists for " + device.getDeviceId());
        }
        long remaining = allocated.addAndGet(-bytes);
        if (remaining < 0) {
            allocated.addAndGet(bytes);
            throw new IllegalStateException(
                    "Allocation accounting underflow for " + device.getDeviceId()
                            + ": releasing " + bytes + " bytes");
        }
    }

    public long allocatedBytes(DeviceDescriptor device) {
        validateDescriptor(device);
        AtomicLong allocated = allocatedBytes.get(device.getDeviceIndex());
        return allocated == null ? 0L : allocated.get();
    }

    private DeviceDescriptor createDescriptor(int deviceId) {
        if (deviceType == DeviceType.CPU) {
            return DeviceDescriptor.cpu(deviceId);
        }
        if (deviceType == DeviceType.CUDA_GPU || deviceType == DeviceType.GPU) {
            return DeviceDescriptor.cuda(deviceId);
        }
        return DeviceDescriptor.accelerator(
                deviceType.getIdentifier(), deviceType, deviceId);
    }

    private void validateDescriptor(DeviceDescriptor device) {
        if (device == null || device.getDeviceType() != deviceType) {
            throw new IllegalArgumentException(
                    "Expected a " + deviceType + " descriptor, got " + device);
        }
        deviceDescriptor(device.getDeviceIndex());
    }

    private void checkNativeError(String action) {
        int errorCode = nativeOps.lastErrorCode();
        if (errorCode != 0) {
            String message = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "Could not " + action + " for " + deviceType
                            + " (native error " + errorCode + "): " + message);
        }
    }
}
