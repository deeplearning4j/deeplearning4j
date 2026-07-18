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

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * AffinityManager implementation for the Vulkan compute backend.
 *
 * <p>Mirrors CUDA's device-abstraction boundary: thread-to-device binding uses
 * the exact Vulkan NativeOps instance, while array attribution and coherency come
 * from each owning VulkanDataBuffer. Cross-device access is always an explicit
 * Vulkan copy or migration and is admitted only when the native device pair
 * reports the required peer capability.</p>
 *
 * <p><b>Discovery contract:</b> a valid zero-device result is reported as zero.
 * Native initialization and enumeration failures remain errors and are never
 * converted into an apparent no-hardware result.</p>
 */
public class VulkanAffinityManager extends BasicAffinityManager {

    private static final Logger log = LoggerFactory.getLogger(VulkanAffinityManager.class);

    /**
     * Backend-scoped affinity state shared by every configured/runtime manager facade.
     * CUDA obtains the same guarantee from its singleton allocator; Vulkan must not let
     * independently-constructed service facades diverge on the current thread's device.
     */
    private static final Map<Long, Integer> affinityMap = new ConcurrentHashMap<>();

    /**
     * Lazily-populated backend device count; -1 until the first query.
     * Guarded by the AtomicInteger's own compare-and-set for initialization.
     */
    private static final AtomicInteger deviceCount = new AtomicInteger(-1);
    private static final AtomicBoolean crossDeviceAccessAllowed = new AtomicBoolean(true);

    private final Nd4jVulkan nativeOps;
    private final VulkanDeviceContextProvider contextProvider;

    /**
     * Constructor used by ND4J's configured Vulkan backend. Installing the provider here
     * mirrors CUDA while keeping auxiliary VulkanRuntime access from replacing another
     * active backend's shared provider in a multi-backend JVM.
     */
    public VulkanAffinityManager() {
        this(VulkanRuntime.getInstance().nativeOps(), true);
    }

    VulkanAffinityManager(Nd4jVulkan nativeOps) {
        this(nativeOps, false);
    }

    private VulkanAffinityManager(Nd4jVulkan nativeOps, boolean installContextProvider) {
        super();
        if (nativeOps == null) {
            throw new IllegalArgumentException("Vulkan NativeOps authority must not be null");
        }
        this.nativeOps = nativeOps;
        this.contextProvider = new VulkanDeviceContextProvider(affinityMap, nativeOps);
        if (installContextProvider) {
            DeviceMemoryManager.getInstance().setContextProvider(contextProvider);
        }
        log.debug("VulkanAffinityManager initialized");
    }

    // -------------------------------------------------------------------------
    // Device count
    // -------------------------------------------------------------------------

    /**
     * Returns the number of enumerated Vulkan devices.
     * Calls the native NativeOps surface (Nd4jVulkan.getAvailableDevices).
     * Safe to call with 0 devices: returns 0, does not throw.
     */
    @Override
    public int getNumberOfDevices() {
        if (deviceCount.get() < 0) {
            int count = nativeOps.getAvailableDevices();
            if (count < 0) {
                throw new IllegalStateException(
                        "Vulkan native discovery returned an invalid device count: " + count);
            }
            deviceCount.compareAndSet(-1, count);
        }
        return deviceCount.get();
    }

    // -------------------------------------------------------------------------
    // Thread -> device binding
    // -------------------------------------------------------------------------

    /**
     * Returns the native Vulkan device selected for the current thread and keeps
     * the shared Java affinity authority synchronized with it.
     */
    @Override
    public Integer getDeviceForCurrentThread() {
        return contextProvider.getCurrentDeviceId();
    }

    @Override
    public Integer getDeviceForThread(long threadId) {
        if (threadId == Thread.currentThread().getId()) {
            return getDeviceForCurrentThread();
        }
        return affinityMap.get(threadId);
    }

    /**
     * Explicitly binds the current thread through the backend's single device-context
     * authority, keeping shared Java and native Vulkan state consistent.
     */
    @Override
    public void setDeviceForCurrentThread(int deviceId) {
        contextProvider.switchDevice(
                deviceId, VulkanAffinityManager.class.getName(), "setDeviceForCurrentThread");
        log.debug("VulkanAffinityManager: thread {} bound to device {}",
                Thread.currentThread().getId(), deviceId);
    }

    // -------------------------------------------------------------------------
    // Array / buffer device attribution
    // -------------------------------------------------------------------------

    /** Returns the exact Vulkan device recorded by the array's owning data buffer. */
    @Override
    public Integer getDeviceForArray(@NonNull INDArray array) {
        return requireVulkanBuffer(array.data()).targetDevice();
    }

    // -------------------------------------------------------------------------
    // Device type helpers
    // -------------------------------------------------------------------------

    @Override
    public boolean isCpuDevice(int deviceId) {
        return deviceId == CPU_DEVICE_ID;
    }

    @Override
    public DeviceType getDeviceType(int deviceId) {
        if (deviceId == CPU_DEVICE_ID) {
            return DeviceType.CPU;
        }
        return DeviceType.VULKAN_GPU;
    }

    // -------------------------------------------------------------------------
    // Coherency and explicit peer-transfer capability
    // -------------------------------------------------------------------------

    @Override
    public void touch(INDArray array) {
        if (array == null) {
            return;
        }
        touch(array.data());
        touch(array.shapeInfoDataBuffer());
    }

    @Override
    public void touch(DataBuffer buffer) {
        if (buffer == null) {
            return;
        }
        VulkanDataBuffer vulkanBuffer = requireVulkanBuffer(buffer);
        int destinationDevice = getDeviceForCurrentThread();
        if (!isPeerCopyAllowed(vulkanBuffer.targetDevice(), destinationDevice)) {
            throw new UnsupportedOperationException(
                    "Vulkan migration is unavailable from device "
                            + vulkanBuffer.targetDevice() + " to device " + destinationDevice);
        }
        vulkanBuffer.ensureAvailableOn(getDeviceDescriptor(destinationDevice));
    }

    @Override
    public void tagLocation(INDArray array, Location location) {
        if (array == null || array.isEmpty()) {
            return;
        }
        tagLocation(array.data(), location);
    }

    @Override
    public void tagLocation(DataBuffer buffer, Location location) {
        if (buffer == null) {
            return;
        }
        VulkanDataBuffer vulkanBuffer = requireVulkanBuffer(buffer);
        switch (location) {
            case HOST:
                vulkanBuffer.markHostDirty();
                break;
            case DEVICE:
                vulkanBuffer.markDeviceDirty();
                break;
            case EVERYWHERE:
                vulkanBuffer.markEverywhere();
                break;
            default:
                throw new IllegalArgumentException("Unknown Vulkan data location " + location);
        }
    }

    @Override
    public void ensureLocation(INDArray array, Location location) {
        if (array == null || array.isEmpty()) {
            return;
        }
        VulkanDataBuffer vulkanBuffer = requireVulkanBuffer(array.data());
        switch (location) {
            case HOST:
                vulkanBuffer.syncToPrimary();
                break;
            case DEVICE:
                vulkanBuffer.syncToSpecial();
                break;
            case EVERYWHERE:
                vulkanBuffer.syncToSpecial();
                vulkanBuffer.syncToPrimary();
                break;
            default:
                throw new IllegalArgumentException("Unknown Vulkan data location " + location);
        }
    }

    @Override
    public Location getActiveLocation(INDArray array) {
        if (array == null || array.isEmpty()) {
            return Location.EVERYWHERE;
        }
        VulkanDataBuffer buffer = requireVulkanBuffer(array.data());
        boolean hostActual = buffer.isValidOn(DeviceDescriptor.cpu());
        boolean deviceActual = buffer.isValidOn(getDeviceDescriptor(buffer.targetDevice()));
        if (hostActual && deviceActual) {
            return Location.EVERYWHERE;
        }
        return deviceActual ? Location.DEVICE : Location.HOST;
    }

    @Override
    public boolean isCrossDeviceAccessSupported() {
        if (!crossDeviceAccessAllowed.get()) {
            return false;
        }
        int count = getNumberOfDevices();
        if (count < 2) {
            return false;
        }
        for (int source = 0; source < count; source++) {
            for (int destination = 0; destination < count; destination++) {
                if (source != destination
                        && !nativeOps.isPeerAccessSupported(source, destination)) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public void allowCrossDeviceAccess(boolean reallyAllow) {
        crossDeviceAccessAllowed.set(reallyAllow);
    }

    @Override
    public INDArray replicateToDevice(Integer deviceId, INDArray array) {
        if (array == null) {
            return null;
        }
        if (deviceId == null) {
            throw new IllegalArgumentException("Target Vulkan device must not be null");
        }
        validateVulkanDevice(deviceId);

        VulkanDataBuffer copiedData =
                (VulkanDataBuffer) replicateToDevice(deviceId, array.data());
        int previousDevice = getDeviceForCurrentThread();
        try {
            if (previousDevice != deviceId) {
                setDeviceForCurrentThread(deviceId);
            }
            DataBuffer copiedShape = VulkanRuntime.getInstance().executioner().createShapeInfo(
                    array.shape(), array.stride(), array.elementWiseStride(), array.ordering(),
                    array.dataType(), array.isEmpty(), array.isView());
            return VulkanNDArray.wrapReplica(copiedData, array, copiedShape);
        } catch (RuntimeException | Error failure) {
            copiedData.close();
            throw failure;
        } finally {
            if (previousDevice != deviceId) {
                setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    @Override
    public DataBuffer replicateToDevice(Integer deviceId, DataBuffer buffer) {
        if (buffer == null) {
            return null;
        }
        if (deviceId == null) {
            throw new IllegalArgumentException("Target Vulkan device must not be null");
        }
        validateVulkanDevice(deviceId);
        return requireVulkanBuffer(buffer).duplicateToDevice(deviceId);
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private static VulkanDataBuffer requireVulkanBuffer(DataBuffer buffer) {
        if (!(buffer instanceof VulkanDataBuffer)) {
            throw new IllegalArgumentException(
                    "Vulkan affinity requires VulkanDataBuffer storage, got "
                            + (buffer == null ? "null" : buffer.getClass().getName()));
        }
        return (VulkanDataBuffer) buffer;
    }

    private void validateVulkanDevice(int deviceId) {
        int count = getNumberOfDevices();
        if (deviceId < 0 || deviceId >= count) {
            throw new IllegalArgumentException(
                    "Vulkan device id " + deviceId + " is outside [0," + count + ")");
        }
    }

    boolean isPeerCopyAllowed(int sourceDevice, int destinationDevice) {
        validateVulkanDevice(sourceDevice);
        validateVulkanDevice(destinationDevice);
        return sourceDevice == destinationDevice
                || (crossDeviceAccessAllowed.get()
                        && nativeOps.isPeerAccessSupported(sourceDevice, destinationDevice));
    }

    VulkanDeviceContextProvider contextProvider() {
        return contextProvider;
    }
}
