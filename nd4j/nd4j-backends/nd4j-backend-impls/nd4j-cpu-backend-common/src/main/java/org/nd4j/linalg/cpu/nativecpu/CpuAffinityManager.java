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

package org.nd4j.linalg.cpu.nativecpu;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * AffinityManager implementation for CPU backend.
 *
 * For the CPU backend, this implementation maps the interface's CPU_DEVICE_ID (-1)
 * to the internal representation of device 0. This ensures that:
 * - When running purely on CPU, device 0 is used internally
 * - When interoperating with CUDA backend, CPU_DEVICE_ID (-1) is recognized
 *
 * This enables seamless data transfer between CPU and GPU backends.
 */
public class CpuAffinityManager extends BasicAffinityManager {
    private static final Logger logger = LoggerFactory.getLogger(CpuAffinityManager.class);

    /**
     * Internal CPU device ID for CPU backend operations.
     * Externally, CPU is represented as CPU_DEVICE_ID (-1) from the interface.
     */
    private static final int INTERNAL_CPU_DEVICE_ID = 0;

    /**
     * Thread affinity map - tracks which threads are associated with the CPU device
     */
    private final Map<Long, Integer> affinityMap = new ConcurrentHashMap<>();

    public CpuAffinityManager() {
        super();
        logger.debug("CpuAffinityManager initialized - CPU acknowledged as device {} (internal: {})",
                CPU_DEVICE_ID, INTERNAL_CPU_DEVICE_ID);
    }

    /**
     * Checks if the given device ID represents the CPU.
     * For CPU backend, both CPU_DEVICE_ID (-1) and internal device 0 are considered CPU.
     *
     * @param deviceId the device ID to check
     * @return true if the device is CPU
     */
    @Override
    public boolean isCpuDevice(int deviceId) {
        return deviceId == CPU_DEVICE_ID || deviceId == INTERNAL_CPU_DEVICE_ID;
    }

    /**
     * Returns the device type for the given device ID.
     * For CPU backend, all valid device IDs are CPU.
     *
     * @param deviceId the device ID to query
     * @return CPU for valid device IDs, CUDA_GPU for GPU-like IDs (which aren't supported)
     */
    @Override
    public DeviceType getDeviceType(int deviceId) {
        // For CPU backend, both the interface CPU_DEVICE_ID (-1) and internal device 0 are CPU
        if (deviceId == CPU_DEVICE_ID || deviceId == INTERNAL_CPU_DEVICE_ID) {
            return DeviceType.CPU;
        }
        // For CPU backend, GPU device IDs are not supported but we return CUDA_GPU for consistency
        return DeviceType.CUDA_GPU;
    }

    /**
     * Returns the device ID for the current thread.
     * For CPU backend, this returns internal device 0.
     *
     * @return internal CPU device ID (0)
     */
    @Override
    public Integer getDeviceForCurrentThread() {
        long threadId = Thread.currentThread().getId();
        Integer deviceId = affinityMap.get(threadId);
        if (deviceId == null) {
            deviceId = INTERNAL_CPU_DEVICE_ID;
            affinityMap.put(threadId, deviceId);
            logger.trace("Thread {} mapped to CPU device {}", threadId, deviceId);
        }
        return deviceId;
    }

    /**
     * Returns the device ID for a given thread.
     * For CPU backend, this returns internal device 0.
     *
     * @param threadId the thread ID to query
     * @return internal CPU device ID (0)
     */
    @Override
    public Integer getDeviceForThread(long threadId) {
        Integer id = affinityMap.get(threadId);
        if (id == null) {
            if (threadId == Thread.currentThread().getId()) {
                return getDeviceForCurrentThread();
            }
            // For CPU, all threads use internal device 0
            return INTERNAL_CPU_DEVICE_ID;
        }
        return id;
    }

    /**
     * Returns the device ID for a given array.
     * For CPU backend, all arrays reside on internal device 0.
     *
     * @param array the array to query
     * @return internal CPU device ID (0)
     */
    @Override
    public Integer getDeviceForArray(@NonNull INDArray array) {
        return INTERNAL_CPU_DEVICE_ID;
    }

    /**
     * Returns the number of available devices.
     * For CPU backend, this is always 1 (the CPU itself).
     *
     * @return 1
     */
    @Override
    public int getNumberOfDevices() {
        return 1;
    }

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     * Has no effect on CPU backend since all data resides on the CPU.
     *
     * @param array the array to touch
     */
    @Override
    public void touch(INDArray array) {
        // no-op - CPU has no device memory to sync
    }

    /**
     * Utility method, to associate DataBuffer with specific device (backend-specific)
     * Has no effect on CPU backend since all data resides on the CPU.
     *
     * @param buffer the buffer to touch
     */
    @Override
    public void touch(DataBuffer buffer) {
        // no-op - CPU has no device memory to sync
    }



    /**
     * Returns the active location for the given array.
     * For CPU backend, data is always in HOST memory and considered EVERYWHERE.
     *
     * @param array the array to query
     * @return Location.EVERYWHERE (CPU memory is both host and device)
     */
    @Override
    public Location getActiveLocation(INDArray array) {
        return Location.EVERYWHERE;
    }

    /**
     * Cross-device access is always supported on CPU since there's only one device.
     *
     * @return true
     */
    @Override
    public boolean isCrossDeviceAccessSupported() {
        return true;
    }

    /**
     * Transfers the given array to CPU.
     * For CPU backend, this returns the array itself or a copy.
     *
     * @param array the array to transfer
     * @param makeCopy if true, always make a copy; if false, returns the same array
     * @return the array (possibly a copy) on CPU
     */
    @Override
    public INDArray transferToCpu(INDArray array, boolean makeCopy) {
        if (array == null) {
            return null;
        }
        // CPU backend - data is already on CPU
        if (makeCopy) {
            return array.dup();
        }
        return array;
    }

    /**
     * Transfers the given array to GPU.
     * For CPU backend, this is a no-op and returns the array as-is.
     *
     * @param array the array to transfer
     * @param gpuDeviceId the target GPU device ID (ignored for CPU backend)
     * @return the same array (CPU backend has no GPU)
     */
    @Override
    public INDArray transferToGpu(INDArray array, int gpuDeviceId) {
        if (array == null) {
            return null;
        }
        // CPU backend has no GPU - log a warning and return as-is
        logger.debug("transferToGpu called on CPU backend - no GPU available, returning array as-is");
        return array;
    }

    /**
     * Replicates the given array to the specified device.
     * For CPU backend, accepts CPU_DEVICE_ID (-1) or internal device 0.
     *
     * @param deviceId target device ID
     * @param array the array to replicate
     * @return a copy of the array
     */
    @Override
    public INDArray replicateToDevice(Integer deviceId, INDArray array) {
        if (array == null) {
            return null;
        }

        // For CPU backend, only CPU devices are valid
        if (!isCpuDevice(deviceId)) {
            logger.warn("replicateToDevice called with GPU device {} on CPU backend - treating as CPU device",
                    deviceId);
        }

        // CPU backend - just duplicate the array
        return array.dup();
    }

    /**
     * Replicates the given buffer to the specified device.
     * For CPU backend, accepts CPU_DEVICE_ID (-1) or internal device 0.
     *
     * @param deviceId target device ID
     * @param buffer the buffer to replicate
     * @return a copy of the buffer
     */
    @Override
    public DataBuffer replicateToDevice(Integer deviceId, DataBuffer buffer) {
        if (buffer == null) {
            return null;
        }

        // For CPU backend, only CPU devices are valid
        if (!isCpuDevice(deviceId)) {
            logger.warn("replicateToDevice called with GPU device {} on CPU backend - treating as CPU device",
                    deviceId);
        }

        // CPU backend - just duplicate the buffer
        return buffer.dup();
    }

    @Override
    public void setDeviceForCurrentThread(int deviceId) {
        long threadId = Thread.currentThread().getId();
        affinityMap.put(threadId, INTERNAL_CPU_DEVICE_ID);
        logger.debug("setDeviceForCurrentThread({}) called for thread {} - mapped to CPU internal device {}",
                deviceId, threadId, INTERNAL_CPU_DEVICE_ID);
    }
}
