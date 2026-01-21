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

package org.nd4j.linalg.api.concurrency;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface AffinityManager {

    /**
     * CPU device ID constant. The CPU is always represented as device -1 in multi-device scenarios.
     * This allows seamless CPU-GPU data transfer when using CUDA backend.
     */
    int CPU_DEVICE_ID = -1;

    enum Location {
        HOST, DEVICE, EVERYWHERE,
    }

    /**
     * This method returns deviceId for current thread
     * @return
     */
    Integer getDeviceForCurrentThread();

    /**
     * This method returns deviceId for a given thread
     * @return
     */
    Integer getDeviceForThread(long threadId);


    /**
     * This method returns id of current device for a given INDArray
     *
     * @param array
     * @return
     */
    Integer getDeviceForArray(INDArray array);

    /**
     * This method returns number of available devices
     * @return
     */
    int getNumberOfDevices();

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     *
     * @param array
     */
    void touch(INDArray array);

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     * 
     * @param buffer
     */
    void touch(DataBuffer buffer);

    /**
     * This method replicates given INDArray, and places it to target device.
     *
     * @param deviceId  target deviceId
     * @param array INDArray to replicate
     * @return
     */
    INDArray replicateToDevice(Integer deviceId, INDArray array);

    /**
     * This method replicates given DataBuffer, and places it to target device.
     *
     * @param deviceId  target deviceId
     * @param buffer
     * @return
     */
    DataBuffer replicateToDevice(Integer deviceId, DataBuffer buffer);

    /**
     * This method tags specific INDArray as "recent" on specified location
     *
     * @param location
     */
    void tagLocation(INDArray array, Location location);

    /**
     * This method tags specific DataBuffer as "recent" on specified location
     *
     * @param location
     */
    void tagLocation(DataBuffer buffer, Location location);


    /**
     * This method propagates given INDArray to specified location
     *
     * @param array
     * @param location
     */
    void ensureLocation(INDArray array, Location location);

    /**
     * This method returns last-updated location for the given INDArray
     * @param array
     * @return
     */
    Location getActiveLocation(INDArray array);

    /**
     * This method forces specific device for current thread.
     *
     * PLEASE NOTE: This method is UNSAFE and should NOT be used with 100% clearance about it.
     *
     * @param deviceId
     */
    void unsafeSetDevice(Integer deviceId);


    /**
     * This method returns TRUE if cross-device access is allowed on this system
     */
    boolean isCrossDeviceAccessSupported();

    /**
     * This method allows to block cross-device access. Mostly suitable for debugging/testing purposes
     *
     * @param reallyAllow
     */
    void allowCrossDeviceAccess(boolean reallyAllow);

    /**
     * Returns the device type for the given device ID.
     * CPU_DEVICE_ID (-1) returns CPU, GPU device IDs (>= 0) return CUDA_GPU.
     *
     * @param deviceId the device ID to query
     * @return the device type
     */
    default DeviceType getDeviceType(int deviceId) {
        return deviceId == CPU_DEVICE_ID ? DeviceType.CPU : DeviceType.CUDA_GPU;
    }

    /**
     * Checks if the given device ID represents the CPU.
     *
     * @param deviceId the device ID to check
     * @return true if the device is CPU, false otherwise
     */
    default boolean isCpuDevice(int deviceId) {
        return deviceId == CPU_DEVICE_ID;
    }

    /**
     * Returns whether CPU is available as a target device for data placement.
     * Always returns true - CPU is always available.
     *
     * @return true
     */
    default boolean isCpuDeviceAvailable() {
        return true;
    }

    /**
     * Transfers the given array to the CPU (host memory) and returns a copy.
     * For CPU backend, this returns the array itself (or a copy if specified).
     * For GPU backend, this syncs data to host and optionally creates a CPU-resident copy.
     *
     * @param array the array to transfer
     * @param makeCopy if true, always make a copy; if false, may return same array if already on CPU
     * @return an array with data on CPU
     */
    default INDArray transferToCpu(INDArray array, boolean makeCopy) {
        return replicateToDevice(CPU_DEVICE_ID, array);
    }

    /**
     * Transfers the given array to the specified GPU device.
     * For CPU backend, this is a no-op (returns the array as-is).
     * For GPU backend, this moves/copies data to the specified GPU.
     *
     * @param array the array to transfer
     * @param gpuDeviceId the target GPU device ID (must be >= 0)
     * @return an array with data on the specified GPU
     */
    default INDArray transferToGpu(INDArray array, int gpuDeviceId) {
        if (gpuDeviceId < 0) {
            throw new IllegalArgumentException("GPU device ID must be >= 0, got: " + gpuDeviceId);
        }
        return replicateToDevice(gpuDeviceId, array);
    }

    /**
     * Ensures the array data is synchronized to CPU (host memory).
     * This is useful for preparing data for CPU-side operations or for transferring to CPU backend.
     *
     * @param array the array to synchronize
     */
    default void ensureOnCpu(INDArray array) {
        ensureLocation(array, Location.HOST);
    }

    /**
     * Ensures the array data is synchronized to GPU (device memory).
     * For CPU backend, this is a no-op.
     *
     * @param array the array to synchronize
     */
    default void ensureOnGpu(INDArray array) {
        ensureLocation(array, Location.DEVICE);
    }

    // =========================================================================
    // DeviceDescriptor Integration
    // =========================================================================

    /**
     * Get a DeviceDescriptor for the given device ID.
     *
     * @param deviceId the device ID (CPU_DEVICE_ID for CPU, >= 0 for GPU)
     * @return the device descriptor
     */
    default DeviceDescriptor getDeviceDescriptor(int deviceId) {
        if (isCpuDevice(deviceId)) {
            return DeviceDescriptor.cpu();
        }
        return DeviceDescriptor.cuda(deviceId);
    }

    /**
     * Get a DeviceDescriptor for the current thread's device.
     *
     * @return the device descriptor for the current thread
     */
    default DeviceDescriptor getCurrentDeviceDescriptor() {
        return getDeviceDescriptor(getDeviceForCurrentThread());
    }

    /**
     * Get a DeviceDescriptor for the CPU.
     *
     * @return the CPU device descriptor
     */
    default DeviceDescriptor getCpuDeviceDescriptor() {
        return DeviceDescriptor.cpu();
    }

    /**
     * Get the device ID from a DeviceDescriptor.
     *
     * @param descriptor the device descriptor
     * @return the device ID (CPU_DEVICE_ID for CPU, device index for GPU)
     */
    default int getDeviceId(DeviceDescriptor descriptor) {
        if (descriptor == null) {
            return CPU_DEVICE_ID;
        }
        if (descriptor.getDeviceType() == DeviceType.CPU) {
            return CPU_DEVICE_ID;
        }
        return descriptor.getDeviceIndex();
    }

    /**
     * Replicate an array to the device specified by a DeviceDescriptor.
     *
     * @param device the target device descriptor
     * @param array the array to replicate
     * @return the replicated array on the target device
     */
    default INDArray replicateToDevice(DeviceDescriptor device, INDArray array) {
        return replicateToDevice(getDeviceId(device), array);
    }

    /**
     * Replicate a buffer to the device specified by a DeviceDescriptor.
     *
     * @param device the target device descriptor
     * @param buffer the buffer to replicate
     * @return the replicated buffer on the target device
     */
    default DataBuffer replicateToDevice(DeviceDescriptor device, DataBuffer buffer) {
        return replicateToDevice(getDeviceId(device), buffer);
    }

    /**
     * Get the DeviceDescriptor for a given array.
     *
     * @param array the array to query
     * @return the device descriptor where the array resides
     */
    default DeviceDescriptor getDeviceDescriptorForArray(INDArray array) {
        Integer deviceId = getDeviceForArray(array);
        return getDeviceDescriptor(deviceId != null ? deviceId : CPU_DEVICE_ID);
    }
}
