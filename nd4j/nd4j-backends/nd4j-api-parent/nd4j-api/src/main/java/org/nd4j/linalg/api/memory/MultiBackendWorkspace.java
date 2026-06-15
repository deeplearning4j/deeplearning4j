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

package org.nd4j.linalg.api.memory;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.conf.DeviceAwareWorkspaceConfiguration;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;

import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

/**
 * Extended MemoryWorkspace interface for multi-backend support.
 * Provides device-aware allocation, cross-device memory management,
 * and coherence tracking.
 */
public interface MultiBackendWorkspace extends MemoryWorkspace {

    /**
     * Get the multi-backend configuration
     * @return device-aware configuration
     */
    DeviceAwareWorkspaceConfiguration getMultiBackendConfiguration();

    /**
     * Get the primary device for this workspace
     * @return primary device
     */
    DeviceDescriptor getPrimaryDevice();

    /**
     * Get all devices this workspace has allocated memory on
     * @return list of devices with allocations
     */
    List<DeviceDescriptor> getActiveDevices();

    /**
     * Allocate memory on a specific device
     * @param requiredMemory size in bytes
     * @param dataType data type for alignment
     * @param device target device
     * @param initialize whether to zero-initialize
     * @return pointer to allocated memory
     */
    PagedPointer allocOnDevice(long requiredMemory, DataType dataType,
                                DeviceDescriptor device, boolean initialize);

    /**
     * Get the current offset for a specific device
     * @param device the device
     * @return current allocation offset
     */
    long getOffsetForDevice(DeviceDescriptor device);

    /**
     * Get the current allocated size for a specific device
     * @param device the device
     * @return allocated size in bytes
     */
    long getAllocatedSizeForDevice(DeviceDescriptor device);

    /**
     * Get available space on a specific device
     * @param device the device
     * @return available bytes
     */
    long getAvailableSpaceForDevice(DeviceDescriptor device);

    /**
     * Get allocations for this cycle per device
     * @return map of device to bytes allocated this cycle
     */
    Map<DeviceDescriptor, Long> getPerDeviceCycleAllocations();

    /**
     * Get total allocations per device
     * @return map of device to total bytes allocated
     */
    Map<DeviceDescriptor, Long> getPerDeviceTotalAllocations();

    /**
     * Transfer workspace data to a different device
     * @param sourceDevice source device
     * @param targetDevice target device
     */
    void transferTo(DeviceDescriptor sourceDevice, DeviceDescriptor targetDevice);

    /**
     * Transfer workspace data to a different device asynchronously
     * @param sourceDevice source device
     * @param targetDevice target device
     * @return future that completes when transfer is done
     */
    Future<Void> transferToAsync(DeviceDescriptor sourceDevice, DeviceDescriptor targetDevice);

    /**
     * Prefetch workspace data to a device
     * @param device target device
     */
    void prefetchTo(DeviceDescriptor device);

    /**
     * Synchronize all pending operations on a device
     * @param device the device to sync
     */
    void syncDevice(DeviceDescriptor device);

    /**
     * Synchronize all devices
     */
    void syncAllDevices();

    /**
     * Get the coherence state for memory at an offset on a device
     * @param device the device
     * @param offset the memory offset
     * @return coherence state
     */
    CoherenceState getCoherenceState(DeviceDescriptor device, long offset);

    /**
     * Set the coherence state for memory at an offset on a device
     * @param device the device
     * @param offset the memory offset
     * @param state the new state
     */
    void setCoherenceState(DeviceDescriptor device, long offset, CoherenceState state);

    /**
     * Invalidate memory on a device (mark as stale)
     * @param device the device
     */
    void invalidateDevice(DeviceDescriptor device);

    /**
     * Invalidate all devices except the specified one
     * @param exceptDevice device to keep valid
     */
    void invalidateAllExcept(DeviceDescriptor exceptDevice);

    /**
     * Check if this workspace has memory on a specific device
     * @param device the device
     * @return true if memory is allocated on that device
     */
    boolean hasMemoryOnDevice(DeviceDescriptor device);

    /**
     * Extend workspace capacity on a specific device
     * @param device the device
     * @param additionalBytes bytes to add
     */
    void extendOnDevice(DeviceDescriptor device, long additionalBytes);

    /**
     * Release memory on a specific device
     * @param device the device
     */
    void releaseOnDevice(DeviceDescriptor device);

    /**
     * Get the underlying device-specific workspace (if exists)
     * @param device the device
     * @return the device workspace, or null if not available
     */
    MemoryWorkspace getDeviceWorkspace(DeviceDescriptor device);

    /**
     * Check if mirroring is enabled for this workspace
     * @return true if cross-device mirroring is enabled
     */
    boolean isMirroringEnabled();

    /**
     * Enable or disable mirroring
     * @param enabled true to enable
     */
    void setMirroringEnabled(boolean enabled);

    /**
     * Get statistics for this workspace
     * @return map of statistic name to value
     */
    Map<String, Object> getStatistics();

    /**
     * Reset statistics
     */
    void resetStatistics();

    /**
     * Compact memory on all devices (defragment if possible)
     */
    void compact();

    /**
     * Compact memory on a specific device
     * @param device the device
     */
    void compactOnDevice(DeviceDescriptor device);
}
