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

package org.nd4j.linalg.api.buffer;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;

/**
 * Interface for data buffers that support multi-device/hybrid execution.
 * A HybridDataBuffer can track which device currently owns the data
 * and manage data transfers between devices.
 *
 * <p>Also supports workspace-based allocation for efficient memory reuse
 * across multiple devices.</p>
 */
public interface HybridDataBuffer extends DataBuffer {

    /**
     * Get the device that currently owns/has the most recent copy of this data.
     *
     * @return the owner device descriptor, or null if on host only
     */
    DeviceDescriptor getOwnerDevice();

    /**
     * Set the device that owns this buffer's data.
     *
     * @param device the new owner device
     */
    void setOwnerDevice(DeviceDescriptor device);

    /**
     * Check if the data is valid on the specified device.
     *
     * @param device the device to check
     * @return true if data is valid and up-to-date on that device
     */
    boolean isValidOn(DeviceDescriptor device);

    /**
     * Mark the data as valid on the specified device.
     *
     * @param device the device where data is now valid
     */
    void markValidOn(DeviceDescriptor device);

    /**
     * Mark the data as invalid (stale) on the specified device.
     *
     * @param device the device where data is now invalid
     */
    void markInvalidOn(DeviceDescriptor device);

    /**
     * Ensure the data is available and up-to-date on the specified device.
     * This may trigger a data transfer if necessary.
     *
     * @param device the target device
     */
    void ensureAvailableOn(DeviceDescriptor device);

    /**
     * Ensure the data is readable on the specified device.
     * Similar to ensureAvailableOn, but specifically for read access.
     *
     * @param device the target device
     */
    default void ensureReadableOn(DeviceDescriptor device) {
        ensureAvailableOn(device);
    }

    /**
     * Prefetch the data to the specified device asynchronously.
     * This hints that the data will be needed on the device soon.
     *
     * @param device the target device
     */
    default void prefetch(DeviceDescriptor device) {
        // Default implementation: synchronous transfer
        ensureAvailableOn(device);
    }

    /**
     * Get the memory location/pointer for this buffer on the specified device.
     *
     * @param device the device
     * @return the device-specific memory address, or 0 if not allocated on that device
     */
    long getDeviceAddress(DeviceDescriptor device);

    @Override
    default boolean isHybrid() {
        return true;
    }

    @Override
    default HybridDataBuffer asHybrid() {
        return this;
    }

    // ========================
    // Workspace Support
    // ========================

    /**
     * Check if this buffer is attached to a multi-backend workspace.
     *
     * @return true if attached to a workspace
     */
    default boolean isAttachedToWorkspace() {
        return getParentWorkspace() != null;
    }

    /**
     * Get the parent multi-backend workspace, if any.
     *
     * @return the parent workspace, or null if not attached
     */
    MultiBackendWorkspace getParentWorkspace();

    /**
     * Attach this buffer to a multi-backend workspace.
     * The buffer's memory will be managed by the workspace.
     *
     * @param workspace the workspace to attach to
     */
    void attachToWorkspace(MultiBackendWorkspace workspace);

    /**
     * Detach this buffer from its workspace, if attached.
     * The buffer may be copied to independent memory.
     */
    void detachFromWorkspace();

    /**
     * Allocate memory for this buffer on the specified device using the parent workspace.
     * If not attached to a workspace, allocates independent memory.
     *
     * @param device the device to allocate on
     * @param requiredSize the required size in bytes
     */
    void allocateOnDevice(DeviceDescriptor device, long requiredSize);

    /**
     * Get the workspace offset for this buffer's allocation on a specific device.
     *
     * @param device the device
     * @return the offset within the workspace, or -1 if not allocated via workspace
     */
    default long getWorkspaceOffset(DeviceDescriptor device) {
        return -1;
    }

    /**
     * Check if this buffer's allocation on a device is workspace-managed.
     *
     * @param device the device
     * @return true if the allocation on that device is from a workspace
     */
    default boolean isWorkspaceAllocation(DeviceDescriptor device) {
        return isAttachedToWorkspace() && getWorkspaceOffset(device) >= 0;
    }

    /**
     * Leverage workspace for a device-to-device copy.
     * This uses the workspace's optimized transfer mechanisms if available.
     *
     * @param sourceDevice the source device
     * @param targetDevice the target device
     */
    default void workspaceCopy(DeviceDescriptor sourceDevice, DeviceDescriptor targetDevice) {
        MultiBackendWorkspace workspace = getParentWorkspace();
        if (workspace != null) {
            workspace.transferTo(sourceDevice, targetDevice);
        } else {
            ensureAvailableOn(targetDevice);
        }
    }
}
