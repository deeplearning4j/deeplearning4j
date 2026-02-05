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
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.nativeblas.OpaqueDataBuffer;

/**
 * Interface for data buffers that support multi-device/hybrid execution.
 * A HybridDataBuffer can track which device currently owns the data
 * and manage data transfers between devices.
 *
 * <p>This interface integrates with the existing {@link OpaqueDataBuffer} infrastructure
 * which provides the underlying native memory management with primary (host) and
 * special (device) buffers.</p>
 *
 * <p>Key integration points with OpaqueDataBuffer:</p>
 * <ul>
 *   <li>{@link OpaqueDataBuffer#primaryBuffer()} - Host memory pointer</li>
 *   <li>{@link OpaqueDataBuffer#specialBuffer()} - Device memory pointer</li>
 *   <li>{@link OpaqueDataBuffer#syncToPrimary()} - Device to host sync</li>
 *   <li>{@link OpaqueDataBuffer#syncToSpecial()} - Host to device sync</li>
 *   <li>{@link OpaqueDataBuffer#deviceId()} - Device affinity</li>
 * </ul>
 *
 * <p>Also supports workspace-based allocation for efficient memory reuse
 * across multiple devices.</p>
 *
 * @see OpaqueDataBuffer
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
     * Get the device this buffer is pinned to, if any.
     * When pinned, ops should be routed to this device and transfers
     * to other devices should be treated as an error.
     *
     * @return the pinned device, or null if not pinned
     */
    default DeviceDescriptor getPinnedDevice() {
        return null;
    }

    /**
     * Pin this buffer to a specific device.
     * Implementations should ensure data is available on that device.
     *
     * @param device the device to pin to
     */
    default void pinTo(DeviceDescriptor device) {
        // no-op by default
    }

    /**
     * Remove any pinning from this buffer.
     */
    default void unpin() {
        // no-op by default
    }

    /**
     * Check if this buffer is pinned to a device.
     *
     * @return true if pinned
     */
    default boolean isPinned() {
        return getPinnedDevice() != null;
    }

    /**
     * Get the effective device for routing decisions.
     * Uses the pinned device if present, otherwise the owner device.
     *
     * @return the effective device, or null if unknown
     */
    default DeviceDescriptor getEffectiveDevice() {
        DeviceDescriptor pinned = getPinnedDevice();
        return pinned != null ? pinned : getOwnerDevice();
    }

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
        return getMultiBackendWorkspace() != null;
    }

    /**
     * Get the parent multi-backend workspace, if any.
     *
     * @return the parent workspace, or null if not attached
     */
    MultiBackendWorkspace getMultiBackendWorkspace();

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
        MultiBackendWorkspace workspace = getMultiBackendWorkspace();
        if (workspace != null) {
            try {
                workspace.transferTo(sourceDevice, targetDevice);
                return;
            } catch (UnsupportedOperationException e) {
                // Fall back to buffer-level transfer if workspace doesn't implement data movement
            }
        }
        ensureAvailableOn(targetDevice);
    }

    // ========================
    // OpaqueDataBuffer Integration
    // ========================

    /**
     * Get the native device ID from the underlying OpaqueDataBuffer.
     * This corresponds to {@link OpaqueDataBuffer#deviceId()}.
     *
     * @return the native device ID, or -1 if not available
     */
    default int getNativeDeviceId() {
        OpaqueDataBuffer opaque = opaqueBuffer();
        return opaque != null ? opaque.deviceId() : -1;
    }

    /**
     * Synchronize data from device (special buffer) to host (primary buffer).
     * This is a wrapper around {@link OpaqueDataBuffer#syncToPrimary()}.
     *
     * <p>Use this when you need to read device data on the host.</p>
     */
    default void syncDeviceToHost() {
        OpaqueDataBuffer opaque = opaqueBuffer();
        if (opaque != null) {
            opaque.syncToPrimary();
        }
    }

    /**
     * Synchronize data from host (primary buffer) to device (special buffer).
     * This is a wrapper around {@link OpaqueDataBuffer#syncToSpecial()}.
     *
     * <p>Use this when you need to push host data to the device.</p>
     */
    default void syncHostToDevice() {
        OpaqueDataBuffer opaque = opaqueBuffer();
        if (opaque != null) {
            opaque.syncToSpecial();
        }
    }

    /**
     * Check if this buffer has a valid device (special) buffer.
     *
     * @return true if special buffer exists and is non-null
     */
    default boolean hasDeviceBuffer() {
        OpaqueDataBuffer opaque = opaqueBuffer();
        if (opaque != null) {
            return opaque.specialBuffer() != null && !opaque.specialBuffer().isNull();
        }
        return false;
    }

    /**
     * Check if this buffer has a valid host (primary) buffer.
     *
     * @return true if primary buffer exists and is non-null
     */
    default boolean hasHostBuffer() {
        OpaqueDataBuffer opaque = opaqueBuffer();
        if (opaque != null) {
            return opaque.primaryBuffer() != null && !opaque.primaryBuffer().isNull();
        }
        return false;
    }

    /**
     * Get the host (primary) buffer address.
     *
     * @return the host buffer address, or 0 if not available
     */
    default long getHostBufferAddress() {
        OpaqueDataBuffer opaque = opaqueBuffer();
        if (opaque != null && opaque.primaryBuffer() != null) {
            return opaque.primaryBuffer().address();
        }
        return 0;
    }

    /**
     * Get the device (special) buffer address.
     *
     * @return the device buffer address, or 0 if not available
     */
    default long getDeviceBufferAddress() {
        OpaqueDataBuffer opaque = opaqueBuffer();
        if (opaque != null && opaque.specialBuffer() != null) {
            return opaque.specialBuffer().address();
        }
        return 0;
    }

    /**
     * Convenience method to ensure data is on the host.
     * Syncs from device if owner is a GPU device.
     */
    default void ensureOnHost() {
        DeviceDescriptor owner = getOwnerDevice();
        if (owner != null && owner.getDeviceType().isGpu()) {
            syncDeviceToHost();
        }
    }

    /**
     * Convenience method to ensure data is on the default GPU.
     * Syncs from host if owner is CPU.
     */
    default void ensureOnDevice() {
        DeviceDescriptor owner = getOwnerDevice();
        if (owner == null || owner.getDeviceType() == DeviceType.CPU) {
            syncHostToDevice();
        }
    }

    /**
     * Check if the data is currently dirty (modified) on the host and needs sync to device.
     * Implementations should track modification state.
     *
     * @return true if host data is dirty
     */
    default boolean isHostDirty() {
        // Default implementation - subclasses should override with actual tracking
        return false;
    }

    /**
     * Check if the data is currently dirty (modified) on the device and needs sync to host.
     * Implementations should track modification state.
     *
     * @return true if device data is dirty
     */
    default boolean isDeviceDirty() {
        // Default implementation - subclasses should override with actual tracking
        return false;
    }

    /**
     * Mark the host buffer as dirty (modified).
     * Call this after modifying data on the host.
     */
    default void markHostDirty() {
        // Default implementation - subclasses should override
    }

    /**
     * Mark the device buffer as dirty (modified).
     * Call this after modifying data on the device.
     */
    default void markDeviceDirty() {
        // Default implementation - subclasses should override
    }
}
