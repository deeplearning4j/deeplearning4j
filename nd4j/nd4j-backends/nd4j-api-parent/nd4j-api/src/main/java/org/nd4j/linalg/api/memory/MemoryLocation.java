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

import org.nd4j.linalg.api.device.DeviceDescriptor;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Represents a memory location on a specific device with coherence tracking.
 * Each MemoryLocation tracks:
 * <ul>
 *   <li>The device where memory is allocated</li>
 *   <li>The memory address/pointer</li>
 *   <li>The coherence state (EXCLUSIVE, SHARED, MODIFIED, INVALID)</li>
 *   <li>A version number for tracking modifications</li>
 *   <li>Reference count for memory management</li>
 * </ul>
 */
public class MemoryLocation {

    private final DeviceDescriptor device;
    private final long address;
    private final long size;

    private volatile CoherenceState state;
    private final AtomicLong version;
    private final AtomicInteger referenceCount;
    private final long creationTime;
    private volatile long lastAccessTime;

    /**
     * Create a new memory location
     * @param device the device where memory is located
     * @param address the memory address
     * @param size the size in bytes
     * @param initialState the initial coherence state
     * @param initialVersion the initial version number
     */
    public MemoryLocation(DeviceDescriptor device, long address, long size,
                          CoherenceState initialState, long initialVersion) {
        this.device = Objects.requireNonNull(device, "Device cannot be null");
        this.address = address;
        this.size = size;
        this.state = initialState;
        this.version = new AtomicLong(initialVersion);
        this.referenceCount = new AtomicInteger(1);
        this.creationTime = System.nanoTime();
        this.lastAccessTime = creationTime;
    }

    /**
     * Create a new memory location with EXCLUSIVE state and version 0
     * @param device the device
     * @param address the memory address
     * @param size the size in bytes
     */
    public MemoryLocation(DeviceDescriptor device, long address, long size) {
        this(device, address, size, CoherenceState.EXCLUSIVE, 0);
    }

    /**
     * Get the device where this memory is located
     * @return the device
     */
    public DeviceDescriptor getDevice() {
        return device;
    }

    /**
     * Get the memory address
     * @return the address
     */
    public long getAddress() {
        return address;
    }

    /**
     * Get the size of this memory region
     * @return size in bytes
     */
    public long getSize() {
        return size;
    }

    /**
     * Get the current coherence state
     * @return the coherence state
     */
    public CoherenceState getState() {
        return state;
    }

    /**
     * Set the coherence state
     * @param newState the new state
     */
    public void setState(CoherenceState newState) {
        this.state = newState;
        this.lastAccessTime = System.nanoTime();
    }

    /**
     * Get the current version
     * @return the version number
     */
    public long getVersion() {
        return version.get();
    }

    /**
     * Increment the version (after modification)
     * @return the new version
     */
    public long incrementVersion() {
        lastAccessTime = System.nanoTime();
        return version.incrementAndGet();
    }

    /**
     * Set the version to match another location
     * @param newVersion the version to set
     */
    public void setVersion(long newVersion) {
        version.set(newVersion);
        lastAccessTime = System.nanoTime();
    }

    /**
     * Get the reference count
     * @return current reference count
     */
    public int getReferenceCount() {
        return referenceCount.get();
    }

    /**
     * Increment the reference count
     * @return the new count
     */
    public int retain() {
        return referenceCount.incrementAndGet();
    }

    /**
     * Decrement the reference count
     * @return the new count
     */
    public int release() {
        return referenceCount.decrementAndGet();
    }

    /**
     * Check if this memory can be freed (reference count is 0)
     * @return true if freeable
     */
    public boolean canFree() {
        return referenceCount.get() <= 0;
    }

    /**
     * Get the creation timestamp
     * @return creation time in nanoseconds
     */
    public long getCreationTime() {
        return creationTime;
    }

    /**
     * Get the last access timestamp
     * @return last access time in nanoseconds
     */
    public long getLastAccessTime() {
        return lastAccessTime;
    }

    /**
     * Update the last access time to now
     */
    public void touch() {
        this.lastAccessTime = System.nanoTime();
    }

    /**
     * Check if this location's data is valid
     * @return true if data is valid
     */
    public boolean isValid() {
        return state.isValid();
    }

    /**
     * Check if this location is the owner of the data
     * @return true if owner
     */
    public boolean isOwner() {
        return state.isOwner();
    }

    /**
     * Check if this location can be written to without sync
     * @return true if writable
     */
    public boolean canWrite() {
        return state.canWrite();
    }

    /**
     * Invalidate this location
     */
    public void invalidate() {
        this.state = CoherenceState.INVALID;
        this.lastAccessTime = System.nanoTime();
    }

    /**
     * Mark as modified after a write
     */
    public void markModified() {
        this.state = CoherenceState.MODIFIED;
        incrementVersion();
    }

    /**
     * Mark as shared (another device has a copy)
     */
    public void markShared() {
        if (this.state == CoherenceState.EXCLUSIVE) {
            this.state = CoherenceState.SHARED;
        }
        this.lastAccessTime = System.nanoTime();
    }

    /**
     * Estimate the transfer time to another device
     * @param target the target device
     * @return estimated time in nanoseconds
     */
    public long estimateTransferTime(DeviceDescriptor target) {
        if (target.equals(device)) {
            return 0;
        }
        long bandwidth = device.getTransferBandwidthTo(target);
        if (bandwidth <= 0) {
            return Long.MAX_VALUE;  // Transfer not possible
        }
        // Time = size / bandwidth (in nanoseconds)
        return (size * 1_000_000_000L) / bandwidth;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MemoryLocation that = (MemoryLocation) o;
        return address == that.address && Objects.equals(device, that.device);
    }

    @Override
    public int hashCode() {
        return Objects.hash(device, address);
    }

    @Override
    public String toString() {
        return String.format("MemoryLocation[device=%s, addr=0x%X, size=%d, state=%s, ver=%d, refs=%d]",
                device.getDeviceId(), address, size, state, version.get(), referenceCount.get());
    }
}
