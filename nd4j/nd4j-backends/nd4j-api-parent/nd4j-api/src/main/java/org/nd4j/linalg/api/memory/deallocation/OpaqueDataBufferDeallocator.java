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

package org.nd4j.linalg.api.memory.deallocation;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.nativeblas.NativeBufferOwner;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Deallocator for OpaqueDataBuffer instances.
 * This class integrates OpaqueDataBuffer with the DeallocatorService,
 * ensuring reliable cleanup of native DataBuffer memory.
 *
 * <p>IMPORTANT: This class implements Deallocatable but NOT Deallocator.
 * The deallocator() method returns a separate BufferDeallocator instance whose
 * native cleanup facade is detached from the public OpaqueDataBuffer and from
 * this phantom referent. DeallocatableReference strongly retains that cleanup
 * action, so any path back to either public object would prevent collection.</p>
 *
 * @author Adam Gibson
 * @see DeallocatorService
 * @see OpaqueDataBuffer
 */
@Slf4j
public class OpaqueDataBufferDeallocator implements Deallocatable {
    private final long uniqueId;
    private final int targetDevice;
    private final BufferDeallocator innerDeallocator;

    /**
     * Creates a new deallocator for the given OpaqueDataBuffer.
     *
     * @param buffer The OpaqueDataBuffer to manage
     * @param uniqueId Unique identifier for tracking
     * @param targetDevice The device this buffer is allocated on
     * @param allocationBytes The size of the allocation in bytes
     */
    public OpaqueDataBufferDeallocator(OpaqueDataBuffer buffer, long uniqueId, int targetDevice, long allocationBytes) {
        this(buffer, uniqueId, targetDevice, allocationBytes,
                requireOwner(buffer), buffer != null ? buffer.allocationDevice() : null);
    }

    /**
     * Creates a deallocator with the exact backend authority and allocation
     * domain captured when the buffer was created.
     */
    public OpaqueDataBufferDeallocator(OpaqueDataBuffer buffer, long uniqueId, int targetDevice,
                                       long allocationBytes, NativeBufferOwner owner,
                                       DeviceDescriptor allocationDevice) {
        if (buffer == null) {
            throw new IllegalArgumentException("OpaqueDataBuffer cannot be null");
        }
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        this.uniqueId = uniqueId;
        this.targetDevice = targetDevice;
        this.innerDeallocator = new BufferDeallocator(
                buffer, uniqueId, allocationBytes, owner, allocationDevice);
    }

    private static NativeBufferOwner requireOwner(OpaqueDataBuffer buffer) {
        if (buffer == null) {
            throw new IllegalArgumentException("OpaqueDataBuffer cannot be null");
        }
        return buffer.backendOwner();
    }

    @Override
    public long getUniqueId() {
        return uniqueId;
    }

    @Override
    public Deallocator deallocator() {
        return innerDeallocator;
    }

    @Override
    public int targetDevice() {
        return targetDevice;
    }

    public boolean isConstant() {
        return innerDeallocator.isConstant();
    }

    public void setConstant(boolean constant) {
        innerDeallocator.setConstant(constant);
    }

    public boolean isDeallocated() {
        return innerDeallocator.isDeallocated();
    }

    public OpaqueDataBuffer getBuffer() {
        return innerDeallocator.getBuffer();
    }

    public long getAllocationBytes() {
        return innerDeallocator.getAllocationBytes();
    }

    /**
     * Marks this deallocator as having completed deallocation.
     * Called by OpaqueDataBuffer.closeBuffer() after it performs dbClose directly.
     */
    public void markDeallocated() {
        innerDeallocator.markDeallocated();
    }

    /** Raw-address facade with no Java reference back to the registered public buffer. */
    private static final class DetachedCleanupBuffer extends OpaqueDataBuffer {
        private DetachedCleanupBuffer(OpaqueDataBuffer source, NativeBufferOwner owner,
                                      DeviceDescriptor allocationDevice) {
            super((Pointer) null);
            this.address = source.address();
            this.position = source.position();
            this.limit = source.limit();
            this.capacity = source.capacity();
            attachOwner(owner, allocationDevice);
        }
    }

    /**
     * Cleanup action retained by DeallocatableReference. It owns only a detached
     * native-pointer facade and therefore has no path back to the phantom referent.
     */
    @Slf4j
    static class BufferDeallocator implements Deallocator {
        private OpaqueDataBuffer buffer;
        private final long uniqueId;
        private final long allocationBytes;
        private final NativeBufferOwner owner;
        private final DeviceDescriptor allocationDevice;
        private final DeallocatorService service;
        private final AtomicBoolean deallocated = new AtomicBoolean(false);
        private volatile boolean constant = false;

        BufferDeallocator(OpaqueDataBuffer buffer, long uniqueId, long allocationBytes,
                          NativeBufferOwner owner, DeviceDescriptor allocationDevice) {
            this.buffer = new DetachedCleanupBuffer(buffer, owner, allocationDevice);
            this.uniqueId = uniqueId;
            this.allocationBytes = allocationBytes;
            this.owner = owner;
            this.allocationDevice = allocationDevice;
            this.service = owner.deallocatorService();
        }

        @Override
        public void deallocate() {
            if (constant || deallocated.get()) {
                return;
            }

            // During JVM shutdown, release native buffers without consulting any
            // process-primary backend. The owning backend remains authoritative.
            if (DeallocatorService.getShutdownInProgress().get()) {
                try {
                    if (buffer != null && !buffer.isNull() && buffer.tryMarkForDeallocation()) {
                        owner.nativeOps().dbFreeBuffersOnly(buffer);
                        buffer.setNull();
                    }
                } catch (Throwable t) {
                    // Ignore - JVM is shutting down, OS will reclaim all memory.
                } finally {
                    markDeallocated();
                }
                return;
            }

            synchronized (this) {
                if (constant || deallocated.get()) {
                    return;
                }

                try {
                    if (buffer != null && !buffer.isNull()) {
                        if (!buffer.tryMarkForDeallocation()) {
                            // Another deallocator (e.g. explicit closeBuffer) already claimed this buffer.
                            return;
                        }

                        boolean deviceBacked = allocationDevice != null
                                && allocationDevice.getDeviceType().isAccelerator();
                        int currentDevice = -1;
                        boolean switchedDevice = false;
                        if (deviceBacked) {
                            int bufferDevice = allocationDevice.getDeviceIndex();
                            int deviceCount = owner.deviceCount();
                            if (bufferDevice < 0 || bufferDevice >= deviceCount) {
                                throw new IllegalStateException(
                                        "Invalid allocation device " + bufferDevice
                                                + " for owning backend with " + deviceCount + " devices");
                            }

                            currentDevice = owner.currentDevice();
                            if (currentDevice != bufferDevice) {
                                owner.setDevice(bufferDevice);
                                switchedDevice = true;
                            }
                        }

                        try {
                            owner.commit();
                            // Narrow the race window between the initial shutdown check
                            // and the shutdown hook setting the flag.
                            if (DeallocatorService.getShutdownInProgress().get()) {
                                owner.nativeOps().dbFreeBuffersOnly(buffer);
                            } else {
                                owner.nativeOps().dbClose(buffer);
                            }
                            buffer.setNull();

                            if (allocationBytes > 0 && allocationDevice != null) {
                                owner.recordDeallocation(allocationDevice, allocationBytes);
                            }
                        } finally {
                            if (switchedDevice) {
                                owner.setDevice(currentDevice);
                            }
                        }
                    }
                } catch (Exception e) {
                    log.error("Error deallocating OpaqueDataBuffer with uniqueId: " + uniqueId, e);
                } finally {
                    markDeallocated();
                }
            }
        }

        @Override
        public boolean isConstant() {
            return constant;
        }

        @Override
        public void setConstant(boolean constant) {
            this.constant = constant;
        }

        boolean isDeallocated() {
            return deallocated.get();
        }

        OpaqueDataBuffer getBuffer() {
            return buffer;
        }

        long getAllocationBytes() {
            return allocationBytes;
        }

        void markDeallocated() {
            synchronized (this) {
                if (this.buffer != null) {
                    this.buffer.setNull();
                }
                this.buffer = null;
                this.deallocated.set(true);
                service.getReferenceMap().remove(uniqueId);
            }
        }
    }
}
