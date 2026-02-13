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
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.BackendRegistry;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Deallocator for OpaqueDataBuffer instances.
 * This class integrates OpaqueDataBuffer with the DeallocatorService,
 * ensuring reliable cleanup of native DataBuffer memory.
 *
 * <p>IMPORTANT: This class implements Deallocatable but NOT Deallocator.
 * The deallocator() method returns a separate BufferDeallocator instance.
 * This separation is critical: DeallocatableReference (PhantomReference) stores
 * the Deallocator returned by deallocator(). If deallocator() returned 'this',
 * the DeallocatableReference would hold a strong reference to the referent,
 * preventing the PhantomReference from ever being enqueued, which means the
 * GC-based cleanup path would never fire.</p>
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
        if (buffer == null) {
            throw new IllegalArgumentException("OpaqueDataBuffer cannot be null");
        }
        this.uniqueId = uniqueId;
        this.targetDevice = targetDevice;
        this.innerDeallocator = new BufferDeallocator(buffer, uniqueId, targetDevice, allocationBytes);
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

    /**
     * Marks this deallocator as having completed deallocation.
     * Called by OpaqueDataBuffer.closeBuffer() after it performs dbClose directly.
     */
    public void markDeallocated() {
        innerDeallocator.markDeallocated();
    }

    /**
     * The actual Deallocator implementation. This is a SEPARATE object from the
     * Deallocatable (OpaqueDataBufferDeallocator) to avoid the strong reference cycle
     * that would prevent PhantomReference enqueuing.
     */
    @Slf4j
    static class BufferDeallocator implements Deallocator {
        private OpaqueDataBuffer buffer;
        private final long uniqueId;
        private final int targetDevice;
        private final long allocationBytes;
        private final AtomicBoolean deallocated = new AtomicBoolean(false);
        private volatile boolean constant = false;

        BufferDeallocator(OpaqueDataBuffer buffer, long uniqueId, int targetDevice, long allocationBytes) {
            this.buffer = buffer;
            this.uniqueId = uniqueId;
            this.targetDevice = targetDevice;
            this.allocationBytes = allocationBytes;
        }

        @Override
        public void deallocate() {
            if (constant || deallocated.get()) {
                return;
            }

            // During JVM shutdown, use GPU-only free (no host free) to avoid SIGABRT
            // from corrupted malloc metadata. C++ op buffer overruns corrupt adjacent
            // heap chunk headers; calling free() discovers this and aborts.
            if (DeallocatorService.getShutdownInProgress().get()) {
                try {
                    if (buffer != null && !buffer.isNull() && buffer.tryMarkForDeallocation()) {
                        Nd4j.getNativeOps().dbFreeBuffersOnly(buffer);
                        buffer.setNull();
                    }
                } catch (Throwable t) {
                    // Ignore - JVM is shutting down, OS will reclaim all memory
                } finally {
                    buffer = null;
                    deallocated.set(true);
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
                            // Another deallocator (e.g. explicit closeBuffer) already claimed this buffer
                            return;
                        }

                        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                        int bufferDevice = targetDevice;
                        try {
                            bufferDevice = buffer.deviceId();
                        } catch (Exception e) {
                            // Fall back to targetDevice if native query fails
                        }

                        int deviceCount = Nd4j.getAffinityManager().getNumberOfDevices();
                        if (deviceCount <= 0 || bufferDevice < 0 || bufferDevice >= deviceCount) {
                            bufferDevice = targetDevice;
                        }
                        if (deviceCount > 0 && (bufferDevice < 0 || bufferDevice >= deviceCount)) {
                            bufferDevice = currentDevice;
                        }
                        boolean switchedDevice = false;

                        if (currentDevice != bufferDevice) {
                            DeviceMemoryManager.getInstance().switchDevice(bufferDevice, "OpaqueDataBufferDeallocator.deallocate", "dealloc");
                            switchedDevice = true;
                        }

                        try {
                            Nd4j.getExecutioner().commit();
                            // Second shutdown check: narrows the race window between the
                            // initial check above and the shutdown hook setting the flag.
                            // If shutdown started while we were setting up, use safe path.
                            if (DeallocatorService.getShutdownInProgress().get()) {
                                Nd4j.getNativeOps().dbFreeBuffersOnly(buffer);
                            } else {
                                Nd4j.getNativeOps().dbClose(buffer);
                            }
                            buffer.setNull();

                            if (allocationBytes > 0) {
                                DeviceDescriptor actualDevice = resolveDeviceDescriptor(bufferDevice);
                                if (actualDevice != null) {
                                    DeviceMemoryManager.getInstance().recordDeallocation(actualDevice, allocationBytes);
                                }
                            }
                        } finally {
                            if (switchedDevice) {
                                DeviceMemoryManager.getInstance().switchDevice(currentDevice, "OpaqueDataBufferDeallocator.deallocate", "restore-device");
                            }
                        }
                    }
                } catch (Exception e) {
                    log.error("Error deallocating OpaqueDataBuffer with uniqueId: " + uniqueId, e);
                } finally {
                    buffer = null;
                    deallocated.set(true);
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

        void markDeallocated() {
            synchronized (this) {
                this.buffer = null;
                this.deallocated.set(true);
            }
        }

        private static DeviceDescriptor resolveDeviceDescriptor(int deviceId) {
            BackendRegistry registry = BackendRegistry.getInstance();
            if (deviceId >= 0) {
                DeviceDescriptor byId = registry.getDevice(DeviceDescriptor.cuda(deviceId).getDeviceId());
                return byId != null ? byId : DeviceDescriptor.cuda(deviceId);
            }

            DeviceDescriptor cpu = registry.getDefaultCpuDevice();
            return cpu != null ? cpu : DeviceDescriptor.cpu();
        }
    }
}
