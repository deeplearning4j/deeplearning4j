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
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.nativeblas.NativeBufferOwner;
import org.nd4j.nativeblas.OpaqueNDArray;

/**
 * Deallocator for OpaqueNDArray instances.
 * This class integrates OpaqueNDArray with the DeallocatorService,
 * ensuring reliable cleanup of native memory without relying on
 * unreliable Java finalizers.
 *
 * <p>When an OpaqueNDArray is created, this registration object is observed
 * through a PhantomReference. The cleanup action retained by that reference
 * owns a detached raw-address facade, never the public array or this phantom
 * referent; otherwise the cleanup action itself would prevent collection.</p>
 *
 * @author Adam Gibson
 * @see DeallocatorService
 * @see OpaqueNDArray
 */
@Slf4j
public class OpaqueNDArrayDeallocator implements Deallocatable {
    private final long uniqueId;
    private final int targetDevice;
    private final ArrayDeallocator innerDeallocator;

    /**
     * Creates a new deallocator for the given OpaqueNDArray.
     *
     * @param array The OpaqueNDArray to manage
     * @param uniqueId Unique identifier for tracking
     * @param targetDevice The device this array is allocated on
     */
    public OpaqueNDArrayDeallocator(OpaqueNDArray array, long uniqueId, int targetDevice) {
        this(array, uniqueId, targetDevice, requireOwner(array));
    }

    public OpaqueNDArrayDeallocator(OpaqueNDArray array, long uniqueId, int targetDevice,
                                    NativeBufferOwner owner) {
        if (array == null) {
            throw new IllegalArgumentException("OpaqueNDArray cannot be null");
        }
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        this.uniqueId = uniqueId;
        this.targetDevice = targetDevice;
        this.innerDeallocator = new ArrayDeallocator(array, uniqueId, targetDevice, owner);
    }

    private static NativeBufferOwner requireOwner(OpaqueNDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("OpaqueNDArray cannot be null");
        }
        return array.backendOwner();
    }

    public void deallocate() {
        innerDeallocator.deallocate();
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

    public OpaqueNDArray getArray() {
        return innerDeallocator.getArray();
    }

    public boolean isConstant() {
        return innerDeallocator.isConstant();
    }

    public void setConstant(boolean constant) {
        innerDeallocator.setConstant(constant);
    }

    public void retainDataBuffers(DataBuffer shapeInfo, DataBuffer data, DataBuffer special) {
        innerDeallocator.retainDataBuffers(shapeInfo, data, special);
    }

    public boolean isDeallocated() {
        return innerDeallocator.isDeallocated();
    }

    /** Raw-address facade with no Java path back to the registered public array. */
    private static final class DetachedCleanupArray extends OpaqueNDArray {
        private LongPointer pointerStorage;

        private DetachedCleanupArray(OpaqueNDArray source, NativeBufferOwner owner) {
            super((Pointer) null);
            // OpaqueNDArray is @ByVal: source.address() is a JavaCPP pointer-cell
            // address, while the actual sd::NDArray* is the cell's first word.
            // Own an independent cell so the cleanup facade never references source.
            pointerStorage = new LongPointer(1L);
            pointerStorage.put(0L, new LongPointer(source).get(0L));
            this.address = pointerStorage.address();
            this.position = 0L;
            this.limit = 1L;
            this.capacity = 1L;
            attachOwner(owner);
        }

        private void releasePointerStorage() {
            if (pointerStorage != null) {
                pointerStorage.close();
                pointerStorage = null;
            }
        }
    }

    /** Cleanup action retained by DeallocatableReference. */
    @Slf4j
    private static final class ArrayDeallocator implements Deallocator {
        private OpaqueNDArray array;
        private final long uniqueId;
        private final int targetDevice;
        private final NativeBufferOwner owner;
        private final DeallocatorService service;
        private DataBuffer shapeInfoBufferRef;
        private DataBuffer dataBufferRef;
        private DataBuffer specialBufferRef;
        private volatile boolean deallocated;
        private volatile boolean constant;

        private ArrayDeallocator(OpaqueNDArray array, long uniqueId, int targetDevice,
                                 NativeBufferOwner owner) {
            this.array = new DetachedCleanupArray(array, owner);
            this.uniqueId = uniqueId;
            this.targetDevice = targetDevice;
            this.owner = owner;
            this.service = owner.deallocatorService();
        }

        private synchronized void retainDataBuffers(
                DataBuffer shapeInfo, DataBuffer data, DataBuffer special) {
            if (deallocated) {
                throw new IllegalStateException("Cannot retain buffers for a deallocated OpaqueNDArray");
            }
            // Retain the DataBuffer referents themselves. Holding only their
            // OpaqueDataBuffer facades does not stop BaseDataBuffer phantom cleanup.
            shapeInfoBufferRef = shapeInfo;
            dataBufferRef = data;
            specialBufferRef = special;
        }

        @Override
        public void deallocate() {
            if (constant || deallocated || DeallocatorService.getShutdownInProgress().get()) {
                return;
            }

            synchronized (this) {
                if (constant || deallocated) {
                    return;
                }

                try {
                    if (array != null && !array.isNull()) {
                        int deviceCount = owner.deviceCount();
                        if (targetDevice < 0 || targetDevice >= deviceCount) {
                            throw new IllegalStateException(
                                    "Invalid allocation device " + targetDevice
                                            + " for owning backend with " + deviceCount + " devices");
                        }

                        int currentDevice = owner.currentDevice();
                        boolean switchedDevice = currentDevice != targetDevice;
                        if (switchedDevice) {
                            owner.setDevice(targetDevice);
                        }

                        try {
                            owner.commit();
                            owner.nativeOps().deleteNDArray(array);
                            array.setNull();
                        } finally {
                            if (switchedDevice) {
                                owner.setDevice(currentDevice);
                            }
                        }
                    }
                } catch (Exception e) {
                    log.error("Error deallocating OpaqueNDArray with uniqueId: " + uniqueId, e);
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
            if (constant) {
                service.getReferenceMap().remove(uniqueId);
            }
        }

        private boolean isDeallocated() {
            return deallocated;
        }

        private OpaqueNDArray getArray() {
            return array;
        }

        private void markDeallocated() {
            if (array != null) {
                array.setNull();
                ((DetachedCleanupArray) array).releasePointerStorage();
            }
            array = null;
            shapeInfoBufferRef = null;
            dataBufferRef = null;
            specialBufferRef = null;
            deallocated = true;
            service.getReferenceMap().remove(uniqueId);
        }
    }
}
