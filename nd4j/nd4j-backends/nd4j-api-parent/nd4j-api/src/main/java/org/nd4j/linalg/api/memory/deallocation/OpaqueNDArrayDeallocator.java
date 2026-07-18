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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
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
 * <p>When an OpaqueNDArray is created, an instance of this deallocator
 * is registered with the DeallocatorService, which will call deallocate()
 * when the Java object becomes unreachable.</p>
 *
 * @author Adam Gibson
 * @see DeallocatorService
 * @see OpaqueNDArray
 */
@Slf4j
public class OpaqueNDArrayDeallocator implements Deallocatable, Deallocator {
    @Getter
    private OpaqueNDArray array;
    private final long uniqueId;
    private final int targetDevice;
    private final NativeBufferOwner owner;
    private volatile boolean deallocated = false;
    private volatile boolean constant = false;

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
        this.array = array;
        this.uniqueId = uniqueId;
        this.targetDevice = targetDevice;
        this.owner = owner;
    }

    private static NativeBufferOwner requireOwner(OpaqueNDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("OpaqueNDArray cannot be null");
        }
        return array.backendOwner();
    }

    @Override
    public void deallocate() {
        // Check constant flag first - constant arrays should never be freed
        // This mirrors the behavior in OpaqueDataBufferDeallocator
        if (constant) {
            return;
        }

        if (deallocated) {
            return;
        }

        // During JVM shutdown, skip native deallocation to avoid calling free()
        // on potentially corrupted heap metadata. The OS reclaims all process memory on exit.
        if (DeallocatorService.getShutdownInProgress().get()) {
            return;
        }

        synchronized (this) {
            if (constant || deallocated) {
                return;
            }

            try {
                if (array != null && !array.isNull()) {
                    if (log.isTraceEnabled()) {
                        log.trace("Deallocating OpaqueNDArray with uniqueId: {}", uniqueId);
                    }

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
                array = null;
                deallocated = true;
                // Remove from referenceMap — mirrors BaseDataBuffer.release() (line 2250)
                // and prevents unbounded refMap growth across model close/reimport cycles.
                // Without this, each model close leaves ~500+ stale OpaqueNDArray entries
                // that persist until GC enqueues the corresponding PhantomReference.
                try {
                    owner.deallocatorService().getReferenceMap().remove(uniqueId);
                } catch (Exception ignored) {
                    // DeallocatorService may be shut down
                }
            }
        }
    }

    @Override
    public long getUniqueId() {
        return uniqueId;
    }

    @Override
    public Deallocator deallocator() {
        return this;
    }

    @Override
    public int targetDevice() {
        return targetDevice;
    }

    @Override
    public boolean isConstant() {
        return constant;
    }

    @Override
    public void setConstant(boolean constant) {
        this.constant = constant;
        // Mirror BaseDataBuffer.setConstant() behavior: when marked constant,
        // remove from DeallocatorService.referenceMap since the deallocator will
        // never fire (deallocate() returns early when constant=true). Without this,
        // OpaqueNDArrays for constant model weights permanently occupy refMap slots.
        if (constant) {
            owner.deallocatorService().getReferenceMap().remove(uniqueId);
        }
    }

    /**
     * Returns whether this deallocator has already been invoked.
     *
     * @return true if deallocate() has been called
     */
    public boolean isDeallocated() {
        return deallocated;
    }

}
