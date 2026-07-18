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
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.nativeblas.NativeBufferOwner;
import org.nd4j.nativeblas.OpaqueNDArray;

/**
 * Phantom-registration object for an owned array of native NDArray wrappers.
 *
 * <p>The registration object and the native cleanup state are deliberately
 * separate. {@link DeallocatableReference} strongly retains the
 * {@link Deallocator} returned by {@link #deallocator()}, so that object must
 * never retain this phantom referent or the {@code OpaqueNDArrayArr} facade
 * that keeps it alive. The detached {@link ResourceState} owns only native
 * resources and the parent arrays required to keep their buffers valid.</p>
 */
@Slf4j
public class OpaqueNDArrayArrDeallocator implements Deallocatable {
    private final long uniqueId;
    private final int targetDevice;
    private final ArrayDeallocator innerDeallocator;

    /**
     * Creates a cleanup registration bound to an exact backend owner and device.
     */
    public OpaqueNDArrayArrDeallocator(ResourceState resources, long uniqueId,
                                       int targetDevice, NativeBufferOwner owner) {
        if (resources == null) {
            throw new IllegalArgumentException("ResourceState cannot be null");
        }
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        int deviceCount = owner.deviceCount();
        if (deviceCount < 1 || targetDevice < 0 || targetDevice >= deviceCount) {
            throw new IllegalArgumentException(
                    "Invalid device " + targetDevice + " for NativeBufferOwner with "
                            + deviceCount + " devices");
        }

        this.uniqueId = uniqueId;
        this.targetDevice = targetDevice;
        this.innerDeallocator =
                new ArrayDeallocator(resources, uniqueId, targetDevice, owner);
    }

    /**
     * Explicitly executes the same cleanup state used by phantom deallocation.
     */
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

    public boolean isConstant() {
        return innerDeallocator.isConstant();
    }

    public void setConstant(boolean constant) {
        innerDeallocator.setConstant(constant);
    }

    public boolean isDeallocated() {
        return innerDeallocator.isDeallocated();
    }

    public INDArray[] getParentArrays() {
        return innerDeallocator.getParentArrays();
    }

    /**
     * Native resources detached from the Java facade observed by the phantom
     * reference. This class must never acquire a reference back to that facade
     * or to {@link OpaqueNDArrayArrDeallocator}.
     */
    public static final class ResourceState {
        private INDArray[] parentArrays;
        private OpaqueNDArray[] opaqueArrays;
        private boolean[] opaqueReferencesReleased;
        private LongPointer pointerValues;
        private PointerPointer<OpaqueNDArray> pointerStorage;
        private final boolean ownsOpaqueArrays;
        private boolean deallocated;

        public ResourceState(INDArray[] parentArrays,
                             OpaqueNDArray[] opaqueArrays,
                             boolean[] opaqueReferencesReleased,
                             LongPointer pointerValues,
                             PointerPointer<OpaqueNDArray> pointerStorage,
                             boolean ownsOpaqueArrays) {
            if (opaqueArrays == null || opaqueArrays.length == 0) {
                throw new IllegalArgumentException(
                        "OpaqueNDArray resources cannot be null or empty");
            }
            if (opaqueReferencesReleased == null
                    || opaqueReferencesReleased.length != opaqueArrays.length) {
                throw new IllegalArgumentException(
                        "OpaqueNDArray release state must match the wrapper count");
            }
            if (pointerValues == null || pointerStorage == null) {
                throw new IllegalArgumentException(
                        "Detached pointer storage cannot be null");
            }
            this.parentArrays = parentArrays;
            this.opaqueArrays = opaqueArrays;
            this.opaqueReferencesReleased = opaqueReferencesReleased;
            this.pointerValues = pointerValues;
            this.pointerStorage = pointerStorage;
            this.ownsOpaqueArrays = ownsOpaqueArrays;
        }

        public synchronized INDArray[] getParentArrays() {
            return parentArrays;
        }

        public synchronized OpaqueNDArray[] getOpaqueArrays() {
            return opaqueArrays;
        }

        public synchronized boolean isDeallocated() {
            return deallocated;
        }

        /**
         * Releases each resource exactly once. Successfully released resources
         * are cleared immediately so a later invocation resumes only the
         * incomplete portion after a transient failure.
         */
        public synchronized void deallocateResources() {
            if (deallocated) {
                return;
            }

            RuntimeException failure = null;
            if (ownsOpaqueArrays && opaqueArrays != null) {
                for (int i = 0; i < opaqueArrays.length; i++) {
                    OpaqueNDArray opaque = opaqueArrays[i];
                    if (opaque == null) {
                        continue;
                    }

                    if (!opaqueReferencesReleased[i]) {
                        try {
                            opaque.releaseReference();
                            opaqueReferencesReleased[i] = true;
                        } catch (RuntimeException e) {
                            failure = appendFailure(failure, e);
                            continue;
                        }
                    }

                    try {
                        opaque.close();
                        opaqueArrays[i] = null;
                    } catch (RuntimeException e) {
                        failure = appendFailure(failure, e);
                    }
                }
            }

            if (pointerStorage != null) {
                try {
                    pointerStorage.close();
                    pointerStorage = null;
                } catch (RuntimeException e) {
                    failure = appendFailure(failure, e);
                }
            }

            if (pointerValues != null) {
                try {
                    pointerValues.close();
                    pointerValues = null;
                } catch (RuntimeException e) {
                    failure = appendFailure(failure, e);
                }
            }

            if (failure != null) {
                throw failure;
            }

            opaqueArrays = null;
            opaqueReferencesReleased = null;
            parentArrays = null;
            deallocated = true;
        }

        private static RuntimeException appendFailure(
                RuntimeException failure, RuntimeException next) {
            if (failure == null) {
                return next;
            }
            failure.addSuppressed(next);
            return failure;
        }
    }

    /**
     * Cleanup action retained by {@link DeallocatableReference}. It owns the
     * detached resources but has no path back to the phantom referent.
     */
    @Slf4j
    private static final class ArrayDeallocator implements Deallocator {
        private final ResourceState resources;
        private final long uniqueId;
        private final int targetDevice;
        private final NativeBufferOwner owner;
        private final DeallocatorService service;
        private volatile boolean deallocated;
        private volatile boolean constant;

        private ArrayDeallocator(ResourceState resources, long uniqueId,
                                 int targetDevice, NativeBufferOwner owner) {
            this.resources = resources;
            this.uniqueId = uniqueId;
            this.targetDevice = targetDevice;
            this.owner = owner;
            this.service = owner.deallocatorService();
        }

        @Override
        public void deallocate() {
            if (constant || deallocated) {
                return;
            }

            synchronized (this) {
                if (constant || deallocated) {
                    return;
                }

                int deviceCount = owner.deviceCount();
                if (targetDevice < 0 || targetDevice >= deviceCount) {
                    throw new IllegalStateException(
                            "Invalid cleanup device " + targetDevice
                                    + " for NativeBufferOwner with "
                                    + deviceCount + " devices");
                }

                if (log.isTraceEnabled()) {
                    INDArray[] parents = resources.getParentArrays();
                    log.trace(
                            "Deallocating OpaqueNDArrayArr resources with uniqueId: {} "
                                    + "(parent count: {})",
                            uniqueId, parents != null ? parents.length : 0);
                }

                int currentDevice = owner.currentDevice();
                boolean switchedDevice = currentDevice != targetDevice;
                RuntimeException failure = null;
                boolean cleanupComplete = false;
                try {
                    if (switchedDevice) {
                        owner.setDevice(targetDevice);
                    }
                    owner.commit();
                    resources.deallocateResources();
                    cleanupComplete = true;
                } catch (RuntimeException e) {
                    failure = e;
                } finally {
                    if (switchedDevice) {
                        try {
                            owner.setDevice(currentDevice);
                        } catch (RuntimeException restoreFailure) {
                            if (failure == null) {
                                failure = restoreFailure;
                            } else {
                                failure.addSuppressed(restoreFailure);
                            }
                        }
                    }
                }

                if (cleanupComplete) {
                    deallocated = true;
                    service.getReferenceMap().remove(uniqueId);
                }
                if (failure != null) {
                    throw failure;
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

        private boolean isDeallocated() {
            return deallocated;
        }

        private INDArray[] getParentArrays() {
            return resources.getParentArrays();
        }
    }
}
