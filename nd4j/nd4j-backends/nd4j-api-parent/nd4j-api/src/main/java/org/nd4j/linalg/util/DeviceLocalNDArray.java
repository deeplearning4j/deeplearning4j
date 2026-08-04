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

package org.nd4j.linalg.util;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

@Slf4j
public class DeviceLocalNDArray extends DeviceLocal<INDArray> {

    // Track if the source array was marked as constant
    private volatile boolean sourceWasConstant = false;

    public DeviceLocalNDArray() {
        this(false);
    }

    public DeviceLocalNDArray(boolean delayedMode) {
        super(delayedMode);
    }

    public DeviceLocalNDArray(INDArray array) {
        this(array, false);
    }

    public DeviceLocalNDArray(INDArray array, boolean delayedMode) {
        super(delayedMode);

        broadcast(array);
    }

    /**
     * Check if the source array's data buffer is marked as constant.
     * This is used to propagate constant flag to device-local copies.
     */
    private boolean isSourceConstant(INDArray array) {
        if (array == null) return false;
        DataBuffer data = array.data();
        if (data != null && data.isConstant()) {
            return true;
        }
        return false;
    }

    /**
     * Propagate constant flag from source to target array.
     * This ensures that device-local copies of constant arrays remain protected from deallocation.
     *
     * This method uses the already-captured sourceWasConstant field instead of
     * re-checking source.data().isConstant(). This prevents a race condition where the source
     * buffer might be in an inconsistent state (GC'd, deallocated) by the time we check.
     * The sourceWasConstant field is set at the START of broadcast() before any dup/detach
     * operations that might affect the source buffer.
     */
    private void propagateConstantFlag(INDArray source, INDArray target) {
        if (target == null) return;

        // Use the already-captured sourceWasConstant field instead of re-checking source
        // This prevents race conditions where source buffer may have been freed
        if (sourceWasConstant) {
            DataBuffer targetData = target.data();
            if (targetData != null) {
                targetData.setConstant(true);
            }
            DataBuffer targetShapeInfo = target.shapeInfoDataBuffer();
            if (targetShapeInfo != null) {
                targetShapeInfo.setConstant(true);
            }
            target.setCloseable(false);
        }
    }

    /**
     * Apply the tracked constant flag to a target array.
     * Used when creating new arrays from delayedArray.
     */
    private void applyConstantFlag(INDArray target) {
        if (!sourceWasConstant || target == null) return;

        DataBuffer targetData = target.data();
        if (targetData != null) {
            targetData.setConstant(true);
        }
        DataBuffer targetShapeInfo = target.shapeInfoDataBuffer();
        if (targetShapeInfo != null) {
            targetShapeInfo.setConstant(true);
        }
        target.setCloseable(false);
    }

    /**
     * This method returns object local to current deviceId
     *
     * @return
     */
    @Override
    public synchronized INDArray get() {
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        if (deviceId >= locksMap.size()) {
            ensureCapacity(deviceId);
        }
        val numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        val sourceId = updatesMap.get(deviceId).get();
        if (sourceId >= 0 && sourceId != deviceId) {
            if (delayedArray == null || delayedArray.data() == null) {
                log.warn("DeviceLocalNDArray.get(): delayedArray is null or has null data for device {}. " +
                        "Clearing stale update flag. This may indicate the array was empty or a race condition occurred.",
                        deviceId);
                updatesMap.get(deviceId).set(deviceId);
                return get(deviceId);
            }

            // Use the backend affinity primitive for the whole cross-device transfer.
            // It owns source synchronization, target allocation, device switching, and
            // copy actuality. Reimplementing those steps here can publish a zero-filled
            // target when the delayed source and current thread belong to different GPUs.
            INDArray newArray;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                newArray = Nd4j.getAffinityManager().replicateToDevice(deviceId, delayedArray);
            }

            // A replica must not become immutable until its contents are initialized.
            // Marking the freshly allocated destination constant before memcpy causes
            // CUDA constant replicas to retain their zero-filled allocation instead of
            // the source values. The local strong reference protects the array while
            // the copy completes; register it only after initialization, then publish it.
            if (sourceWasConstant) {
                newArray = Nd4j.getDeallocatorService().registerPendingConstant(newArray);
            }
            applyConstantFlag(newArray);
            if (sourceWasConstant) {
                Nd4j.getDeallocatorService().releasePendingConstant(newArray);
            }

            backingMap.put(deviceId, newArray);

            // reset updates flag
            updatesMap.get(deviceId).set(deviceId);


            // also check if all updates were consumed
            boolean allUpdated = true;
            for (int e = 0; e < numDevices; e++) {
                if (updatesMap.get(e).get() != e) {
                    allUpdated = false;
                    break;
                }
            }

            if (allUpdated)
                delayedArray = null;
        }
        INDArray result = get(deviceId);
        if (result == null) {
            // Fallback: check other devices for an empty array that wasn't replicated.
            // Empty arrays are device-agnostic (no data buffer), so any copy is valid.
            for (int i = 0; i < numDevices; i++) {
                if (i != deviceId) {
                    INDArray other = get(i);
                    if (other != null && other.isEmpty()) {
                        set(deviceId, other);
                        return other;
                    }
                }
            }
        }
        return result;
    }

    /**
     * This method duplicates array, and stores it to all devices
     *
     * PLEASE NOTE: this method is NOT atomic, so you must be sure no other threads are using this instance during the update
     * @param array
     */
    public synchronized void broadcast(INDArray array) {
        if (array == null)
            return;

        Preconditions.checkArgument(!array.isView() || array.elementWiseStride() != 1, "View can't be used in DeviceLocalNDArray");

        Nd4j.getExecutioner().commit();

        // Track if the source array is constant - this will be used to propagate
        // the constant flag to all device-local copies
        sourceWasConstant = isSourceConstant(array);

        val numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        if (!delayedMode) {
            // in immediate mode we put data in

            for (int i = 0; i < numDevices; i++) {
                // if current thread equal to this device - we just save it, without duplication
                if (deviceId == i) {
                    INDArray detached;
                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        detached = sourceWasConstant ?
                            Nd4j.getDeallocatorService().registerPendingConstant(array.detach()) :
                            array.detach();
                    }
                    // Propagate constant flag to detached array
                    propagateConstantFlag(array, detached);
                    if (sourceWasConstant) {
                        Nd4j.getDeallocatorService().releasePendingConstant(detached);
                    }
                    set(i, detached);
                } else {
                    INDArray replicated;
                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        replicated = sourceWasConstant ?
                            Nd4j.getDeallocatorService().registerPendingConstant(
                                Nd4j.getAffinityManager().replicateToDevice(i, array)) :
                            Nd4j.getAffinityManager().replicateToDevice(i, array);
                    }
                    // Propagate constant flag to replicated array
                    propagateConstantFlag(array, replicated);
                    if (sourceWasConstant) {
                        Nd4j.getDeallocatorService().releasePendingConstant(replicated);
                    }
                    set(i, replicated);
                }

            }
        } else {
            // we're only updating this device
            INDArray detachedForCurrentDevice;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                detachedForCurrentDevice = sourceWasConstant ?
                    Nd4j.getDeallocatorService().registerPendingConstant(array.detach()) :
                    array.detach();
            }
            propagateConstantFlag(array, detachedForCurrentDevice);
            if (sourceWasConstant) {
                Nd4j.getDeallocatorService().releasePendingConstant(detachedForCurrentDevice);
            }
            set(deviceId, detachedForCurrentDevice);
           if(!array.isEmpty() && array.data() != null && numDevices > 1) {
               // Bidirectional sync so HOST has current data for cross-device copies.
               // ensureHostAccess does H→D then force D→H, handling both device-only
               // data (compact copies) and host-only data (deserialized arrays).
               DeviceMemoryManager.getInstance().ensureHostAccess(detachedForCurrentDevice);
               delayedArray = detachedForCurrentDevice;

               for (int i = 0; i < numDevices; i++) {
                   if (i != deviceId) {
                       updatesMap.get(i).set(deviceId);
                   }
               }
           } else if (array.isEmpty() && numDevices > 1) {
               // Empty arrays have no data buffer — they are device-agnostic.
               // Store the same empty array for all devices so get() never returns null.
               for (int i = 0; i < numDevices; i++) {
                   if (i != deviceId) {
                       set(i, detachedForCurrentDevice);
                   }
               }
           }
        }

    }

    /**
     * This method updates
     *
     * PLEASE NOTE: this method is NOT atomic, so you must be sure no other threads are using this instance during the update
     * @param array
     */
    public synchronized void update(@NonNull INDArray array) {
        Preconditions.checkArgument(!array.isView() || array.elementWiseStride() != 1, "View can't be used in DeviceLocalNDArray");

        // Update the constant flag tracking
        if (isSourceConstant(array)) {
            sourceWasConstant = true;
        }

        val numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        val device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        val currentArray = backingMap.get(device);
        boolean wasDelayed = false;

        if (Arrays.equals(currentArray.shapeInfoJava(), array.shapeInfoJava())) {
            // if arrays are the same - we'll just issue memcpy
            for (int k = 0; k < numDevices; k++) {
                val lock = locksMap.get(k);
                try {
                    lock.writeLock().lock();
                    val v = backingMap.get(k);
                    if (v == null) {
                        if (!wasDelayed) {
                            boolean isConstant = isSourceConstant(array);
                            INDArray delayed;
                            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                delayed = isConstant ?
                                    Nd4j.getDeallocatorService().registerPendingConstant(array.dup(array.ordering()).detach()) :
                                    array.dup(array.ordering()).detach();
                            }
                            // Propagate constant flag to delayed array
                            propagateConstantFlag(array, delayed);
                            if (isConstant) {
                                Nd4j.getDeallocatorService().releasePendingConstant(delayed);
                            }
                            delayedArray = delayed;
                            wasDelayed = true;
                        }
                        updatesMap.get(k).set(device);
                        continue;
                    }

                    Nd4j.getMemoryManager().memcpy(v.data(), array.data());
                    Nd4j.getExecutioner().commit();
                } finally {
                    lock.writeLock().unlock();
                }
            }
        } else {
            // if arrays are not the same - we'll issue broadcast call
            broadcast(array);
        }
    }
}
