/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.util;

import edu.umd.cs.findbugs.annotations.Nullable;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;
import org.nd4j.linalg.profiler.ProfilerConfig;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * DeviceLocal implementation for INDArray, with special broadcast method
 * @author raver119@gmail.com
 */
@Slf4j
public class DeviceLocalNDArray extends DeviceLocal<INDArray> {

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
     * This method returns object local to current deviceId
     *
     * @return
     */
    @Nullable
    @Override
    public synchronized INDArray get() {
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        val numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        val sourceId = updatesMap.get(deviceId).get();
        if (sourceId >= 0 && sourceId != deviceId) {
            // if updates map contains some deviceId - we should take updated array from there
            val newArray = Nd4j.create(delayedArray.dataType(), delayedArray.shape(), delayedArray.stride(), delayedArray.ordering());
            Nd4j.getMemoryManager().memcpy(newArray.data(), delayedArray.data());
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
        return get(deviceId);
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

        val config = OpProfiler.getInstance().getConfig();
        val locality = config.isCheckLocality();

        if (locality)
            config.setCheckLocality(false);
        val numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        if (!delayedMode) {
            // in immediate mode we put data in

            for (int i = 0; i < numDevices; i++) {
                // if current thread equal to this device - we just save it, without duplication
                if (deviceId == i) {
                    set(i, array.detach());
                } else {
                    set(i, Nd4j.getAffinityManager().replicateToDevice(i, array));
                }

            }
        } else {
            // we're only updating this device
            set(Nd4j.getAffinityManager().getDeviceForCurrentThread(), array);
            delayedArray = array.dup(array.ordering()).detach();

            // and marking all other devices as stale, and provide id of device with the most recent array
            for (int i = 0; i < numDevices; i++) {
                if (i != deviceId) {
                    updatesMap.get(i).set(deviceId);
                }
            }
        }

        config.setCheckLocality(locality);
    }

    /**
     * This method updates
     *
     * PLEASE NOTE: this method is NOT atomic, so you must be sure no other threads are using this instance during the update
     * @param array
     */
    public synchronized void update(@NonNull INDArray array) {
        Preconditions.checkArgument(!array.isView() || array.elementWiseStride() != 1, "View can't be used in DeviceLocalNDArray");

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
                            delayedArray = array.dup(array.ordering()).detach();
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
