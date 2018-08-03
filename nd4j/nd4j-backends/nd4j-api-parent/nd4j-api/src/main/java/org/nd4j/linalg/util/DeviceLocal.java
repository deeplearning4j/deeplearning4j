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

import org.nd4j.linalg.factory.Nd4j;

import edu.umd.cs.findbugs.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Class similar to Java ThreadLocal class, but locality is preserved with respect to device used
 *
 * @author raver119@gmail.com
 */
public class DeviceLocal<T extends Object> {
    private Map<Integer, T> backingMap = new ConcurrentHashMap<>();
    private List<ReentrantReadWriteLock> locksMap = new ArrayList<>();

    public DeviceLocal() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int i = 0; i < numDevices; i++) {
            locksMap.add(new ReentrantReadWriteLock());
        }
    }

    /**
     * This method returns object local to current deviceId
     *
     * @return
     */
    @Nullable
    public T get() {
        return get(Nd4j.getAffinityManager().getDeviceForCurrentThread());
    }

    /**
     * This method returns object local to target device
     *
     * @param deviceId
     * @return
     */
    @Nullable
    public T get(int deviceId) {
        try {
            locksMap.get(deviceId).readLock().lock();
            return backingMap.get(deviceId);
        } finally {
            locksMap.get(deviceId).readLock().unlock();
        }
    }

    /**
     * This method sets object for specific device
     *
     * @param deviceId
     * @param object
     */
    public void set(int deviceId, T object) {
        try {
            locksMap.get(deviceId).writeLock().lock();
            backingMap.put(deviceId, object);
        } finally {
            locksMap.get(deviceId).writeLock().unlock();
        }
    }

    /**
     * This method sets object for current device
     *
     * @param object
     */
    public void set(T object) {
        set(Nd4j.getAffinityManager().getDeviceForCurrentThread(), object);
    }


    /**
     * This method removes object stored for current device
     *
     */
    public void clear() {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        try {
            locksMap.get(deviceId).writeLock().lock();
            backingMap.remove(deviceId);
        } finally {
            locksMap.get(deviceId).writeLock().unlock();
        }
    }
}
