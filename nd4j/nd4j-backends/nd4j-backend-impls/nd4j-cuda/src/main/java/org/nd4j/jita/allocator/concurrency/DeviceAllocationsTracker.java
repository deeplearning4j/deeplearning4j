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

package org.nd4j.jita.allocator.concurrency;

import lombok.NonNull;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 *
 *
 * @author raver119@gmail.com
 */
public class DeviceAllocationsTracker {
    private Configuration configuration;

    private final ReentrantReadWriteLock globalLock = new ReentrantReadWriteLock();

    private final Map<Integer, ReentrantReadWriteLock> deviceLocks = new ConcurrentHashMap<>();

    private final Map<Integer, AtomicLong> memoryTackled = new ConcurrentHashMap<>();

    private final Map<Integer, AtomicLong> reservedSpace = new ConcurrentHashMap<>();

    private static Logger log = LoggerFactory.getLogger(DeviceAllocationsTracker.class);

    public DeviceAllocationsTracker(@NonNull Configuration configuration) {
        this.configuration = configuration;

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        for (int device = 0; device < numDevices; device++) {
            deviceLocks.put(device, new ReentrantReadWriteLock());
        }
    }

    protected void ensureThreadRegistered(Long threadId, Integer deviceId) {
        globalLock.readLock().lock();

        //  boolean contains = allocationTable.contains(deviceId, threadId);

        globalLock.readLock().unlock();

        if (!memoryTackled.containsKey(deviceId)) {
            globalLock.writeLock().lock();

            //contains = allocationTable.contains(deviceId, threadId);
            //if (!contains) {
            //allocationTable.put(deviceId, threadId, new AtomicLong(0));

            if (!memoryTackled.containsKey(deviceId)) {
                memoryTackled.put(deviceId, new AtomicLong(0));
            }

            if (!reservedSpace.containsKey(deviceId)) {
                reservedSpace.put(deviceId, new AtomicLong(0));
            }
            //}
            globalLock.writeLock().unlock();
        }
    }

    public long addToAllocation(@NonNull Long threadId, Integer deviceId, long memorySize) {
        ensureThreadRegistered(threadId, deviceId);
        try {
            deviceLocks.get(deviceId).readLock().lock();

            long res = memoryTackled.get(deviceId).addAndGet(memorySize);

            subFromReservedSpace(deviceId, memorySize);

            return res; //allocationTable.get(deviceId, threadId).addAndGet(memorySize);
        } finally {
            deviceLocks.get(deviceId).readLock().unlock();
        }
    }

    public long subFromAllocation(Long threadId, Integer deviceId, long memorySize) {
        ensureThreadRegistered(threadId, deviceId);

        try {
            deviceLocks.get(deviceId).writeLock().lock();

            AtomicLong val2 = memoryTackled.get(deviceId);
            //long before = val2.get();
            val2.addAndGet(memorySize * -1);

            //long after = memoryTackled.get(deviceId).get();

            //log.info("Memory reduction on device [{}], memory size: [{}], before: [{}], after [{}]", deviceId, memorySize, before, after);

            //            AtomicLong val = allocationTable.get(deviceId, threadId);

            //            val.addAndGet(memorySize * -1);

            return val2.get();
        } finally {
            deviceLocks.get(deviceId).writeLock().unlock();
        }
    }

    /**
     * This method "reserves" memory within allocator
     *
     * @param threadId
     * @param deviceId
     * @param memorySize
     * @return
     */
    public boolean reserveAllocationIfPossible(Long threadId, Integer deviceId, long memorySize) {
        ensureThreadRegistered(threadId, deviceId);
        try {
            deviceLocks.get(deviceId).writeLock().lock();
            /*
            if (getAllocatedSize(deviceId) + memorySize + getReservedSpace(deviceId)> environment.getDeviceInformation(deviceId).getTotalMemory() * configuration.getMaxDeviceMemoryUsed()) {
                return false;
            } else {
                addToReservedSpace(deviceId, memorySize);
                return true;
            }
            */
            addToReservedSpace(deviceId, memorySize);
            return true;
        } finally {
            deviceLocks.get(deviceId).writeLock().unlock();
        }
    }

    public long getAllocatedSize(Long threadId, Integer deviceId) {
        ensureThreadRegistered(threadId, deviceId);

        try {
            deviceLocks.get(deviceId).readLock().lock();

            return getAllocatedSize(deviceId); /// allocationTable.get(deviceId, threadId).get();
        } finally {
            deviceLocks.get(deviceId).readLock().unlock();
        }
    }


    public long getAllocatedSize(Integer deviceId) {
        if (!memoryTackled.containsKey(deviceId))
            return 0L;
        try {
            deviceLocks.get(deviceId).readLock().lock();
            return memoryTackled.get(deviceId).get();
        } finally {
            deviceLocks.get(deviceId).readLock().unlock();
        }
    }

    public long getReservedSpace(Integer deviceId) {
        return reservedSpace.get(deviceId).get();
    }

    protected void addToReservedSpace(Integer deviceId, long memorySize) {
        ensureThreadRegistered(Thread.currentThread().getId(), deviceId);

        reservedSpace.get(deviceId).addAndGet(memorySize);
    }

    protected void subFromReservedSpace(Integer deviceId, long memorySize) {
        ensureThreadRegistered(Thread.currentThread().getId(), deviceId);

        reservedSpace.get(deviceId).addAndGet(memorySize * -1);
    }
}
