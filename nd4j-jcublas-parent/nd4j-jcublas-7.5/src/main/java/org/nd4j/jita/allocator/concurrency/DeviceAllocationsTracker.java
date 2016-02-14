package org.nd4j.jita.allocator.concurrency;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import lombok.NonNull;
import org.nd4j.jita.conf.CudaEnvironment;

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
    private CudaEnvironment environment;

    private final ReentrantReadWriteLock globalLock = new ReentrantReadWriteLock();

    private final Map<Integer, ReentrantReadWriteLock> deviceLocks = new ConcurrentHashMap<>();

    private final Table<Integer, Long, AtomicLong> allocationTable = HashBasedTable.create();

    public DeviceAllocationsTracker(@NonNull CudaEnvironment environment) {
        this.environment = environment;

        for (Integer device: environment.getAvailableDevices().keySet()) {
            deviceLocks.put(device, new ReentrantReadWriteLock());
        }
    }

    protected void ensureThreadRegistered(Long threadId, Integer deviceId) {
        globalLock.readLock().lock();

        boolean contains = allocationTable.contains(deviceId, threadId);

        globalLock.readLock().unlock();

        if (!contains) {
            globalLock.writeLock().lock();

            contains = allocationTable.contains(deviceId, threadId);
            if (!contains) {
                allocationTable.put(deviceId, threadId, new AtomicLong(0));
            }
                  globalLock.writeLock().unlock();
        }
    }

    public long addToAllocation(Long threadId, Integer deviceId, long memorySize) {
        ensureThreadRegistered(threadId, deviceId);
        try {
            deviceLocks.get(deviceId).readLock().lock();

            return allocationTable.get(deviceId, threadId).addAndGet(memorySize);
        } finally {
            deviceLocks.get(deviceId).readLock().unlock();
        }
    }

    public long subFromAllocation(Long threadId, Integer deviceId, long memorySize) {
        ensureThreadRegistered(threadId, deviceId);
        try {
            deviceLocks.get(deviceId).writeLock().lock();

            AtomicLong val = allocationTable.get(deviceId, threadId);

            val.set(val.get() - memorySize);

            return val.get();
        } finally {
            deviceLocks.get(deviceId).writeLock().unlock();
        }
    }


    public boolean reserveAllocationIfPossible(Long threadId, Integer deviceId, Long memorySize) {
        ensureThreadRegistered(threadId, deviceId);
        return false;
    }

    public long getAllocatedSize(Long threadId, Integer deviceId) {
        ensureThreadRegistered(threadId, deviceId);

        try {
            deviceLocks.get(deviceId).readLock().lock();

            return allocationTable.get(deviceId, threadId).get();
        } finally {
            deviceLocks.get(deviceId).readLock().unlock();
        }
    }


    public long getAllocatedSize(Integer deviceId) {
        try {
            deviceLocks.get(deviceId).readLock().lock();

            Map<Long, AtomicLong> map =allocationTable.row(deviceId);
            long sum = 0;
            for (AtomicLong alloc: map.values()) {
                sum += alloc.get();
            }

            return sum;
        } finally {
            deviceLocks.get(deviceId).readLock().unlock();
        }
    }
}
