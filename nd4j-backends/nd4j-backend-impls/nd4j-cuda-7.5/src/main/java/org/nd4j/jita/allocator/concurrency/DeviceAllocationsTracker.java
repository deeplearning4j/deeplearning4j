package org.nd4j.jita.allocator.concurrency;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import lombok.NonNull;
import org.nd4j.jita.conf.Configuration;
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
    private Configuration configuration;

    private final ReentrantReadWriteLock globalLock = new ReentrantReadWriteLock();

    private final Map<Integer, ReentrantReadWriteLock> deviceLocks = new ConcurrentHashMap<>();

    private final Table<Integer, Long, AtomicLong> allocationTable = HashBasedTable.create();

    private final Map<Integer, AtomicLong> memoryTackled = new ConcurrentHashMap<>();

    private final Map<Integer, AtomicLong> reservedSpace = new ConcurrentHashMap<>();

    public DeviceAllocationsTracker(@NonNull CudaEnvironment environment, @NonNull Configuration configuration) {
        this.environment = environment;
        this.configuration = configuration;

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

                if (!memoryTackled.containsKey(deviceId)) {
                    memoryTackled.put(deviceId, new AtomicLong(0));
                }

                if (!reservedSpace.containsKey(deviceId)) {
                    reservedSpace.put(deviceId, new AtomicLong(0));
                }
            }
                  globalLock.writeLock().unlock();
        }
    }

    public long addToAllocation(Long threadId, Integer deviceId, long memorySize) {
        ensureThreadRegistered(threadId, deviceId);
        try {
            deviceLocks.get(deviceId).readLock().lock();

            memoryTackled.get(deviceId).addAndGet(memorySize);

            subFromReservedSpace(deviceId, memorySize);

            return allocationTable.get(deviceId, threadId).addAndGet(memorySize);
        } finally {
            deviceLocks.get(deviceId).readLock().unlock();
        }
    }

    public long subFromAllocation(Long threadId, Integer deviceId, long memorySize) {
        ensureThreadRegistered(threadId, deviceId);

        AtomicLong val2 = memoryTackled.get(deviceId);
        val2.addAndGet(memorySize * -1);

        try {
            deviceLocks.get(deviceId).writeLock().lock();

            AtomicLong val = allocationTable.get(deviceId, threadId);

            val.addAndGet(memorySize * -1);

            return val.get();
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

            if (getAllocatedSize(deviceId) + memorySize + getReservedSpace(deviceId)> environment.getDeviceInformation(deviceId).getTotalMemory() * configuration.getMaxDeviceMemoryUsed()) {
                return false;
            } else {
                addToReservedSpace(deviceId, memorySize);
                return true;
            }
        } finally {
            deviceLocks.get(deviceId).writeLock().unlock();
        }
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
        if (!memoryTackled.containsKey(deviceId))
            return 0L;
        try {
            deviceLocks.get(deviceId).readLock().lock();
            return memoryTackled.get(deviceId).get();
        } finally {
            deviceLocks.get(deviceId).readLock().unlock();
        }
    }

    protected long getReservedSpace(Integer deviceId) {
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
