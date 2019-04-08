package org.nd4j.jita.allocator.impl;

import java.util.*;

public class MemoryTracker {

    private List<AtomicLong> allocatedPerDevice = new ArrayList<>();
    private List<AtomicLong> cachedPerDevice = new ArrayList<>();
    private List<AtomicLong> totalPerDevice = new ArrayList<>();
    private static MemoryTracker INSTANCE = new MemoryTracker();

    private MemoryTracker() {
        for (int i = 0; i < Nd4j.getAffinityManager().getNumberOfDevices(); ++i) {
            allocatedPerDevice.get(i).set(0);
            cachedPerDevice.get(i).set(0);
            totalPerDevice.get(i).set(0);
        }
    }

    public static MemoryTracker getInstance() {
        return INSTANCE;
    }

    public AtomicLong getAllocated(int deviceId) {
        return allocatedPerDevice.get(deviceId);
    }

    public AtomicLong getCached(int deviceId) {
        return cachedPerDevice.get(deviceId);
    }

    public AtomicLong getTotal(int deviceId) {
        return cachedPerDevice.get(deviceId);
    }

    public synchronized void incrementAllocated(int deviceId) {
        allocatedPerDevice.get(deviceId).getAndIncrement();
        totalPerDevice.get(deviceId).getAndIncrement();
    }

    public synchronized void incrementCached(int deviceId) {
        cachedPerDevice.get(deviceId).getAndIncrement();
        totalPerDevice.get(deviceId).getAndIncrement();
    }
}
