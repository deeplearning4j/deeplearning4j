package org.nd4j.jita.allocator.impl;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import org.nd4j.linalg.factory.Nd4j;

public class MemoryTracker {

    private List<AtomicLong> allocatedPerDevice = new ArrayList<>();
    private List<AtomicLong> cachedPerDevice = new ArrayList<>();
    private List<AtomicLong> totalPerDevice = new ArrayList<>();
    private static MemoryTracker INSTANCE = new MemoryTracker();

    private MemoryTracker() {
        for (int i = 0; i < Nd4j.getAffinityManager().getNumberOfDevices(); ++i) {
            allocatedPerDevice.add(i, new AtomicLong(0));
            cachedPerDevice.add(i, new AtomicLong(0));
            totalPerDevice.add(i, new AtomicLong(0));
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
        return totalPerDevice.get(deviceId);
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
