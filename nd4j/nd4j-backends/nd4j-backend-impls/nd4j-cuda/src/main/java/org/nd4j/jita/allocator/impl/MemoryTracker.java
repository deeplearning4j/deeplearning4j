package org.nd4j.jita.allocator.impl;

import java.util.*;

public class MemoryTracker {

    private List<Long> allocatedPerDevice = new ArrayList<>();
    private List<Long> cachedPerDevice = new ArrayList<>();
    private static MemoryTracker INSTANCE = new MemoryTracker();

    public static MemoryTracker getInstance() {
        return INSTANCE;
    }

    public Long getAllocated(int deviceId) {
        return allocatedPerDevice.get(deviceId);
    }

    public Long getCached(int deviceId) {
        return cachedPerDevice.get(deviceId);
    }

    public synchronized void incrementAllocated(int deviceId) {
        long cnt  = allocatedPerDevice.get(deviceId);
	allocatedPerDevice.set(deviceId, ++cnt);
    }

    public synchronized void incrementCached(int deviceId) {
        long cnt = cachedPerDevice.get(deviceId);
	cachedPerDevice.set(deviceId, ++cnt);
    }
}
