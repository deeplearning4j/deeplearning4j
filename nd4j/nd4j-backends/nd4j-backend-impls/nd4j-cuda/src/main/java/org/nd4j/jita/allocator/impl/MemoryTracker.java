package org.nd4j.jita.allocator.impl;

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

    public void incrementAllocated(int deviceId) {
        allocatedPerDevice.get(deviceId) += 1;
    }

    public void incrementCached(int deviceId) {
        cachedPerDevice.get(deviceId) += 1;
    }
}