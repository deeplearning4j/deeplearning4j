package org.nd4j.jita.allocator.impl;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import org.nd4j.linalg.factory.Nd4j;

public class MemoryTracker {

    private List<AtomicLong> allocatedPerDevice = new ArrayList<>();
    private List<AtomicLong> cachedPerDevice = new ArrayList<>();
    private List<AtomicLong> totalPerDevice = new ArrayList<>();
    private List<AtomicLong> workspacesPerDevice = new ArrayList<>();
    private final static MemoryTracker INSTANCE = new MemoryTracker();

    private MemoryTracker() {
        for (int i = 0; i < Nd4j.getAffinityManager().getNumberOfDevices(); ++i) {
            allocatedPerDevice.add(i, new AtomicLong(0));
            cachedPerDevice.add(i, new AtomicLong(0));
	    workspacesPerDevice.add(i, new AtomicLong(0));
            totalPerDevice.add(i, new AtomicLong(0));
        }
    }

    public static MemoryTracker getInstance() {
        return INSTANCE;
    }

    public long getAllocatedAmount(int deviceId) {
        return allocatedPerDevice.get(deviceId).get();
    }

    public long getCachedAmount(int deviceId) {
        return cachedPerDevice.get(deviceId).get();
    }

    public long getWorkspaceAllocatedAmount(int deviceId) {
        return workspacesPerDevice.get(deviceId).get();
    }

    public long getTotalMemory(int deviceId) {
        return totalPerDevice.get(deviceId).get();
    }

    public void incrementAllocatedAmount(int deviceId, long memoryAdded) {
        allocatedPerDevice.get(deviceId).getAndAdd(memoryAdded);
    }

    public void incrementCachedAmount(int deviceId, long memoryAdded) {
        cachedPerDevice.get(deviceId).getAndAdd(memoryAdded);
    }

    public void decrementAllocatedAmount(int deviceId, long memoryAdded) {
        allocatedPerDevice.get(deviceId).getAndAdd(-memoryAdded);
    }

    public void decrementCachedAmount(int deviceId, long memorySubtracted) {
        cachedPerDevice.get(deviceId).getAndAdd(-memorySubtracted);
    }

    public void incrementWorkspaceAllocatedAmount(int deviceId, long memoryAdded) {
        workspacesPerDevice.get(deviceId).getAndAdd(memoryAdded);
    }

    public void decrementWorkspaceAmount(int deviceId, long memorySubtracted) {
        workspacesPerDevice.get(deviceId).getAndAdd(-memorySubtracted);
    }

    private void setTotalPerDevice(int device, long memoryAvailable) {
        totalPerDevice.add(device, new AtomicLong(memoryAvailable));
    }
}
