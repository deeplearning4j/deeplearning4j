package org.nd4j.jita.allocator.impl;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

@Slf4j
public class MemoryTracker {

    private List<AtomicLong> allocatedPerDevice = new ArrayList<>();
    private List<AtomicLong> cachedPerDevice = new ArrayList<>();
    private List<AtomicLong> totalPerDevice = new ArrayList<>();
    private List<AtomicLong> freePerDevice = new ArrayList<>();
    private List<AtomicLong> workspacesPerDevice = new ArrayList<>();
    private AtomicLong cachedHost = new AtomicLong(0);
    private AtomicLong allocatedHost = new AtomicLong(0);
    private final static MemoryTracker INSTANCE = new MemoryTracker();

    public MemoryTracker() {
        for (int i = 0; i < Nd4j.getAffinityManager().getNumberOfDevices(); ++i) {
            allocatedPerDevice.add(i, new AtomicLong(0));
            cachedPerDevice.add(i, new AtomicLong(0));
	        workspacesPerDevice.add(i, new AtomicLong(0));

	        val ptr = new CudaPointer(i);
            totalPerDevice.add(i, new AtomicLong(NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceTotalMemory(ptr)));

            val f = new AtomicLong(NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemory(ptr));

            log.debug("Free memory on device_{}: {}", i, f);
            freePerDevice.add(i, f);
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

    /**
     * This method returns number of bytes currently cached from host memory
     * @return
     */
    public long getCachedHostAmount() {
        return cachedHost.get();
    }

    /**
     * This method returns number of bytes currently allocated from host memory
     * @return
     */
    public long getAllocatedHostAmount() {
        return allocatedHost.get();
    }

    /**
     * This method returns number of bytes allocated and cached in host ram
     * @return
     */
    public long getActiveHostAmount() {
        return getAllocatedHostAmount() + getCachedHostAmount();
    }

    public void incrementCachedHostAmount(long numBytes) {
        cachedHost.addAndGet(numBytes);
    }

    public void incrementAllocatedHostAmount(long numBytes) {
        allocatedHost.addAndGet(numBytes);
    }

    public void decrementCachedHostAmount(long numBytes) {
        cachedHost.addAndGet(-numBytes);
    }

    public void decrementAllocatedHostAmount(long numBytes) {
        allocatedHost.addAndGet(-numBytes);
    }

    public long getWorkspaceAllocatedAmount(int deviceId) {
        return workspacesPerDevice.get(deviceId).get();
    }

    public long getTotalMemory(int deviceId) {
        return totalPerDevice.get(deviceId).get();
    }

    public long getFreeMemory(int deviceId) {
        return freePerDevice.get(deviceId).get();
    }

    /**
     * This method returns approximate free memory on specified device
     * @param deviceId
     * @return
     */
    public long getApproximateFreeMemory(int deviceId) {
        val externalAllocations = getTotalMemory(deviceId) - getFreeMemory(deviceId);
        val active = getActiveMemory(deviceId);
        val free = getTotalMemory(deviceId) - (active + externalAllocations);
        return free;
    }

    /**
     * This method returns precise amount of free memory on specified device
     * @param deviceId
     * @return
     */
    public long getPreciseFreeMemory(int deviceId) {
        // we refresh free memory on device
        val extFree = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemory(new CudaPointer(deviceId));
        //freePerDevice.get(deviceId).set(extFree);

        return extFree;
    }

    /**
     * This method returns delta between total memory and free memory
     * @param deviceId
     * @return
     */
    public long getUsableMemory(int deviceId) {
        return getTotalMemory(deviceId) - getFreeMemory(deviceId);
    }

    /**
     * This method returns total amount of device memory allocated on specified device
     *
     * Includes: workspace memory, cached memory, regular memory
     * @param deviceId
     * @return
     */
    public long getActiveMemory(int deviceId) {
        return getWorkspaceAllocatedAmount(deviceId) +  getAllocatedAmount(deviceId) + getCachedAmount(deviceId);
    }

    /**
     * This method returns amount of memory that relies on JVM GC
     *
     * Includes: cached memory, regular allocated memory
     *
     * @param deviceId
     * @return
     */
    public long getManagedMemory(int deviceId) {
        return getAllocatedAmount(deviceId) + getCachedAmount(deviceId);
    }

    /**
     * This method increments amount of regular allocated memory
     *
     * @param deviceId
     * @param memoryAdded
     */
    public void incrementAllocatedAmount(int deviceId, long memoryAdded) {
        allocatedPerDevice.get(deviceId).getAndAdd(matchBlock(memoryAdded));
    }

    /**
     * This method increments amount of cached memory
     *
     * @param deviceId
     * @param memoryAdded
     */
    public void incrementCachedAmount(int deviceId, long memoryAdded) {
        cachedPerDevice.get(deviceId).getAndAdd(matchBlock(memoryAdded));
    }

    /**
     * This method decrements amount of regular allocated memory
     *
     * @param deviceId
     * @param memorySubtracted
     */
    public void decrementAllocatedAmount(int deviceId, long memorySubtracted) {
        allocatedPerDevice.get(deviceId).getAndAdd(-matchBlock(memorySubtracted));
    }

    /**
     * This method decrements amount of cached memory
     *
     * @param deviceId
     * @param memorySubtracted
     */
    public void decrementCachedAmount(int deviceId, long memorySubtracted) {
        cachedPerDevice.get(deviceId).getAndAdd(-matchBlock(memorySubtracted));
    }

    /**
     * This method increments amount of memory allocated within workspaces
     *
     * @param deviceId
     * @param memoryAdded
     */
    public void incrementWorkspaceAllocatedAmount(int deviceId, long memoryAdded) {
        workspacesPerDevice.get(deviceId).getAndAdd(matchBlock(memoryAdded));
    }

    /**
     * This method decrements amount of memory allocated within workspaces
     *
     * @param deviceId
     * @param memorySubtracted
     */
    public void decrementWorkspaceAmount(int deviceId, long memorySubtracted) {
        workspacesPerDevice.get(deviceId).getAndAdd(-matchBlock(memorySubtracted));
    }


    private void setTotalPerDevice(int device, long memoryAvailable) {
        totalPerDevice.add(device, new AtomicLong(memoryAvailable));
    }


    private long matchBlock(long numBytes) {
        //int align = 65536 * 2;
        //return numBytes + (align - (numBytes % align));
        return numBytes;
    }
}
