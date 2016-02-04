package org.nd4j.jita.allocator.impl;

import lombok.NonNull;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.mover.Mover;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This is going to be basic JITA implementation.
 *
 * PLEASE NOTE: WORK IN PROGRESS, DO NOT EVER USE IT!
 *
 * @author raver119@gmail.com
 */
public final class BasicAllocator implements Allocator {
    private static final BasicAllocator INSTANCE = new BasicAllocator();
    private Configuration configuration = new Configuration();

    private transient Mover mover;

    private Map<Long, AllocationPoint> allocationPoints = new ConcurrentHashMap<>();


    private static Logger log = LoggerFactory.getLogger(BasicAllocator.class);

    protected BasicAllocator() {
        //
    }

    public static BasicAllocator getInstance() {
        return INSTANCE;
    }


    /**
     * Consume and apply configuration passed in as argument
     *
     * @param configuration configuration bean to be applied
     */
    @Override
    public void applyConfiguration(Configuration configuration) {
        // TODO: global lock to be implemented
        this.configuration = configuration;
    }

    /**
     * Returns current Allocator configuration
     *
     * @return current configuration
     */
    @Override
    public Configuration getConfiguration() {
        // TODO: global lock to be implemented
        return configuration;
    }

    /**
     * This method registers buffer within allocator instance
     *
     * @param buffer
     */
    @Override
    public void pickupSpan(DataBuffer buffer) {

    }

    /**
     * This method registers array's buffer within allocator instance
     *
     * @param array
     */
    @Override
    public void pickupSpan(INDArray array) {

    }

    protected void registerSpan(Long objectId, @NonNull AllocationShape shape) {
        // TODO: object-level lock is HIGHLY required here, for multithreaded safety
        if (!allocationPoints.containsKey(objectId)) {
            AllocationPoint allocationPoint = new AllocationPoint();
            allocationPoint.setAccessHost(System.nanoTime());
            allocationPoint.setAllocationStatus(AllocationStatus.UNDEFINED);
            allocationPoint.setShape(shape);

            allocationPoint.addShape(shape);

            allocationPoints.put(objectId, allocationPoint);
        } else {
            AllocationPoint allocationPoint = allocationPoints.get(objectId);
            if (shape.equals(allocationPoint.getShape())) {
                // that's temporary exception, since such scenario is theoretically possible
                throw new IllegalStateException("Double register called on the same id and same shape");
            } else {
                // just suballocation. check for buffer overflow and attach new shape
                allocationPoint.addShape(shape);
            }
        }
    }

    /**
     * Returns allocation point for specified object ID
     *
     * @param objectId
     * @return
     */
    protected AllocationPoint getAllocationPoint(Long objectId) {
        return allocationPoints.get(objectId);
    }

    /**
     * This method hints allocator, that specific object was accessed on host side.
     * This includes putRow, putScalar etc methods as well as initial object instantiation.
     *
     * @param objectId unique object ID
     */
    @Override
    public void tickHost(Long objectId) {
        // TODO: provide object-level lock here
        AllocationPoint point = allocationPoints.get(objectId);
        point.setAccessHost(System.nanoTime());
    }

    /**
     * This methods hints allocator, that specific object was accessed on device side.
     *
     * @param objectId unique object ID
     * @param deviceId device ID
     */
    @Override
    public void tickDevice(Long objectId, Integer deviceId) {
        // TODO: provide object-level lock here
        AllocationPoint point = allocationPoints.get(objectId);
        point.setAccessDevice(System.nanoTime());
        point.tickDevice();
    }

    /**
     * This method returns actual device pointer valid for current object
     *
     * @param objectId
     */
    @Override
    public Object getDevicePointer(Long objectId) {
        // TODO: this method should return pointer at some point later
        // TODO: provide object-level lock here
        AllocationPoint point = allocationPoints.get(objectId);

        return getDevicePointer(objectId, point.getShape());
    }

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param objectId
     * @param shape
     */
    @Override
    public Object getDevicePointer(Long objectId, AllocationShape shape) {
         /*
            Basic plan here:
            1. we should check, if any devicePointer exists for this object.
                1. if it doesn't exist - create new one, and proceed to p.3
                2. if it exist, go to next plan point
            2. we should check, if there was any changes in data on host site
                1. if any changes were made - memory on device side should be updated using mover
            3. update access information, to reflect current state
            4. return devicePointer
         */
        AllocationPoint point = allocationPoints.get(objectId);

        Object pointer = point.getDevicePointer();

        if (pointer == null) {
            // we don't have actual pointer, so we can assume initial allocation here
            if (point.getShape().equals(shape)) {
                // we're allocating the whole original buffer

                pointer = mover.alloc(AllocationStatus.ZERO, point.getShape(), 0);
                point.setDevicePointer(pointer);
                point.setAllocationStatus(AllocationStatus.ZERO);
                point.tickDescendant(point.getShape());
            } else {
                // we're allocating the part of original array
                // we have to decide here, if we can allocate full buffer
                // for now we hardcode FALSE, restricting allocation to specific chunk
                if (1 > 0) {
                    pointer = mover.alloc(AllocationStatus.ZERO, shape, 0);
                    point.setDevicePointer(pointer);
                    point.setAllocationStatus(AllocationStatus.ZERO);
                    point.tickDescendant(shape);
                }
            }
        } else {
            // TODO: we have pointer, and we should check it for synchronization
            // we should check, if it's offest requested or it matches original shape
            if (!point.getShape().equals(shape)) {
                // we assume that it's suballocation
                // actually all we need to do, is shift original pointer by offset  * elementSize
                // like pointer += (shape.getOffset * (shape.getDataType() == DOUBLE) ? 8 : 4;
                point.addShape(shape);
                point.tickDescendant(shape);
            }
        }

        // p.3
        this.tickDevice(objectId, 1);

        // p.4
        return pointer;
    }

    /**
     * This method should be called to make sure that data on host size is actualized
     *
     * @param objectId
     */
    @Override
    public void validateHostData(Long objectId) {
        AllocationPoint point = allocationPoints.get(objectId);


        if (!getHostMemoryState(objectId).equals(SyncState.SYNC)) {
            // if data was accessed by device, it could be changed somehow
            mover.copyback(point);
        } else {
            // if data wasn't accessed on device side, we don't have to do anything for validation
            // i.e: multiple putRow calls, or putScalar, or whatever else
            ;
        }
    }

    /**
     * This method returns current host memory state
     *
     * @param objectId
     * @return
     */
    @Override
    public SyncState getHostMemoryState(Long objectId) {
        AllocationPoint point = allocationPoints.get(objectId);
        if (point.getAccessHost() >= point.getAccessDevice()) {
            point.setHostMemoryState(SyncState.SYNC);
        } else {
            point.setHostMemoryState(SyncState.DESYNC);
        }
        return point.getHostMemoryState();
    }

    /**
     * This method returns the number of top-level memory allocation.
     * No descendants are included in this result.
     *
     * @return number of allocated top-level memory chunks
     */
    @Override
    public int tableSize() {
        return allocationPoints.size();
    }

    /**
     * This method forces allocator to shutdown gracefully
     */
    protected void shutdown() {
        // TODO: to be implemented
    }

    /**
     * This method resets allocator state
     *
     * PLEASE NOTE: This method is unsafe, do not use it until you 100% sure what are you doing
     */
    protected void reset() {
        // TODO: to be implemented
    }

    /**
     * This method relocates memory from one point to another
     *
     * @param objectId
     * @param targetStatus
     */
    protected void relocateMemory(@NonNull Long objectId, @NonNull AllocationStatus targetStatus) {
        // TODO: implement object-level lock here
        AllocationPoint point = allocationPoints.get(objectId);

        mover.relocate(point.getAllocationStatus(), targetStatus, point);
    }

    /**
     *
     * @param mover Mover implementation to be used for data transfers
     */
    protected void setMover(@NonNull Mover mover) {
        this.mover = mover;
    }
}
