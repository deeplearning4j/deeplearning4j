package org.nd4j.jita.allocator.impl;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This class describes top-level allocation unit.
 * Every buffer passed into CUDA wii have allocation point entry, describing allocation state.
 *
 * @author raver119@gmail.com
 */
@Data
public class AllocationPoint {
    private static Logger log = LoggerFactory.getLogger(AllocationPoint.class);

    // TODO: change this to Pointer later
    private Object devicePointer;
    private Object hostPointer;

    private Long objectId;

    private AllocationStatus allocationStatus = AllocationStatus.UNDEFINED;
    private SyncState hostMemoryState = SyncState.UNDEFINED;

    // corresponding access time in nanoseconds
    private long accessHost = 0;
    private long accessDevice = 0;

    /*
     device, where memory was allocated.
     0 for host, -1 for deallocated/undefined
    */
    private Integer deviceId = -1;

    /*
        We assume 1D memory chunk allocations.
    */
    private AllocationShape shape;

    private AtomicLong deviceTicks = new AtomicLong(0);
    private AtomicLong descendantsTicks = new AtomicLong(0);

    private Map<AllocationShape, AtomicLong> usedChunks = new ConcurrentHashMap<>();

    public long getDeviceTicks() {
        return deviceTicks.get();
    }

    public long getDescendantsTicks() {
        return descendantsTicks.get();
    }

    public long getDescendantTicks(@NonNull AllocationShape shape) {
        if (usedChunks.containsKey(shape)) {
            return usedChunks.get(shape).get();
        } else {
            // FIXME: remove this in production use
            //throw new IllegalStateException("Descendant shape not found: " + shape);
            return -1;
        }
    }

    public void tickDevice() {
        this.deviceTicks.incrementAndGet();
    }

    public void tickDescendant(AllocationShape shape) {
        this.descendantsTicks.incrementAndGet();
        this.usedChunks.get(shape).incrementAndGet();
    }

    public int getNumberOfDescendants() {
        return usedChunks.size();
    };

    /**
     * Adds suballocation shape to tracking list
     *
     * @param shape
     */
    public void addShape(@NonNull AllocationShape shape) {
        if (!usedChunks.containsKey(shape)) {
            this.usedChunks.put(shape, new AtomicLong(0));
        }
    }

    /**
     * Removes suballocation shape from tracking list
     *
     * @param shape
     */
    public void dropShape(@NonNull AllocationShape shape) {
        if (!usedChunks.containsKey(shape))
            throw new IllegalStateException("Shape [" + shape + "] was NOT found on dropShape() call");

        usedChunks.remove(shape);
    }
}
