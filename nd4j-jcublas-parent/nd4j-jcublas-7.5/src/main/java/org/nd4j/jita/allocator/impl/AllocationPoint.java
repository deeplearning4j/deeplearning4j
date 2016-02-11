package org.nd4j.jita.allocator.impl;

import jcuda.Pointer;
import lombok.Data;
import lombok.NonNull;
import org.nd4j.jita.allocator.concurrency.AtomicState;
import org.nd4j.jita.allocator.enums.AccessState;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.time.RateTimer;
import org.nd4j.jita.allocator.time.impl.SimpleTimer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
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
    private Pointer cudaPointer;

    private Object hostPointer;

    private Long objectId;

    private AllocationStatus allocationStatus = AllocationStatus.UNDEFINED;
    private SyncState hostMemoryState = SyncState.UNDEFINED;

    // corresponding access time in nanoseconds
    private long accessHost = 0;
    private long accessDevice = 0;

    // TODO: timer should be instantiated externally
    private RateTimer timerShort = new SimpleTimer(10, TimeUnit.SECONDS); //new BinaryTimer(5, TimeUnit.SECONDS);
    private RateTimer timerLong = new SimpleTimer(60, TimeUnit.SECONDS);

    /*
     device, where memory was/will be allocated.
    Valid integer >= 0 is deviceId, null for undefined
    */
    private Integer deviceId = null;

    /*
        We assume 1D memory chunk allocations.
    */
    private AllocationShape shape;

    private AtomicLong deviceTicks = new AtomicLong(0);
    private AtomicLong descendantsTicks = new AtomicLong(0);
    private AtomicLong descendantsTacks = new AtomicLong(0);

    private Map<AllocationShape, NestedPoint> usedChunks = new ConcurrentHashMap<>();

    private AtomicState accessState = new AtomicState();

    private transient WeakReference<DataBuffer> originalDataBuffer;

    public long getDeviceTicks() {
        return deviceTicks.get();
    }

    public long getDescendantsTicks() {
        return descendantsTicks.get();
    }

    public long getDescendantsTacks() {
        return descendantsTacks.get();
    }

    public long getDescendantTicks(@NonNull AllocationShape shape) {
        if (usedChunks.containsKey(shape)) {
            return usedChunks.get(shape).getTicks();
        } else {
            // FIXME: remove this in production use
            //throw new IllegalStateException("Descendant shape not found: " + shape);
            return -1;
        }
    }

    public void tickDevice() {
        this.deviceTicks.incrementAndGet();
        this.timerShort.triggerEvent();
        this.timerLong.triggerEvent();
    }

    public void tackDevice() {
        //this.deviceTicks.incrementAndGet();
    }

    public void tickDescendant(AllocationShape shape) {
        this.descendantsTicks.incrementAndGet();
        this.usedChunks.get(shape).tick();
    }

    public void tackDescendant(AllocationShape shape) {
        this.descendantsTacks.incrementAndGet();
        this.usedChunks.get(shape).tack();
    }

    public boolean confirmNoActiveDescendants() {
        // TODO: point-wise lock should be assumed here
        return descendantsTicks.get() == descendantsTacks.get();
    }

    public int getNumberOfDescendants() {
        return usedChunks.size();
    };

    /**
     * Adds suballocation shape to tracking list
     *
     * @param point
     */
    public void addShape(@NonNull NestedPoint point) {
        if (!usedChunks.containsKey(point.getShape())) {
            this.usedChunks.put(point.getShape(), point);
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

    public void dropShape(@NonNull NestedPoint point) {
        if (!usedChunks.containsKey(point.getShape()))
            throw new IllegalStateException("Shape [" + shape + "] was NOT found on dropShape() call");

        usedChunks.remove(point.getShape());
    }

    public boolean containsShape(@NonNull AllocationShape shape) {
        return usedChunks.containsKey(shape);
    }

    public NestedPoint getNestedPoint(@NonNull AllocationShape shape) {
        if (containsShape(shape))
            return usedChunks.get(shape);
        else  throw new IllegalStateException("Shape [" + shape + "] was NOT found on getNestedPoint() call");
    }
}
