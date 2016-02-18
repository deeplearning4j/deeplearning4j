package org.nd4j.jita.allocator.impl;

import jcuda.Pointer;
import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.jita.allocator.concurrency.AtomicState;
import org.nd4j.jita.allocator.enums.AccessState;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.time.RateTimer;
import org.nd4j.jita.allocator.time.TimeProvider;
import org.nd4j.jita.allocator.time.impl.SimpleTimer;
import org.nd4j.jita.allocator.time.providers.MillisecondsProvider;
import org.nd4j.jita.allocator.time.providers.OperativeProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.DevicePointerInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This class describes top-level allocation unit.
 * Every buffer passed into CUDA wii have allocation point entry, describing allocation state.
 *
 * @author raver119@gmail.com
 */

public class AllocationPoint {
    private static Logger log = LoggerFactory.getLogger(AllocationPoint.class);

    // thread safety is guaranteed by cudaLock
    private volatile DevicePointerInfo pointerInfo;

    @Getter @Setter private Long objectId;

    // thread safety is guaranteed by allocLock
    private volatile AllocationStatus allocationStatus = AllocationStatus.UNDEFINED;

    private SyncState hostMemoryState = SyncState.UNDEFINED;

    private transient TimeProvider timeProvider = new OperativeProvider();
    private transient TimeProvider realTimeProvider = new MillisecondsProvider();

    // corresponding access times in TimeProvider quants
    private final AtomicLong accessHostRead = new AtomicLong(0);
    private final AtomicLong accessDeviceRead = new AtomicLong(0);
    private final AtomicLong accessHostWrite = new AtomicLong(0);
    private final AtomicLong accessDeviceWrite = new AtomicLong(0);

    // real time here
    private final AtomicLong deviceAccessTime = new AtomicLong(0);

    // TODO: timer should be instantiated externally
    @Getter private final RateTimer timerShort = new SimpleTimer(10, TimeUnit.SECONDS); //new BinaryTimer(5, TimeUnit.SECONDS);
    @Getter private final RateTimer timerLong = new SimpleTimer(60, TimeUnit.SECONDS);

    /*
     device, where memory was/will be allocated.
    Valid integer >= 0 is deviceId, null for undefined
    */
    @Getter @Setter private volatile Integer deviceId;

    /*
        We assume 1D memory chunk allocations.
    */
    @Getter @Setter private AllocationShape shape;

    private AtomicLong deviceTicks = new AtomicLong(0);
    private AtomicLong descendantsTicks = new AtomicLong(0);
    private AtomicLong descendantsTacks = new AtomicLong(0);

    private Map<AllocationShape, NestedPoint> usedChunks = new ConcurrentHashMap<>();

    @Getter private AtomicState accessState = new AtomicState();

    private volatile WeakReference<DataBuffer> originalDataBufferReference;

    private ReentrantReadWriteLock cudaLock = new ReentrantReadWriteLock();
    private ReentrantReadWriteLock allocLock = new ReentrantReadWriteLock();

    /**
     * This method stores WeakReference to original BaseCudaDataBuffer
     *
     * @param buffer
     */
    public void attachBuffer(@NonNull DataBuffer buffer) {
        originalDataBufferReference = new WeakReference<DataBuffer>(buffer);
    }

    /**
     * This method returns previously stored BaseCudaDataBuffer instance
     *
     * PLEASE NOTE: Return value CAN be null
     *
     * @return
     */
    public DataBuffer getBuffer() {
        if (originalDataBufferReference != null) {
            return originalDataBufferReference.get();
        } else return null;
    }

    /**
     * This method returns current AllocationStatus for this point
     * @return
     */
    public AllocationStatus getAllocationStatus() {
        try {
            allocLock.readLock().lock();

            return allocationStatus;
        } finally {
            allocLock.readLock().unlock();
        }
    }

    /**
     * This method sets specified AllocationStatus for this point
     * @param status
     */
    public void setAllocationStatus(@NonNull AllocationStatus status) {
        try {
            allocLock.writeLock().lock();

            allocationStatus = status;
        } finally {
            allocLock.writeLock().unlock();
        }
    }

    /**
     * This method returns CUDA pointer object for this allocation.
     * It can be either device pointer or pinned memory pointer, or null.
     *
     * PLEASE NOTE: Thread safety is guaranteed by reentrant read/write lock
     * @return
     */
    public Pointer getCudaPointer() {
        try {
            cudaLock.readLock().lock();

            if (pointerInfo == null)
                return null;

            if (pointerInfo.getPointers() == null)
                return null;

            return pointerInfo.getPointers().getDevicePointer();
        } finally {
            cudaLock.readLock().unlock();
        }
    }

    /**
     * This method returns CUDA pointer object for this allocation.
     * It can be either device pointer or pinned memory pointer, or null.
     *
     * PLEASE NOTE: Thread safety is guaranteed by reentrant read/write lock
     * @return
     */
    public Pointer getHostPointer() {
        try {
            cudaLock.readLock().lock();

            if (pointerInfo == null)
                return null;

            if (pointerInfo.getPointers() == null)
                return null;

            return pointerInfo.getPointers().getHostPointer();
        } finally {
            cudaLock.readLock().unlock();
        }
    }

    /**
     * This method sets CUDA pointer for this allocation.
     * It can be either device pointer, or pinned memory pointer, or null.
     *
     * PLEASE NOTE: Thread safety is guaranteed by reentrant read/write lock
     * @param pointerInfo CUDA pointers wrapped into DevicePointerInfo
     */
    public void setCudaPointers(DevicePointerInfo pointerInfo) {
        try {
            cudaLock.writeLock().lock();

            this.pointerInfo = pointerInfo;
        } finally {
            cudaLock.writeLock().unlock();
        }
    }

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
        this.deviceAccessTime.set(realTimeProvider.getCurrentTime());
        this.accessDeviceRead.set(timeProvider.getCurrentTime());
    }

    public void tackDevice() {
        //this.deviceTicks.incrementAndGet();
        this.accessDeviceRead.set(timeProvider.getCurrentTime());
        this.deviceAccessTime.set(realTimeProvider.getCurrentTime());
    }

    public void tickDescendant(AllocationShape shape) {
        this.descendantsTicks.incrementAndGet();
        this.usedChunks.get(shape).tick();
    }

    public void tackDescendant(AllocationShape shape) {
        this.descendantsTacks.incrementAndGet();
        this.usedChunks.get(shape).tack();
    }

    @Deprecated
    public boolean confirmNoActiveDescendants() {
        /*
            This method is probably deprecated, and probably will be removed, since we have TickTackToe tracking now.
         */
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

    /**
     * Removes suballocation shape from tracking list
     *
     * @param point
     */
    public void dropShape(@NonNull NestedPoint point) {
        if (!usedChunks.containsKey(point.getShape()))
            throw new IllegalStateException("Shape [" + shape + "] was NOT found on dropShape() call");

        usedChunks.remove(point.getShape());
    }

    /**
     * Checks, if we have specific suballocation shape registered
     * @param shape
     * @return
     */
    public boolean containsShape(@NonNull AllocationShape shape) {
        return usedChunks.containsKey(shape);
    }

    /**
     * This method returns suballocation description for specific shape
     *
     * @param shape
     * @return
     */
    public NestedPoint getNestedPoint(@NonNull AllocationShape shape) {
        if (containsShape(shape))
            return usedChunks.get(shape);
        else  throw new IllegalStateException("Shape [" + shape + "] was NOT found on getNestedPoint() call");
    }

    /**
     * Returns time, in milliseconds, when this point was accessed on host side
     *
     * @return
     */
    public long getHostAccessTime() {
        return accessHostRead.get();
    }


    public long getRealDeviceAccessTime() {
        return deviceAccessTime.get();
    }

    /**
     * Returns time, in milliseconds, when this point was accessed on device side
     *
     * @return
     */
    public long getDeviceAccessTime() {
        return accessDeviceRead.get();
    }

    /**
     * Returns time when point was written on device last time
     *
     * @return
     */
    public long getDeviceWriteTime() {
        return accessDeviceWrite.get();
    }

    public void tickHostRead() {
        accessHostRead.set(timeProvider.getCurrentTime());
    }

    /**
     * This method sets time when this point was changed on device
     *
     */
    public void tickDeviceWrite() {
        deviceAccessTime.set(realTimeProvider.getCurrentTime());
        accessDeviceWrite.set(timeProvider.getCurrentTime());
    }

    /**
     * This method sets time when this point was changed on host
     */
    public void tickHostWrite() {
        accessHostWrite.set(timeProvider.getCurrentTime());
    }

    /**
     * This method returns, if host side has actual copy of data
     *
     * @return true, if data is actual, false otherwise
     */
    public boolean isActualOnHostSide() {
        return getHostAccessTime() >= getDeviceWriteTime();
    }

    /**
     * This method returns, if device side has actual copy of data
     *
     * @return
     */
    public boolean isActualOnDeviceSide() {
        return accessHostWrite.get() <= getDeviceAccessTime();
    }

    /**
     * This method sets device access time equal to host write time
     */
    public void tickDeviceToHost() {
        accessDeviceRead.set(accessHostRead.get());
        this.deviceAccessTime.set(realTimeProvider.getCurrentTime());
    }
}
