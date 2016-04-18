package org.nd4j.jita.allocator.impl;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaEvent_t;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.concurrency.AtomicState;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.garbage.GarbageReference;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.time.RateTimer;
import org.nd4j.jita.allocator.time.TimeProvider;
import org.nd4j.jita.allocator.time.impl.SimpleTimer;
import org.nd4j.jita.allocator.time.providers.MillisecondsProvider;
import org.nd4j.jita.allocator.time.providers.OperativeProvider;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.Reference;
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
    private volatile PointersPair pointerInfo;

    @Getter @Setter private Long objectId;
    @Getter @Setter private Long bucketId;

    // thread safety is guaranteed by allocLock
    private volatile AllocationStatus allocationStatus = AllocationStatus.UNDEFINED;

    private transient TimeProvider timeProvider = new OperativeProvider();
    private transient TimeProvider realTimeProvider = new MillisecondsProvider();

    // corresponding access times in TimeProvider quants
    private final AtomicLong accessHostRead = new AtomicLong(0);
    private final AtomicLong accessDeviceRead = new AtomicLong(0);

    private final AtomicLong accessHostWrite = new AtomicLong(0);
    private final AtomicLong accessDeviceWrite = new AtomicLong(0);

    // real time here
    private final AtomicLong deviceAccessTime = new AtomicLong(0);

    @Getter @Setter private boolean constant;

    // TODO: timer should be instantiated externally
//    @Getter private final RateTimer timerShort = new SimpleTimer(10, TimeUnit.SECONDS); //new BinaryTimer(5, TimeUnit.SECONDS);
//    @Getter private final RateTimer timerLong = new SimpleTimer(60, TimeUnit.SECONDS);

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

//    private Map<AllocationShape, NestedPoint> usedChunks = new ConcurrentHashMap<>();

//    @Getter private AtomicState accessState = new AtomicState();

    private volatile WeakReference<BaseDataBuffer> originalDataBufferReference;

    private volatile GarbageReference garbageReference;

    private cudaEvent_t lastEvent;


    public void setLastEvent(cudaEvent_t event) {
        if (event != null) {
            if (lastEvent != null)
                JCuda.cudaEventDestroy(lastEvent);
        }
        lastEvent = event;
    }

    public cudaEvent_t getLastEvent() {
        return lastEvent;
    }


    /**
     * This method stores WeakReference to original BaseCudaDataBuffer
     *
     * @param buffer
     */
    public void attachBuffer(@NonNull BaseDataBuffer buffer) {
        originalDataBufferReference = new WeakReference<BaseDataBuffer>(buffer);
    }

    public void attachReference(GarbageReference reference) {
        garbageReference = reference;
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
            return allocationStatus;
    }

    /**
     * This method sets specified AllocationStatus for this point
     * @param status
     */
    public void setAllocationStatus(@NonNull AllocationStatus status) {
            allocationStatus = status;
    }

    /**
     * This method returns CUDA pointer object for this allocation.
     * It can be either device pointer or pinned memory pointer, or null.
     *
     * PLEASE NOTE: Thread safety is guaranteed by reentrant read/write lock
     * @return
     */
    public Pointer getDevicePointer() {
            if (pointerInfo == null) {
                log.info("pointerInfo is null");
                return null;
            }
            return pointerInfo.getDevicePointer();
    }

    /**
     * This method returns CUDA pointer object for this allocation.
     * It can be either device pointer or pinned memory pointer, or null.
     *
     * PLEASE NOTE: Thread safety is guaranteed by reentrant read/write lock
     * @return
     */
    public Pointer getHostPointer() {
            if (pointerInfo == null)
                return null;

            return pointerInfo.getHostPointer();
    }

    /**
     * This method sets CUDA pointer for this allocation.
     * It can be either device pointer, or pinned memory pointer, or null.
     *
     * PLEASE NOTE: Thread safety is guaranteed by reentrant read/write lock
     * @param pointerInfo CUDA pointers wrapped into DevicePointerInfo
     */
    public void setPointers(@NonNull PointersPair pointerInfo) {
        this.pointerInfo = pointerInfo;
    }

    public PointersPair getPointers() {
            return this.pointerInfo;
    }

    public long getDeviceTicks() {
        return deviceTicks.get();
    }

    public void tickDeviceRead() {
//        this.deviceTicks.incrementAndGet();
//        this.timerShort.triggerEvent();
//        this.timerLong.triggerEvent();
        //this.deviceAccessTime.set(realTimeProvider.getCurrentTime());
        this.accessDeviceRead.set(timeProvider.getCurrentTime());
    }

    public void tackDevice() {
        //this.deviceTicks.incrementAndGet();
        this.accessDeviceRead.set(timeProvider.getCurrentTime());
        this.deviceAccessTime.set(realTimeProvider.getCurrentTime());
    }

    /**
     * Returns time, in milliseconds, when this point was accessed on host side
     *
     * @return
     */
    public long getHostReadTime() {
        return accessHostRead.get();
    }

    public long getHostWriteTime() {
        return accessHostWrite.get();
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
//        deviceAccessTime.set(realTimeProvider.getCurrentTime());
        tickDeviceRead();
        accessDeviceWrite.set(timeProvider.getCurrentTime());
    }

    /**
     * This method sets time when this point was changed on host
     */
    public void tickHostWrite() {
        tickHostRead();
        accessHostWrite.set(timeProvider.getCurrentTime());
    }

    /**
     * This method returns, if host side has actual copy of data
     *
     * @return true, if data is actual, false otherwise
     */
    public boolean isActualOnHostSide() {
        //log.info("isActuialOnHostSide() -> Host side: [{}], Device side: [{}]", accessHostRead.get(), accessDeviceRead.get());
        boolean result = accessHostWrite.get() >= accessDeviceWrite.get() || accessHostRead.get() >= accessDeviceWrite.get();
        //log.info("isActuialOnHostSide() -> {}, shape: {}", result, shape);
        return result;
    }

    /**
     * This method returns, if device side has actual copy of data
     *
     * @return
     */
    public boolean isActualOnDeviceSide() {
        //log.info("isActuialOnDeviceSide() -> Host side: [{}], Device side: [{}]", accessHostWrite.get(), accessDeviceWrite.get());
        boolean result = accessDeviceWrite.get() >= accessHostWrite.get() || accessDeviceRead.get() >= accessHostWrite.get(); //accessHostWrite.get() <= getDeviceAccessTime();
//        log.info("isActuialOnDeviceSide() -> {} ({}), Shape: {}", result, objectId, shape);
        return result;
    }

    /**
     * This method sets device access time equal to host write time
     */
    public void tickDeviceToHost() {
        accessDeviceRead.set(accessHostRead.get());
        this.deviceAccessTime.set(realTimeProvider.getCurrentTime());
    }

    @Override
    public String toString() {
        return "AllocationPoint{" +
                "deviceId=" + deviceId +
                ", objectId=" + objectId +
                ", shape=" + shape +
                '}';
    }
}
