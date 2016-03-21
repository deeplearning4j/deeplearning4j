package org.nd4j.jita.allocator.impl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import lombok.NonNull;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.concurrency.DeviceAllocationsTracker;
import org.nd4j.jita.allocator.enums.AccessState;
import org.nd4j.jita.allocator.enums.Aggressiveness;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.time.Ring;
import org.nd4j.jita.allocator.time.rings.LockedRing;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.mover.Mover;
import org.nd4j.jita.mover.CudaZeroMover;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Just-in-Time Allocator for CUDA
 *
 * This method is a basement for pre-allocated memory management for cuda.
 * Basically that's sophisticated garbage collector for both zero-copy memory, and multiple device memory.
 *
 * There's multiple possible data movement directions, but general path is:
 * host memory (issued on JVM side) ->
 *          zero-copy pinned memory (which is allocated for everything out there) ->
 *                  device memory (where data gets moved from zero-copy, if used actively enough)
 *
 * And the backward movement, if memory isn't used anymore (like if originating INDArray was trashed by JVM GC), or it's not popular enough to hold in device memory
 *
 * Mechanism is as lock-free, as possible. This achieved using three-state memory state signalling: Tick/Tack/Toe.
 * Tick: memory chunk (or its part) is accessed on on device
 * Tack: memory chink (or its part) device access session was finished
 * Toe: memory chunk is locked for some reason. Possible reasons:
 *              Memory synchronization is ongoing, host->gpu or gpu->host
 *              Memory relocation is ongoing, zero->gpu, or gpu->zero, or gpu->host
 *              Memory removal is ongoing.
 *
 * So, basically memory being used for internal calculations, not interfered with manual changes (aka putRow etc), are always available without locks
 *
 * @author raver119@gmail.com
 */
public class AtomicAllocator implements Allocator {
    private static final AtomicAllocator INSTANCE = new AtomicAllocator();

    private Configuration configuration = new Configuration();
    private CudaEnvironment environment;
    private transient Mover mover = new CudaZeroMover();
    private AtomicLong allocationsCounter = new AtomicLong(0);

    private AtomicLong objectsTracker = new AtomicLong(Long.MIN_VALUE);

    // tracker for thread->device affinity
    protected Map<Long, Integer> devicesAffinity = new ConcurrentHashMap<>();

    // simple counter to track allocated host-memory
    protected final AtomicLong zeroUseCounter = new AtomicLong(0);

    // another simple counter, to track allocated device memory on per-thread per-device basis
    protected volatile DeviceAllocationsTracker deviceMemoryTracker;

    // we have single tracking point for allocation points, since we're not going to cycle through it it any time soon
    private Map<Long, AllocationPoint> allocationsMap = new ConcurrentHashMap<>();

    /*
        table for Thread, Device, Object allocations of device memory. Objects should be used to grab Allocation point from allocationsMap
    */
    // TODO: proper thread-safe implementation would be nice to have here :(
    // FIXME: CopyOnWriteArrayList is BAD here. Really BAD. B A D.
    // Table thread safety is guaranteed by reentrant read/write locks :(
    private Table<Long, Integer, ConcurrentHashMap<Long, Long>> deviceAllocations = HashBasedTable.create();

    /*
        map for Thread, Object allocations in zero memory.
    */
    // CopyOnWriteArrayList performance to be investigated in this use case
    // Map thread safety is guaranteed by exclusive writeLock in getDeviceId() method, because we can't use putIfAbsent on j7
    // FIXME: at j7 -> j8 transition, this one could be changed to ConcurrentHashMap
    private Map<Long, ConcurrentHashMap<Long, Long>> zeroAllocations = new HashMap<>();

    /*
        WeakHashMap for buffer->id conversion. If DataBuffer get's removed by jvm GC, we'll know that.
        So, just a short way for reverse lookup, that causes no GC issues.
     */
    private Map<DataBuffer, Long> externalBuffers = Collections.synchronizedMap(new WeakHashMap<DataBuffer, Long>());

    // simple pool for cublas contexts
    private Map<Long, CudaContext> contextPool = new ConcurrentHashMap<>();

    private static Logger log = LoggerFactory.getLogger(AtomicAllocator.class);

    // list of banned devices to be used
    private List<Integer> bannedDevices = new ArrayList<>();

    /*
        locks for internal resources
     */
    private ReentrantReadWriteLock deviceLock = new ReentrantReadWriteLock();
    private ReentrantReadWriteLock globalLock = new ReentrantReadWriteLock();
    private ReentrantReadWriteLock externalsLock = new ReentrantReadWriteLock();

    /*
        here we have handles for garbage collector threads
        ThreadId, GarbageCollector
     */
    private Map<Long, ZeroGarbageCollectorThread> collectorsZero = new ConcurrentHashMap<>();
    private Map<Long, DeviceGarbageCollectorThread> collectorsDevice = new ConcurrentHashMap<>();

    private final AtomicBoolean shouldStop = new AtomicBoolean(false);

    private final AtomicBoolean wasInitialised = new AtomicBoolean(false);


    private final Ring deviceLong = new LockedRing(30);
    private final Ring deviceShort = new LockedRing(30);

    private final Ring zeroLong = new LockedRing(30);
    private final Ring zeroShort = new LockedRing(30);

    // pointer to next deviceId
    private final AtomicInteger devPtr = new AtomicInteger(0);


    public static AtomicAllocator getInstance() {
        return INSTANCE;
    }

    protected AtomicAllocator() {
        environment = new CudaEnvironment(configuration);

        this.deviceMemoryTracker = new DeviceAllocationsTracker(this.environment, this.configuration);
    }

    /**
     * This method returns CudaContext for current thread
     *
     * @return
     */
    @Override
    public CudaContext getCudaContext() {
        // FIXME: proper lock avoidance required here

        if (!contextPool.containsKey(Thread.currentThread().getId())) {
            initCudaContextForThread(Thread.currentThread().getId());
        }
        return contextPool.get(Thread.currentThread().getId());
    }

    /**
     * This method specifies Mover implementation to be used internally
     * @param mover
     */
    @Override
    public void setMover(@NonNull Mover mover) {
        globalLock.writeLock().lock();

        this.mover = mover;
        this.mover.init(configuration, environment, this);

        globalLock.writeLock().unlock();
    }

    /**
     * Consume and apply configuration passed in as argument
     *
     * PLEASE NOTE: This method should only be used BEFORE any calculations were started.
     *
     * @param configuration configuration bean to be applied
     */
    @Override
    public void applyConfiguration(@NonNull Configuration configuration) {
        if (!wasInitialised.get()) {
            globalLock.writeLock().lock();

            this.configuration = configuration;
            this.environment = new CudaEnvironment(this.configuration);

            this.deviceMemoryTracker = new DeviceAllocationsTracker(this.environment, this.configuration);

            globalLock.writeLock().unlock();
        }
    }

    /**
     * This method allows you to exclude specific device from being used for calculations
     * <p>
     * Please note: you can call this method multiple times, to ban multiple devices
     *
     * @param deviceId deviceId to be banned
     */
    public void banDevice(@NonNull Integer deviceId) {
        globalLock.writeLock().lock();

        environment.banDevice(deviceId);

        globalLock.writeLock().unlock();
    }

    /**
     * Set active CUDA environment
     *
     * @param environment
     */
    @Override
    public void setEnvironment(@NonNull CudaEnvironment environment) {
        globalLock.writeLock().lock();
        this.environment = environment;

        this.deviceMemoryTracker = new DeviceAllocationsTracker(this.environment, this.configuration);

        globalLock.writeLock().unlock();
    }

    /**
     * Returns current Allocator configuration
     *
     * @return current configuration
     */
    @Override
    public Configuration getConfiguration() {
        try {
            globalLock.readLock().lock();
            return configuration;
        } finally {
            globalLock.readLock().unlock();
        }
    }

    /**
     * This method registers buffer within allocator instance
     *
     * @param buffer DataBuffer object to be picked & tracked
     */

    protected Long pickupSpan(DataBuffer buffer, AllocationShape shape) {
        //log.info("pickupSpan(BaseCudaDataBuffer)");
//        if (1> 0) throw new RuntimeException("");
        try {
            externalsLock.writeLock().lock();

            if (externalBuffers.containsKey(buffer)) {
                /*
                    We have such buffer already. It's either the Nested allocation, or something like that.
                    Just throw exception for now.
                 */
                throw new IllegalStateException("Buffer is already registered");
                //return buffer.getAllocatorPointer();
            } else {
                /*
                    We don't have such buffer registered somehow. Probably that's new allocation
                 */
                AllocationPoint point = new AllocationPoint();
                //point.setShape(AllocationUtils.buildAllocationShape(buffer.originalDataBuffer()));

                // set device ID -> current thread
                point.setDeviceId(getDeviceId());

                Long allocPointer = objectsTracker.getAndIncrement();

                point.setObjectId(allocPointer);
                point.setShape(shape);

                /*
                    we don't keep strong references Allocator -> Buffer, but we store Buffer -> Allocator references instead :)
                  */
                //buffer.setAllocationPoint(point);
                buffer.setTrackingPoint(allocPointer);

                // we storing buffer instance as WeakReference here for future copybacks
                point.attachBuffer(buffer);

                // the only
                externalBuffers.put(buffer, allocPointer);

                allocationsMap.put(allocPointer, point);

                return allocPointer;
            }

        } finally {
            externalsLock.writeLock().unlock();
        }
    }

    /**
     * This method registers array's buffer within allocator instance
     *
     * @param array INDArray object to be picked & tracked
     */
    @Override
    public Long pickupSpan(INDArray array) {
        //log.info("pickupSpan(INDarray)");
        /*
         while working on array level, we actually immediately downgrade to buffer level, with AllocationShape defined by this array
          */
        //if (!(array.data() instanceof BaseCudaDataBuffer)) throw new IllegalStateException("Underlying buffer isn't instance of BaseCudaDataBuffer");

        // For buffer registration we're always using full underlying buffer
        AllocationShape shape = AllocationUtils.buildAllocationShape(array); /*new AllocationShape();
        shape.setOffset(0);
        shape.setStride(1);
        shape.setLength(array.data().length());
        shape.setDataType(Nd4j.dataType());
*/

        DataBuffer buffer = array.data().originalDataBuffer() == null ? array.data() : array.data().originalDataBuffer();

        return pickupSpan(buffer, shape);
    }

    /**
     * This method hints allocator, that specific object was accessed on host side.
     * This includes putRow, putScalar;
     *
     * @param array
     */
    @Override
    public void tickHost(INDArray array) {
        // TODO: to be implemented, probably
    }

    /**
     * This methods hints allocator, that specific object was accessed on device side.
     *
     * @param array
     */
    @Override
    public void tickDevice(INDArray array) {
        // TODO: to be implemented, probably
    }

    /**
     * This method registers array's buffer within allocator instance
     *
     * PLEASE NOTE: This is debug method, and it shouldn't be used in any circumstances besides tests.
     *
     * @param object Object to be picked & tracked
     */
    protected Long pickupSpan(Object object) {
        return null;
    }


    /**
     * This method hints allocator, that specific object was released on device side
     *
     * @param array
     */
    @Override
    public void tackDevice(INDArray array) {
//        log.info("tackDevice(INDArray)");

        DataBuffer buffer = array.data().originalDataBuffer() == null ? array.data() : array.data().originalDataBuffer();

        tackDevice(buffer, AllocationUtils.buildAllocationShape(array));

        if (array.shapeInfoDataBuffer().getTrackingPoint() != null) {
            tackDevice(array.shapeInfoDataBuffer(), AllocationUtils.buildAllocationShape(array.shapeInfoDataBuffer()));
        }
    }




    /**
     * This method hints allocator, that specific object was released on device side
     *
     * @param buffer
     * @param shape
     */
    protected void tackDevice(DataBuffer buffer, AllocationShape shape) {
        AllocationPoint point = getAllocationPoint(buffer, shape, true);

        point.getAccessState().requestTack();

        //point.tickDeviceWrite();
    }


    /**
     * This method notifies allocator, that specific object was changed on device side
     *
     * @param array
     */
    @Override
    public void tickDeviceWrite(INDArray array) {
        //log.info("Tick device write!");
        DataBuffer buffer = array.data().originalDataBuffer() == null ? array.data() : array.data().originalDataBuffer();

        AllocationPoint point = getAllocationPoint(buffer, AllocationUtils.buildAllocationShape(array), true);

        point.tickDeviceWrite();
    }

    /**
     * This method notifies allocator, that specific object was changed on host side
     *
     * @param array
     */
    @Override
    public void tickHostWrite(INDArray array) {
        DataBuffer buffer = array.data().originalDataBuffer() == null ? array.data() : array.data().originalDataBuffer();

        AllocationPoint point = getAllocationPoint(buffer, AllocationUtils.buildAllocationShape(array), true);

        if (point == null) {
//            log.info("tickHostWrite INDarray");
            pickupSpan(array);
        }

        point.tickHostWrite();
    }

    /**
     * This method returns actual device pointer valid for current object
     *
     * TODO: this method should be removed.
     * @param objectId
     */
    @Override
    @Deprecated
    public Pointer getPointer(DataBuffer objectId) {
        return getPointer(objectId, AllocationUtils.buildAllocationShape(objectId), false);
    }

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param buffer
     * @param shape
     * @param isView
     */
    @Override
    public Pointer getPointer(DataBuffer buffer, AllocationShape shape, boolean isView) {
     //   log.info("requesting pointer for: [" + shape + "]; isView: [" + isView +"]");
        /*
            We assume that object is registered within allocator
         */
        AllocationPoint point = getAllocationPoint(buffer, shape, true);

        boolean isNewAllocation = false;

        Long trackingPoint = buffer.getTrackingPoint();

        // we're checking, if cuda pointer is null without any locks. but if it's null, we'll request Toe state on this allocation, to make sure nothing can mess with it
        if (point.getDevicePointer() == null) {
            //log.info("Building pointer");
            // at this point memory becomes read/write-locked for a few ms, to make sure cudaPointer exists
            point.getAccessState().requestToe();

            if (point.getDevicePointer() == null) {
                /*
                    If pointer is null, that means we're on first stage of allocation, so we need to allocate Zero memory
                    PLEASE NOTE: Also, if this is a view - we allocate full underlying buffer on first call, not a shape
                */

                AllocationShape internalShape = isView? AllocationUtils.buildAllocationShape(buffer) : shape;

                /*
                    Before allocating anything, we must ensure that we have enough space left
                 */
                long requiredMemory = AllocationUtils.getRequiredMemory(internalShape);
                while (zeroUseCounter.get() > configuration.getMaximumZeroAllocation() - (configuration.getMaximumZeroAllocation() / 10)) {
                    log.warn("No free host memory available. Starting GC manually with [URGENT] agressiveness");
//                    if (zeroUseCounter.get() > configuration.getMaximumZeroAllocation() - (configuration.getMaximumZeroAllocation() / 10)) {
                    long freedMemory = seekUnusedZero(Thread.currentThread().getId(), Aggressiveness.URGENT);
//                    } else {

//                    }
                }
                /*
                    We intentionally update counter prior to allocation
                 */
                zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(internalShape));

                /*
                    now it's ALMOST safe to allocate zero-copy memory.
                    Technically it's still possible to fail there, with oom or CUDA-originated exception
                 */
                point.setAllocationStatus(AllocationStatus.HOST);

                PointersPair info = mover.alloc(AllocationStatus.HOST, point, internalShape);

                long allocCnt = allocationsCounter.incrementAndGet();
                zeroAllocations.get(Thread.currentThread().getId()).put(trackingPoint, trackingPoint);
                if (allocCnt % 10000 == 0)
                    log.debug("Total zero allocations happened: [" + allocCnt + "]; active zero allocations: ["+ zeroAllocations.get(Thread.currentThread().getId()).size()+"]");

                /*
                    it's safe to remove this check in production environment
                 */
                if (info == null)
                    throw new IllegalStateException("Zero-copy allocation failed");

                point.setPointers(info);




                /*
                    Copy data from host buffer to device
                 */
                // We don't need separate call here, we wrap that inside alloc call
                // mover.copyforward(point);
            } else {
                /*
                    do nothing here, the only possible reason for us to get in this scope, is concurrent getPointer access, so it was stopped by TTT barrier, and now we're here after everything being done
                  */
                ;
            }

            isNewAllocation = true;

            point.getAccessState().releaseToe();
        };

        /*
            Before coming to promotion, we should decide, if we need to synchronize data on device
         */
        if (!isNewAllocation) {
            if (!point.isActualOnDeviceSide()) {
                // update data in Toe state
                point.getAccessState().requestToe();

                if (!point.isActualOnDeviceSide()) {
                    //log.info("Calling for copyforward on: " + shape);
                    mover.copyforward(point, shape);
                }
                // we set device access time equal to host write time
                point.tickDeviceToHost();

                point.getAccessState().releaseToe();
            }
        }

        /*
            So, right now we are guaranteed to have cudaPointer. We can decide now, if this memory chunk should be promoted or not.
         */
        if (!isNewAllocation && !isView) {
            // we check promotion only for existant allocations. just ignore new allocations here :)
            // TODO: add memory check all the way here
            long requiredMemory = AllocationUtils.getRequiredMemory(shape);
            if (point.getDeviceTicks() > configuration.getMinimumRelocationThreshold() && point.getAllocationStatus() == AllocationStatus.HOST && requiredMemory < configuration.getMaximumSingleAllocation()) {

                // before doing actual promotion, we check to our tracker, to minimize cuda driver calls as well
                if (deviceMemoryTracker.reserveAllocationIfPossible(Thread.currentThread().getId(), point.getDeviceId(), requiredMemory) && mover.pingDeviceForFreeMemory(point.getDeviceId(), requiredMemory)) {
                    point.getAccessState().requestToe();
                    //     log.info("Starting promotion");

                    // moving memory from ZERO to DEVICE
                    promoteObject(trackingPoint, point, shape);

                    point.getAccessState().releaseToe();
                }
            }
        }

        /*
            after everything was done here - register tick, and return the pointer to outer context
         */
        point.getAccessState().requestTick();
        point.tickDevice();

        /*
            Now we store use rates
         */

        if (point.getAllocationStatus() == AllocationStatus.HOST) {
            zeroLong.store(point.getTimerLong().getFrequencyOfEvents());
            zeroShort.store(point.getTimerShort().getFrequencyOfEvents());
        } else {
            deviceLong.store(point.getTimerLong().getFrequencyOfEvents());
            deviceShort.store(point.getTimerShort().getFrequencyOfEvents());
        }

        /*
            Now we should return pointer with proper offset
         */
        Pointer pointer = null;
        if (shape.getOffset() > 0) {
        //    log.info("Offest: " + AllocationUtils.getByteOffset(shape));
            //withByteOffset(AllocationUtils.getByteOffset(shape));
            // FIXME: get back offset considerations ^^^
            pointer = point.getDevicePointer();
        } else pointer = point.getDevicePointer();

    //    log.info("Pointer GO: " + pointer.getNativePointer());

        return pointer;
    }

    /**
     * This method returns actual device pointer valid for specified INDArray
     *
     * @param array
     */
    @Override
    public Pointer getPointer(INDArray array) {
        AllocationShape shape = AllocationUtils.buildAllocationShape(array);

        DataBuffer buffer = array.data().originalDataBuffer() != null ? array.data().originalDataBuffer() : array.data();

        if (buffer.getTrackingPoint() == null) {
            pickupSpan(array);
        }

        return getPointer(buffer, shape, array.isView());
    }

    /**
     * This method moves specific object from zero-copy memory to device memory
     *
     * PLEASE NOTE: This method should be kept private, and never exposed to public
     *
     * @param trackingPoint
     * @param point
     * @param shape
     * @return
     */
    // TODO: this method should be moved into CudaZeroMover implementation
    protected boolean promoteObject(Long trackingPoint, AllocationPoint point, AllocationShape shape) {
        try {
            long threadId = Thread.currentThread().getId();

            PointersPair newPointers = mover.alloc(AllocationStatus.DEVICE, point, shape);

            point.setAllocationStatus(AllocationStatus.DEVICE);
            point.setPointers(newPointers);

            deviceLock.readLock().lock();

            deviceAllocations.get(threadId, point.getDeviceId()).put(trackingPoint, trackingPoint);

            deviceLock.readLock().unlock();

            zeroAllocations.get(threadId).remove(trackingPoint);

            deviceMemoryTracker.addToAllocation(threadId, point.getDeviceId(), AllocationUtils.getRequiredMemory(shape));

            zeroUseCounter.set(zeroUseCounter.get() - AllocationUtils.getRequiredMemory(point.getShape()));

//                    log.info("Relocation happened!");
        } catch (Exception e){
            if (1>0) throw new RuntimeException(e);
            return false;
        }

        return true;
    }

    /**
     * This method returns actual host pointer, valid for specified shape of current object
     *
     * @param array
     * @return
     */

    @Deprecated
    public Pointer getHostPointer(INDArray array) {
        /*
        if(array.data().allocationMode() == DataBuffer.AllocationMode.DIRECT || array.data().allocationMode() == DataBuffer.AllocationMode.JAVACPP)
            return Pointer.to(array.data().asNio());
        else {
           switch(array.data().dataType()) {
               case INT: return Pointer.to(array.data().asInt());
               case DOUBLE: return Pointer.to(array.data().asDouble());
               case FLOAT: return Pointer.to(array.data().asFloat());
           }
        }
        */
        throw new UnsupportedOperationException("getHostPointer() was deprecated");
    }


    /**
     * This method should be called to make sure that data on host side is actualized
     *
     * @param buffer
     */
    protected void synchronizeHostData(DataBuffer buffer, AllocationShape shape) {
        AllocationPoint point = getAllocationPoint(buffer, shape, true);
        //log.info("Synchronize called on buffer with shape: " + shape);

        /*
            We set memory state to Toe, and issue copyback if required
         */

//        log.info("Current state: " + point.getAccessState().getCurrentState());
        if (!point.isActualOnHostSide() || point.getAccessState().getCurrentState() != AccessState.TACK) {

            point.getAccessState().requestToe();

            if (!point.isActualOnHostSide()) {
                //log.info("Data isn't actual on host side, copyback() started");
                mover.copyback(point, shape);

                // update the timer for hostRead
                point.tickHostRead();
            }// else log.info("Data is actual 2 , skipping sync");

            point.getAccessState().releaseToe();
        }// else log.info("Data is actual 1, skipping sync");
    }

    /**
     * This method should be callsd to make sure that data on host side is actualized.
     * However, this method only tries to lock data before synchronization.
     * <p>
     * PLEASE NOTE: This method is considered UNSAFE.
     *
     * @param syncBuffer
     */
    @Override
    public void trySynchronizeHostData(DataBuffer syncBuffer) {
        DataBuffer buffer =  syncBuffer.originalDataBuffer() == null ? syncBuffer : syncBuffer.originalDataBuffer();

        AllocationPoint point = getAllocationPoint(buffer, AllocationUtils.buildAllocationShape(buffer), false);
        //log.info("trySync on shape: " + AllocationUtils.buildAllocationShape(buffer));

        if (point != null && !point.isActualOnHostSide()) {
            //log.info("Try hit");
            if (point.getAccessState().tryRequestToe()) {
                // log.info("Try copyback");
                mover.copyback(point, AllocationUtils.buildAllocationShape(buffer));

                // update the timer for hostRead
                point.tickHostRead();

                point.getAccessState().releaseToe();
            }// else log.info("Toe is busy, skipping");
        }
    }

    /**
     * This method should be called to make sure that data on host side is actualized
     *
     * @param array
     */
    @Override
    public void synchronizeHostData(INDArray array) {
        //log.info("Synchronize called on array");
        DataBuffer buffer = array.data().originalDataBuffer() == null ? array.data() : array.data().originalDataBuffer();

        AllocationPoint point = getAllocationPoint(buffer, AllocationUtils.buildAllocationShape(array), true);

        if (point == null) {
            pickupSpan(array);
        }

        synchronizeHostData(buffer, AllocationUtils.buildAllocationShape(array));
    }

    /**
     * This method should be callsd to make sure that data on host side is actualized
     *
     * @param buffer
     */

    @Override
    public void synchronizeHostData(DataBuffer buffer) {
        //log.info("Synchronize called on buffer");
        DataBuffer fbuffer = buffer.originalDataBuffer() == null ? buffer : buffer.originalDataBuffer();

        AllocationPoint point = getAllocationPoint(fbuffer, AllocationUtils.buildAllocationShape(fbuffer), true);

        if (point == null) {
            pickupSpan(fbuffer);
        }

        synchronizeHostData(fbuffer, AllocationUtils.buildAllocationShape(fbuffer));
    }

    /**
     * This method returns current host memory state
     *
     * @param array
     * @return
     */
    @Override
    public SyncState getHostMemoryState(INDArray array) {
        /*
            basically we just want to compare two access time values: device & host.
            we can't know, if memory was changed on device side or not
          */

        /*
            TODO: improvement is possible here ->
             as soon as we'll have partial allocations available, we can have partially synced memory
         */
        AllocationPoint point = getAllocationPoint(null);
        if (point.isActualOnHostSide() == true) {
            return SyncState.SYNC;
        } else {
            return SyncState.DESYNC;
        }
    }

    /**
     * This method returns the number of top-level memory allocation.
     * No descendants are included in this result.
     *
     * @return number of allocated top-level memory chunks
     */
    @Override
    public int tableSize() {
        return allocationsMap.size();
    }

    /**
     * This method returns CUDA deviceId for specified buffer
     *
     * @param array
     * @return
     */
    @Override
    public Integer getDeviceId(INDArray array) {
        AllocationPoint point = getAllocationPoint(array.data().originalDataBuffer(), AllocationUtils.buildAllocationShape(array), true);

        if (point == null || point.getDeviceId() == null)
            throw new IllegalStateException("deviceId for point is undefined");

        return point.getDeviceId();
    }

    @Deprecated
    protected void initCudaContextForThread(Long threadId) {
        /*
            this method is called from write-locked region, but its backward call falls into lock-free branch, since deviceAffinity is already defined
         */

        // we set device to be used prior to stream creation
        /*
        JCuda.cudaSetDevice(getDeviceId());

        CudaContext context = new CudaContext();
        context.initHandle();
        context.initOldStream();
        context.initStream();
        context.associateHandle();
        contextPool.put(threadId, context);
        */
    }

    /**
     * This method returns CUDA deviceId for current thread
     *
     * @return
     */
    @Override
    public Integer getDeviceId() {
        Long threadId = Thread.currentThread().getId();

        if (!devicesAffinity.containsKey(threadId)) {
            try {
                deviceLock.writeLock().lock();

                if (!devicesAffinity.containsKey(threadId)) {
                    wasInitialised.compareAndSet(false, true);

                    /*
                    // Random-based device selection
                    List<Integer> devices = new ArrayList<>(environment.getAvailableDevices().keySet());
                    Random rnd = new Random();
                    Integer device = devices.get(rnd.nextInt(devices.size()));
                    */

                    // sequental device selection for better balance
                    List<Integer> devices = new ArrayList<>(environment.getAvailableDevices().keySet());
                    Integer device = devices.get(devPtr.getAndIncrement());
                    if (devPtr.get() >= devices.size())
                        devPtr.set(0);


                    devicesAffinity.put(threadId, device);

                    if (!zeroAllocations.containsKey(threadId)) {
                        // TODO: investigate CopyOnWriteArrayList here, _PROBABLY_ we could replace it with synchronized list, without backing
                        zeroAllocations.put(threadId, new ConcurrentHashMap<Long, Long>());
                    }

                    if (!deviceAllocations.contains(threadId, device)) {
                        deviceAllocations.put(threadId, device, new ConcurrentHashMap<Long, Long>());
                    }

                    log.info("Mapping device [" + device + "] to thread [" + Thread.currentThread().getId() + "]");

                    //initCudaContextForThread(threadId);
                    mover.initializeDevice(threadId, device);



                    ZeroGarbageCollectorThread thread = new ZeroGarbageCollectorThread(threadId, device, shouldStop);
                    thread.start();
                    collectorsZero.put(threadId, thread);

                    DeviceGarbageCollectorThread dThread = new DeviceGarbageCollectorThread(threadId, device, shouldStop);
                    dThread.start();
                    collectorsDevice.put(threadId, dThread);
                }
                return devicesAffinity.get(threadId);
            } finally {
                deviceLock.writeLock().unlock();
            }
        } else devicesAffinity.get(Thread.currentThread().getId());

        return devicesAffinity.get(threadId);
    }

    /**
     * This method allocates required chunk of memory
     *
     * @param requiredMemory
     */
    @Override
    public AllocationPoint allocateMemory(AllocationShape requiredMemory) {
        return allocateMemory(requiredMemory, mover.getInitialLocation());
    }

    /**
     * This method allocates required chunk of memory in specific location
     * <p>
     * PLEASE NOTE: Do not use this method, unless you're 100% sure what you're doing
     *
     * @param requiredMemory
     * @param location
     */
    @Override
    public AllocationPoint allocateMemory(AllocationShape requiredMemory, AllocationStatus location) {
        throw new UnsupportedOperationException("Not implemented yet");
    }


    protected AllocationPoint getAllocationPoint(DataBuffer buffer, AllocationShape shape, boolean catchNewAllocations) {
        Long trackingPointer = buffer.getTrackingPoint();

        if (trackingPointer == null) { // AllocationUtils.buildAllocationShape(objectId)
            if (catchNewAllocations) {
//                log.info("Registering");
                trackingPointer = pickupSpan(buffer, shape);
            } else return null;
        }

        // that's a temporary exception, we'll change that to re-ack later
        if (trackingPointer == null)
            throw new IllegalStateException("trackingPointer is NULL");


        AllocationPoint point = getAllocationPoint(trackingPointer);

        // temporary exception too
        if (point == null)
            throw new IllegalStateException("AllocationPoint is NULL");


        return point;
    }

    protected AllocationPoint getAllocationPoint(Long objectId) {
        return allocationsMap.get(objectId);
    }

    /**
     * This method explicitly removes object from zero-copy memory.
     *
     * @param threadId
     * @param objectId
     * @param copyback  if TRUE, corresponding memory block on JVM side will be updated, if FALSE - memory will be just discarded
     */
    protected void purgeZeroObject(Long threadId, Long objectId, AllocationPoint point, boolean copyback) {
        if (copyback) {

            // copyback here
            mover.copyback(point, point.getShape());

            externalsLock.writeLock().lock();

            externalBuffers.remove(point.getBuffer());

            externalsLock.writeLock().unlock();
        }
        zeroAllocations.get(threadId).remove(objectId);
        allocationsMap.remove(objectId);

        // we call for caseless deallocation here
        mover.free(point, point.getAllocationStatus());
        point.setAllocationStatus(AllocationStatus.DEALLOCATED);

        zeroUseCounter.set(zeroUseCounter.get() - AllocationUtils.getRequiredMemory(point.getShape()));
    }

    /**
     * This method explicitly removes object from device memory.
     *
     * @param threadId
     * @param objectId
     * @param copyback  if TRUE, corresponding memory block on JVM side will be updated, if FALSE - memory will be just discarded
     */
    protected void purgeDeviceObject(Long threadId, Integer deviceId, Long objectId, AllocationPoint point, boolean copyback) {
        if (copyback) {
            // copyback here basically means that we're gonna have new zero allocation right now
            mover.fallback(point, point.getShape());

            zeroAllocations.get(threadId).put(objectId, objectId);
            point.tickDevice();
            point.tickDeviceWrite();
            point.setAllocationStatus(AllocationStatus.HOST);
            zeroUseCounter.set(zeroUseCounter.get() - AllocationUtils.getRequiredMemory(point.getShape()));
        }

        deviceLock.readLock().lock();
        Map<Long, Long> allocations = deviceAllocations.get(threadId, deviceId);
        deviceLock.readLock().unlock();

        allocations.remove(objectId);

        deviceMemoryTracker.subFromAllocation(threadId, deviceId, AllocationUtils.getRequiredMemory(point.getShape()));

        if (!copyback) {
            allocationsMap.remove(objectId);
            mover.free(point, AllocationStatus.DEVICE);
        }

        environment.trackAllocatedMemory(deviceId, AllocationUtils.getRequiredMemory(point.getShape()));
    }

    /**
     * This method seeks for unused zero-copy memory allocations
     *
     * @param threadId Id of the thread, retrieved via Thread.currentThread().getId()
     * @return size of memory that was deallocated
     */
    protected synchronized long seekUnusedZero(Long threadId, Aggressiveness aggressiveness) {
        /*
            This method is blocking on thread basis, just to prevent parallel calls

            TODO: To prevent cyclic calls we need something smart here
         */
        AtomicLong freeSpace = new AtomicLong(0);

        int totalElements = zeroAllocations.get(threadId).size();
        log.debug("Total zero elements to be checked: [" + totalElements + "]; zeroUsed: ["+ zeroUseCounter.get()+"]");

        float shortAverage = zeroShort.getAverage();
        float longAverage = zeroLong.getAverage();

        float shortThreshold = shortAverage / (Aggressiveness.values().length - aggressiveness.ordinal());
        float longThreshold = longAverage / (Aggressiveness.values().length - aggressiveness.ordinal());



        AtomicInteger elementsDropped = new AtomicInteger(0);

        for (Long object: zeroAllocations.get(threadId).keySet()) {
            AllocationPoint point = getAllocationPoint(object);

            if (point.getAccessState().isToeAvailable()) {
                point.getAccessState().requestToe();

                /*
                    Check if memory points to non-existant buffer, using externals.
                    If externals don't have specified buffer - delete reference.
                 */
                if (point.getBuffer() == null) {
                    //log.info("Ghost reference removed: " + object);

                    purgeZeroObject(threadId, object, point, false);
                    freeSpace.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()));
                    elementsDropped.incrementAndGet();
                    continue;
                }

                /*
                    Check, if memory can be removed from allocation.
                    To check it, we just compare average rates for few tens of latest calls
                 */

                long millisecondsTTL = configuration.getMinimumTTLMilliseconds();
                if (point.getRealDeviceAccessTime() < System.currentTimeMillis() - millisecondsTTL) {
                    // we could remove device allocation ONLY if it's older then minimum TTL
                    if (point.getTimerLong().getFrequencyOfEvents() < longThreshold && point.getTimerShort().getFrequencyOfEvents() < shortThreshold) {
                        //log.info("Removing object: " + object);

                        purgeZeroObject(threadId, object, point, true);
                        freeSpace.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()));
                        elementsDropped.incrementAndGet();
                    }
                }

                point.getAccessState().releaseToe();
            }
        }



        log.debug("Short average: ["+shortAverage+"], Long average: [" + longAverage + "]");
        log.debug("Aggressiveness: ["+ aggressiveness+"]; Short threshold: ["+shortThreshold+"]; Long threshold: [" + longThreshold + "]");
        log.debug("Zero elements deleted: " + elementsDropped.get());

        /*
            o.n.j.a.i.AtomicAllocator - Short average: [2.29], Long average: [0.3816667]
            o.n.j.a.i.AtomicAllocator - Aggressiveness: [PEACEFUL]; Short threshold: [0.5725]; Long threshold: [0.09541667]
            o.n.j.a.i.AtomicAllocator - Elements deleted: 17485


            o.n.j.a.i.AtomicAllocator - Short average: [1.0566667], Long average: [0.14388889]
            o.n.j.a.i.AtomicAllocator - Aggressiveness: [REASONABLE]; Short threshold: [0.35222223]; Long threshold: [0.047962964]
            o.n.j.a.i.AtomicAllocator - Elements deleted: 18214

            o.n.j.a.i.AtomicAllocator - Short average: [1.4866666], Long average: [0.19944443]
            o.n.j.a.i.AtomicAllocator - Aggressiveness: [URGENT]; Short threshold: [0.7433333]; Long threshold: [0.099722214]
            o.n.j.a.i.AtomicAllocator - Elements deleted: 18933

            o.n.j.a.i.AtomicAllocator - Short average: [1.6933333], Long average: [0.28222224]
            o.n.j.a.i.AtomicAllocator - Aggressiveness: [IMMEDIATE]; Short threshold: [1.6933333]; Long threshold: [0.28222224]
            o.n.j.a.i.AtomicAllocator - Elements deleted: 18169
         */


        //log.info("Total zero elements left: " + zeroAllocations.get(threadId).size());
        return freeSpace.get();
    }

    /**
     * This method seeks for unused device memory allocations, for specified thread and device
     *
     * @param threadId Id of the thread, retrieved via Thread.currentThread().getId()
     * @param deviceId Id of the device
     * @return size of memory that was deallocated
     */
    protected long seekUnusedDevice(Long threadId, Integer deviceId, Aggressiveness aggressiveness) {
        AtomicLong freeSpace = new AtomicLong(0);

        deviceLock.readLock().lock();
        Map<Long,Long> allocations = deviceAllocations.get(threadId, deviceId);
        deviceLock.readLock().unlock();

        float shortAverage = deviceShort.getAverage();
        float longAverage = deviceLong.getAverage();

        float shortThreshold = shortAverage / (Aggressiveness.values().length - aggressiveness.ordinal());
        float longThreshold = longAverage / (Aggressiveness.values().length - aggressiveness.ordinal());

        log.debug("Total device elements: " + allocations.size());

        AtomicInteger elementsDropped = new AtomicInteger(0);
        AtomicInteger elementsMoved = new AtomicInteger(0);

        for (Long object: allocations.keySet()) {
            AllocationPoint point = getAllocationPoint(object);

            if (point.getAccessState().isToeAvailable()) {
                point.getAccessState().requestToe();

                /*
                    Check if memory points to non-existant buffer, using externals.
                    If externals don't have specified buffer - delete reference.
                 */
                if (point.getBuffer() == null) {
                    //log.info("Ghost reference removed: " + object);

                    purgeDeviceObject(threadId, deviceId, object, point, false);
                    freeSpace.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()));

                    elementsDropped.incrementAndGet();
                    continue;
                }

                /*
                    Check, if memory can be removed from allocation.
                    To check it, we just compare average rates for few tens of latest calls
                 */

                long millisecondsTTL = configuration.getMinimumTTLMilliseconds();
                if (point.getRealDeviceAccessTime() < System.currentTimeMillis() - millisecondsTTL) {
                    // we could remove device allocation ONLY if it's older then minimum TTL
                    if (point.getTimerLong().getFrequencyOfEvents() < longThreshold && point.getTimerShort().getFrequencyOfEvents() < shortThreshold) {
                        //log.info("Removing object: " + object);

                        purgeDeviceObject(threadId, deviceId, object, point, true);

                        freeSpace.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()));

                        elementsMoved.incrementAndGet();

                        purgeZeroObject(threadId, object, point, true);
                    }
                }

                point.getAccessState().releaseToe();
            }
        }

        log.debug("Thread/Device ["+ threadId+"/"+deviceId+"] elements purged: [" + elementsDropped.get()+"]; Relocated: ["+ elementsMoved.get()+"]; Device objects left: ["+allocations.size()+"]");

        return freeSpace.get();
    }


    private class ZeroGarbageCollectorThread extends Thread implements Runnable {

        private final Long threadId;
        private final Integer deviceId;
        private final AtomicBoolean terminate;

        public ZeroGarbageCollectorThread(Long threadId, Integer deviceId, AtomicBoolean terminate) {
            this.threadId = threadId;
            this.deviceId = deviceId;
            this.terminate = terminate;

            this.setName("zero gc thread " + threadId);
            this.setDaemon(true);
        }

        @Override
        public void run() {
            log.debug("Starting zero GC for device: " + deviceId);
            while (!terminate.get()) {

                /*
                    Check for zero-copy garbage
                 */
                //   log.info("ZeroGC started...");
                /*
                    We want allocations to take in account multiple things:
                    1. average access rates for last X objects
                    2. total number of currently allocated objects
                    3. total allocated memory size
                    4. desired aggressiveness
                */
                try {
                    Thread.sleep(Math.max(configuration.getMinimumTTLMilliseconds(), 5000));
                } catch (Exception e) {
                    // we can have interruption here, to force gc
                    ;
                }

                Aggressiveness aggressiveness = configuration.getHostDeallocAggressiveness();

                // if we have too much objects, or total allocated memory has met 75% of max allocation - use urgent mode
                if ((zeroAllocations.get(threadId).size() > 500000 || zeroUseCounter.get() > (configuration.getMaximumZeroAllocation() * 0.75)) && aggressiveness.ordinal() < Aggressiveness.URGENT.ordinal())
                    aggressiveness = Aggressiveness.URGENT;

                if (zeroUseCounter.get() > (configuration.getMaximumZeroAllocation() * 0.85))
                    aggressiveness = Aggressiveness.IMMEDIATE;

                if (zeroUseCounter.get() < (configuration.getMaximumZeroAllocation() * 0.25) && (zeroAllocations.get(threadId).size() < 500)) {
                    ; // i don't want deallocation to be fired on lower thresholds. just no sense locking stuff
                    //log.debug("Skipping zero GC round: ["+zeroUseCounter.get()+"/" +zeroAllocations.get(threadId).size() + "]");
                }  else seekUnusedZero(threadId, aggressiveness);
            }
        }
    }

    private class DeviceGarbageCollectorThread extends Thread implements Runnable {

        private final Long threadId;
        private final Integer deviceId;
        private final AtomicBoolean terminate;

        public DeviceGarbageCollectorThread(Long threadId, Integer deviceId, AtomicBoolean terminate) {
            this.threadId = threadId;
            this.deviceId = deviceId;
            this.terminate = terminate;
            this.setName("device gc thread " + threadId + "/" + deviceId);
            this.setDaemon(true);
        }

        @Override
        public void run() {
            log.debug("Starting device GC for device: " + deviceId);
            while (!terminate.get()) {
                /*
                    Check for device garbage
                 */

                try {
                    Thread.sleep(Math.max(configuration.getMinimumTTLMilliseconds(), 5000));
                } catch (Exception e) {
                    // we can have interruption here, to force gc

                }


                /*
                if(deviceMemoryTracker == null)
                    continue;
                */

                //log.info("DeviceGC started...");
                Aggressiveness aggressiveness = configuration.getGpuDeallocAggressiveness();

                // if we have too much objects, or total allocated memory has met 75% of max allocation - use urgent mode
                if (deviceAllocations != null && deviceAllocations.contains(threadId,deviceId) && (deviceAllocations.get(threadId, deviceId).size() > 100000 || deviceMemoryTracker.getAllocatedSize(threadId, deviceId)> (configuration.getMaximumDeviceAllocation() * 0.75)) && aggressiveness.ordinal() < Aggressiveness.URGENT.ordinal())
                    aggressiveness = Aggressiveness.URGENT;

                if (deviceMemoryTracker.getAllocatedSize(threadId, deviceId) > (configuration.getMaximumDeviceAllocation() * 0.85))
                    aggressiveness = Aggressiveness.IMMEDIATE;

                if (deviceMemoryTracker.getAllocatedSize(threadId, deviceId) < (configuration.getMaximumDeviceAllocation() * 0.25) && (deviceAllocations.get(threadId, deviceId).size()  < 500)) {
                    // i don't want deallocation to be fired on lower thresholds. just no sense locking stuff
              //      log.debug("Skipping device GC round: ["+deviceMemoryTracker.getAllocatedSize(threadId, deviceId) +"/"+deviceAllocations.get(threadId, deviceId).size()+"]");
                } else seekUnusedDevice(this.threadId, this.deviceId, aggressiveness);


            }
        }
    }


    /**
     * This method returns the number of tracked zero-copy allocations
     *
     * @return
     */
    protected int getTotalZeroAllocations() {
        if (zeroAllocations.get(Thread.currentThread().getId()) != null) {
            return zeroAllocations.get(Thread.currentThread().getId()).size();
        } else return 0;
    }

    /**
     * This method returns the number of all tracked memory chunks
     *
     * @return
     */
    protected int getTotalTrackingPoints() {
        return allocationsMap.size();
    }

    protected long getTotalAllocatedDeviceMemory(Integer deviceId) {
        return deviceMemoryTracker.getAllocatedSize(deviceId);
    }
}
