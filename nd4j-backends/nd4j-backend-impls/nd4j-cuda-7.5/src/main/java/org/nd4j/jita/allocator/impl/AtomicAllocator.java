package org.nd4j.jita.allocator.impl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import lombok.Getter;
import lombok.NonNull;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.concurrency.DeviceAllocationsTracker;
import org.nd4j.jita.allocator.context.ExternalContext;
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
import org.nd4j.jita.mover.MemoryHandler;
import org.nd4j.jita.mover.CudaZeroHandler;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
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
    @Getter private transient MemoryHandler memoryHandler;
    private AtomicLong allocationsCounter = new AtomicLong(0);

    private AtomicLong objectsTracker = new AtomicLong(Long.MIN_VALUE);

    // we have single tracking point for allocation points, since we're not going to cycle through it it any time soon
    private Map<Long, AllocationPoint> allocationsMap = new ConcurrentHashMap<>();

    /*
        WeakHashMap for buffer->id conversion. If DataBuffer get's removed by jvm GC, we'll know that.
        So, just a short way for reverse lookup, that causes no GC issues.
     */
    private Map<DataBuffer, Long> externalBuffers = Collections.synchronizedMap(new WeakHashMap<DataBuffer, Long>());


    private static Logger log = LoggerFactory.getLogger(AtomicAllocator.class);

    /*
        locks for internal resources
     */
    private ReentrantReadWriteLock globalLock = new ReentrantReadWriteLock();
    private ReentrantReadWriteLock externalsLock = new ReentrantReadWriteLock();

    /*
        here we have handles for garbage collector threads
        ThreadId, GarbageCollector
     */
    private Map<Long, ZeroGarbageCollectorThread> collectorsZero = new ConcurrentHashMap<>();
    private Map<Integer, DeviceGarbageCollectorThread> collectorsDevice = new ConcurrentHashMap<>();

    private final AtomicBoolean shouldStop = new AtomicBoolean(false);

    private final AtomicBoolean wasInitialised = new AtomicBoolean(false);

    private final Ring deviceLong = new LockedRing(30);
    private final Ring deviceShort = new LockedRing(30);

    private final Ring zeroLong = new LockedRing(30);
    private final Ring zeroShort = new LockedRing(30);

    public static AtomicAllocator getInstance() {
        return INSTANCE;
    }

    protected AtomicAllocator() {
        environment = new CudaEnvironment(configuration);
        this.memoryHandler = new CudaZeroHandler();
        this.memoryHandler.init(configuration, environment, this);

        initDeviceCollectors();
    }

    protected void initDeviceCollectors() {
        for (Integer deviceId : this.memoryHandler.getAvailableDevices()) {

            DeviceGarbageCollectorThread dThread = new DeviceGarbageCollectorThread(deviceId, shouldStop);
            dThread.start();
            collectorsDevice.put(deviceId, dThread);
        }
    }

    /**
     * This method returns CudaContext for current thread
     *
     * @return
     */
    @Override
    public ExternalContext getDeviceContext() {
        // FIXME: proper lock avoidance required here
        return memoryHandler.getDeviceContext();
/*

        */
    }

    /**
     * This method specifies Mover implementation to be used internally
     * @param memoryHandler
     */
    public void setMemoryHandler(@NonNull MemoryHandler memoryHandler) {
        globalLock.writeLock().lock();

        this.memoryHandler = memoryHandler;
        this.memoryHandler.init(configuration, environment, this);

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

//        this.deviceMemoryTracker = new DeviceAllocationsTracker(this.environment, this.configuration);

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
     * This method returns actual device pointer valid for current object
     *
     * @param objectId
     */
    @Override
    public Pointer getPointer(DataBuffer buffer) {
        AllocationPoint point = ((BaseCudaDataBuffer) buffer).getAllocationPoint();

        if (point.getAllocationStatus() == AllocationStatus.HOST) {
            //zeroLong.store(point.getTimerLong().getFrequencyOfEvents());
            //zeroShort.store(point.getTimerShort().getFrequencyOfEvents());
        } else {
            deviceLong.store(point.getTimerLong().getFrequencyOfEvents());
            deviceShort.store(point.getTimerShort().getFrequencyOfEvents());
        }

        return memoryHandler.getDevicePointer(buffer);
        //return getPointer(buffer, AllocationUtils.buildAllocationShape(buffer), false);
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
        if (1 > 0) return memoryHandler.getDevicePointer(buffer);

        AllocationPoint point = ((BaseCudaDataBuffer) buffer).getAllocationPoint();

        if (point.getAllocationStatus() == AllocationStatus.HOST) {
       //     zeroLong.store(point.getTimerLong().getFrequencyOfEvents());
       //     zeroShort.store(point.getTimerShort().getFrequencyOfEvents());
        } else {
            deviceLong.store(point.getTimerLong().getFrequencyOfEvents());
            deviceShort.store(point.getTimerShort().getFrequencyOfEvents());
        }


        //log.info("requesting pointer for: [" + shape + "]; isView: [" + isView +"]");
        /*
            We assume that object is registered within allocator
         */

        Long trackingPoint = buffer.getTrackingPoint();

   //     log.info("Tracking Point for request: " + trackingPoint);

        //AllocationPoint point = getAllocationPoint(trackingPoint);

        boolean isNewAllocation = false;

        // we're checking, if cuda pointer is null without any locks. but if it's null, we'll request Toe state on this allocation, to make sure nothing can mess with it
        if (point.getDevicePointer() == null) {
            log.info("Building pointer");
            // at this point memory becomes read/write-locked for a few ms, to make sure cudaPointer exists
            point.getAccessState().requestToe();

            if (point.getDevicePointer() == null) {
                /*
                    If pointer is null, that means we're on first stage of allocation, so we need to allocate Zero memory
                    PLEASE NOTE: Also, if this is a view - we allocate full underlying buffer on first call, not a shape
                */

                AllocationShape internalShape = isView? AllocationUtils.buildAllocationShape(buffer) : shape;
                /*
                    now it's ALMOST safe to allocate zero-copy memory.
                    Technically it's still possible to fail there, with oom or CUDA-originated exception
                 */
                point.setAllocationStatus(AllocationStatus.HOST);

                PointersPair info = memoryHandler.alloc(AllocationStatus.HOST, point, internalShape);

                long allocCnt = allocationsCounter.incrementAndGet();
                //zeroAllocations.get(Thread.currentThread().getId()).put(trackingPoint, trackingPoint);
                //if (allocCnt % 10000 == 0)
                    //log.debug("Total zero allocations happened: [" + allocCnt + "]; active zero allocations: ["+ zeroAllocations.get(Thread.currentThread().getId()).size()+"]");

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
        /*
        if (!isNewAllocation) {
            if (!point.isActualOnDeviceSide()) {
                // update data in Toe state
                log.info("Requesting toe");
                point.getAccessState().requestToe();

                if (!point.isActualOnDeviceSide()) {
                    log.info("Calling for copyforward on: " + shape);
                    mover.copyforward(point, shape);
                }
                // we set device access time equal to host write time
                point.tickDeviceToHost();

                point.getAccessState().releaseToe();
            }
        }
        */

        /*
            So, right now we are guaranteed to have cudaPointer. We can decide now, if this memory chunk should be promoted or not.
         */
        /*
        if (!isNewAllocation && !isView) {
            // we check promotion only for existant allocations. just ignore new allocations here :)
            // TODO: add memory check all the way here
            long requiredMemory = AllocationUtils.getRequiredMemory(shape);
            if (point.getDeviceTicks() > configuration.getMinimumRelocationThreshold() && point.getAllocationStatus() == AllocationStatus.HOST && requiredMemory < configuration.getMaximumSingleAllocation()) {

                // before doing actual promotion, we check to our tracker, to minimize cuda driver calls as well
                if (deviceMemoryTracker.reserveAllocationIfPossible(Thread.currentThread().getId(), point.getDeviceId(), requiredMemory) && memoryHandler.pingDeviceForFreeMemory(point.getDeviceId(), requiredMemory)) {
                    point.getAccessState().requestToe();
                    //     log.info("Starting promotion");

                    // moving memory from ZERO to DEVICE
                    //promoteObject(trackingPoint, point, shape);

                    point.getAccessState().releaseToe();
                }
            }
        }
        */

        /*
            after everything was done here - register tick, and return the pointer to outer context
         */
        point.getAccessState().requestTick();
        point.tickDevice();

        /*
            Now we store use rates
         */



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

        return memoryHandler.getDevicePointer(array.data());
/*
        AllocationPoint point = ((BaseCudaDataBuffer) array.data()).getAllocationPoint();

        if (point.getAllocationStatus() == AllocationStatus.HOST) {
            point.getTimerLong().triggerEvent();
            point.getTimerShort().triggerEvent();

            zeroLong.store(point.getTimerLong().getFrequencyOfEvents());
            zeroShort.store(point.getTimerShort().getFrequencyOfEvents());
        } else {
            point.getTimerLong().triggerEvent();
            point.getTimerShort().triggerEvent();

            deviceLong.store(point.getTimerLong().getFrequencyOfEvents());
            deviceShort.store(point.getTimerShort().getFrequencyOfEvents());
        }
        */
    }



    /**
     * This method returns actual host pointer, valid for specified shape of current object
     *
     * @param array
     * @return
     */


    /**
     * This method should be called to make sure that data on host side is actualized
     *
     * @param array
     */
    @Override
    public void synchronizeHostData(INDArray array) {
        DataBuffer buffer = array.data().originalDataBuffer() == null ? array.data() : array.data().originalDataBuffer();
        synchronizeHostData(buffer);
    }

    /**
     * This method should be callsd to make sure that data on host side is actualized
     *
     * @param buffer
     */

    @Override
    public void synchronizeHostData(DataBuffer buffer) {
        if (memoryHandler.isDeviceDependant()) {
            AllocationPoint point = getAllocationPoint(buffer.getTrackingPoint());
            memoryHandler.synchronizeThreadDevice(Thread.currentThread().getId(), memoryHandler.getDeviceId(), point);
        }
    }


    /**
     * This method returns CUDA deviceId for specified buffer
     *
     * @param array
     * @return
     */
    public Integer getDeviceId(INDArray array) {
        return memoryHandler.getDeviceId();
    }


    /**
     * This method allocates required chunk of memory
     *
     * @param requiredMemory
     */
    @Override
    public AllocationPoint allocateMemory(AllocationShape requiredMemory) {
        if (!collectorsZero.containsKey(Thread.currentThread().getId())) {
            ZeroGarbageCollectorThread zThread = new ZeroGarbageCollectorThread(Thread.currentThread().getId(), memoryHandler.getDeviceId(), shouldStop);
            zThread.start();

            collectorsZero.put(Thread.currentThread().getId(), zThread);
        }

        AllocationPoint point = allocateMemory(requiredMemory, memoryHandler.getInitialLocation());

        return point;
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
        AllocationPoint point = new AllocationPoint();

        Long allocId = objectsTracker.getAndIncrement();

        point.setObjectId(allocId);
        point.setShape(requiredMemory);

        PointersPair pair = memoryHandler.alloc(location, point, requiredMemory);

        point.setPointers(pair);

        allocationsMap.put(allocId, point);

//        log.info("AllocationPoint 1: " + point);

        return point;
        //throw new UnsupportedOperationException("Not implemented yet");
    }


    protected AllocationPoint getAllocationPoint(DataBuffer buffer, AllocationShape shape, boolean catchNewAllocations) {
        Long trackingPointer = buffer.getTrackingPoint();

        if (trackingPointer == null) { // AllocationUtils.buildAllocationShape(objectId)
            if (catchNewAllocations) {
                log.info("Registering...");
                throw new IllegalStateException("WTF?");
                //trackingPointer = pickupSpan(buffer, shape);
            } else return null;
        }

        // that's a temporary exception, we'll change that to re-ack later
        if (trackingPointer == null)
            throw new IllegalStateException("trackingPointer is NULL");


        AllocationPoint point = getAllocationPoint(trackingPointer);
//        log.info("AllocationPoint 2: " + point);
        // temporary exception too
        if (point == null)
            throw new IllegalStateException("AllocationPoint is NULL");


        return point;
    }

    protected AllocationPoint getAllocationPoint(Long objectId) {
        return allocationsMap.get(objectId);
    }


    protected void purgeZeroObject(Long threadId, Long objectId, AllocationPoint point, boolean copyback) {
        // TODO: to be implemented
        if (copyback) {

            // copyback here
            memoryHandler.copyback(point, point.getShape());

            externalsLock.writeLock().lock();

            externalBuffers.remove(point.getBuffer());

            externalsLock.writeLock().unlock();
        }

        allocationsMap.remove(objectId);

        memoryHandler.purgeZeroObject(threadId, objectId, point, copyback);
    }

    protected void purgeDeviceObject(Long threadId, Integer deviceId, Long objectId, AllocationPoint point, boolean copyback) {
        // TODO: to be implemented
        if (!copyback) {
            allocationsMap.remove(objectId);
        }

        memoryHandler.purgeDeviceObject(threadId, deviceId, objectId, point, copyback);
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

        int totalElements = (int) memoryHandler.getAllocatedHostObjects(threadId);
        log.info("Total zero elements to be checked: [" + totalElements + "]; zeroUsed: ["+ memoryHandler.getAllocatedHostMemory() +"]");

        float shortAverage = zeroShort.getAverage();
        float longAverage = zeroLong.getAverage();

        float shortThreshold = shortAverage / (Aggressiveness.values().length - aggressiveness.ordinal());
        float longThreshold = longAverage / (Aggressiveness.values().length - aggressiveness.ordinal());



        AtomicInteger elementsDropped = new AtomicInteger(0);

        for (Long object: memoryHandler.getHostTrackingPoints(threadId)) {
            AllocationPoint point = getAllocationPoint(object);

            if (point == null)
                throw new RuntimeException("WTF???");

            if (point.getAccessState().isToeAvailable()) {
                point.getAccessState().requestToe();

                /*
                    Check if memory points to non-existant buffer, using externals.
                    If externals don't have specified buffer - delete reference.
                 */
                if (point.getBuffer() == null) {
       //             log.info("Ghost reference removed: " + object);

                    purgeZeroObject(threadId, object, point, false);
                    freeSpace.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()));

                    elementsDropped.incrementAndGet();
                    continue;
                }

                /*
                    Check, if memory can be removed from allocation.
                    To check it, we just compare average rates for few tens of latest calls
                 */
                /*
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
            */
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


        Set<Long> allocations = memoryHandler.getDeviceTrackingPoints(deviceId);


        float shortAverage = deviceShort.getAverage();
        float longAverage = deviceLong.getAverage();

        float shortThreshold = shortAverage / (Aggressiveness.values().length - aggressiveness.ordinal());
        float longThreshold = longAverage / (Aggressiveness.values().length - aggressiveness.ordinal());

        log.debug("Total device elements: " + allocations.size());

        AtomicInteger elementsDropped = new AtomicInteger(0);
        AtomicInteger elementsMoved = new AtomicInteger(0);

        for (Long object: allocations) {
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

                        purgeDeviceObject(threadId, deviceId, object, point, true);
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
            log.debug("Starting zero GC for thread: " + threadId);
            long lastCheck = System.currentTimeMillis();
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
                if ((memoryHandler.getAllocatedHostObjects(threadId) > 500000 || memoryHandler.getAllocatedHostMemory() > (configuration.getMaximumZeroAllocation() * 0.75)) && aggressiveness.ordinal() < Aggressiveness.URGENT.ordinal())
                    aggressiveness = Aggressiveness.URGENT;

                if (memoryHandler.getAllocatedHostMemory()> (configuration.getMaximumZeroAllocation() * 0.85))
                    aggressiveness = Aggressiveness.IMMEDIATE;

                if (memoryHandler.getAllocatedHostMemory() < (configuration.getMaximumZeroAllocation() * 0.25) && (memoryHandler.getAllocatedHostObjects(threadId) < 500) && lastCheck > System.currentTimeMillis() - 30000) {
                    ; // i don't want deallocation to be fired on lower thresholds. just no sense locking stuff
                    //log.debug("Skipping zero GC round: ["+zeroUseCounter.get()+"/" +zeroAllocations.get(threadId).size() + "]");
                }  else {
                    lastCheck = System.currentTimeMillis();
                    seekUnusedZero(threadId, aggressiveness);
                }
            }
        }
    }

    private class DeviceGarbageCollectorThread extends Thread implements Runnable {

        private final Integer deviceId;
        private final AtomicBoolean terminate;

        public DeviceGarbageCollectorThread(Integer deviceId, AtomicBoolean terminate) {
            this.deviceId = deviceId;
            this.terminate = terminate;
            this.setName("device gc thread ["+ deviceId +"]");
            this.setDaemon(true);
        }

        @Override
        public void run() {
            log.info("Starting device GC for device: " + deviceId);
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
                if ((memoryHandler.getAllocatedDeviceObjects(deviceId) > 100000 || memoryHandler.getAllocatedDeviceMemory(deviceId)> (configuration.getMaximumDeviceAllocation() * 0.75)) && aggressiveness.ordinal() < Aggressiveness.URGENT.ordinal())
                    aggressiveness = Aggressiveness.URGENT;

                if (memoryHandler.getAllocatedDeviceMemory(deviceId) > (configuration.getMaximumDeviceAllocation() * 0.85))
                    aggressiveness = Aggressiveness.IMMEDIATE;

                if (memoryHandler.getAllocatedDeviceMemory(deviceId)< (configuration.getMaximumDeviceAllocation() * 0.25) && (memoryHandler.getAllocatedDeviceObjects(deviceId) < 500)) {
                    // i don't want deallocation to be fired on lower thresholds. just no sense locking stuff
              //      log.debug("Skipping device GC round: ["+deviceMemoryTracker.getAllocatedSize(threadId, deviceId) +"/"+deviceAllocations.get(threadId, deviceId).size()+"]");
                } else seekUnusedDevice(0L, this.deviceId, aggressiveness);


            }
        }
    }


    /**
     * This method returns the number of tracked zero-copy allocations
     *
     * @return
     */
    public long getTotalAllocatedHostMemory() {
        return memoryHandler.getAllocationStatistics().row(AllocationStatus.HOST).get(0);
    }

    /**
     * This method returns the number of all tracked memory chunks
     *
     * @return
     */
    protected int getTotalTrackingPoints() {
        return allocationsMap.size();
    }

    public long getTotalAllocatedDeviceMemory(Integer deviceId) {
        return memoryHandler.getAllocationStatistics().row(AllocationStatus.DEVICE).get(deviceId);
    }

    @Override
    public void memcpyAsync(DataBuffer dstBuffer, jcuda.Pointer srcPointer, long length, long dstOffset) {
        this.memoryHandler.memcpyAsync(dstBuffer, srcPointer, length, dstOffset);
    }

    @Override
    public void memcpyBlocking(DataBuffer dstBuffer, jcuda.Pointer srcPointer, long length, long dstOffset) {
        this.memoryHandler.memcpyBlocking(dstBuffer, srcPointer, length, dstOffset);
    }

    @Override
    public void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer) {
        this.memoryHandler.memcpy(dstBuffer, srcBuffer);
    }

    /**
     * This method returns deviceId for current thread
     * All values >= 0 are considered valid device IDs, all values < 0 are considered stubs.
     *
     * @return
     */
    @Override
    public Integer getDeviceId() {
        return memoryHandler.getDeviceId();
    }
}
