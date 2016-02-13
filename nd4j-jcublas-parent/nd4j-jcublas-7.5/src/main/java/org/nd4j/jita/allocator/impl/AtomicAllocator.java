package org.nd4j.jita.allocator.impl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import jcuda.Pointer;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import lombok.NonNull;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.concurrency.Lock;
import org.nd4j.jita.allocator.enums.AccessState;
import org.nd4j.jita.allocator.enums.Aggressiveness;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.time.Ring;
import org.nd4j.jita.allocator.time.rings.LockedRing;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.mover.Mover;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.JCublasNDArray;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.DevicePointerInfo;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Just-in-Time Allocator for CUDA
 *
 * TODO: add description and basic algorithm ideas here
 *
 * @author raver119@gmail.com
 */
public class AtomicAllocator implements Allocator {
    private static final AtomicAllocator INSTANCE = new AtomicAllocator();

    private Configuration configuration = new Configuration();
    private CudaEnvironment environment;
    private transient Mover mover;
    private AtomicLong allocationsCounter = new AtomicLong(0);

    private AtomicLong objectsTracker = new AtomicLong(Long.MIN_VALUE);

    // tracker for thread->device affinity
    protected Map<Long, Integer> devicesAffinity = new ConcurrentHashMap<>();

    // simple counter to track allocated host-memory
    protected volatile AtomicLong zeroUseCounter = new AtomicLong(0);

    // we have single tracking point for allocation points, since we're not going to cycle through it it any time soon
    private Map<Long, AllocationPoint> allocationsMap = new ConcurrentHashMap<>();

    /*
        table for Thread, Device, Object allocations of device memory. Objects should be used to grab Allocation point from allocationsMap
    */
    // TODO: proper thread-safe implementation would be nice to have here :(
    // Table thread safety is guaranteed by reentrant read/write locks :(
    private Table<Long, Integer, CopyOnWriteArrayList<Long>> deviceAllocations = HashBasedTable.create();

    /*
        map for Thread, Object allocations in zero memory.
    */
    // CopyOnWriteArrayList performance to be investigated in this use case
    // Map thread safety is guaranteed by exclusive writeLock in getDeviceId() method, because we can't use putIfAbsent on j7
    // FIXME: at j7 -> j8 transition, this one could be changed to ConcurrentHashMap
    private Map<Long, CopyOnWriteArrayList<Long>> zeroAllocations = new HashMap<>();

    /*
        WeakHashMap for buffer->id conversion. If DataBuffer get's removed by jvm GC, we'll know that.
        So, just a short way for reverse lookup, that causes no GC issues.
     */
    private Map<BaseCudaDataBuffer, Long> externalBuffers = Collections.synchronizedMap(new WeakHashMap<BaseCudaDataBuffer, Long>());

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



    public static AtomicAllocator getInstance() {
        return INSTANCE;
    }

    protected AtomicAllocator() {
        environment = new CudaEnvironment(configuration);
    }

    /**
     * This method returns cublasHandle for current thread
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

    @Override
    public void setMover(@NonNull Mover mover) {
        globalLock.writeLock().lock();

        this.mover = mover;
        this.mover.init(configuration, environment);

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
    @Override
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
    @Override
    public Long pickupSpan(@NonNull BaseCudaDataBuffer buffer, @NonNull AllocationShape shape) {

        try {
            externalsLock.writeLock().lock();

            if (externalBuffers.containsKey(buffer)) {
                /*
                    We have such buffer already. It's either the Nested allocation, or something like that.
                    Just throw exception for now.
                 */
                throw new IllegalStateException("Buffer is already registered");
            } else {
                /*
                    We don't have such buffer registered somehow. Probably that's new allocation
                 */
                AllocationPoint point = new AllocationPoint();
                point.setShape(shape);

                // set device ID -> current thread
                point.setDeviceId(getDeviceId());

                Long allocPointer = objectsTracker.getAndIncrement();

                point.setObjectId(allocPointer);

                /*
                    we don't keep strong references Allocator -> Buffer, but we store Buffer -> Allocator references instead :)
                  */
                buffer.setAllocationPoint(point);
                buffer.setAllocatorPointer(allocPointer);

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

        /*
         while working on array level, we actually immediately downgrade to buffer level, with AllocationShape defined by this array
          */
        if (!(array.data() instanceof BaseCudaDataBuffer)) throw new IllegalStateException("Underlying buffer isn't instance of BaseCudaDataBuffer");

        // TODO: confirm that these values are fetched properly
        AllocationShape shape = new AllocationShape();
        shape.setOffset(array.offset());
        shape.setStride(array.elementWiseStride());
        shape.setLength(array.length());
        shape.setDataType(Nd4j.dataType());

        return pickupSpan((BaseCudaDataBuffer) array.data(), shape);
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
     * This method hints allocator, that specific object was accessed on host side.
     * This includes putRow, putScalar;
     *
     * @param objectId unique object ID
     */
    @Override
    public void tickHost(BaseCudaDataBuffer objectId) {

    }

    /**
     * This methods hints allocator, that specific object was accessed on device side.
     *
     * @param objectId unique object ID
     * @param shape    shape
     */
    @Override
    @Deprecated
    public void tickDevice(BaseCudaDataBuffer objectId, AllocationShape shape) {
        throw new UnsupportedOperationException("tickDevice should be removed!!!11one");
    }

    /**
     * This method hints allocator, that specific object was released on device side
     *
     * @param array
     */
    @Override
    public void tackDevice(INDArray array) {
        tackDevice((BaseCudaDataBuffer) array.data(), AllocationUtils.buildAllocationShape(array));
    }



    @Override
    public void tackDevice(BaseCudaDataBuffer objectId, AllocationShape shape) {
        AllocationPoint point = getAllocationPoint(objectId);

        point.getAccessState().requestTack();
    }


    /**
     * This method notifies allocator, that specific object was changed on device side
     *
     * @param array
     */
    @Override
    public void tickDeviceWrite(INDArray array) {
        AllocationPoint point = getAllocationPoint((BaseCudaDataBuffer) array.data());

        point.tickDeviceWrite();
    }

    /**
     * This method notifies allocator, that specific object was changed on host side
     *
     * @param array
     */
    @Override
    public void tickHostWrite(INDArray array) {
        AllocationPoint point = getAllocationPoint((BaseCudaDataBuffer) array.data());

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
    public Pointer getDevicePointer(BaseCudaDataBuffer objectId) {
        throw new UnsupportedOperationException("getDevicePointer() method should be removed");
    }

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param objectId
     * @param shape
     */
    @Override
    public Pointer getDevicePointer(BaseCudaDataBuffer objectId, AllocationShape shape) {
        /*
            We assume that object is registered within allocator
         */
        AllocationPoint point = getAllocationPoint(objectId);

        boolean isNewAllocation = false;

        Long trackingPoint = objectId.getAllocatorPointer();

        // we're checking, if cuda pointer is null without any locks. but if it's null, we'll request Toe state on this allocation, to make sure nothing can mess with it
        if (point.getCudaPointer() == null) {
            // at this point memory becomes read/write-locked for a few ms, to make sure cudaPointer exists
            point.getAccessState().requestToe();

            if (point.getCudaPointer() == null) {
                /*
                    If pointer is null, that means we're on first stage of allocation, so we need to allocate Zero memory
                */

                /*
                    Before allocating anything, we must ensure that we have enough space left
                 */
                long requiredMemory = AllocationUtils.getRequiredMemory(shape);
                while (zeroUseCounter.get() > configuration.getMaximumZeroAllocation() - (configuration.getMaximumZeroAllocation() / 10)) {
                        log.info("No free host memory available. Startig GC manually with [URGENT] agressiveness");
//                    if (zeroUseCounter.get() > configuration.getMaximumZeroAllocation() - (configuration.getMaximumZeroAllocation() / 10)) {
                        long freedMemory = seekUnusedZero(Thread.currentThread().getId(), Aggressiveness.URGENT);
//                    } else {

//                    }
                }
                /*
                    We intentionally update counter prior to allocation
                 */
                zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(shape));

                /*
                    now it's ALMOST safe to allocate zero-copy memory.
                    Technically it's still possible to fail there, with oom or CUDA-originated exception
                 */
                point.setAllocationStatus(AllocationStatus.ZERO);
                DevicePointerInfo info = mover.alloc(AllocationStatus.ZERO, point, shape);
                long allocCnt = allocationsCounter.incrementAndGet();
                if (allocCnt % 1000 == 0) log.info("Total zero allocations happened: [" + allocCnt + "]; active zero allocations: ["+ zeroAllocations.get(Thread.currentThread().getId()).size()+"]");

                /*
                    it's safe to remove this check in production environment
                 */
                if (info == null)
                    throw new IllegalStateException("Zero-copy allocation failed");

                point.setCudaPointers(info);
                zeroAllocations.get(Thread.currentThread().getId()).add(trackingPoint);


                /*
                    Copy data from host buffer to device
                 */
                // We don't need separate call here, we wrap that inside alloc call
                // mover.copyforward(point);
            } else {
                /*
                    do nothing here, the only possible reason for us to get in this scope, is concurrent getDevicePointer access, so it was stopped by TTT barrier, and now we're here after everything being done
                  */
                ;
            }

            isNewAllocation = true;

            point.getAccessState().releaseToe();
        }

        /*
            Before coming to promotion, we should decide, if we need to synchronize data on device
         */
        if (!isNewAllocation) {
            if (!point.isActualOnDeviceSide()) {
                // update data in Toe state
                point.getAccessState().requestToe();

                if (!point.isActualOnDeviceSide()) {
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
        if (!isNewAllocation) {
            // we check promotion only for existant allocations. just ignore new allocations here :)
            // TODO: add memory check all the way here
     /*       if (point.getDeviceTicks() > 5 && point.getAllocationStatus() == AllocationStatus.ZERO) {
                point.getAccessState().requestToe();

                try {
                    DevicePointerInfo newPointers = mover.alloc(AllocationStatus.DEVICE, point, shape);

                    point.setAllocationStatus(AllocationStatus.DEVICE);
                    point.setCudaPointers(newPointers);
                    log.info("Relocation happened!");
                } catch (Exception e){

                }
                point.getAccessState().releaseToe();
            }*/
        }

        /*
            after everything was done here - register tick, and return the pointer to outer context
         */
        point.getAccessState().requestTick();
        point.tickDevice();

        /*
            Now we store use rates
         */

        if (point.getAllocationStatus() == AllocationStatus.ZERO) {
            zeroLong.store(point.getTimerLong().getFrequencyOfEvents());
            zeroShort.store(point.getTimerShort().getFrequencyOfEvents());
        } else {
            deviceLong.store(point.getTimerLong().getFrequencyOfEvents());
            deviceShort.store(point.getTimerShort().getFrequencyOfEvents());
        }

        return point.getCudaPointer();
    }

    /**
     * This method returns actual host pointer, valid for specified shape of current object
     *
     * @param buffer
     * @param shape
     * @return
     */
    @Override
    public Pointer getHostPointer(BaseCudaDataBuffer buffer, AllocationShape shape) {
        return null;
    }

    /**
     * This method should be called to make sure that data on host side is actualized
     *
     * @param objectId
     */
    @Override
    public void synchronizeHostData(BaseCudaDataBuffer objectId, AllocationShape shape) {
        AllocationPoint point = getAllocationPoint(objectId);

        /*
            We set memory state to Toe, and issue copyback if required
         */

        if (!point.isActualOnHostSide() || point.getAccessState().getCurrentState() != AccessState.TACK) {

            point.getAccessState().requestToe();

            if (!point.isActualOnHostSide()) {
                mover.copyback(point, shape);

                // update the timer for hostRead
                point.tickHostRead();
            } else log.info("Data is actual, skipping sync");

            point.getAccessState().releaseToe();
        }


    }

    /**
     * This method should be called to make sure that data on host side is actualized
     *
     * @param array
     */
    @Override
    public void synchronizeHostData(JCublasNDArray array) {
        synchronizeHostData((BaseCudaDataBuffer) array.data(), AllocationUtils.buildAllocationShape(array));
    }

    /**
     * This method returns current host memory state
     *
     * @param objectId
     * @return
     */
    @Override
    public SyncState getHostMemoryState(BaseCudaDataBuffer objectId) {
        /*
            basically we just want to compare two access time values: device & host.
            we can't know, if memory was changed on device side or not
          */

        /*
            TODO: improvement is possible here ->
             as soon as we'll have partial allocations available, we can have partially synced memory
         */
        AllocationPoint point = getAllocationPoint(objectId);
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
     * @param objectId
     * @return
     */
    @Override
    public Integer getDeviceId(BaseCudaDataBuffer objectId) {
        return null;
    }

    protected void initCudaContextForThread(Long threadId) {
        CudaContext context = new CudaContext();
        context.initHandle();
        contextPool.put(threadId, context);
    }

    /**
     * This method returns CUDA deviceId for current thread
     *
     * @return
     */
    @Override
    public Integer getDeviceId() {
        try {
            deviceLock.writeLock().lock();

            Long threadId = Thread.currentThread().getId();

            if (!devicesAffinity.containsKey(threadId)) {
                wasInitialised.compareAndSet(false, true);

                List<Integer> devices = new ArrayList<>(environment.getAvailableDevices().keySet());
                Random rnd = new Random();
                Integer device = devices.get(rnd.nextInt(devices.size()));
                devicesAffinity.put(threadId, device );

                if (!zeroAllocations.containsKey(threadId)) {
                    // TODO: investigate CopyOnWriteArrayList here, _PROBABLY_ we could replace it with synchronized list, without backing
                    zeroAllocations.put(threadId, new CopyOnWriteArrayList<Long>());
                }

                if (!deviceAllocations.contains(threadId, device)) {
                    deviceAllocations.put(threadId, device, new CopyOnWriteArrayList<Long>());
                }

                initCudaContextForThread(threadId);

                log.info("Mapping device ["+ device+"] to thread [" + Thread.currentThread().getId() + "]");

                ZeroGarbageCollectorThread thread = new ZeroGarbageCollectorThread(threadId, device, shouldStop);
                thread.start();
                collectorsZero.put(threadId, thread);

                DeviceGarbageCollectorThread dThread = new DeviceGarbageCollectorThread(threadId, device, shouldStop);
                dThread.start();
                collectorsDevice.put(threadId, dThread);
            }
            return devicesAffinity.get(Thread.currentThread().getId());
        } finally {
            deviceLock.writeLock().unlock();
        }
    }

    protected AllocationPoint getAllocationPoint(BaseCudaDataBuffer objectId) {
        Long trackingPointer = objectId.getAllocatorPointer();

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
     *
     * @param threadId
     * @param objectId
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
        //log.info("Total zero elements to be checked: [" + totalElements + "]; zeroUsed: ["+ zeroUseCounter.get()+"]");

        float shortAverage = zeroShort.getAverage();
        float longAverage = zeroLong.getAverage();

        float shortThreshold = shortAverage / (Aggressiveness.values().length - aggressiveness.ordinal());
        float longThreshold = longAverage / (Aggressiveness.values().length - aggressiveness.ordinal());



        AtomicInteger elementsDropped = new AtomicInteger(0);

        for (Long object: zeroAllocations.get(threadId)) {
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



        log.info("Short average: ["+shortAverage+"], Long average: [" + longAverage + "]");
        log.info("Aggressiveness: ["+ aggressiveness+"]; Short threshold: ["+shortThreshold+"]; Long threshold: [" + longThreshold + "]");
        log.info("Elements deleted: " + elementsDropped.get());

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
    protected long seekUnusedDevice(Long threadId, Integer deviceId) {
        AtomicLong freeSpace = new AtomicLong(0);

        deviceLock.readLock().lock();
        List<Long> allocations = deviceAllocations.get(threadId, deviceId);
        deviceLock.readLock().unlock();


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
        }

        @Override
        public void run() {
            while (!terminate.get()) {

                /*
                    Check for zero-copy garbage
                 */
                log.info("ZeroGC started...");
                /*
                    We want allocations to take in account multiple things:
                    1. average access rates for last X objects
                    2. total number of currently allocated objects
                    3. total allocated memory size
                    4. desired aggressiveness
                */

                Aggressiveness aggressiveness = configuration.getDeallocAggressiveness();

                // if we have too much objects, or total allocated memory has met 75% of max allocation - use urgent mode
                if ((zeroAllocations.get(threadId).size() > 500000 || zeroUseCounter.get() > (configuration.getMaximumZeroAllocation() * 0.75)) && aggressiveness.ordinal() < Aggressiveness.URGENT.ordinal())
                    aggressiveness = Aggressiveness.URGENT;

                if (zeroUseCounter.get() > (configuration.getMaximumZeroAllocation() * 0.85))
                    aggressiveness = Aggressiveness.IMMEDIATE;

                seekUnusedZero(threadId, aggressiveness);

                try {
                    Thread.sleep(Math.max(configuration.getMinimumTTLMilliseconds(), 5000));
                } catch (Exception e) {
                    // we can have interruption here, to force gc
                  ;
                }
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
        }

        @Override
        public void run() {
            while (!terminate.get()) {
                /*
                    Check for device garbage
                 */
                log.info("DeviceGC started...");
                seekUnusedDevice(this.threadId, this.deviceId);


                try {
                    Thread.sleep(Math.max(configuration.getMinimumTTLMilliseconds(), 5000));
                } catch (Exception e) {
                    // we can have interruption here, to force gc
                    ;
                }
            }
        }
    }
}
