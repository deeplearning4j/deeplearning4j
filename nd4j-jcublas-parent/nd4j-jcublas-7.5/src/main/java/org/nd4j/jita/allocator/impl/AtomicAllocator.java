package org.nd4j.jita.allocator.impl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import jcuda.Pointer;
import lombok.NonNull;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
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
    private CudaEnvironment environment = new CudaEnvironment();
    private transient Mover mover;

    private AtomicLong objectsTracker = new AtomicLong(Long.MIN_VALUE);

    // tracker for thread->device affinity
    protected Map<Long, Integer> devicesAffinity = new ConcurrentHashMap<>();

    // simple counter to track allocated host-memory
    protected AtomicLong zeroUseCounter = new AtomicLong(0);

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


    private static Logger log = LoggerFactory.getLogger(AtomicAllocator.class);


    /*
        locks for internal resources
     */
    private ReentrantReadWriteLock deviceLock = new ReentrantReadWriteLock();
    private ReentrantReadWriteLock globalLock = new ReentrantReadWriteLock();
    private ReentrantReadWriteLock externalsLock = new ReentrantReadWriteLock();

    public static AtomicAllocator getInstance() {
        return INSTANCE;
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
     * @param configuration configuration bean to be applied
     */
    @Override
    public void applyConfiguration(@NonNull Configuration configuration) {
        globalLock.writeLock().lock();

        this.configuration = configuration;

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
                    // TODO: call for zero copy memory copyback & deallocation

                    long freedMemory = seekUnusedZero(Thread.currentThread().getId());
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
                // update data
                point.getAccessState().requestToe();

                if (!point.isActualOnDeviceSide()) {
                    mover.copyforward(point, shape);
                }
                point.tickDeviceToHost();

                point.getAccessState().releaseToe();
            }
        }

        /*
            So, right now we are guaranteed to have cudaPointer. We can decide now, if this memory chunk should be promoted or not.
         */
        if (!isNewAllocation) {
            // we check promotion only for existant allocations. just ignore new allocations here :)

        }

        /*
            after everything was done here - register tick, and return the pointer to outer context
         */
        point.getAccessState().requestTick();
        point.tickDevice();

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


        point.getAccessState().requestToe();

        if (!point.isActualOnHostSide()) {
            mover.copyback(point, shape);

            // update the timer for hostRead
            point.tickHostRead();
        } else log.info("Data is actual, skipping sync");



        point.getAccessState().releaseToe();
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

                log.info("Mapping device ["+ device+"] to thread [" + Thread.currentThread().getId() + "]");
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
     * This method seeks for unused zero-copy memory allocations
     *
     * @param threadId Id of the thread, retrieved via Thread.currentThread().getId()
     * @return size of memory that was deallocated
     */
    protected long seekUnusedZero(Long threadId) {
        /*
            This method is blocking on thread basis, just to prevent parallel calls

            TODO: To prevent cyclic calls we need something smart here
         */
        AtomicLong freeSpace = new AtomicLong(0);

        for (Long object: zeroAllocations.get(threadId)) {
            AllocationPoint point = getAllocationPoint(object);
            if (point.getAccessState().isToeAvailable()) {
                point.getAccessState().requestToe();

                /*
                    Check if memory points to non-existant buffer, using externals.
                    If externals don't have specified buffer - delete reference.
                 */

                /*
                    Check, if memory can be removed from allocation.
                    To check it, we just compare average rates for few tens of latest calls
                 */

                point.getAccessState().releaseToe();
            }
        }

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


    private class GarbageCollectorThread extends Thread implements Runnable {

        private final Long threadId;
        private final Integer deviceId;
        private final AtomicBoolean terminate;

        public GarbageCollectorThread(Long threadId, Integer deviceId, AtomicBoolean terminate) {
            this.threadId = threadId;
            this.deviceId = deviceId;
            this.terminate = terminate;
        }

        @Override
        public void run() {
            while (!terminate.get()) {
                /*
                    Check for device garbage
                 */
                seekUnusedDevice(this.threadId, this.deviceId);

                /*
                    Check for zero-copy garbage
                 */
                seekUnusedZero(threadId);

                try {
                    Thread.sleep(30000);
                } catch (Exception e) {
                    // we can have interruption here, to force gc
                  ;
                }
            }
        }
    }
}
