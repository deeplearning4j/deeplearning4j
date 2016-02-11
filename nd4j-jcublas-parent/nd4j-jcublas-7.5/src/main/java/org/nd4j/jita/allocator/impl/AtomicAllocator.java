package org.nd4j.jita.allocator.impl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import jcuda.Pointer;
import lombok.NonNull;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.mover.Mover;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
public class AtomicAllocator implements Allocator {
    private static final AllocatorPrototype INSTANCE = new AllocatorPrototype();

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
    private Table<Long, Integer, Long> deviceAllocations = HashBasedTable.create();

    /*
        map for Thread, Object allocations in zero memory.
    */
    private Map<Long, Long> zeroAllocations = new ConcurrentHashMap<>();

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
                Long allocPoiner = objectsTracker.getAndIncrement();
                externalBuffers.put(buffer, allocPoiner);
                return allocPoiner;
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

    @Override
    public void tackDevice(BaseCudaDataBuffer objectId, AllocationShape shape) {
        AllocationPoint point = getAllocationPoint(objectId);

        point.getAccessState().requestTack();
    }

    /**
     * This method returns actual device pointer valid for current object
     *
     * TODO: this method should be removed.
     * @param objectId
     */
    @Override
    @Deprecated
    public Object getDevicePointer(BaseCudaDataBuffer objectId) {
        return null;
    }

    /**
     * This method returns actual device pointer valid for specified shape of current object
     *
     * @param objectId
     * @param shape
     */
    @Override
    public Object getDevicePointer(BaseCudaDataBuffer objectId, AllocationShape shape) {
        /*
            We assume that object is registered within allocator
         */
        AllocationPoint point = getAllocationPoint(objectId);

        // we're checking, if cuda pointer is null without any locks. but if it's null, we'll request Toe state on this allocation, to make sure nothing can mess with it
        if (point.getCudaPointer() == null) {
            // at this point memory becomes read/write-locked for a few ms, to make sure cudaPointer exists
            point.getAccessState().requestToe();

            if (point.getCudaPointer() == null) {
                /*
                    If pointer is null, that means we're on first stage of allocation, so we need to allocate Zero memory
                */
                Pointer cudaPointer = mover.alloc(AllocationStatus.ZERO, shape);
                point.setCudaPointer(cudaPointer);

                /*
                    Copy data from host buffer to device
                 */
                mover.copyforward(point);
            } else {
                /*
                    do nothing here, the only possible reason for us to get in this scope, is concurrent getDevicePointer access, so it was stopped by TTT barrier, and now we're here after everything being done
                  */
                ;
            }

            point.getAccessState().releaseToe();
        }


        /*
            So, right now we are guaranteed to have cudaPointer. We can decide now, if this memory chunk should be promoted or not.
         */


        /*
            after everything was done here - register tick, and return the pointer to outer context
         */
        point.getAccessState().requestTick();

        return point.getCudaPointer();
    }

    /**
     * This method should be called to make sure that data on host size is actualized
     *
     * @param objectId
     */
    @Override
    public void synchronizeHostData(BaseCudaDataBuffer objectId) {

    }

    /**
     * This method returns current host memory state
     *
     * @param objectId
     * @return
     */
    @Override
    public SyncState getHostMemoryState(BaseCudaDataBuffer objectId) {
        return null;
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

            if (!devicesAffinity.containsKey(Thread.currentThread().getId())) {
                List<Integer> devices = new ArrayList<>(environment.getAvailableDevices().keySet());
                Random rnd = new Random();
                Integer device = devices.get(rnd.nextInt(devices.size()));
                devicesAffinity.put(Thread.currentThread().getId(), device );
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
}
