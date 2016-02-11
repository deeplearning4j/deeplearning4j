package org.nd4j.jita.allocator.impl;

import lombok.NonNull;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AccessState;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.time.RateTimer;
import org.nd4j.jita.allocator.time.impl.SimpleTimer;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.balance.Balancer;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.mover.Mover;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This is going to be basic JITA implementation.
 *
 * We assume that we have 3 states for each INDArray underlying memory:
 *
 * HOST: memory is detached from GPU use.
 * ZERO: memory is used as zero-copy memory
 * DEVICE: memory is used on device
 *
 * PLEASE NOTE: WORK IN PROGRESS, DO NOT EVER USE IT!
 *
 * @author raver119@gmail.com
 */
public final class AllocatorPrototype implements Allocator {
    private static final AllocatorPrototype INSTANCE = new AllocatorPrototype();

    private Configuration configuration = new Configuration();
    private CudaEnvironment environment = new CudaEnvironment();

    // Balancer will be merged with mover.
    private transient Balancer balancer;
    private transient Mover mover;
//    private transient Lock locker = new RRWLock();

    private Map<Long, AllocationPoint> allocationPoints = new ConcurrentHashMap<>();

    private Map<Long, AllocationPoint> deviceAllocations = new ConcurrentHashMap<>();
    private Map<Long, AllocationPoint> zeroAllocations = new ConcurrentHashMap<>();

    // device/zero new allocations rate tracking
    private transient RateTimer devicePressure = new SimpleTimer(10, TimeUnit.SECONDS);
    private transient RateTimer zeroPressure = new SimpleTimer(10, TimeUnit.SECONDS);

    // here we track Thread <-> Device affinity to avoid cross-device access as much as possible
    protected Map<Long, Integer> devicesAffinity = new ConcurrentHashMap<>();

    // simple counter to track allocated host-memory
    protected AtomicLong zeroUseCounter = new AtomicLong(0);

    // list of objects originated allocations
    private WeakHashMap<Long, AllocationPoint> externals = new WeakHashMap<>();

    private static Logger log = LoggerFactory.getLogger(AllocatorPrototype.class);

    protected AllocatorPrototype() {
        //
    }

    public static AllocatorPrototype getInstance() {
        return INSTANCE;
    }


    protected long getHostAllocatedMemory() {
        return zeroUseCounter.get();
    }

    /**
     * Consume and apply configuration passed in as argument
     *
     * @param configuration configuration bean to be applied
     */
    @Override
    public void applyConfiguration(Configuration configuration) {
        try {
//            locker.globalWriteLock();
            this.configuration = configuration;
        } finally {
//            locker.globalWriteUnlock();
        }
    }

    /**
     * Returns current Allocator configuration
     *
     * @return current configuration
     */
    @Override
    public Configuration getConfiguration() {
        // TODO: global lock to be implemented
        try {
//            locker.globalReadLock();
            return configuration;
        } finally {
//            locker.globalReadUnlock();
        }
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

    /**
     * Returns deviceId to be used from some thread.
     *
     * @return
     */
    protected Integer getNewDeviceId() {
        try {
//            locker.globalReadLock();

            if (!devicesAffinity.containsKey(Thread.currentThread().getId())) {
                List<Integer> devices = new ArrayList<>(environment.getAvailableDevices().keySet());
                Random rnd = new Random();
                Integer device = devices.get(rnd.nextInt(devices.size()));
                devicesAffinity.put(Thread.currentThread().getId(), device );
                log.info("Mapping device ["+ device+"] to thread [" + Thread.currentThread().getId() + "]");
            }
            return devicesAffinity.get(Thread.currentThread().getId());
        } finally {
//            locker.globalReadUnlock();
        }
    }

    protected void registerSpan(Long objectId, @NonNull AllocationShape shape) {
        // TODO: object-level lock is HIGHLY required here, for multithreaded safety
        try {
       //     locker.globalReadLock();

            if (!allocationPoints.containsKey(objectId)) {
//                locker.attachObject(objectId);
                try {
//                    locker.objectWriteLock(objectId);


                    AllocationPoint allocationPoint = new AllocationPoint();
                    allocationPoint.setAccessHost(System.nanoTime());
                    allocationPoint.setAllocationStatus(AllocationStatus.UNDEFINED);
                    allocationPoint.setShape(shape);
                    allocationPoint.setObjectId(objectId);

                    // once deviceId is set, all future relocations should respect it
                    allocationPoint.setDeviceId(getNewDeviceId());

                    //allocationPoint.addShape(shape);

                    // we add this buffer to tracking
//                    locker.externalsWriteLock();
                    externals.put(objectId, allocationPoint);
//                    locker.externalsWriteUnlock();

                    allocationPoints.put(objectId, allocationPoint);
                } finally {
//                    locker.objectWriteUnlock(objectId);
                }
            } else {
                log.info("Registering older object");
                try {
//                    locker.objectWriteLock(objectId);

                    AllocationPoint allocationPoint = allocationPoints.get(objectId);
                    if (shape.equals(allocationPoint.getShape())) {
                        // that's temporary exception, since such scenario is theoretically possible
                        throw new IllegalStateException("Double register called on the same id and same shape");
                    } else {
                        // just suballocation. check for buffer overflow and attach new shape
                        //allocationPoint.addShape(shape);
                        NestedPoint nestedPoint = new NestedPoint(shape);
                        allocationPoint.addShape(nestedPoint);
                    }
                } finally {
//                    locker.objectWriteUnlock(objectId);
                }
            }
        } finally {
        //    locker.globalReadUnlock();
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
        try {
//            locker.globalReadLock();

            AllocationPoint point = allocationPoints.get(objectId);
            point.setAccessHost(System.nanoTime());
        } finally {
  //          locker.globalReadUnlock();
        }
    }

    /**
     * This methods hints allocator, that specific object started access on device side.
     *
     * @param objectId unique object ID
     * @param shape shape
     */
    @Override
    public void tickDevice(Long objectId, @NonNull AllocationShape shape) {
        // TODO: provide object-level lock here

            AllocationPoint point = getAllocationPoint(objectId);
            if (shape.equals(point.getShape())) {
                try {
//                    locker.objectWriteLock(objectId);

                    point.setAccessDevice(System.nanoTime());
                    point.tickDevice();
                    point.setAccessState(AccessState.TICK);
                } finally {
  //                  locker.objectWriteUnlock(objectId);
                }
            } else {

                // TODO: decide on lock here
                point.tickDescendant(shape);

                if (point.containsShape(shape)) {
                    try {
//                        locker.shapeWriteLock(objectId, shape);

                        NestedPoint nestedPoint = point.getNestedPoint(shape);
                        nestedPoint.setAccessState(AccessState.TICK);
                    } finally {
//                        locker.shapeWriteUnlock(objectId, shape);
                    }
                    //nestedPoint.tick();
                } else throw new IllegalStateException("Shape [" + shape + "] wasn't found at tickDevice()");
            }
    }

    /**
     * This method hints allocator, that specific object finished access on device side
     *
     * @param objectId
     * @param shape
     */
    @Override
    public void tackDevice(Long objectId, @NonNull AllocationShape shape) {
            AllocationPoint point = getAllocationPoint(objectId);
            if (shape.equals(point.getShape())) {
                try {
//                    locker.objectWriteLock(objectId);

                    point.setAccessDevice(System.nanoTime());
                    point.tackDevice();
                    point.setAccessState(AccessState.TACK);
                } finally {
//                    locker.objectWriteUnlock(objectId);
                }
            } else {
                // TODO: to be decided on lock here
                point.tackDescendant(shape);
                if (point.containsShape(shape)) {
                    try {
//                        locker.shapeWriteLock(objectId, shape);

                        NestedPoint nestedPoint = point.getNestedPoint(shape);
                        nestedPoint.setAccessState(AccessState.TACK);
                    } finally {
//                        locker.shapeWriteUnlock(objectId, shape);
                    }
                    //  nestedPoint.tack();
                } else throw new IllegalStateException("Shape [" + shape + "] wasn't found at tickDevice()");
            }
    }

    /**
     * This method returns actual device pointer valid for current object
     *
     * @param objectId
     */
    @Override
    @Deprecated
    public Object getDevicePointer(Long objectId) {
        // TODO: this method should return pointer at some point later
        // TODO: provide object-level lock here
            AllocationPoint point = getAllocationPoint(objectId);

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
            AllocationPoint point = getAllocationPoint(objectId);

            if (point == null) {
                // TODO: blind invocation event here. something better would be nice to have here
                registerSpan(objectId, shape);
                point = getAllocationPoint(objectId);
            }

            Object pointer = null; //point.getDevicePointer();

            if (point.getShape().equals(shape)) {
                // this is the same alloc shape
                try {
                    // FIXME: wrong lock here, consider something better like j8 stampedLock

                    /*
                        We should check allocation bounds here
                     */

//                    locker.objectWriteLock(objectId);
                    pointer = point.getCudaPointer();

                    if (pointer == null) {
                        try {
                       //     locker.externalsWriteLock();

                            /*
                                we set 10% barrier of max zero alloc size.
                                if memory use is >= of that - start copyback and deallocation
                            */
                            if (zeroUseCounter.get() >= configuration.getMaximumZeroAllocation() - (configuration.getMaximumZeroAllocation() / 10)) {

                                syncbackUnusedZero(configuration.getMaximumZeroAllocation() / 5);
                            }

                        } finally {
                        //    locker.externalsWriteUnlock();
                        }

                        pointer = mover.alloc(AllocationStatus.ZERO, point.getShape());
                        point.setCudaPointer(pointer);
                        point.setAllocationStatus(AllocationStatus.ZERO);

                        zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(shape));

                        zeroAllocations.put(objectId, point);
                    }
                } finally {
//                    locker.objectWriteUnlock(objectId);
                }

//                locker.globalReadLock();
                int minThreshold = configuration.getMinimumRelocationThreshold();
//                locker.globalReadUnlock();

                if (point.getDeviceTicks() > minThreshold && point.getTimerShort().getFrequencyOfEvents() > 0 && point.getAccessState() == AccessState.TACK) {
                    // try relocation
                    try {
//                        locker.objectWriteLock(objectId);

                        if (point.getAccessState() == AccessState.TACK) {
                            AllocationStatus target = makePromoteDecision(objectId, point.getShape());
                            if (target == AllocationStatus.DEVICE) {
//                                log.info("Trying relocation: " + objectId);
                                relocateMemory(objectId, target);
                                pointer = point.getCudaPointer();

                                // put this one to device allocations bucket
                                deviceAllocations.put(objectId, point);

                                zeroAllocations.remove(objectId);

                                zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(shape) * -1);
                            }
                        }
                    } finally {
///                        locker.objectWriteUnlock(objectId);
                    }
                }
            } else {
                // this is suballocation
                if (point.containsShape(shape)) {
                    try {
//                        locker.shapeWriteLock(objectId, shape);
                        pointer = point.getNestedPoint(shape).getDevicePointer();
                        if (pointer == null) {
                            pointer = new Object();
                            NestedPoint nestedPoint = new NestedPoint(shape);
                            nestedPoint.setDevicePointer(pointer);

                            point.addShape(nestedPoint);
                            //point.tickDescendant(shape);
                        }
                    } finally {
//                        locker.shapeWriteUnlock(objectId, shape);
                    }
                } else throw new IllegalStateException("Shape isn't existant: " + shape);
            }

            // p.3
            this.tickDevice(objectId, shape);

            // p.4
            return pointer;
    }

    /**
     * This method should be called to make sure that data on host size is actualized
     *
     * @param objectId
     */
    @Override
    public void synchronizeHostData(Long objectId) {
            AllocationPoint point = getAllocationPoint(objectId);

            if (!getHostMemoryState(objectId).equals(SyncState.SYNC)) {
                // if data was accessed by device, it could be changed somehow
                try {
//                    locker.objectWriteLock(objectId);

                    // TODO: target buffer should be passed here
                    mover.copyback(point);
                } finally {
//                    locker.objectWriteUnlock(objectId);
                }
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
        try {
            // FIXME: wrong sync here, consider something better
//            locker.objectReadLock(objectId);

            AllocationPoint point = getAllocationPoint(objectId);
            if (point.getAccessHost() >= point.getAccessDevice()) {
                point.setHostMemoryState(SyncState.SYNC);
            } else {
                point.setHostMemoryState(SyncState.DESYNC);
            }
            return point.getHostMemoryState();
        } finally {
//            locker.objectReadUnlock(objectId);
        }
    }

    /**
     * This method returns the number of top-level memory allocations.
     * No descendants are included in this result.
     *
     * @return number of allocated top-level memory chunks
     */
    @Override
    public int tableSize() {
        return allocationPoints.size();
    }

    /**
     * This method returns CUDA deviceId for specified buffer
     *
     * @param objectId
     * @return
     */
    @Override
    public Integer getDeviceId(Long objectId) {
        AllocationPoint point = getAllocationPoint(objectId);

        return point.getDeviceId();
    }

    /**
     * This method returns CUDA deviceId for current thread
     *
     * @return
     */
    @Override
    public Integer getDeviceId() {
        return getNewDeviceId();
    }


    protected int zeroTableSize() {
        return zeroAllocations.size();
    }

    protected int deviceTableSize() {
        return deviceAllocations.size();
    }

    /**
     * This method forces allocator to shutdown gracefully
     */
    protected void shutdown() {
        // TODO: to be implemented
    }

    protected void deallocatePoint(Long object, AllocationPoint point) {
        try {
//            locker.objectWriteLock(object);

            AllocationStatus status = point.getAllocationStatus();

            if (status == AllocationStatus.ZERO) {
                zeroAllocations.remove(object);
            } else if (status == AllocationStatus.DEVICE) {
                deviceAllocations.remove(object);
            }

            synchronizeHostData(object);
            mover.free(point);

            long usedMemory = AllocationUtils.getRequiredMemory(point.getShape());

//            locker.globalWriteLock();

            if (status == AllocationStatus.DEVICE) {
                environment.trackAllocatedMemory(1, usedMemory * -1);
            }

//            locker.globalWriteUnlock();

        } finally {
//            locker.objectWriteUnlock(object);
        }
    }

    /*
        This method build list of allocationPoints that should be removed from tracking, since their originators were GCd
    */
    protected List<Long> updateGcState() {
        List<Long> result = new ArrayList<>();
        for (Long objectId: allocationPoints.keySet()) {
//            locker.externalsReadLock();

            if (!externals.containsKey(objectId))
                result.add(objectId);

//            locker.externalsReadUnlock();
        }
        return result;
    }

    /*
        Seeks for unused device memory chunks, up to targetFreeMemory size.
        Set to Long.MAX_VALUE to demote all unused device memory
     */
    protected synchronized long demoteUnused(Integer deviceId, long targetFreeMemory, double minShortFrequency, double minLongFrequency) {
        long sumCandidates = 0;
        for (Long object: deviceAllocations.keySet()) {
            AllocationPoint point = getAllocationPoint(object);

            if ((point.getTimerLong().getFrequencyOfEvents() > minLongFrequency && point.getTimerShort().getFrequencyOfEvents() > minShortFrequency ) || point.getAccessState() != AccessState.TACK) continue;

            sumCandidates += AllocationUtils.getRequiredMemory(point.getShape());

  //          locker.objectWriteLock(object);

             relocateMemory(object, AllocationStatus.ZERO);

            /*
            if (point.getAllocationStatus() != AllocationStatus.ZERO)
                throw new IllegalStateException("Not a ZERO!");
*/
            deviceAllocations.remove(object);

            zeroAllocations.put(object, point);
//            locker.objectWriteUnlock(object);

            if (sumCandidates >= targetFreeMemory) break;
        }

        if (sumCandidates > 0)
            log.info("Demote candidates sum: " + sumCandidates);

        return sumCandidates;
    }

    /**
     * Seeks for oldest memory chunks on host, and syncs them back to host (if originator is alive, though)
     * if originator isn't alive anymore - memory get's discarded
     */
    protected synchronized void syncbackUnusedZero(long targetMemorySize) {
        // for future locks removal

        List<Long> deadReferences = new ArrayList<>();
        AtomicInteger deadCnt = new AtomicInteger(0);
        AtomicInteger relocCnt = new AtomicInteger(0);

        for (Long object: zeroAllocations.keySet()) {

//                locker.objectReadLock(object);

                AllocationPoint point = getAllocationPoint(object);

                if (point == null || point.getAccessState() == AccessState.TICK) {
//                    locker.objectReadUnlock(object);
                    continue;
                } //else locker.objectReadUnlock(object);

//                locker.externalsReadLock();
                if (!externals.containsKey(object)) {
//                    locker.externalsReadUnlock();
                    // object got expired, throw it away

                    releaseMemory(object, point.getShape());
                    zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()) * -1);

                    zeroAllocations.remove(object);
                    allocationPoints.remove(object);
                    deadReferences.add(object);
                    deadCnt.incrementAndGet();
                } else {
//                    locker.externalsReadUnlock();

                    // object is alive, so we could push it back to JVM, and let it be it's own problem

//                    locker.objectWriteLock(object);
                    if (point.getTimerLong().getFrequencyOfEvents() < 1) {

                        synchronizeHostData(object);

                        releaseMemory(object, point.getShape());

                        zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()) * -1);

                        zeroAllocations.remove(object);
                        allocationPoints.remove(object);
                        deadReferences.add(object);
                        relocCnt.incrementAndGet();

                    }
//                    locker.objectWriteUnlock(object);

                }



     //      locker.objectReadUnlock(object);
/*            for (Long dead: deadReferences) {
                locker.detachObject(dead);
            }
*/
        }

       // log.info("Zero cnt: [" + zeroAllocations.size() + "]; Dead cnt: [" + deadCnt.get() + "]; Reloc cnt: [" + relocCnt.get() + "]");
    }

    protected synchronized void debug_remove_reference(Long object) {
        externals.remove(object);
    }

    protected void deallocateUnused() {
        for (Long object: allocationPoints.keySet()) {
            try {
//                locker.objectReadLock(object);
                AllocationPoint point = allocationPoints.get(object);

                // skip if not in trackable region of memory
                if (point.getAllocationStatus() != AllocationStatus.ZERO && point.getAllocationStatus() != AllocationStatus.DEVICE) continue;

                // skip if not in TACK state
                if (point.getTimerLong().getNumberOfEvents() == 0 && point.getAccessState() == AccessState.TACK) {
//                    locker.objectReadUnlock(object);

                    deallocatePoint(object, point);

//                    locker.objectReadLock(object);
                }
            } finally {
//                locker.objectReadUnlock(object);
            }
        }
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
        try {
            AllocationPoint point = getAllocationPoint(objectId);

    //        locker.objectWriteLock(objectId);

            mover.relocate(point.getAllocationStatus(), targetStatus, point);

            // FIXME: this is WRONG way to tag stuff. Fix this on next iteration.
            if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
                deviceAllocations.put(objectId, point);
                zeroAllocations.remove(objectId);
            } else if (point.getAllocationStatus() == AllocationStatus.ZERO) {
                deviceAllocations.remove(objectId);
                zeroAllocations.put(objectId, point);
            }
        } finally {
    //        locker.objectWriteUnlock(objectId);
        }
    }

    /**
     * This method releases memory targeted by specific shape
     *
     * @param objectId
     */
    protected void releaseMemory(@NonNull Long objectId, @NonNull AllocationShape shape) {
        try {
         //   locker.globalReadLock();

            AllocationPoint point = allocationPoints.get(objectId);

            if (!point.confirmNoActiveDescendants())
                throw new IllegalStateException("There's still some mode descendantds");

            if (point.getAccessState().equals(AccessState.TICK))
                throw new IllegalStateException("Memory is still in use");

            if (shape.equals(point.getShape())) {
                if (point.getNumberOfDescendants() == 0) {
                    mover.free(point);
                    allocationPoints.remove(objectId);
                }
            } else {
                // this is sub-allocation event. we could just remove one of descendants and forget that
                point.dropShape(shape);
            }
        } finally {
         //   locker.globalReadUnlock();
        }
    }

    /**
     * Specifies balancer instance for memory management
     *
     * @param balancer
     */
    protected void setBalancer(@NonNull Balancer balancer) {
        try {
//            locker.globalWriteLock();

            this.balancer = balancer;
            this.balancer.init(configuration, environment);
        } finally {
//            locker.globalWriteUnlock();
        }
    }

    /**
     * Specifies Mover implementation to be used for data transfers
     *
     * @param mover Mover implementation to be used for data transfers
     */
    protected void setMover(@NonNull Mover mover) {
        try {
//            locker.globalWriteLock();

            this.mover = mover;
            this.mover.init(configuration, environment);
        } finally {
//            locker.globalWriteUnlock();
        }
    }

    /**
     * This method sets CUDA environment
     *
     * @param environment
     */
    protected void setEnvironment(@NonNull CudaEnvironment environment) {
        try {
//            locker.globalWriteLock();
            this.environment = environment;
        } finally {
//            locker.globalWriteUnlock();
        }
    }

    /**
     * This method checks, if specific memory region could be moved to device
     *
     * @param objectId
     * @return
     */
    protected AllocationStatus makePromoteDecision(@NonNull Long objectId, @NonNull AllocationShape shape) {
        /*
            There's few possible reasons to promote memory region:
                1. Memory chunk is accessed often
                2. There's enough device memory
                3. There's no more better candidates for device memory.

            There's also few possible reasons to decline promotion:
                1. There's not enough device memory.
                2. Memory was already relocated somehow
                3. Memory is enough, but there's better candidates for the same device memory chunk
        */

        AllocationPoint point = getAllocationPoint(objectId);

        long requiredMemory = AllocationUtils.getRequiredMemory(point.getShape());


        if (point.getAllocationStatus().equals(AllocationStatus.DEVICE)) return AllocationStatus.DEVICE;


//            locker.globalReadLock();

            // first, we check if memory is enough

            // TODO: balancer, affinity & locks should be considered here

            long availableMemory = environment.getAvailableMemoryForDevice(1);

            long allocatedMemory = environment.getAllocatedMemoryForDevice(1);

            long maximumAllocation = configuration.getMaximumDeviceAllocation();

//            locker.globalReadUnlock();
/*
        log.info("Req memory: " + requiredMemory);
        log.info("Available memory:" + availableMemory);
        log.info("Allocated memory: " + allocatedMemory);
        log.info("Maxumum allocation: " + configuration.getMaximumAllocation());
*/
            if (availableMemory > requiredMemory && requiredMemory + allocatedMemory < maximumAllocation) {
                // we have available memory, let's consider allocation allowed
                return AllocationStatus.DEVICE;
            } else {
                // we don't have available memory

                double minShortFrequency = point.getTimerShort().getFrequencyOfEvents();
                double minLongFrequency = point.getTimerLong().getFrequencyOfEvents();

                long unused = demoteUnused(point.getDeviceId(), Long.MAX_VALUE, minShortFrequency, minLongFrequency);
                if (unused >= requiredMemory) return makePromoteDecision(objectId, shape);

                return AllocationStatus.ZERO;
            }
    }

    /**
     * This method checks, if specific memory region could be moved from device to host
     *
     * @param objectId
     * @return
     */
    protected AllocationStatus makeDemoteDecision(@NonNull Long objectId, @NonNull AllocationShape shape) {
        try {
//            locker.globalReadLock();

            AllocationPoint point = getAllocationPoint(objectId);

            if (shape.equals(point.getShape())) {
                if (point.getAccessState().equals(AccessState.TICK)) {
                    log.info("Not a tick");
                    return point.getAllocationStatus();
                }

                // number of ticks should be equal to tacks
                // NOTE: this should be done in single atomic comparison to avoid concurrent breakouts
                if (!point.confirmNoActiveDescendants()) {
                    log.info("There's active descendants");
                    return point.getAllocationStatus();
                }
            } else {
                NestedPoint nestedPoint = point.getNestedPoint(shape);
                if (nestedPoint.getNestedStatus().equals(AllocationStatus.NESTED))
                    throw new IllegalStateException("Can't release [NESTED] shape: [" + shape + "]");

                if (nestedPoint.getAccessState().equals(AccessState.TICK)) return nestedPoint.getNestedStatus();
            }

            return balancer.makeDemoteDecision(1, point, shape);
        } finally {
//            locker.globalReadUnlock();
        }
    }


    protected Map<Long, AllocationPoint> getZeroAllocations() {
        return zeroAllocations;
    }

    protected Map<Long, AllocationPoint> getDeviceAllocations() {
        return deviceAllocations;
    }


    private static class Observer {
        public Observer(long delay) {

        }
    }
}
