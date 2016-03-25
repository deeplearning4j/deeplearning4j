package org.nd4j.jita.handler.impl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import lombok.NonNull;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.concurrency.DeviceAllocationsTracker;
import org.nd4j.jita.allocator.context.ExternalContext;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.*;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.memory.impl.CudaCachingProvider;
import org.nd4j.jita.memory.MemoryProvider;
import org.nd4j.jita.handler.MemoryHandler;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.NioUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This Mover implementation uses following techs:
 * 1. Unified Memory Architecture
 * 2. Zero-Copy Pinned Memory (if available)
 * 3. Pageable memory (if zero-copy pinned memory isn't supported by device)
 *
 * Current drawbacks:
 * 1. For each allocation it's using it's own separate malloc call.
 * 2. Result arrays/scalars are note covered yet.
 *
 * @author raver119@gmail.com
 */
public class CudaZeroHandler implements MemoryHandler {
    private Configuration configuration;
    private CudaEnvironment environment;
    private static Allocator allocator = AtomicAllocator.getInstance();

    private static Logger log = LoggerFactory.getLogger(CudaZeroHandler.class);

    // simple counter to track allocated host-memory
    protected final AtomicLong zeroUseCounter = new AtomicLong(0);

    // simple pool for cublas contexts
    private Map<Long, CudaContext> contextPool = new ConcurrentHashMap<>();


    // another simple counter, to track allocated device memory on per-thread per-device basis
    protected volatile DeviceAllocationsTracker deviceMemoryTracker;

    // tracker for thread->device affinity
    protected Map<Long, Integer> devicesAffinity = new ConcurrentHashMap<>();

    private ReentrantReadWriteLock deviceLock = new ReentrantReadWriteLock();

    private AtomicInteger devPtr = new AtomicInteger(0);

    private final AtomicBoolean wasInitialised = new AtomicBoolean(false);

    private final MemoryProvider provider = new CudaCachingProvider();

    /*
    table for Thread, Device, Object allocations of device memory. Objects should be used to grab Allocation point from allocationsMap
*/
    // TODO: proper thread-safe implementation would be nice to have here :(
    // FIXME: CopyOnWriteArrayList is BAD here. Really BAD. B A D.
    // Table thread safety is guaranteed by reentrant read/write locks :(
    //private Table<Long, Integer, ConcurrentHashMap<Long, Long>> deviceAllocations = HashBasedTable.create();
    private Map<Integer, ConcurrentHashMap<Long, Long>> deviceAllocations = new ConcurrentHashMap<>();

    /*
        map for Thread, Object allocations in zero memory.
    */
    // CopyOnWriteArrayList performance to be investigated in this use case
    // Map thread safety is guaranteed by exclusive writeLock in getDeviceId() method, because we can't use putIfAbsent on j7
    // FIXME: at j7 -> j8 transition, this one could be changed to ConcurrentHashMap
    private Map<Long, ConcurrentHashMap<Long, Long>> zeroAllocations = new HashMap<>();



    public CudaZeroHandler() {
        allocator = AtomicAllocator.getInstance();
    }

    @Override
    public void init(@NonNull Configuration configuration, @NonNull CudaEnvironment environment, @NonNull Allocator allocator) {
        this.configuration = configuration;
        this.environment = environment;
        this.allocator = allocator;

        this.deviceMemoryTracker = new DeviceAllocationsTracker(this.environment, this.configuration);
        initCudaContextForThread(Thread.currentThread().getId());
    }

    /**
     * Allocate specified memory chunk on specified device/host
     *
     * @param targetMode valid arguments are DEVICE, ZERO
     * @param shape
     * @return
     */
    @Override
    public PointersPair alloc(AllocationStatus targetMode, AllocationPoint point, AllocationShape shape) {
        //log.info("Alloc called for shape: " + shape);
        //if (shape.getLength() == 757) throw new RuntimeException("757");
        //log.info("Memory required: " + AllocationUtils.getRequiredMemory(shape));
        switch (targetMode) {
            case HOST: {
                PointersPair pair = provider.malloc(shape, point, targetMode);

                zeroAllocations.get(Thread.currentThread().getId()).put(point.getObjectId(), point.getObjectId());
                zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(shape));

                return pair;
            }
            case DEVICE: {
                PointersPair pair = provider.malloc(shape, point, targetMode);

                return pair;
            }
            default:
                throw new IllegalStateException("Can't allocate memory on target [" + targetMode + "]");
        }
    }

    /**
     * This method checks if specified device has free memory
     *
     * @param deviceId
     * @param requiredMemory
     * @return
     */
    @Override
    public boolean pingDeviceForFreeMemory(Integer deviceId, long requiredMemory) {
        long[] totalMem = new long[1];
        long[] freeMem = new long[1];

        JCuda.cudaMemGetInfo(freeMem, totalMem);

        long free = freeMem[0];
        long total = totalMem[0];
        long used = total - free;

        /*
            We don't want to allocate memory if it's too close to the end of available ram.
         */
        if (configuration != null && used > total * configuration.getMaxDeviceMemoryUsed()) return false;

        if (configuration != null && free + requiredMemory < total * configuration.getMaxDeviceMemoryUsed())
            return true;
        else return false;
    }

    /**
     * Copies specific chunk of memory from one storage to another
     *
     * Possible directions:  DEVICE -> ZERO, ZERO -> DEVICE, ZERO -> HOST, DEVICE -> HOST
     *
     * @param currentStatus
     * @param targetStatus
     * @param point
     */
    @Override
    public void relocate(AllocationStatus currentStatus, AllocationStatus targetStatus, AllocationPoint point, AllocationShape shape) {
        //log.info("RELOCATE CALLED: [" +currentStatus+ "] -> ["+targetStatus+"]");

     if (currentStatus == AllocationStatus.DEVICE && targetStatus == AllocationStatus.HOST) {
            // DEVICE -> HOST
            DataBuffer targetBuffer = point.getBuffer();
            if (targetBuffer == null)
                throw new IllegalStateException("Target buffer is NULL!");

            Pointer devicePointer = new Pointer(point.getPointers().getDevicePointer().address());

            CudaContext context = getCudaContext();

            // we must be sure, no calculations are pending within these streams before copyback
            context.syncOldStream();
            context.syncStream();

            JCuda.cudaMemcpyAsync(
                    PointerUtil.getHostPointer(targetBuffer),
                    devicePointer,
                    AllocationUtils.getRequiredMemory(shape),
                    cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    context.getOldStream()
            );

            context.syncOldStream();

        } else if (currentStatus == AllocationStatus.HOST && targetStatus == AllocationStatus.DEVICE) {
            // HOST -> DEVICE

            if (point.getPointers().getDevicePointer() == null) {
                 throw new IllegalStateException("devicePointer is NULL!");
            }

            Pointer devicePointer = new Pointer(point.getPointers().getDevicePointer().address());

            Pointer hostPointer = new Pointer(point.getPointers().getHostPointer().address());

            CudaContext context = getCudaContext();

            JCuda.cudaMemcpyAsync(
                 devicePointer,
                 hostPointer,
                 AllocationUtils.getRequiredMemory(shape),
                 cudaMemcpyKind.cudaMemcpyHostToDevice,
                 context.getOldStream()
             );

            context.syncOldStream();

        } else throw new UnsupportedOperationException("Can't relocate data in requested direction: [" + currentStatus + "] -> [" + targetStatus + "]");
    }

    /**
     * Copies memory from device to host, if needed.
     * Device copy is preserved as is.
     *
     * @param point
     */
    @Override
    public void copyback(AllocationPoint point, AllocationShape shape) {
        /*
            Technically that's just a case for relocate, with source as point.getAllocationStatus() and target HOST
         */
        //   log.info("copyback() called on shape: " + point.getShape());
        relocate(point.getAllocationStatus(), AllocationStatus.HOST, point, shape);
    }

    /**
     * Copies memory from host buffer to device.
     * Host copy is preserved as is.
     *
     * @param point
     */
    @Override
    public void copyforward(AllocationPoint point, AllocationShape shape) {
        /*
            Technically that's just a case for relocate, with source as HOST and target point.getAllocationStatus()
         */
 //       log.info("copyforward() called on tp["+point.getObjectId()+"], shape: " + point.getShape());
        relocate(AllocationStatus.HOST, point.getAllocationStatus(), point, shape);
    }

    /**
     * Copies memory from device to zero-copy memory
     *
     * @param point
     * @param shape
     */
    @Override
    public void fallback(AllocationPoint point, AllocationShape shape) {
        if (point.getAllocationStatus() != AllocationStatus.DEVICE)
            throw new IllegalStateException("Can't fallback from ["+point.getAllocationStatus()+"]");

        PointersPair pair = point.getPointers();

        CudaContext context = getCudaContext();

        Pointer devPtr = new Pointer(pair.getDevicePointer().address());
        Pointer hostPointer = new Pointer(pair.getHostPointer().address());
        long reqMem =AllocationUtils.getRequiredMemory(shape);

        JCuda.cudaMemcpyAsync(
                hostPointer,
                devPtr,
                reqMem,
                cudaMemcpyKind.cudaMemcpyDeviceToHost,
                context.getOldStream()
        );

        context.syncOldStream();

        JCuda.cudaFree(devPtr);

        Pointer newDevPtr = new Pointer();

        JCuda.cudaHostGetDevicePointer(
                newDevPtr,
                hostPointer,
                0);

        pair.setDevicePointer(new CudaPointer(devPtr, reqMem));

        /*
        DevicePointerInfo info = alloc(AllocationStatus.ZERO, point, shape);

        CudaContext context = allocator.getCudaContext();

        JCuda.cudaMemcpyAsync(
                info.getPointers().getHostPointer(),
                point.getDevicePointer(),
                AllocationUtils.getRequiredMemory(shape),
                cudaMemcpyKind.cudaMemcpyDeviceToHost,
                context.getOldStream()
        );

        context.syncOldStream();

        JCuda.cudaFree(point.getDevicePointer());

        point.setPointers(info);
        */
    }

    /**
     * This method frees memory chunk specified by pointer and location
     *
     * @param point Pointer
     */
    @Override
    public void free(AllocationPoint point, AllocationStatus target) {
        provider.free(point);
    }

    /**
     * This method returns initial allocation location. So, it can be HOST, or DEVICE if environment allows that.
     *
     * @return
     */
    @Override
    public AllocationStatus getInitialLocation() {
        return AllocationStatus.HOST;
    }

    /**
     * This method initializes specific device for current thread
     *
     * @param threadId
     * @param deviceId
     */
    @Override
    public void initializeDevice(Long threadId, Integer deviceId) {
        JCuda.cudaSetDevice(deviceId);

        CudaContext context = new CudaContext();
        context.initHandle();
        context.initOldStream();
        context.initStream();
        context.associateHandle();

        // FIXME:  context should be treated within mover
        contextPool.put(threadId, context);
    }

    private NioUtil.BufferType getBufferType(DataBuffer buffer) {
        switch(buffer.dataType()) {
            case DOUBLE: return NioUtil.BufferType.DOUBLE;
            case INT: return NioUtil.BufferType.FLOAT;
            case FLOAT: return NioUtil.BufferType.FLOAT;
            default: throw new UnsupportedOperationException("Unsupported data buffer type");
        }
    }

    @Override
    public void memcpyAsync(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset) {
        CudaContext context = getCudaContext();
        AllocationPoint point = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();
        // we update host memory regardless.
        Pointer dP = new Pointer(point.getPointers().getHostPointer().address() + dstOffset);
//        Pointer sP = new Pointer(srcPointer.getNativePointer());

//        if (length > 4)
//            log.info("memcpyAsync:  ["+ srcPointer.getNativePointer()+"] -> ["+ dP.getNativePointer()+"], length: [" + length+ "], offset: ["+ dstOffset+"], dstBufferOffset: ["+(dstBuffer.getElementSize() * dstBuffer.offset()) + "/" + dstBuffer.offset() +"]");

        JCuda.cudaMemcpyAsync(
                dP,
                srcPointer,
                length,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
                context.getOldStream()
        );

        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            Pointer rDP = new Pointer(point.getPointers().getDevicePointer().address() + dstOffset);

            JCuda.cudaMemcpyAsync(
                    rDP,
                    dP,
                    length,
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                    context.getOldStream()
            );
        }

        // TODO: to be removed
        //context.syncOldStream();
    }

    @Override
    public void memcpyBlocking(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset) {
        CudaContext context = getCudaContext();
        memcpyAsync(dstBuffer, srcPointer, length, dstOffset);
        context.syncOldStream();
    }

    @Override
    public void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer) {
        CudaContext context = getCudaContext();
        AllocationPoint dstPoint = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();
        AllocationPoint srcPoint = ((BaseCudaDataBuffer) srcBuffer).getAllocationPoint();

        Pointer dP = new Pointer(dstPoint.getPointers().getHostPointer().address());
        Pointer sP = null;

        if (srcPoint.getAllocationStatus() == AllocationStatus.DEVICE) {
            sP = new Pointer(srcPoint.getPointers().getDevicePointer().address());

            JCuda.cudaMemcpyAsync(
                    dP,
                    sP,
                    srcBuffer.length(),
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                    context.getOldStream()
            );
        } else {
            sP = new Pointer(srcPoint.getPointers().getHostPointer().address());

            JCuda.cudaMemcpyAsync(
                    dP,
                    sP,
                    srcBuffer.length(),
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                    context.getOldStream()
            );
        }

        if (dstPoint.getAllocationStatus() == AllocationStatus.DEVICE) {
            // TODO:  device replication to be implemented
        }

        context.syncOldStream();
    }

    /**
     * PLEASE NOTE: Specific implementation, on systems without special devices can return HostPointer here
     *
     * @param buffer
     * @return
     */
    @Override
    public org.bytedeco.javacpp.Pointer getDevicePointer(DataBuffer buffer) {
        AllocationPoint dstPoint = ((BaseCudaDataBuffer) buffer).getAllocationPoint();

        // here's the place, where we do care about promotion
        if (dstPoint.getAllocationStatus() == AllocationStatus.HOST ) {
            if (dstPoint.getDeviceTicks() > configuration.getMinimumRelocationThreshold()) {
                // at this point we know, that this request is done withing some existent context
                long requiredMemory = AllocationUtils.getRequiredMemory(dstPoint.getShape());
                if (deviceMemoryTracker.reserveAllocationIfPossible(Thread.currentThread().getId(), getDeviceId(), requiredMemory) && pingDeviceForFreeMemory(getDeviceId(), requiredMemory)) {
                    // so, memory is reserved
                    promoteObject(dstPoint.getObjectId(), dstPoint, dstPoint.getShape());
                }
            }
        } else {
            // if that's device state, we probably might want to update device memory state
            if (dstPoint.isActualOnHostSide()) {
//                relocate(AllocationStatus.HOST, AllocationStatus.DEVICE, dstPoint, dstPoint.getShape());
                copyforward(dstPoint, dstPoint.getShape());
            }
        }

        dstPoint.tickDevice();


        return new CudaPointer(dstPoint.getPointers().getDevicePointer(), buffer.length(),  (buffer.offset() * buffer.getElementSize()));
    }

    /**
     * PLEASE NOTE: This method always returns pointer within OS memory space
     *
     * @param buffer
     * @return
     */
    @Override
    public org.bytedeco.javacpp.Pointer getHostPointer(DataBuffer buffer) {
        AllocationPoint dstPoint = ((BaseCudaDataBuffer) buffer).getAllocationPoint();

        return new CudaPointer(dstPoint.getPointers().getHostPointer(), buffer.length(), (buffer.offset() * buffer.getElementSize()));
    }

    /**
     * This method moves specific object from zero-copy memory to device memory
     *
     * PLEASE NOTE:  DO NOT EVER USE THIS METHOD, UNLESS YOU 100% HAVE TO
     *
     * @param trackingPoint
     * @param point
     * @param shape
     * @return
     */
    public boolean promoteObject(Long trackingPoint, AllocationPoint point, AllocationShape shape) {
        try {
            long threadId = Thread.currentThread().getId();

            point.setDeviceId(getDeviceId());



            PointersPair newPointers = alloc(AllocationStatus.DEVICE, point, shape);

            relocate(AllocationStatus.HOST, AllocationStatus.DEVICE, point, shape);

            point.setAllocationStatus(AllocationStatus.DEVICE);
            point.setPointers(newPointers);


            //deviceLock.readLock().lock();

            deviceAllocations.get(point.getDeviceId()).put(trackingPoint, trackingPoint);

            //deviceLock.readLock().unlock();

            zeroAllocations.get(threadId).remove(trackingPoint);

            deviceMemoryTracker.addToAllocation(threadId, point.getDeviceId(), AllocationUtils.getRequiredMemory(shape));

            zeroUseCounter.addAndGet(-1 * AllocationUtils.getRequiredMemory(point.getShape()));

                    log.info("Relocation happened!");
        } catch (Exception e){
            throw new RuntimeException(e);
        }

        return true;
    }

    /**
     * This method returns total amount of memory allocated within system
     *
     * @return
     */
    @Override
    public Table<AllocationStatus, Integer, Long> getAllocationStatistics() {
        Table<AllocationStatus, Integer, Long> table = HashBasedTable.create();
        table.put(AllocationStatus.HOST, 0 , zeroUseCounter.get());

        return table;
    }

    @Override
    public long getAllocatedDeviceMemory(Integer device) {
        return deviceMemoryTracker.getAllocatedSize(device);
    }

    @Override
    public long getAllocatedHostMemory() {
        return zeroUseCounter.get();
    }

    @Override
    public long getAllocatedDeviceObjects(Integer deviceId) {
        return deviceAllocations.get(deviceId).size();
    }

    @Override
    public long getAllocatedHostObjects(Long threadId) {
        return zeroAllocations.get(threadId).size();
    }

    @Override
    public long getAllocatedHostObjects() {
        AtomicLong counter = new AtomicLong(0);
        for (Long threadId: zeroAllocations.keySet()) {
            counter.addAndGet(zeroAllocations.get(threadId).size());
        }
        return counter.get();
    }

    @Override
    public Set<Long> getDeviceTrackingPoints(Integer deviceId) {
        return deviceAllocations.get(deviceId).keySet();
    }

    @Override
    public Set<Long> getHostTrackingPoints(Long threadId) {
        return zeroAllocations.get(threadId).keySet();
    }


    /**
     * This method explicitly removes object from device memory.
     *
     * @param threadId
     * @param objectId
     * @param copyback  if TRUE, corresponding memory block on JVM side will be updated, if FALSE - memory will be just discarded
     */
    @Override
    public void purgeDeviceObject(Long threadId, Integer deviceId, Long objectId, AllocationPoint point, boolean copyback) {
        if (point.getAllocationStatus() == AllocationStatus.HOST)
            return;

        if (copyback) {
            // copyback here basically means that we're gonna have new zero allocation right now
            fallback(point, point.getShape());

//            zeroAllocations.get(threadId).put(objectId, objectId);

            point.setAllocationStatus(AllocationStatus.HOST);
   //         zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()));
        }


        deviceAllocations.get(deviceId).remove(objectId);

        deviceMemoryTracker.subFromAllocation(threadId, deviceId, AllocationUtils.getRequiredMemory(point.getShape()));

        if (!copyback) {
            free(point, AllocationStatus.DEVICE);
        }

        environment.trackAllocatedMemory(deviceId, AllocationUtils.getRequiredMemory(point.getShape()));
    }

    /**
     * This method explicitly removes object from zero-copy memory.
     *
     * @param threadId
     * @param objectId
     * @param copyback  if TRUE, corresponding memory block on JVM side will be updated, if FALSE - memory will be just discarded
     */
    @Override
    public void purgeZeroObject(Long threadId, Long objectId, AllocationPoint point, boolean copyback) {
        if (copyback) {
//            copyback(point, point.getShape());
        }
        zeroAllocations.get(threadId).remove(objectId);


        // we call for caseless deallocation here
        free(point, point.getAllocationStatus());

        point.setAllocationStatus(AllocationStatus.DEALLOCATED);

        zeroUseCounter.addAndGet(AllocationUtils.getRequiredMemory(point.getShape()) * -1);
    }


    /**
     * This method returns CUDA deviceId for current thread
     *
     * @return
     */
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

                    if (!deviceAllocations.containsKey(device)) {
                        deviceAllocations.put(device, new ConcurrentHashMap<Long, Long>());
                    }

                    log.info("Mapping device [" + device + "] to thread [" + Thread.currentThread().getId() + "]");

                    initCudaContextForThread(threadId);
                    initializeDevice(threadId, device);


/*
                    ZeroGarbageCollectorThread thread = new ZeroGarbageCollectorThread(threadId, device, shouldStop);
                    thread.start();
                    collectorsZero.put(threadId, thread);

                    DeviceGarbageCollectorThread dThread = new DeviceGarbageCollectorThread(threadId, device, shouldStop);
                    dThread.start();
                    collectorsDevice.put(threadId, dThread);
                    */
                }
                return devicesAffinity.get(threadId);
            } finally {
                deviceLock.writeLock().unlock();
            }
        } else devicesAffinity.get(Thread.currentThread().getId());

        return devicesAffinity.get(threadId);
    }

    public Set<Integer> getAvailableDevices() {
        return environment.getAvailableDevices().keySet();
    }

    @Override
    public ExternalContext getDeviceContext() {
        return new ExternalContext(getCudaContext());
    }

    public CudaContext getCudaContext() {
        if (!contextPool.containsKey(Thread.currentThread().getId())) {
            initCudaContextForThread(Thread.currentThread().getId());

        }
        return contextPool.get(Thread.currentThread().getId());
    }


    @Deprecated
    protected void initCudaContextForThread(Long threadId) {

        // we set device to be used prior to stream creation

        JCuda.cudaSetDevice(getDeviceId());

        CudaContext context = new CudaContext();
        context.initHandle();
        context.initOldStream();
        context.initStream();
        context.associateHandle();
        contextPool.put(threadId, context);
    }

    @Override
    public boolean isDeviceDependant() {
        return true;
    }

    @Override
    public void synchronizeThreadDevice(Long threadId, Integer deviceId, AllocationPoint point) {
        if (!point.isActualOnHostSide()) {
            //log.info("Calling sync...");
            CudaContext context = getCudaContext();
            context.syncOldStream();
            point.tickHostRead();
            if (point.getAllocationStatus() == AllocationStatus.DEVICE && !point.isActualOnHostSide()) {
                JCuda.cudaMemcpyAsync(
                        new Pointer(point.getHostPointer().address()),
                        new Pointer(point.getDevicePointer().address()),
                        AllocationUtils.getRequiredMemory(point.getShape()),
                        cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        context.getOldStream()
                );

                context.syncOldStream();
            }
        }
    }
}