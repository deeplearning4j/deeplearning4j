package org.nd4j.jita.handler.impl;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.concurrency.DeviceAllocationsTracker;
import org.nd4j.jita.allocator.context.ContextPool;
import org.nd4j.jita.allocator.context.ExternalContext;
import org.nd4j.jita.allocator.context.impl.LimitedContextPool;
import org.nd4j.jita.allocator.context.impl.PackedContextPool;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.jita.flow.impl.AsynchronousFlowController;
import org.nd4j.jita.flow.impl.GridFlowController;
import org.nd4j.jita.handler.MemoryHandler;
import org.nd4j.jita.memory.MemoryProvider;
import org.nd4j.jita.memory.impl.CudaCachingZeroProvider;
import org.nd4j.jita.memory.impl.CudaDirectProvider;
import org.nd4j.jita.memory.impl.CudaFullCachingProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
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
 *
 * @author raver119@gmail.com
 */
public class CudaZeroHandler implements MemoryHandler {
    private static Configuration configuration = CudaEnvironment.getInstance().getConfiguration();

    private static Logger log = LoggerFactory.getLogger(CudaZeroHandler.class);

    // simple counter to track allocated host-memory
    protected final AtomicLong zeroUseCounter = new AtomicLong(0);

    // another simple counter, to track allocated device memory on per-thread per-device basis
    protected volatile DeviceAllocationsTracker deviceMemoryTracker;

    // tracker for thread->device affinity
    protected Map<Long, Integer> devicesAffinity = new ConcurrentHashMap<>();

    private ReentrantReadWriteLock deviceLock = new ReentrantReadWriteLock();

    private AtomicInteger devPtr = new AtomicInteger(0);

    private final AtomicBoolean wasInitialised = new AtomicBoolean(false);

    private final ContextPool contextPool;

    @Getter
    private final MemoryProvider memoryProvider;

    private final FlowController flowController;

    private final AllocationStatus INITIAL_LOCATION;

    private final AffinityManager affinityManager = Nd4j.getAffinityManager();

    /*
    table for Thread, Device, Object allocations of device memory. Objects should be used to grab Allocation point from allocationsMap
    */
    // TODO: proper thread-safe implementation would be nice to have here :(
    // FIXME: CopyOnWriteArrayList is BAD here. Really BAD. B A D.
    // Table thread safety is guaranteed by reentrant read/write locks :(
    //private Table<Long, Integer, ConcurrentHashMap<Long, Long>> deviceAllocations = HashBasedTable.create();
    //private final Map<Integer, ConcurrentHashMap<Long, Long>> deviceAllocations = new ConcurrentHashMap<>();
    private final List<ConcurrentHashMap<Long, Long>> deviceAllocations = new ArrayList<>();

    /*
        map for Thread, Object allocations in zero memory.
    */
    // CopyOnWriteArrayList performance to be investigated in this use case
    // Map thread safety is guaranteed by exclusive writeLock in getDeviceId() method, because we can't use putIfAbsent on j7
    // FIXME: at j7 -> j8 transition, this one could be changed to ConcurrentHashMap
    private final Map<Long, ConcurrentHashMap<Long, Long>> zeroAllocations = new ConcurrentHashMap<>();


    private AtomicLong zeroCounter = new AtomicLong(0);

    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    public CudaZeroHandler() {

        configuration.setInitialized();

        this.INITIAL_LOCATION = configuration.getFirstMemory();

        switch (configuration.getExecutionModel()) {
            case OPTIMIZED:
            case ASYNCHRONOUS: {
                this.flowController = new AsynchronousFlowController();
                this.contextPool = new PackedContextPool();
            }
                break;
            case SEQUENTIAL: {
                this.flowController = new GridFlowController();
                this.contextPool = new LimitedContextPool();
            }
                break;
            default:
                throw new RuntimeException("Unknown ExecutionModel: [" + configuration.getExecutionModel() + "]");
        }

        switch (configuration.getAllocationModel()) {
            case CACHE_ALL:
                this.memoryProvider = new CudaFullCachingProvider();
                break;
            case CACHE_HOST:
                this.memoryProvider = new CudaCachingZeroProvider();
                break;
            case DIRECT:
                this.memoryProvider = new CudaDirectProvider();
                break;
            default:
                throw new RuntimeException("Unknown AllocationModel: [" + configuration.getAllocationModel() + "]");
        }

        int numDevices = NativeOpsHolder.getInstance().getDeviceNativeOps().getAvailableDevices();
        for (int i = 0; i < numDevices; i++) {
            deviceAllocations.add(new ConcurrentHashMap<Long, Long>());
        }

        if (NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceMajor(new CudaPointer(0)) < 3) {
            throw new ND4JIllegalStateException("CUDA backend requires compute capatibility of 3.0 and above to run.");
        }
    }

    /**
     * This method gets called from Allocator, during Allocator/MemoryHandler initialization
     *
     * @param configuration
     * @param allocator
     */
    @Override
    public void init(@NonNull Configuration configuration, @NonNull Allocator allocator) {
        this.configuration = configuration;

        this.deviceMemoryTracker = new DeviceAllocationsTracker(this.configuration);
        this.flowController.init(allocator);
    }

    private void pickupHostAllocation(AllocationPoint point) {
        int numBuckets = configuration.getNumberOfGcThreads();
        long bucketId = RandomUtils.nextInt(0, numBuckets);

        long reqMemory = AllocationUtils.getRequiredMemory(point.getShape());

        zeroUseCounter.addAndGet(reqMemory);

        point.setBucketId(bucketId);

        if (!zeroAllocations.containsKey(bucketId)) {
            log.debug("Creating bucketID: " + bucketId);
            synchronized (this) {
                if (!zeroAllocations.containsKey(bucketId)) {
                    zeroAllocations.put(bucketId, new ConcurrentHashMap<Long, Long>());
                }
            }
        }

        zeroAllocations.get(bucketId).put(point.getObjectId(), point.getObjectId());
    }


    /**
     * Allocate specified memory chunk on specified device/host
     *
     * @param targetMode valid arguments are DEVICE, ZERO
     * @param shape
     * @return
     */
    @Override
    public PointersPair alloc(AllocationStatus targetMode, AllocationPoint point, AllocationShape shape,
                    boolean initialize) {

        long reqMemory = AllocationUtils.getRequiredMemory(shape);
        CudaContext context = getCudaContext();
        switch (targetMode) {
            case HOST: {
                if (zeroUseCounter.get() + reqMemory >= configuration.getMaximumZeroAllocation()) {
                    if (reqMemory > configuration.getMaximumZeroAllocation()) {
                        throw new IllegalStateException(
                                        "You can't allocate more memory, then allowed with configured value: ["
                                                        + configuration.getMaximumZeroAllocation() + "]");
                    }


                    while (zeroUseCounter.get() + reqMemory >= configuration.getMaximumZeroAllocation()) {
                        try {
                            log.warn("No available [HOST] memory, sleeping for a while...");
                            log.debug("Currently used: [" + zeroUseCounter.get() + "], allocated objects: ["
                                            + zeroAllocations.get(0) + "]");

                            Nd4j.getMemoryManager().invokeGc();
                            Thread.sleep(1000);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                }


                PointersPair pair = memoryProvider.malloc(shape, point, targetMode);


                if (initialize) {
                    org.bytedeco.javacpp.Pointer.memset(pair.getHostPointer(), 0, reqMemory);
                    point.tickHostWrite();
                }


                pickupHostAllocation(point);

                return pair;
            }
            case DEVICE: {
                int deviceId = getDeviceId();

                PointersPair returnPair = new PointersPair();
                PointersPair tmpPair = new PointersPair();

                // if the initial memory location is device, there's a chance we don't have zero memory allocated
                if (point.getPointers() == null || point.getPointers().getHostPointer() == null) {
                    tmpPair = alloc(AllocationStatus.HOST, point, point.getShape(), initialize);

                    returnPair.setDevicePointer(tmpPair.getHostPointer());
                    returnPair.setHostPointer(tmpPair.getHostPointer());

                    point.setAllocationStatus(AllocationStatus.HOST);
                    point.setPointers(tmpPair);
                }
/*
                if (reqMemory < configuration.getMaximumSingleHostAllocation()
                                && deviceMemoryTracker.getAllocatedSize(deviceId) + reqMemory < configuration
                                                .getMaximumDeviceAllocation()) {
*/
                    if (deviceMemoryTracker.reserveAllocationIfPossible(Thread.currentThread().getId(), deviceId,
                                    reqMemory)) {
                        point.setDeviceId(deviceId);
                        PointersPair pair = memoryProvider.malloc(shape, point, targetMode);
                        if (pair != null) {
                            //  log.info("PEWPEW");
                            returnPair.setDevicePointer(pair.getDevicePointer());

                            point.setAllocationStatus(AllocationStatus.DEVICE);

                            if (point.getPointers() == null)
                                throw new RuntimeException("WTF?");

                            point.getPointers().setDevicePointer(pair.getDevicePointer());

                            deviceAllocations.get(deviceId).put(point.getObjectId(), point.getObjectId());


                            val p = point.getBucketId();

                            if (p != null) {
                                val m = zeroAllocations.get(point.getBucketId());

                                // m can be null, if that's point from workspace - just no bucketId for it
                                if (m != null)
                                    m.remove(point.getObjectId());
                            }

                            deviceMemoryTracker.addToAllocation(Thread.currentThread().getId(), deviceId, reqMemory);

                            //  point.tickDeviceWrite();
                            point.tickHostWrite();

                            if (!initialize) {
                                point.tickDeviceWrite();
                                point.tickHostRead();
                            } else {
                                //CudaContext ctx = AtomicAllocator.getInstance().getFlowController().prepareAction(point);

                                nativeOps.memsetAsync(pair.getDevicePointer(), 0, reqMemory, 0,
                                                context.getSpecialStream());
                                context.getSpecialStream().synchronize();

                                point.tickDeviceWrite();
                                point.tickHostRead();

                                //AtomicAllocator.getInstance().getFlowController().registerAction(ctx, point);
                            }
                        } else {
                            log.warn("Out of [DEVICE] memory, host memory will be used instead: deviceId: [{}], requested bytes: [{}]",
                                            deviceId, reqMemory);
                            // if device memory allocation failed (aka returned NULL), keep using host memory instead

                            returnPair.setDevicePointer(tmpPair.getHostPointer());

                            point.setAllocationStatus(AllocationStatus.HOST);

                            Nd4j.getMemoryManager().invokeGc();
                            try {
                                Thread.sleep(100);
                            } catch (Exception e) {

                            }
                        }
                    } else {
                        log.warn("Hard limit on [DEVICE] memory hit, please consider tuning memory parameters, deviceId [{}]",
                                        deviceId);

                        Nd4j.getMemoryManager().invokeGc();
                        try {
                            Thread.sleep(100);
                        } catch (Exception e) {

                        }
                    }
               /* } else {
                    log.warn("Soft limit on [DEVICE] memory hit, please consider tuning memory parameters, deviceId [{}]",
                                    deviceId);

                    Nd4j.getMemoryManager().invokeGc();
                    try {
                        Thread.sleep(100);
                    } catch (Exception e) {

                    }
                }*/

                return returnPair;
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
        return memoryProvider.pingDeviceForFreeMemory(deviceId, requiredMemory);
    }

    /**
     * Copies specific chunk of memory from one storage to another
     *
     * Possible directions:  HOST -> DEVICE, DEVICE -> HOST
     *
     * @param currentStatus
     * @param targetStatus
     * @param point
     */
    @Override
    public void relocate(AllocationStatus currentStatus, AllocationStatus targetStatus, AllocationPoint point,
                    AllocationShape shape, CudaContext context) {
        //log.info("RELOCATE CALLED: [" +currentStatus+ "] -> ["+targetStatus+"]");

        if (currentStatus == AllocationStatus.DEVICE && targetStatus == AllocationStatus.HOST) {
            // DEVICE -> HOST
            DataBuffer targetBuffer = point.getBuffer();
            if (targetBuffer == null)
                throw new IllegalStateException("Target buffer is NULL!");

            Pointer devicePointer = new CudaPointer(point.getPointers().getDevicePointer().address());

        } else if (currentStatus == AllocationStatus.HOST && targetStatus == AllocationStatus.DEVICE) {
            // HOST -> DEVICE


            // TODO: this probably should be removed
            if (point.isConstant()) {
                //log.info("Skipping relocation for constant");
                return;
            }

            if (point.getPointers().getDevicePointer() == null) {
                throw new IllegalStateException("devicePointer is NULL!");
            }

            val profD = PerformanceTracker.getInstance().helperStartTransaction();

            if (nativeOps.memcpyAsync(point.getPointers().getDevicePointer(), point.getPointers().getHostPointer(),
                            AllocationUtils.getRequiredMemory(shape), CudaConstants.cudaMemcpyHostToDevice,
                            context.getSpecialStream()) == 0)
                throw new IllegalStateException("MemcpyAsync relocate H2D failed: [" + point.getHostPointer().address()
                                + "] -> [" + point.getDevicePointer().address() + "]");

            flowController.commitTransfer(context.getSpecialStream());

            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), profD, point.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

            //context.syncOldStream();

        } else
            throw new UnsupportedOperationException("Can't relocate data in requested direction: [" + currentStatus
                            + "] -> [" + targetStatus + "]");
    }

    /**
     * Copies memory from device to host, if needed.
     * Device copy is preserved as is.
     *
     * @param point
     */
    @Override
    @Deprecated
    public void copyback(AllocationPoint point, AllocationShape shape) {
        /*
            Technically that's just a case for relocate, with source as point.getAllocationStatus() and target HOST
         */
        //   log.info("copyback() called on shape: " + point.getShape());
        //  relocate(point.getAllocationStatus(), AllocationStatus.HOST, point, shape);
        throw new UnsupportedOperationException("Deprecated call");
    }

    /**
     * Copies memory from host buffer to device.
     * Host copy is preserved as is.
     *
     * @param point
     */
    @Override
    @Deprecated
    public void copyforward(AllocationPoint point, AllocationShape shape) {
        /*
            Technically that's just a case for relocate, with source as HOST and target point.getAllocationStatus()
         */
        log.info("copyforward() called on tp[" + point.getObjectId() + "], shape: " + point.getShape());
        //relocate(AllocationStatus.HOST, point.getAllocationStatus(), point, shape);
        throw new UnsupportedOperationException("Deprecated call");
    }

    /**
     * Copies memory from device to zero-copy memory
     *
     * @param point
     * @param shape
     */
    @Override
    @Deprecated
    public void fallback(AllocationPoint point, AllocationShape shape) {
        throw new IllegalStateException("Can't fallback from [" + point.getAllocationStatus() + "]");
    }

    /**
     * This method frees memory chunk specified by pointer and location
     *
     * @param point Pointer
     */
    @Override
    public void free(AllocationPoint point, AllocationStatus target) {
        //if (point.getAllocationStatus() == AllocationStatus.DEVICE)
        //deviceAllocations.get(point.getDeviceId()).remove(point.getObjectId());

        //zeroAllocations.get(point.getBucketId()).remove(point.getObjectId());
        if (point.getAllocationStatus() == AllocationStatus.DEVICE)
            deviceMemoryTracker.subFromAllocation(Thread.currentThread().getId(), point.getDeviceId(),
                            AllocationUtils.getRequiredMemory(point.getShape()));

        memoryProvider.free(point);
    }

    /**
     * This method returns initial allocation location. So, it can be HOST, or DEVICE if environment allows that.
     *
     * @return
     */
    @Override
    public AllocationStatus getInitialLocation() {
        return INITIAL_LOCATION;
    }

    /**
     * This method initializes specific device for current thread
     *
     * @param threadId
     * @param deviceId
     */
    @Override
    public void initializeDevice(Long threadId, Integer deviceId) {
        /*
        JCuda.cudaSetDevice(deviceId);
        
        CudaContext context = new CudaContext();
        context.initHandle();
        context.initOldStream();
        //        context.initStream();
        context.associateHandle();
        
        contextPool.put(threadId, context);
        */
    }

    /**
     * Asynchronous version of memcpy
     *
     * PLEASE NOTE: This is device-dependent method, if it's not supported in your environment, blocking call will be used instead.
     *
     * @param dstBuffer
     * @param srcPointer
     * @param length
     * @param dstOffset
     */
    @Override
    public void memcpyAsync(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset) {
        AllocationPoint point = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();
        // we update host memory regardless.
        //Pointer dP = new Pointer((point.getAllocationStatus() == AllocationStatus.DEVICE ? point.getPointers().getDevicePointer().address() : point.getPointers().getHostPointer().address()) + dstOffset);
        Pointer dP = new CudaPointer((point.getPointers().getHostPointer().address()) + dstOffset);
        //        Pointer sP = new Pointer(srcPointer.getNativePointer());
        //log.info("Location: " + point.getAllocationStatus());
        //        if (length > 4)
        //log.info("memcpyAsync:  ["+ srcPointer.getNativePointer()+"] -> ["+ dP.getNativePointer()+"], length: [" + length+ "], offset: ["+ dstOffset+"], dstBufferOffset: ["+(dstBuffer.getElementSize() * dstBuffer.offset()) + "/" + dstBuffer.offset() +"]");

        CudaContext tContext = null;

        if (dstBuffer.isConstant()) {

            org.bytedeco.javacpp.Pointer dstPointer =
                            new CudaPointer(point.getPointers().getHostPointer().address() + dstOffset, 0L);
            org.bytedeco.javacpp.Pointer srcPointerJ = new CudaPointer(srcPointer, length);

            //   log.info("JCPP Memcpy: [{}] -> [{}], length: [{}]", srcPointerJ.address(), dstPointer.address(), length);

            val profD = PerformanceTracker.getInstance().helperStartTransaction();

            org.bytedeco.javacpp.Pointer.memcpy(dstPointer, srcPointerJ, length);

            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), profD, point.getNumberOfBytes(), MemcpyDirection.HOST_TO_HOST);


            point.tickHostRead();
        } else {
            //log.info("Memcpy pointers: [{}] -> [{}]", srcPointer.address(),  dP.address());

            CudaContext context = flowController.prepareAction(point);
            tContext = context;

            val prof = PerformanceTracker.getInstance().helperStartTransaction();

            if (nativeOps.memcpyAsync(dP, srcPointer, length, CudaConstants.cudaMemcpyHostToHost,
                            context.getSpecialStream()) == 0)
                throw new IllegalStateException(
                                "MemcpyAsync H2H failed: [" + srcPointer.address() + "] -> [" + dP.address() + "]");

            flowController.commitTransfer(tContext.getSpecialStream());

            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), prof, point.getNumberOfBytes(),MemcpyDirection.HOST_TO_HOST);

            if (point.getAllocationStatus() == AllocationStatus.HOST)
                flowController.registerAction(context, point);
        }

        // if we're copying something into host memory, but we're on device - we need to provide exact copy to device as well
        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            // TODO: this sounds wrong, and probably memcpy whould check initial direction, like relocate did before
            Pointer rDP = new CudaPointer(point.getPointers().getDevicePointer().address() + dstOffset);

            if (tContext == null)
                tContext = flowController.prepareAction(point);
            //log.info("MemcpyAsync to device... [{}] -> [{}]", dP.getNativePointer(), rDP.getNativePointer());

            val prof = PerformanceTracker.getInstance().helperStartTransaction();

            if (nativeOps.memcpyAsync(rDP, dP, length, CudaConstants.cudaMemcpyHostToDevice,
                            tContext.getSpecialStream()) == 0)
                throw new IllegalStateException(
                                "MemcpyAsync H2D failed: [" + dP.address() + "] -> [" + rDP.address() + "]");

            flowController.commitTransfer(tContext.getSpecialStream());

            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), prof, point.getNumberOfBytes(),MemcpyDirection.HOST_TO_DEVICE);

            flowController.registerAction(tContext, point);


        }
        point.tickDeviceWrite();
    }

    @Override
    public void memcpyDevice(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset,
                    CudaContext context) {
        AllocationPoint point = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();

        Pointer dP = new CudaPointer((point.getPointers().getDevicePointer().address()) + dstOffset);

        if (nativeOps.memcpyAsync(dP, srcPointer, length, CudaConstants.cudaMemcpyDeviceToDevice, context.getOldStream()) == 0)
            throw new ND4JIllegalStateException("memcpyAsync failed");

        point.tickDeviceWrite();
    }

    /**
     * Special memcpy version, addressing shapeInfoDataBuffer copies
     *
     * PLEASE NOTE: Blocking H->H, Async H->D
     *
     * @param dstBuffer
     * @param srcPointer
     * @param length
     * @param dstOffset
     */
    @Override
    public void memcpySpecial(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset) {
        CudaContext context = getCudaContext();
        AllocationPoint point = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();

        Pointer dP = new CudaPointer((point.getPointers().getHostPointer().address()) + dstOffset);

        val profH = PerformanceTracker.getInstance().helperStartTransaction();

        if (nativeOps.memcpyAsync(dP, srcPointer, length, CudaConstants.cudaMemcpyHostToHost, context.getOldStream()) == 0)
            throw new ND4JIllegalStateException("memcpyAsync failed");

        PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), profH, point.getNumberOfBytes(),MemcpyDirection.HOST_TO_HOST);

        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            Pointer rDP = new CudaPointer(point.getPointers().getDevicePointer().address() + dstOffset);

            val profD = PerformanceTracker.getInstance().helperStartTransaction();

            if (nativeOps.memcpyAsync(rDP, dP, length, CudaConstants.cudaMemcpyHostToDevice, context.getOldStream()) == 0)
                throw new ND4JIllegalStateException("memcpyAsync failed");

            context.syncOldStream();

            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), profD, point.getNumberOfBytes(),MemcpyDirection.HOST_TO_DEVICE);
        }

        context.syncOldStream();


        point.tickDeviceWrite();
    }



    /**
     *  Synchronous version of memcpy.
     *
     *
     * @param dstBuffer
     * @param srcPointer
     * @param length
     * @param dstOffset
     */
    @Override
    public void memcpyBlocking(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset) {
        // internally it's just memcpyAsync + sync
        CudaContext context = getCudaContext();
        memcpyAsync(dstBuffer, srcPointer, length, dstOffset);
        context.syncOldStream();
    }

    /**
     *  Synchronous version of memcpy.
     *
     *
     * @param dstBuffer
     * @param srcBuffer
     */
    @Override
    public void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer) {
        //log.info("Buffer MemCpy called");
        //log.info("Memcpy buffer: {} bytes ", dstBuffer.length() * dstBuffer.getElementSize());
        CudaContext context = getCudaContext();
        AllocationPoint dstPoint = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();
        AllocationPoint srcPoint = ((BaseCudaDataBuffer) srcBuffer).getAllocationPoint();

        Pointer dP = null; //new CudaPointer(dstPoint.getPointers().getHostPointer().address());
        Pointer sP = null;
        MemcpyDirection direction = null;

        val profDH = PerformanceTracker.getInstance().helperStartTransaction();



        Nd4j.getExecutioner().push();

        if (srcPoint.isActualOnDeviceSide()) {
            sP = AtomicAllocator.getInstance().getPointer(srcBuffer, context);
            dP = AtomicAllocator.getInstance().getPointer(dstBuffer, context);

            if (nativeOps.memcpyAsync(dP, sP, srcBuffer.length() * srcBuffer.getElementSize(),
                    CudaConstants.cudaMemcpyDeviceToDevice, context.getOldStream()) == 0) {
                throw new ND4JIllegalStateException("memcpyAsync failed");
            }

            dstPoint.tickDeviceWrite();
            direction = MemcpyDirection.DEVICE_TO_DEVICE;
        } else {
            sP = AtomicAllocator.getInstance().getHostPointer(srcBuffer);
            dP = AtomicAllocator.getInstance().getPointer(dstBuffer, context);

            if (nativeOps.memcpyAsync(dP, sP, srcBuffer.length() * srcBuffer.getElementSize(),
                    CudaConstants.cudaMemcpyHostToDevice, context.getOldStream()) == 0) {
                throw new ND4JIllegalStateException("memcpyAsync failed");
            }

            direction = MemcpyDirection.HOST_TO_DEVICE;
        }

        dstPoint.tickDeviceWrite();

        // it has to be blocking call
        context.syncOldStream();

        PerformanceTracker.getInstance().helperRegisterTransaction(srcPoint.getDeviceId(), profDH / 2, dstPoint.getNumberOfBytes(), direction);
//        PerformanceTracker.getInstance().helperRegisterTransaction(dstPoint.getDeviceId(), profDH / 2, dstPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);
    }

    /**
     * PLEASE NOTE: Specific implementation, on systems without special devices can return HostPointer here
     *
     * @param buffer
     * @return
     */
    @Override
    public org.bytedeco.javacpp.Pointer getDevicePointer(DataBuffer buffer, CudaContext context) {
        // TODO: It would be awesome to get rid of typecasting here
        //getCudaContext().syncOldStream();
        AllocationPoint dstPoint = ((BaseCudaDataBuffer) buffer).getAllocationPoint();

        //log.info("getDevicePointer called");
        /*
        if (configuration.getMemoryModel() == Configuration.MemoryModel.DELAYED && dstPoint.getAllocationStatus() == AllocationStatus.HOST) {
        
            // if we have constant buffer (aka shapeInfo or other constant stuff)
            if (buffer.isConstant()) {
                Nd4j.getConstantHandler().moveToConstantSpace(buffer);
            } else {
                PointersPair pair = memoryProvider.malloc(dstPoint.getShape(), dstPoint, AllocationStatus.DEVICE);
        
                if (pair != null) {
                    Integer deviceId = getDeviceId();
        
                    dstPoint.getPointers().setDevicePointer(pair.getDevicePointer());
                    dstPoint.setAllocationStatus(AllocationStatus.DEVICE);
        
                    deviceAllocations.get(deviceId).put(dstPoint.getObjectId(), dstPoint.getObjectId());
        
                    zeroAllocations.get(dstPoint.getBucketId()).remove(dstPoint.getObjectId());
                    deviceMemoryTracker.addToAllocation(Thread.currentThread().getId(), deviceId, AllocationUtils.getRequiredMemory(dstPoint.getShape()));
        
        
                    dstPoint.tickHostWrite();
                }
            }
        }
        */
        // here's the place, where we do care about promotion. but we only care about promotion of original  buffers
        if (dstPoint.getAllocationStatus() == AllocationStatus.HOST && buffer.offset() == 0 && 1 < 0) {
            if (dstPoint.getDeviceTicks() > configuration.getMinimumRelocationThreshold()) {
                // at this point we know, that this request is done withing some existent context
                long requiredMemory = AllocationUtils.getRequiredMemory(dstPoint.getShape());
                if (deviceMemoryTracker.reserveAllocationIfPossible(Thread.currentThread().getId(), getDeviceId(),
                                requiredMemory) && pingDeviceForFreeMemory(getDeviceId(), requiredMemory)) {
                    // so, memory is reserved
                    promoteObject(buffer);
                }
            }
        }


        // if that's device state, we probably might want to update device memory state
        if (dstPoint.getAllocationStatus() == AllocationStatus.DEVICE) {
            if (!dstPoint.isActualOnDeviceSide()) {
                //                log.info("Relocating to GPU");
                relocate(AllocationStatus.HOST, AllocationStatus.DEVICE, dstPoint, dstPoint.getShape(), context);
            } else {
                //  log.info("Buffer is actual on device side: " + dstPoint.getShape());
            }
        } //else log.info("Not on [DEVICE]");


        //  we update memory use counter, to announce that it's somehow used on device
        dstPoint.tickDeviceRead();

        // return pointer with offset if needed. length is specified for constructor compatibility purposes
        CudaPointer p = new CudaPointer(dstPoint.getPointers().getDevicePointer(), buffer.length(),
                        (buffer.offset() * buffer.getElementSize()));
        switch (buffer.dataType()) {
            case DOUBLE:
                return p.asDoublePointer();
            case FLOAT:
                return p.asFloatPointer();
            case INT:
                return p.asIntPointer();
            case HALF:
                return p.asShortPointer();
            default:
                return p;
        }
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

        // return pointer with offset if needed. length is specified for constructor compatibility purposes
        if (dstPoint.getPointers().getHostPointer() == null) {
            log.info("DevicePointer: " + dstPoint.getPointers().getDevicePointer());
            log.info("HostPointer: " + dstPoint.getPointers().getHostPointer());
            log.info("AllocStatus: " + dstPoint.getAllocationStatus());
            throw new RuntimeException("pointer is null");
        }
        //dstPoint.tickHostWrite();
        //dstPoint.tickHostRead();
        //log.info("Requesting host pointer for {}", buffer);
        //getCudaContext().syncOldStream();
        synchronizeThreadDevice(Thread.currentThread().getId(), dstPoint.getDeviceId(), dstPoint);

        CudaPointer p = new CudaPointer(dstPoint.getPointers().getHostPointer(), buffer.length(),
                        (buffer.offset() * buffer.getElementSize()));
        switch (buffer.dataType()) {
            case DOUBLE:
                return p.asDoublePointer();
            case FLOAT:
                return p.asFloatPointer();
            case INT:
                return p.asIntPointer();
            case HALF:
                return p.asShortPointer();
            default:
                return p;
        }
    }

    @Override
    public synchronized void relocateObject(DataBuffer buffer) {
        AllocationPoint dstPoint = AtomicAllocator.getInstance().getAllocationPoint(buffer);

        // we don't relocate non-DEVICE buffers (i.e HOST or CONSTANT)
        if (dstPoint.getAllocationStatus() != AllocationStatus.DEVICE)
            return;

        int deviceId = getDeviceId();


        if (dstPoint.getDeviceId() >= 0 && dstPoint.getDeviceId() == deviceId ) {
            return;
        }

        // FIXME: cross-thread access, might cause problems
        if (!dstPoint.isActualOnHostSide())
            AtomicAllocator.getInstance().synchronizeHostData(buffer);

        if (!dstPoint.isActualOnHostSide())
            throw new RuntimeException("Buffer synchronization failed");

        if (buffer.isAttached() || dstPoint.isAttached()) {
            // if this buffer is Attached, we just relocate to new workspace

            MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();

            if (workspace == null) {
                // if we're out of workspace, we should mark our buffer as detached, so gc will pick it up eventually
                alloc(AllocationStatus.DEVICE, dstPoint, dstPoint.getShape(), false);

                CudaContext context = getCudaContext();

                val profD = PerformanceTracker.getInstance().helperStartTransaction();

                if (nativeOps.memcpyAsync(dstPoint.getDevicePointer(), dstPoint.getHostPointer(),
                        buffer.length() * buffer.getElementSize(), 1, context.getSpecialStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");

                context.syncSpecialStream();

                PerformanceTracker.getInstance().helperRegisterTransaction(dstPoint.getDeviceId(), profD / 2, dstPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

                // updating host pointer now
                alloc(AllocationStatus.HOST, dstPoint, dstPoint.getShape(), false);

                // marking it as detached
                dstPoint.setAttached(false);

                // marking it as proper on device
                dstPoint.tickHostRead();
                dstPoint.tickDeviceWrite();
            } else {
                // this call will automagically take care of workspaces, so it'll be either
                //log.info("Relocating to deviceId [{}], workspace [{}]...", deviceId, workspace.getId());
                BaseCudaDataBuffer nBuffer = (BaseCudaDataBuffer) Nd4j.createBuffer(buffer.length());

                Nd4j.getMemoryManager().memcpy(nBuffer, buffer);

                dstPoint.getPointers().setDevicePointer(nBuffer.getAllocationPoint().getDevicePointer());
                dstPoint.getPointers().setHostPointer(nBuffer.getAllocationPoint().getHostPointer());
                dstPoint.setDeviceId(deviceId);

                dstPoint.tickDeviceRead();
                dstPoint.tickHostRead();
            }


            return;
        }

        if (buffer.isConstant()) {
            // we can't relocate or modify buffers
            throw new RuntimeException("Can't relocateObject() for constant buffer");
        } else {
            //                log.info("Free relocateObject: deviceId: {}, pointer: {}", deviceId, dstPoint.getPointers().getDevicePointer().address());
            memoryProvider.free(dstPoint);
            deviceMemoryTracker.subFromAllocation(Thread.currentThread().getId(), dstPoint.getDeviceId(), AllocationUtils.getRequiredMemory(dstPoint.getShape()));

            // we replace original device pointer with new one
            alloc(AllocationStatus.DEVICE, dstPoint, dstPoint.getShape(), false);

            val profD = PerformanceTracker.getInstance().helperStartTransaction();

            CudaContext context = getCudaContext();
            if (nativeOps.memcpyAsync(dstPoint.getDevicePointer(), dstPoint.getHostPointer(),
                            buffer.length() * buffer.getElementSize(), 1, context.getSpecialStream()) == 0)
                throw new ND4JIllegalStateException("memcpyAsync failed");

            context.syncSpecialStream();

            PerformanceTracker.getInstance().helperRegisterTransaction(dstPoint.getDeviceId(), profD, dstPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

            dstPoint.tickDeviceRead();
            dstPoint.tickHostRead();
        }
    }

    /**
     * This method moves specific object from zero-copy memory to device memory
     *
     * PLEASE NOTE:  DO NOT EVER USE THIS METHOD MANUALLY, UNLESS YOU 100% HAVE TO
     *
     * @return
     */
    @Override
    public boolean promoteObject(DataBuffer buffer) {
        AllocationPoint dstPoint = AtomicAllocator.getInstance().getAllocationPoint(buffer);

        if (dstPoint.getAllocationStatus() != AllocationStatus.HOST)
            return false;

        if (configuration.getMemoryModel() == Configuration.MemoryModel.DELAYED
                        && dstPoint.getAllocationStatus() == AllocationStatus.HOST) {


            // if we have constant buffer (aka shapeInfo or other constant stuff)
            if (buffer.isConstant()) {
                Nd4j.getConstantHandler().moveToConstantSpace(buffer);
            } else {

                PointersPair pair = memoryProvider.malloc(dstPoint.getShape(), dstPoint, AllocationStatus.DEVICE);

                if (pair != null) {
                    Integer deviceId = getDeviceId();
                    //               log.info("Promoting object to device: [{}]", deviceId);

                    dstPoint.getPointers().setDevicePointer(pair.getDevicePointer());
                    dstPoint.setAllocationStatus(AllocationStatus.DEVICE);

                    deviceAllocations.get(deviceId).put(dstPoint.getObjectId(), dstPoint.getObjectId());

                    zeroAllocations.get(dstPoint.getBucketId()).remove(dstPoint.getObjectId());
                    deviceMemoryTracker.addToAllocation(Thread.currentThread().getId(), deviceId,
                                    AllocationUtils.getRequiredMemory(dstPoint.getShape()));


                    dstPoint.tickHostWrite();
                } else
                    throw new RuntimeException("PewPew");

            }
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
        table.put(AllocationStatus.HOST, 0, zeroUseCounter.get());
        for (Integer deviceId : configuration.getAvailableDevices()) {
            table.put(AllocationStatus.DEVICE, deviceId, getAllocatedDeviceMemory(deviceId));
        }
        return table;
    }

    /**
     * This method returns total amount of memory allocated at specified device
     *
     * @param device
     * @return
     */
    @Override
    public long getAllocatedDeviceMemory(Integer device) {
        return deviceMemoryTracker.getAllocatedSize(device);
    }

    /**
     * This method returns total amount of host memory allocated within this MemoryHandler
     *
     * @return
     */
    @Override
    public long getAllocatedHostMemory() {
        return zeroUseCounter.get();
    }

    /**
     * This method returns total number of object allocated on specified device
     *
     * @param deviceId
     * @return
     */
    @Override
    public long getAllocatedDeviceObjects(Integer deviceId) {
        return deviceAllocations.get(deviceId).size();
    }

    /**
     * This method returns number of allocated objects within specific bucket
     *
     * @param bucketId
     * @return
     */
    @Override
    public long getAllocatedHostObjects(Long bucketId) {
        if (zeroAllocations.containsKey(bucketId))
            return zeroAllocations.get(bucketId).size();
        else
            return 0L;
    }

    /**
     * This method returns total number of allocated objects in host memory
     * @return
     */
    @Override
    public long getAllocatedHostObjects() {
        AtomicLong counter = new AtomicLong(0);
        for (Long threadId : zeroAllocations.keySet()) {
            counter.addAndGet(zeroAllocations.get(threadId).size());
        }
        return counter.get();
    }

    /**
     * This method returns set of allocation tracking IDs for specific device
     *
     * @param deviceId
     * @return
     */
    @Override
    public Set<Long> getDeviceTrackingPoints(Integer deviceId) {
        return deviceAllocations.get(deviceId).keySet();
    }

    /**
     * This method returns sets of allocation tracking IDs for specific bucket
     *
     * @param bucketId
     * @return
     */
    @Override
    public Set<Long> getHostTrackingPoints(Long bucketId) {
        if (!zeroAllocations.containsKey(bucketId)) {
            return new HashSet<>();
        }
        return zeroAllocations.get(bucketId).keySet();
    }


    /**
     * This method explicitly removes object from device memory.
     *
     * @param threadId
     * @param objectId
     * @param copyback  if TRUE, corresponding memory block on JVM side will be updated, if FALSE - memory will be just discarded
     */
    @Override
    public void purgeDeviceObject(Long threadId, Integer deviceId, Long objectId, AllocationPoint point,
                    boolean copyback) {
        if (point.getAllocationStatus() != AllocationStatus.DEVICE)
            return;

        flowController.waitTillReleased(point);

        free(point, AllocationStatus.DEVICE);

        if (!deviceAllocations.get(deviceId).containsKey(objectId))
            throw new IllegalStateException("Can't happen ever");

        forget(point, AllocationStatus.DEVICE);

        if (deviceAllocations.get(deviceId).containsKey(objectId))
            throw new IllegalStateException("Can't happen ever");

        deviceMemoryTracker.subFromAllocation(threadId, deviceId, AllocationUtils.getRequiredMemory(point.getShape()));

        point.setAllocationStatus(AllocationStatus.HOST);

        //environment.trackAllocatedMemory(deviceId, AllocationUtils.getRequiredMemory(point.getShape()));
    }

    /**
     * This method explicitly removes object from zero-copy memory.
     *
     * @param bucketId
     * @param objectId
     * @param copyback  if TRUE, corresponding memory block on JVM side will be updated, if FALSE - memory will be just discarded
     */
    @Override
    public void purgeZeroObject(Long bucketId, Long objectId, AllocationPoint point, boolean copyback) {
        forget(point, AllocationStatus.HOST);

        flowController.waitTillReleased(point);

        // we call for caseless deallocation here
        //JCudaDriver.cuCtxSetCurrent(contextPool.getCuContextForDevice(0));
        free(point, AllocationStatus.HOST);

        point.setAllocationStatus(AllocationStatus.DEALLOCATED);

        long reqMem = AllocationUtils.getRequiredMemory(point.getShape()) * -1;
        zeroUseCounter.addAndGet(reqMem);
    }

    @Override
    public void forget(AllocationPoint point, AllocationStatus location) {
        if (location == AllocationStatus.DEVICE) {
            deviceAllocations.get(point.getDeviceId()).remove(point.getObjectId());
        } else if (location == AllocationStatus.HOST) {
            zeroAllocations.get(point.getBucketId()).remove(point.getObjectId());
        }
    }


    /**
     * This method returns CUDA deviceId for current thread
     *
     * @return
     */
    public Integer getDeviceId() {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        return deviceId;
    }

    /** Returns {@link #getDeviceId()} wrapped as a {@link Pointer}. */
    @Override
    public Pointer getDeviceIdPointer() {
        return new CudaPointer(getDeviceId());
    }

    /**
     * This method returns set of available devices
     * @return
     */
    @Override
    public Set<Integer> getAvailableDevices() {
        return new HashSet<>(configuration.getAvailableDevices());
    }

    /**
     * This method returns ExternalContext wrapper (if applicable)
     * @return
     */
    @Override
    public ExternalContext getDeviceContext() {
        return new ExternalContext(getCudaContext());
    }

    /**
     * This method returns CudaContext for current thread. If context doesn't exist - it gets created first.
     * @return
     */
    public CudaContext getCudaContext() {
        // FIXME: remove this before release
        Integer deviceId = getDeviceId();
        return contextPool.acquireContextForDevice(deviceId);
    }


    /**
     * This method does initialization for thread.
     *
     *
     * @param threadId
     */
    protected void initCudaContextForThread(Long threadId) {

        // we set device to be used prior to stream creation

        nativeOps.setDevice(getDeviceIdPointer());

        CudaContext context = new CudaContext();
        context.initHandle();
        context.initOldStream();
        context.initStream();
        context.associateHandle();
        //contextPool.put(threadId, context);
    }

    /**
     * This method returns if this MemoryHandler instance is device-dependant (i.e. CUDA)
     *
     * @return TRUE if dependant, FALSE otherwise
     */
    @Override
    public boolean isDeviceDependant() {
        // this is always TRUE for current implementation
        return true;
    }

    /**
     * This method causes memory synchronization on host side.
     *  Viable only for Device-dependant MemoryHandlers
     *
     * @param threadId
     * @param deviceId
     * @param point
     */
    @Override
    public void synchronizeThreadDevice(Long threadId, Integer deviceId, AllocationPoint point) {
        // we synchronize only if this AllocationPoint was used within device context, so for multiple consequent syncs only first one will be issued
        flowController.synchronizeToHost(point);
    }

    @Override
    public void registerAction(CudaContext context, INDArray result, INDArray... operands) {
        flowController.registerAction(context, result, operands);
    }

    @Override
    public FlowController getFlowController() {
        return flowController;
    }

    @Override
    public ContextPool getContextPool() {
        return contextPool;
    }


}
