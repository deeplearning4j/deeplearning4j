/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.jita.handler.impl;

import lombok.var;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;
import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.concurrency.DeviceAllocationsTracker;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.impl.MemoryTracker;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.jita.flow.impl.GridFlowController;
import org.nd4j.jita.handler.MemoryHandler;
import org.nd4j.jita.memory.MemoryProvider;
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
import org.nd4j.linalg.profiler.OpProfiler;
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

    private final FlowController flowController;

    private final AllocationStatus INITIAL_LOCATION;

    private final List<cublasHandle_t> cublasHandles = new ArrayList<>();

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
            case SEQUENTIAL: {
                this.flowController = new GridFlowController();
            }
                break;
            default:
                throw new RuntimeException("Unknown ExecutionModel: [" + configuration.getExecutionModel() + "]");
        }

        int numDevices = NativeOpsHolder.getInstance().getDeviceNativeOps().getAvailableDevices();
        for (int i = 0; i < numDevices; i++) {
            deviceAllocations.add(new ConcurrentHashMap<Long, Long>());
            cublasHandles.add(null);
        }

        if (NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceMajor(0) < 3) {
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

        long reqMemory = point.getNumberOfBytes();

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

            throw new UnsupportedOperationException();
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
        return true;
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
        val point = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();
        CudaContext tContext = null;

        if (dstBuffer.isConstant()) {
            org.bytedeco.javacpp.Pointer dstPointer = new CudaPointer(point.getHostPointer().address() + dstOffset, 0L);
            org.bytedeco.javacpp.Pointer srcPointerJ = new CudaPointer(srcPointer, length);

            val profD = PerformanceTracker.getInstance().helperStartTransaction();
            org.bytedeco.javacpp.Pointer.memcpy(dstPointer, srcPointerJ, length);
            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), profD, point.getNumberOfBytes(), MemcpyDirection.HOST_TO_HOST);

            point.tickHostRead();
        } else {
            // if we're copying something into host memory, but we're on device - we need to provide exact copy to device as well
            Pointer rDP = new CudaPointer(point.getDevicePointer().address() + dstOffset);

            if (tContext == null)
                tContext = flowController.prepareAction(point);

            var prof = PerformanceTracker.getInstance().helperStartTransaction();

            flowController.commitTransfer(tContext.getSpecialStream());

            if (nativeOps.memcpyAsync(rDP, srcPointer, length, CudaConstants.cudaMemcpyHostToDevice, tContext.getSpecialStream()) == 0)
                throw new IllegalStateException("MemcpyAsync H2D failed: [" + srcPointer.address() + "] -> [" + rDP.address() + "]");

            flowController.commitTransfer(tContext.getSpecialStream());

            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), prof, point.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

            flowController.registerAction(tContext, point);
            point.tickDeviceWrite();

            // we optionally copy to host memory
            if (point.getHostPointer() != null) {
                Pointer dP = new CudaPointer((point.getHostPointer().address()) + dstOffset);

                CudaContext context = flowController.prepareAction(point);
                tContext = context;

                prof = PerformanceTracker.getInstance().helperStartTransaction();

                if (nativeOps.memcpyAsync(dP, srcPointer, length, CudaConstants.cudaMemcpyHostToHost, context.getSpecialStream()) == 0)
                    throw new IllegalStateException("MemcpyAsync H2H failed: [" + srcPointer.address() + "] -> [" + dP.address() + "]");

                flowController.commitTransfer(tContext.getSpecialStream());

                PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), prof, point.getNumberOfBytes(), MemcpyDirection.HOST_TO_HOST);

                if (point.getAllocationStatus() == AllocationStatus.HOST)
                    flowController.registerAction(context, point);

                point.tickHostRead();
            }
        }
    }

    @Override
    public void memcpyDevice(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset,
                    CudaContext context) {
        AllocationPoint point = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();

        Pointer dP = new CudaPointer((point.getDevicePointer().address()) + dstOffset);

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

        Pointer dP = new CudaPointer((point.getHostPointer().address()) + dstOffset);

        val profH = PerformanceTracker.getInstance().helperStartTransaction();

        if (nativeOps.memcpyAsync(dP, srcPointer, length, CudaConstants.cudaMemcpyHostToHost, context.getOldStream()) == 0)
            throw new ND4JIllegalStateException("memcpyAsync failed");

        PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), profH, point.getNumberOfBytes(),MemcpyDirection.HOST_TO_HOST);

        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            Pointer rDP = new CudaPointer(point.getDevicePointer().address() + dstOffset);

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
        val dstPoint = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();
        val srcPoint = ((BaseCudaDataBuffer) srcBuffer).getAllocationPoint();

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
        AllocationPoint dstPoint = ((BaseCudaDataBuffer) buffer).getAllocationPoint();

        // if that's device state, we probably might want to update device memory state
        if (dstPoint.getAllocationStatus() == AllocationStatus.DEVICE) {
            if (!dstPoint.isActualOnDeviceSide()) {
                //relocate(AllocationStatus.HOST, AllocationStatus.DEVICE, dstPoint, dstPoint.getShape(), context);
                throw new UnsupportedOperationException("Pew-pew");
            }
        }

        if (dstPoint.getDevicePointer() == null)
            return null;


        // return pointer. length is specified for constructor compatibility purposes. Offset is accounted at C++ side
        val p = new CudaPointer(dstPoint.getDevicePointer(), buffer.length(), 0);

        if (OpProfiler.getInstance().getConfig().isCheckLocality())
             NativeOpsHolder.getInstance().getDeviceNativeOps().tryPointer(context.getOldStream(), p, 1);

        switch (buffer.dataType()) {
            case DOUBLE:
                return p.asDoublePointer();
            case FLOAT:
                return p.asFloatPointer();
            case UINT32:
            case INT:
                return p.asIntPointer();
            case SHORT:
            case UINT16:
            case HALF:
            case BFLOAT16:
                return p.asShortPointer();
            case UINT64:
            case LONG:
                return p.asLongPointer();
            case UTF8:
            case UBYTE:
            case BYTE:
                return p.asBytePointer();
            case BOOL:
                return p.asBooleanPointer();
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
        if (dstPoint.getHostPointer() == null) {
            return null;
        }

        synchronizeThreadDevice(Thread.currentThread().getId(), dstPoint.getDeviceId(), dstPoint);

        CudaPointer p = new CudaPointer(dstPoint.getHostPointer(), buffer.length(), 0);

        switch (buffer.dataType()) {
            case DOUBLE:
                return p.asDoublePointer();
            case FLOAT:
                return p.asFloatPointer();
            case UINT32:
            case INT:
                return p.asIntPointer();
            case SHORT:
            case UINT16:
            case BFLOAT16:
            case HALF:
                return p.asShortPointer();
            case UINT64:
            case LONG:
                return p.asLongPointer();
            default:
                return p;
        }
    }

    @Override
    public synchronized void relocateObject(DataBuffer buffer) {
        AllocationPoint dstPoint = AtomicAllocator.getInstance().getAllocationPoint(buffer);

        if (1 > 0)
            throw new UnsupportedOperationException("Pew-pew");

        // we don't relocate non-DEVICE buffers (i.e HOST or CONSTANT)
        if (dstPoint.getAllocationStatus() != AllocationStatus.DEVICE)
            return;

        int deviceId = getDeviceId();


        if (dstPoint.getDeviceId() >= 0 && dstPoint.getDeviceId() == deviceId ) {
            return;
        }

        val okDevice = dstPoint.isActualOnDeviceSide();
        val okHost = dstPoint.isActualOnHostSide();

        val odPtr = dstPoint.getDevicePointer();
        val ohPtr = dstPoint.getHostPointer();

        // FIXME: cross-thread access, might cause problems
        if (dstPoint.getHostPointer() != null && !dstPoint.isActualOnHostSide())
            AtomicAllocator.getInstance().synchronizeHostData(buffer);

        if (dstPoint.getHostPointer() != null && !dstPoint.isActualOnHostSide())
            throw new RuntimeException("Buffer synchronization failed");

        if (buffer.isAttached() || dstPoint.isAttached()) {
            // if this buffer is Attached, we just relocate to new workspace

            MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();

            if (workspace == null) {
                // if we're out of workspace, we should mark our buffer as detached, so gc will pick it up eventually
                // host part is optional
                if (dstPoint.getHostPointer() != null) {
                    //val pairH = alloc(AllocationStatus.HOST, dstPoint, dstPoint.getShape(), false);
                    //dstPoint.getPointers().setHostPointer(pairH.getHostPointer());
                }

                //val pairD = alloc(AllocationStatus.DEVICE, dstPoint, dstPoint.getShape(), false);
                //dstPoint.getPointers().setDevicePointer(pairD.getDevicePointer());

                ////log.info("New host pointer: {}; Old host pointer: {}", dstPoint.getHostPointer().address(), ohPtr.address());

                CudaContext context = getCudaContext();

                val profD = PerformanceTracker.getInstance().helperStartTransaction();

                if (okDevice) {
                    if (nativeOps.memcpyAsync(dstPoint.getDevicePointer(), odPtr, buffer.length() * buffer.getElementSize(), CudaConstants.cudaMemcpyDeviceToDevice, context.getSpecialStream()) == 0)
                        throw new ND4JIllegalStateException("memcpyAsync failed");

                    context.syncSpecialStream();
                    PerformanceTracker.getInstance().helperRegisterTransaction(dstPoint.getDeviceId(), profD / 2, dstPoint.getNumberOfBytes(), MemcpyDirection.DEVICE_TO_DEVICE);
                } else {
                    if (nativeOps.memcpyAsync(dstPoint.getDevicePointer(), ohPtr, buffer.length() * buffer.getElementSize(), CudaConstants.cudaMemcpyHostToDevice, context.getSpecialStream()) == 0)
                        throw new ND4JIllegalStateException("memcpyAsync failed");

                    context.syncSpecialStream();
                    PerformanceTracker.getInstance().helperRegisterTransaction(dstPoint.getDeviceId(), profD / 2, dstPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);
                }
                // marking it as detached
                dstPoint.setAttached(false);

                // marking it as proper on device
                dstPoint.tickDeviceWrite();
            } else {
                // this call will automagically take care of workspaces, so it'll be either
                //log.info("Relocating to deviceId [{}], workspace [{}]...", deviceId, workspace.getId());
                BaseCudaDataBuffer nBuffer = (BaseCudaDataBuffer) Nd4j.createBuffer(buffer.length());

                Nd4j.getMemoryManager().memcpy(nBuffer, buffer);

                //dstPoint.getPointers().setDevicePointer(nBuffer.getAllocationPoint().getDevicePointer());

                if (dstPoint.getHostPointer() != null) {
                  //  dstPoint.getPointers().setHostPointer(nBuffer.getAllocationPoint().getHostPointer());
                }

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
            val context = getCudaContext();
            if (dstPoint.getHostPointer() == null) {
                ((BaseCudaDataBuffer) buffer).lazyAllocateHostPointer();

                if (nativeOps.memcpyAsync(dstPoint.getHostPointer(), dstPoint.getDevicePointer(),
                        buffer.length() * buffer.getElementSize(), 2, context.getSpecialStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");

                context.syncSpecialStream();
            }

            //deviceMemoryTracker.subFromAllocation(Thread.currentThread().getId(), dstPoint.getDeviceId(), AllocationUtils.getRequiredMemory(dstPoint.getShape()));

            // we replace original device pointer with new one
            //alloc(AllocationStatus.DEVICE, dstPoint, dstPoint.getShape(), false);

            val profD = PerformanceTracker.getInstance().helperStartTransaction();

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

        if (1 > 0)
            throw new UnsupportedOperationException("Pew-pew");

        if (dstPoint.getAllocationStatus() != AllocationStatus.HOST)
            return false;

        if (configuration.getMemoryModel() == Configuration.MemoryModel.DELAYED
                        && dstPoint.getAllocationStatus() == AllocationStatus.HOST) {


            // if we have constant buffer (aka shapeInfo or other constant stuff)
            if (buffer.isConstant()) {
                Nd4j.getConstantHandler().moveToConstantSpace(buffer);
            } else {

                PointersPair pair = null; //memoryProvider.malloc(dstPoint.getShape(), dstPoint, AllocationStatus.DEVICE);

                if (pair != null) {
                    Integer deviceId = getDeviceId();
                    //               log.info("Promoting object to device: [{}]", deviceId);

                    //dstPoint.setDevicePointer(pair.getDevicePointer());
                    dstPoint.setAllocationStatus(AllocationStatus.DEVICE);

                    deviceAllocations.get(deviceId).put(dstPoint.getObjectId(), dstPoint.getObjectId());

                    zeroAllocations.get(dstPoint.getBucketId()).remove(dstPoint.getObjectId());
                    //deviceMemoryTracker.addToAllocation(Thread.currentThread().getId(), deviceId, AllocationUtils.getRequiredMemory(dstPoint.getShape()));


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

        //deviceMemoryTracker.subFromAllocation(threadId, deviceId, AllocationUtils.getRequiredMemory(point.getShape()));

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
        if (1 > 0)
            throw new UnsupportedOperationException("Pew-pew");

        forget(point, AllocationStatus.HOST);

        flowController.waitTillReleased(point);

        // we call for caseless deallocation here
        if (point.getHostPointer() != null) {
            free(point, AllocationStatus.HOST);

            //long reqMem = AllocationUtils.getRequiredMemory(point.getShape()) * -1;
            //zeroUseCounter.addAndGet(reqMem);
        }

        point.setAllocationStatus(AllocationStatus.DEALLOCATED);
    }

    @Override
    public void forget(AllocationPoint point, AllocationStatus location) {
        if (location == AllocationStatus.DEVICE) {
            deviceAllocations.get(point.getDeviceId()).remove(point.getObjectId());
        } else if (location == AllocationStatus.HOST) {
            if (point.getHostPointer() != null)
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
    public CudaContext getDeviceContext() {
        return getCudaContext();
    }

    //
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    protected cublasHandle_t getCudaCublasHandle(OpaqueLaunchContext lc) {
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        try {
            lock.writeLock().lock();

            if (cublasHandles.get(deviceId) == null) {
                cublasHandles.remove(deviceId);
                cublasHandles.add(deviceId, new cublasHandle_t(nativeOps.lcBlasHandle(lc)));
            }

            return cublasHandles.get(deviceId);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * This method returns CudaContext for current thread. If context doesn't exist - it gets created first.
     * @return
     */
    public CudaContext getCudaContext() {
        val lc = nativeOps.defaultLaunchContext();

        return CudaContext.builder()
                .bufferScalar(nativeOps.lcScalarPointer(lc))
                .bufferReduction(nativeOps.lcReductionPointer(lc))
                .bufferAllocation(nativeOps.lcAllocationPointer(lc))
                .bufferSpecial(nativeOps.lcScalarPointer(lc))
                .oldStream(new cudaStream_t(nativeOps.lcExecutionStream(lc)))
                .specialStream(new cudaStream_t(nativeOps.lcCopyStream(lc)))
                .cublasHandle(getCudaCublasHandle(lc))
                .solverHandle(new cusolverDnHandle_t(nativeOps.lcSolverHandle(lc)))
                .build();
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
    public MemoryProvider getMemoryProvider() {
        return null;
    }
}
