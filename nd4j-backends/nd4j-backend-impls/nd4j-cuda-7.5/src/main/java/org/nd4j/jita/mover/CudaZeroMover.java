package org.nd4j.jita.mover;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import lombok.NonNull;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.*;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.NioUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

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
public class CudaZeroMover implements Mover {
    private Configuration configuration;
    private CudaEnvironment environment;
    private static Allocator allocator = AtomicAllocator.getInstance();

    private static Logger log = LoggerFactory.getLogger(CudaZeroMover.class);

    public CudaZeroMover() {
        allocator = AtomicAllocator.getInstance();
    }

    @Override
    public void init(@NonNull Configuration configuration, @NonNull CudaEnvironment environment, @NonNull Allocator allocator) {
        this.configuration = configuration;
        this.environment = environment;
        this.allocator = allocator;
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
        log.info("Alloc called for shape: " + shape);
        //if (shape.getLength() == 757) throw new RuntimeException("757");
        //log.info("Memory required: " + AllocationUtils.getRequiredMemory(shape));
        switch (targetMode) {
            case HOST: {
                    /*
                        TODO: we need to implement pool here, to avoid multiple consequent cudaHostAlloc calls here, since we could just use one or few managed pools here.
                     */

                // cudaMallocHost call, or cudaHostAlloc, depending on device properties
                // TODO: add device capability dependant code, based on device properties from CudaEnvironment

                Pointer devicePointer = new Pointer();
                Pointer hostPointer = new Pointer();
                long reqMem = AllocationUtils.getRequiredMemory(shape);

                JCuda.cudaHostAlloc(
                        hostPointer,
                        reqMem,
                        JCuda.cudaHostAllocMapped);

                JCuda.cudaHostGetDevicePointer(
                        devicePointer,
                        hostPointer,
                        0);

                /*
                DevicePointerInfo devicePointerInfo = new DevicePointerInfo(
                        new HostDevicePointer(hostPointer,devicePointer),
                        shape.getLength(),
                        shape.getStride(),
                        shape.getOffset(),
                        false);
            */

                PointersPair devicePointerInfo = new PointersPair();
                devicePointerInfo.setDevicePointer(new CudaPointer(devicePointer, reqMem));
                devicePointerInfo.setHostPointer(new CudaPointer(hostPointer, reqMem));


                // copy data from
                /*
                We don't need copy anymore, since we assume that memory will be filled in later

                ByteBuffer pointer = hostPointer.getByteBuffer(0, AllocationUtils.getRequiredMemory(shape));
                pointer.order(ByteOrder.nativeOrder());
                NioUtil.copyAtStride(shape.getLength(),getBufferType(point.getBuffer()), point.getBuffer().asNio(), shape.getOffset(), shape.getStride(), pointer,0,1);
                */
                point.setAllocationStatus(AllocationStatus.HOST);
                return devicePointerInfo;
            }
            case DEVICE: {
                point.setAllocationStatus(AllocationStatus.DEVICE);
                // cudaMalloc call
                Pointer devicePointer = new Pointer();
                Pointer hostPointer = new Pointer();

                CudaContext context = allocator.getCudaContext();

                long reqMem = AllocationUtils.getRequiredMemory(shape);
                JCuda.cudaMalloc(devicePointer, reqMem);

                /*
                DevicePointerInfo devicePointerInfo = new DevicePointerInfo(
                        new HostDevicePointer(hostPointer,devicePointer),
                        shape.getLength(),
                        shape.getStride(),
                        shape.getOffset(),
                        false);
                */

                PointersPair devicePointerInfo = point.getPointers();
                devicePointerInfo.setDevicePointer(new CudaPointer(devicePointer, reqMem));

                JCuda.cudaMemcpyAsync(
                        devicePointer,
                        new Pointer(devicePointerInfo.getHostPointer().address()),
                        reqMem,
                        cudaMemcpyKind.cudaMemcpyHostToDevice,
                        context.getOldStream()
                );

                context.syncOldStream();

                // In new meta, we can't free this pointer anymore, since it's still used on java side
                //free(point, AllocationStatus.ZERO);

                return devicePointerInfo;
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

            CudaContext context = allocator.getCudaContext();

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

            CudaContext context = allocator.getCudaContext();

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
        //   log.info("copyforward() called on shape: " + point.getShape());
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

        CudaContext context = allocator.getCudaContext();

        Pointer devPtr = new Pointer(pair.getDevicePointer().address());

        JCuda.cudaMemcpyAsync(
                new Pointer(pair.getHostPointer().address()),
                devPtr,
                AllocationUtils.getRequiredMemory(shape),
                cudaMemcpyKind.cudaMemcpyDeviceToHost,
                context.getOldStream()
        );

        context.syncOldStream();

        JCuda.cudaFree(devPtr);

        pair.setDevicePointer(null);

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
        switch (target) {
            case HOST: {
                // cudaFreeHost call here
                JCuda.cudaFreeHost(new Pointer(point.getPointers().getHostPointer().address()));
            }
            break;
            case DEVICE: {
                // cudaFree call
                JCuda.cudaFree(new Pointer(point.getPointers().getDevicePointer().address()));
            }
            break;
            default:
                throw new IllegalStateException("Can't free memory on target [" + target + "]");
        }
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
    public void initializeDevice(Long threadId, Integer deviceId, Map<Long, CudaContext> contextPool) {
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
        CudaContext context = AtomicAllocator.getInstance().getCudaContext();
        AllocationPoint point = ((BaseCudaDataBuffer) dstBuffer).getAllocationPoint();
        // we update host memory regardless.
        Pointer dP = new Pointer(point.getPointers().getHostPointer().address() + dstOffset);
//        Pointer sP = new Pointer(srcPointer.getNativePointer());

        log.info("memcpyAsync:  ["+ srcPointer.getNativePointer()+"] -> ["+ dP.getNativePointer()+"], length: [" + length+ "], offset: ["+ dstOffset+"]");

        JCuda.cudaMemcpyAsync(
                dP,
                srcPointer,
                length,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
                context.getOldStream()
        );

        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            // TODO:  device replication to be implemented
        }

        // TODO: to be removed
        context.syncOldStream();
    }

    @Override
    public void memcpyBlocking(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset) {
        CudaContext context = AtomicAllocator.getInstance().getCudaContext();
        memcpyAsync(dstBuffer, srcPointer, length, dstOffset);
        context.syncOldStream();
    }

    @Override
    public void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer) {
        CudaContext context = allocator.getCudaContext();
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
}