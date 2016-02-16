package org.nd4j.jita.mover;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import lombok.NonNull;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.DevicePointerInfo;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.buffer.allocation.HostDevicePointer;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.NioUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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
public class UmaMover implements Mover {
    private Configuration configuration;
    private CudaEnvironment environment;

    private static Logger log = LoggerFactory.getLogger(UmaMover.class);

    @Override
    public void init(Configuration configuration, CudaEnvironment environment) {
        this.configuration = configuration;
        this.environment = environment;
    }

    /**
     * Allocate specified memory chunk on specified device/host
     *
     * @param targetMode valid arguments are DEVICE, ZERO
     * @param shape
     * @return
     */
    @Override
    public DevicePointerInfo alloc(@NonNull AllocationStatus targetMode,@NonNull AllocationPoint point,  @NonNull AllocationShape shape) {
        log.info("Alloc called for shape: " + shape);
        switch (targetMode) {
            case ZERO: {
                    /*
                        TODO: we need to implement pool here, to avoid multiple consequent cudaHostAlloc calls here, since we could just use one or few managed pools here.
                     */

                    // cudaMallocHost call, or cudaHostAlloc, depending on device properties
                    // TODO: add device capability dependant code, based on device properties from CudaEnvironment

                Pointer devicePointer = new Pointer();
                Pointer hostPointer = new Pointer();
                JCuda.cudaHostAlloc(
                        hostPointer,
                        AllocationUtils.getRequiredMemory(shape),
                        JCuda.cudaHostAllocMapped);

                JCuda.cudaHostGetDevicePointer(
                        devicePointer,
                        hostPointer,
                        0);

                DevicePointerInfo devicePointerInfo = new DevicePointerInfo(
                        new HostDevicePointer(hostPointer,devicePointer),
                        shape.getLength(),
                        shape.getStride(),
                        shape.getOffset(),
                        false);



                // copy data from
                ByteBuffer pointer = hostPointer.getByteBuffer(0, AllocationUtils.getRequiredMemory(shape));
                pointer.order(ByteOrder.nativeOrder());
                NioUtil.copyAtStride(shape.getLength(),getBufferType(point.getBuffer()), point.getBuffer().asNio(), shape.getOffset(), shape.getStride(), pointer,0,1);

                point.setAllocationStatus(AllocationStatus.ZERO);
                return devicePointerInfo;
            }
            case DEVICE: {
                point.setAllocationStatus(AllocationStatus.DEVICE);
                    // cudaMalloc call
                Pointer devicePointer = new Pointer();
                Pointer hostPointer = new Pointer();


                JCuda.cudaMalloc(devicePointer, AllocationUtils.getRequiredMemory(shape));

                DevicePointerInfo devicePointerInfo = new DevicePointerInfo(
                    new HostDevicePointer(hostPointer,devicePointer),
                    shape.getLength(),
                    shape.getStride(),
                    shape.getOffset(),
                    false);

                // FIXME: async copy is required here
                JCuda.cudaMemcpy(devicePointer, point.getHostPointer(), AllocationUtils.getRequiredMemory(shape), cudaMemcpyKind.cudaMemcpyHostToDevice);

                free(point, AllocationStatus.ZERO);

                return devicePointerInfo;
            }
            default:
                throw new IllegalStateException("Can't allocate memory on target [" + targetMode + "]");
        }
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
        log.info("RELOCATE CALLED: [" +currentStatus+ "] -> ["+targetStatus+"]");
        if (currentStatus == AllocationStatus.ZERO && targetStatus == AllocationStatus.DEVICE) {
            // ZERO -> DEVICE
        } else if (currentStatus == AllocationStatus.DEVICE && targetStatus == AllocationStatus.ZERO) {
            // DEVICE -> ZERO
        } else if (currentStatus == AllocationStatus.ZERO && targetStatus == AllocationStatus.HOST) {
            // ZERO -> HOST
            Pointer hostPointer = point.getHostPointer();

            if (hostPointer == null)
                throw new IllegalStateException("HostPointer is null, can't relocate!");

            DataBuffer targetBuffer = point.getBuffer();
            if (targetBuffer == null)
                throw new IllegalStateException("Target buffer is NULL!");

            // FIXME: this is wrong. We MUST take AllocationShape into account, to avoid unneccessary copybacks, also breaking partial allocations
            ByteBuffer pointer = hostPointer.getByteBuffer(0, targetBuffer.getElementSize() * targetBuffer.length()).order(ByteOrder.nativeOrder());
            ByteBuffer bufferNio = targetBuffer.asNio();
            NioUtil.copyAtStride(shape.getLength(),getBufferType(targetBuffer),pointer, 0,1,bufferNio,shape.getOffset(),1);

        } else if (currentStatus == AllocationStatus.DEVICE && targetStatus == AllocationStatus.HOST) {
            // DEVICE -> HOST
            DataBuffer targetBuffer = point.getBuffer();
            if (targetBuffer == null)
                throw new IllegalStateException("Target buffer is NULL!");

            Pointer devicePointer = point.getCudaPointer();

            // FIXME: ASYNC should be here
            JCuda.cudaMemcpy(PointerUtil.getHostPointer(targetBuffer), devicePointer, AllocationUtils.getRequiredMemory(shape), cudaMemcpyKind.cudaMemcpyDeviceToHost );

        } else if (currentStatus == AllocationStatus.HOST && targetStatus == AllocationStatus.ZERO) {
            // HOST -> ZERO
            Pointer hostPointer = point.getHostPointer();

            if (hostPointer == null)
                throw new IllegalStateException("HostPointer is null, can't relocate!");

            ByteBuffer pointer = hostPointer.getByteBuffer(0, AllocationUtils.getRequiredMemory(shape));
            pointer.order(ByteOrder.nativeOrder());
            log.info("copyforward HOST->ZERO shape: " + shape);
            NioUtil.copyAtStride(
                    shape.getLength(), // copy length
                    getBufferType(point.getBuffer()),  // buffer type
                    point.getBuffer().asNio(), // copy from
                    shape.getOffset(), // copy from offset
                    shape.getStride(), // copy from stride
                    pointer, // copy to
                    0, // dst offset
                    shape.getStride() // dst stride
            );

        } else if (currentStatus == AllocationStatus.HOST && targetStatus == AllocationStatus.DEVICE) {
            // HOST -> DEVICE
            DataBuffer hostBuffer = point.getBuffer();
            if (hostBuffer == null)
                throw new IllegalStateException("Target buffer is NULL!");

            Pointer devicePointer = point.getCudaPointer();

            // FIXME: ASYNC should be here
            JCuda.cudaMemcpy(devicePointer, PointerUtil.getHostPointer(hostBuffer), AllocationUtils.getRequiredMemory(shape), cudaMemcpyKind.cudaMemcpyHostToDevice );

        }  else throw new UnsupportedOperationException("Can't relocate data in requested direction: [" + currentStatus + "] -> [" + targetStatus + "]");
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


        DevicePointerInfo info = alloc(AllocationStatus.ZERO, point, shape);

        JCuda.cudaMemcpy(info.getPointers().getHostPointer(), point.getCudaPointer(), AllocationUtils.getRequiredMemory(shape), cudaMemcpyKind.cudaMemcpyDeviceToHost);

        JCuda.cudaFree(point.getCudaPointer());

        point.setCudaPointers(info);
    }

    /**
     * This method frees memory chunk specified by pointer and location
     *
     * @param point Pointer
     */
    @Override
    public void free(@NonNull AllocationPoint point, AllocationStatus target) {
        switch (target) {
            case ZERO: {
                    // cudaFreeHost call here
                    JCuda.cudaFreeHost(point.getCudaPointer());
                }
                break;
            case DEVICE: {
                    // cudaFree call
                    JCuda.cudaFree(point.getCudaPointer());
                }
                break;
            default:
                throw new IllegalStateException("Can't free memory on target [" + target + "]");
        }
    }

    private NioUtil.BufferType getBufferType(DataBuffer buffer) {
        switch(buffer.dataType()) {
            case DOUBLE: return NioUtil.BufferType.DOUBLE;
            case INT: return NioUtil.BufferType.FLOAT;
            case FLOAT: return NioUtil.BufferType.FLOAT;
            default: throw new UnsupportedOperationException("Unsupported data buffer type");
        }
    }
}
