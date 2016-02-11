package org.nd4j.jita.mover;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import lombok.NonNull;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.jcublas.buffer.DevicePointerInfo;
import org.nd4j.linalg.jcublas.buffer.allocation.HostDevicePointer;

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
    public Pointer alloc(@NonNull AllocationStatus targetMode,@NonNull AllocationPoint point,  @NonNull AllocationShape shape) {
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



                    point.setAllocationStatus(AllocationStatus.ZERO);
                    return null;
                }
            case DEVICE: {
                point.setAllocationStatus(AllocationStatus.DEVICE);
                    // cudaMalloc call

                    return null;
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
    public void relocate(AllocationStatus currentStatus, AllocationStatus targetStatus, AllocationPoint point) {
        if (currentStatus == AllocationStatus.ZERO && targetStatus == AllocationStatus.DEVICE) {
            // ZERO -> DEVICE
        } else if (currentStatus == AllocationStatus.DEVICE && targetStatus == AllocationStatus.ZERO) {
            // DEVICE -> ZERO
        } else if (currentStatus == AllocationStatus.ZERO && targetStatus == AllocationStatus.HOST) {
            // ZERO -> HOST
        } else if (currentStatus == AllocationStatus.DEVICE && targetStatus == AllocationStatus.HOST) {
            // DEVICE -> HOST
        } else throw new UnsupportedOperationException("Can't relocate data in requested direction: [" + currentStatus + "] -> [" + targetStatus + "]");
    }

    /**
     * Copies memory from device to host, if needed.
     * Device copy is preserved as is.
     *
     * @param point
     */
    @Override
    public void copyback(AllocationPoint point) {
        /*
            Technically that's just a case for relocate, with source as point.getAllocationStatus() and target HOST
         */

    }

    /**
     * Copies memory from host buffer to device.
     * Host copy is preserved as is.
     *
     * @param point
     */
    @Override
    public void copyforward(AllocationPoint point) {
        /*
            Technically that's just a case for relocate, with source as HOST and target point.getAllocationStatus()
         */
    }

    /**
     * This method frees memory chunk specified by pointer and location
     *
     * @param pointer Pointer
     * @param location AllocationStatus
     */
    @Override
    public void free(@NonNull Pointer pointer, @NonNull AllocationStatus location) {
        switch (location) {
            case ZERO: {
                    // cudaFreeHost call here
                }
                break;
            case DEVICE: {
                    // cudaFree call
                }
                break;
            default:
                throw new IllegalStateException("Can't free memory on target [" + location + "]");
        }
    }
}
