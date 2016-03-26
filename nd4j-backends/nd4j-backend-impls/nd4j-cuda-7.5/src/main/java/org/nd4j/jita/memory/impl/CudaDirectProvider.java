package org.nd4j.jita.memory.impl;

import jcuda.Pointer;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.memory.MemoryProvider;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author raver119@gmail.com
 */
public class CudaDirectProvider implements MemoryProvider {

    private static Logger log = LoggerFactory.getLogger(CudaDirectProvider.class);

    /**
     * This method provides PointersPair to memory chunk specified by AllocationShape
     *
     * @param shape shape of desired memory chunk
     * @param point target AllocationPoint structure
     * @param location either HOST or DEVICE
     * @return
     */
    @Override
    public PointersPair malloc(AllocationShape shape, AllocationPoint point, AllocationStatus location) {
        switch (location) {
            case HOST: {

                Pointer devicePointer = new Pointer();
                Pointer hostPointer = new Pointer();
                long reqMem = AllocationUtils.getRequiredMemory(shape);

                if (reqMem >= Integer.MAX_VALUE)
                    throw new UnsupportedOperationException("Sorry, you can't allocate > 2GB of memory");

                //log.info("Requested memory size: " + reqMem);

                JCuda.cudaHostAlloc(
                        hostPointer,
                        reqMem,
                        JCuda.cudaHostAllocMapped | JCuda.cudaHostAllocPortable );


                JCuda.cudaHostGetDevicePointer(
                        devicePointer,
                        hostPointer,
                        0);

                PointersPair devicePointerInfo = new PointersPair();
                devicePointerInfo.setDevicePointer(new CudaPointer(devicePointer, reqMem));
                devicePointerInfo.setHostPointer(new CudaPointer(hostPointer, reqMem));

                point.setAllocationStatus(AllocationStatus.HOST);
                return devicePointerInfo;
            }
            case DEVICE: {
                // cudaMalloc call
                Pointer devicePointer = new Pointer();
                Pointer hostPointer = new Pointer();

//                CudaContext context = getCudaContext();

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
/*
                JCuda.cudaMemcpyAsync(
                        devicePointer,
                        new Pointer(devicePointerInfo.getHostPointer().address()),
                        reqMem,
                        cudaMemcpyKind.cudaMemcpyHostToDevice,
                        context.getOldStream()
                );

                context.syncOldStream();
*/
                // In new meta, we can't free this pointer anymore, since it's still used on java side
                //free(point, AllocationStatus.ZERO);

                return devicePointerInfo;
            }
            default:
                throw new IllegalStateException("Unsupported location for malloc: ["+ location+"]");
        }
    }

    /**
     * This method frees specific chunk of memory, described by AllocationPoint passed in
     *
     * @param point
     */
    @Override
    public void free(AllocationPoint point) {
        switch (point.getAllocationStatus()) {
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
                throw new IllegalStateException("Can't free memory on target [" + point.getAllocationStatus() + "]");
        }
    }

    /**
     * This method checks specified device for specified amount of memory
     *
     * @param deviceId
     * @param requiredMemory
     * @return
     */
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
        //if (configuration != null && used > total * configuration.getMaxDeviceMemoryUsed()) return false;

        if (free + requiredMemory < total * 0.90)
            return true;
        else return false;
    }
}
