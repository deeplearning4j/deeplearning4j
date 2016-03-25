package org.nd4j.jita.memory.impl;

import jcuda.Pointer;
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

/**
 * @author raver119@gmail.com
 */
public class CudaDirectProvider implements MemoryProvider {
    @Override
    public PointersPair malloc(AllocationShape shape, AllocationPoint point, AllocationStatus location) {
        switch (location) {
            case HOST: {

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

                PointersPair devicePointerInfo = new PointersPair();
                devicePointerInfo.setDevicePointer(new CudaPointer(devicePointer, reqMem));
                devicePointerInfo.setHostPointer(new CudaPointer(hostPointer, reqMem));

                point.setAllocationStatus(AllocationStatus.HOST);
                return devicePointerInfo;
            }
            case DEVICE: {
                point.setAllocationStatus(AllocationStatus.DEVICE);
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
}
