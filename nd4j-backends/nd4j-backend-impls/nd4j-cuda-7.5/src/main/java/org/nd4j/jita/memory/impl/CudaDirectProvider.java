package org.nd4j.jita.memory.impl;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.memory.MemoryProvider;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.ops.executioner.JCudaExecutioner;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
public class CudaDirectProvider implements MemoryProvider {

    protected static final long DEVICE_RESERVED_SPACE = 1024 * 1024 * 50L;
    private static Logger log = LoggerFactory.getLogger(CudaDirectProvider.class);
    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

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
                long reqMem = AllocationUtils.getRequiredMemory(shape);

                // FIXME: this is WRONG, and directly leads to memleak
                if (reqMem < 1)
                    reqMem = 1;

               long pointer = nativeOps.mallocHost(reqMem, 0);
                if (pointer == 0)
                    throw new RuntimeException("Can't allocate [HOST] memory: " + reqMem);

                Pointer hostPointer = new CudaPointer(pointer);
/*
                JCuda.cudaHostGetDevicePointer(
                        devicePointer,
                        hostPointer,
                        0);
*/
                PointersPair devicePointerInfo = new PointersPair();
                devicePointerInfo.setDevicePointer(new CudaPointer(hostPointer, reqMem));
                devicePointerInfo.setHostPointer(new CudaPointer(hostPointer, reqMem));

                point.setPointers(devicePointerInfo);

                point.setAllocationStatus(AllocationStatus.HOST);
                return devicePointerInfo;
            }
            case DEVICE: {
                // cudaMalloc call

                long reqMem = AllocationUtils.getRequiredMemory(shape);

                // FIXME: this is WRONG, and directly leads to memleak
                if (reqMem < 1)
                    reqMem = 1;

                // FIXME: it would be nice to get rid of typecasting here


                long pointer = nativeOps.mallocDevice(reqMem, 0, 0);
                if (pointer == 0)
                    return null;
                    //throw new RuntimeException("Can't allocate [DEVICE] memory!");

                Pointer devicePointer = new CudaPointer(pointer);

                PointersPair devicePointerInfo = point.getPointers();
                if (devicePointerInfo == null)
                    devicePointerInfo = new PointersPair();
                devicePointerInfo.setDevicePointer(new CudaPointer(devicePointer, reqMem));

                point.setAllocationStatus(AllocationStatus.DEVICE);

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
                // FIXME: it would be nice to get rid of typecasting here
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

                long result = nativeOps.freeHost(point.getPointers().getHostPointer().address());
                //JCuda.cudaFreeHost(new Pointer(point.getPointers().getHostPointer().address()));
                if (result == 0)
                    throw new RuntimeException("Can't deallocate [HOST] memory...");
            }
            break;
            case DEVICE: {
                // cudaFree call
                //JCuda.cudaFree(new Pointer(point.getPointers().getDevicePointer().address()));

                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

                long result = nativeOps.freeDevice(point.getPointers().getDevicePointer().address(), 0);
                if (result == 0)
                    throw new RuntimeException("Can't deallocate [DEVICE] memory...");
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
        /*
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

        /*
        if (free + requiredMemory < total * 0.85)
            return true;
        else return false;
        */
        long freeMem = nativeOps.getDeviceFreeMemory(-1);
        if (freeMem - requiredMemory < DEVICE_RESERVED_SPACE)
            return false;
        else return true;

    }
}
