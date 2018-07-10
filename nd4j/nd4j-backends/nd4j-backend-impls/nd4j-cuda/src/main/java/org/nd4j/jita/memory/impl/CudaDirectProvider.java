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

package org.nd4j.jita.memory.impl;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.memory.MemoryProvider;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class CudaDirectProvider implements MemoryProvider {

    protected static final long DEVICE_RESERVED_SPACE = 1024 * 1024 * 50L;
    private static Logger log = LoggerFactory.getLogger(CudaDirectProvider.class);
    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    protected volatile ConcurrentHashMap<Long, Integer> validator = new ConcurrentHashMap<>();


    private AtomicLong emergencyCounter = new AtomicLong(0);

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

        //log.info("shape onCreate: {}, target: {}", shape, location);

        switch (location) {
            case HOST: {
                Pointer devicePointer = new Pointer();
                long reqMem = AllocationUtils.getRequiredMemory(shape);

                // FIXME: this is WRONG, and directly leads to memleak
                if (reqMem < 1)
                    reqMem = 1;

                Pointer pointer = nativeOps.mallocHost(reqMem, 0);
                if (pointer == null)
                    throw new RuntimeException("Can't allocate [HOST] memory: " + reqMem + "; threadId: "
                                    + Thread.currentThread().getId());

                //                log.info("Host allocation, Thread id: {}, ReqMem: {}, Pointer: {}", Thread.currentThread().getId(), reqMem, pointer != null ? pointer.address() : null);

                Pointer hostPointer = new CudaPointer(pointer);

                PointersPair devicePointerInfo = new PointersPair();
                devicePointerInfo.setDevicePointer(new CudaPointer(hostPointer, reqMem));
                devicePointerInfo.setHostPointer(new CudaPointer(hostPointer, reqMem));

                point.setPointers(devicePointerInfo);

                point.setAllocationStatus(AllocationStatus.HOST);
                return devicePointerInfo;
            }
            case DEVICE: {
                // cudaMalloc call
                int deviceId = AtomicAllocator.getInstance().getDeviceId();
                long reqMem = AllocationUtils.getRequiredMemory(shape);

                // FIXME: this is WRONG, and directly leads to memleak
                if (reqMem < 1)
                    reqMem = 1;

                //                if (CudaEnvironment.getInstance().getConfiguration().getDebugTriggered() == 119)
                //                    throw new RuntimeException("Device allocation happened");


                Pointer pointer = nativeOps.mallocDevice(reqMem, null, 0);
                //log.info("Device [{}] allocation, Thread id: {}, ReqMem: {}, Pointer: {}", AtomicAllocator.getInstance().getDeviceId(), Thread.currentThread().getId(), reqMem, pointer != null ? pointer.address() : null);


                if (pointer == null)
                    return null;
                //throw new RuntimeException("Can't allocate [DEVICE] memory!");

                Pointer devicePointer = new CudaPointer(pointer);

                PointersPair devicePointerInfo = point.getPointers();
                if (devicePointerInfo == null)
                    devicePointerInfo = new PointersPair();
                devicePointerInfo.setDevicePointer(new CudaPointer(devicePointer, reqMem));

                point.setAllocationStatus(AllocationStatus.DEVICE);
                point.setDeviceId(deviceId);

                return devicePointerInfo;
            }
            default:
                throw new IllegalStateException("Unsupported location for malloc: [" + location + "]");
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
                long reqMem = AllocationUtils.getRequiredMemory(point.getShape());

                //  log.info("Deallocating {} bytes on [HOST]", reqMem);

                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

                long result = nativeOps.freeHost(point.getPointers().getHostPointer());
                //JCuda.cudaFreeHost(new Pointer(point.getPointers().getHostPointer()));
                if (result == 0)
                    throw new RuntimeException("Can't deallocate [HOST] memory...");
            }
                break;
            case DEVICE: {
                // cudaFree call
                //JCuda.cudaFree(new Pointer(point.getPointers().getDevicePointer().address()));
                if (point.isConstant())
                    return;

                long reqMem = AllocationUtils.getRequiredMemory(point.getShape());

                //       log.info("Deallocating {} bytes on [DEVICE]", reqMem);

                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

                long result = nativeOps.freeDevice(point.getPointers().getDevicePointer(), new CudaPointer(0));
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
        long freeMem = nativeOps.getDeviceFreeMemory(new CudaPointer(-1));
        if (freeMem - requiredMemory < DEVICE_RESERVED_SPACE)
            return false;
        else
            return true;
    }

    protected void freeHost(Pointer pointer) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.freeHost(pointer);
    }

    protected void freeDevice(Pointer pointer, int deviceId) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.freeDevice(pointer, new CudaPointer(0));
    }

    @Override
    public void purgeCache() {
        // no-op
    }
}
