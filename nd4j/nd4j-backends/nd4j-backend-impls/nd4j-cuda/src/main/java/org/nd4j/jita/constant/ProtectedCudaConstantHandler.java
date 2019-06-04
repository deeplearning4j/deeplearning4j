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

package org.nd4j.jita.constant;

import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.*;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by raver on 08.06.2016.
 */
public class ProtectedCudaConstantHandler implements ConstantHandler {
    private static ProtectedCudaConstantHandler ourInstance = new ProtectedCudaConstantHandler();

    protected Map<Integer, AtomicLong> constantOffsets = new HashMap<>();
    protected Map<Integer, Semaphore> deviceLocks = new ConcurrentHashMap<>();

    protected Map<Integer, Map<ArrayDescriptor, DataBuffer>> buffersCache = new HashMap<>();
    protected Map<Integer, Pointer> deviceAddresses = new HashMap<>();
    protected AtomicLong bytes = new AtomicLong(0);
    protected FlowController flowController;

    protected static final ConstantProtector protector = ConstantProtector.getInstance();

    private static Logger logger = LoggerFactory.getLogger(ProtectedCudaConstantHandler.class);

    private static final int MAX_CONSTANT_LENGTH = 49152;
    private static final int MAX_BUFFER_LENGTH = 272;

    protected Semaphore lock = new Semaphore(1);
    private boolean resetHappened = false;


    public static ProtectedCudaConstantHandler getInstance() {
        return ourInstance;
    }

    private ProtectedCudaConstantHandler() {}

    /**
     * This method removes all cached constants
     */
    @Override
    public void purgeConstants() {
        buffersCache = new HashMap<>();

        protector.purgeProtector();

        resetHappened = true;
        logger.info("Resetting Constants...");

        for (Integer device : constantOffsets.keySet()) {
            constantOffsets.get(device).set(0);
            buffersCache.put(device, new ConcurrentHashMap<ArrayDescriptor, DataBuffer>());
        }
    }

    /**
     * Method suited for debug purposes only
     *
     * @return
     */
    protected int amountOfEntries(int deviceId) {
        ensureMaps(deviceId);
        return buffersCache.get(0).size();
    }

    /**
     * This method moves specified dataBuffer to CUDA constant memory space.
     *
     * PLEASE NOTE: CUDA constant memory is limited to 48KB per device.
     *
     * @param dataBuffer
     * @return
     */
    @Override
    public synchronized long moveToConstantSpace(DataBuffer dataBuffer) {
        if (1 > 0)
            throw new RuntimeException("This code shouldn't be called, ever");

        // now, we move things to constant memory
        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();
        ensureMaps(deviceId);

        AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(dataBuffer);

        long requiredMemoryBytes = AllocationUtils.getRequiredMemory(point.getShape());
        val originalBytes = requiredMemoryBytes;
        requiredMemoryBytes += 8 - (requiredMemoryBytes % 8);

        val div = requiredMemoryBytes / 4;
        if (div % 2 != 0)
            requiredMemoryBytes += 4;

        //logger.info("shape: " + point.getShape());
        // and release device memory :)

        AllocationsTracker.getInstance().markAllocated(AllocationKind.CONSTANT, deviceId, requiredMemoryBytes);

        long currentOffset = constantOffsets.get(deviceId).get();
        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        if (currentOffset + requiredMemoryBytes >= MAX_CONSTANT_LENGTH || requiredMemoryBytes > MAX_BUFFER_LENGTH) {
            if (point.getAllocationStatus() == AllocationStatus.HOST
                            && CudaEnvironment.getInstance().getConfiguration().getMemoryModel() == Configuration.MemoryModel.DELAYED) {
                AtomicAllocator.getInstance().getMemoryHandler().alloc(AllocationStatus.DEVICE, point, point.getShape(),
                                false);
            }

            val profD = PerformanceTracker.getInstance().helperStartTransaction();

            if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(point.getPointers().getDevicePointer(), point.getPointers().getHostPointer(), originalBytes, 1, context.getSpecialStream()) == 0) {
                throw new ND4JIllegalStateException("memcpyAsync failed");
            }
            flowController.commitTransfer(context.getSpecialStream());

            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), profD, point.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

            point.setConstant(true);
            point.tickDeviceWrite();
            point.tickHostRead();
            point.setDeviceId(deviceId);

            protector.persistDataBuffer(dataBuffer);

            return 0;
        }

        long bytes = requiredMemoryBytes;
        currentOffset = constantOffsets.get(deviceId).getAndAdd(bytes);

        if (currentOffset >= MAX_CONSTANT_LENGTH) {
            if (point.getAllocationStatus() == AllocationStatus.HOST
                            && CudaEnvironment.getInstance().getConfiguration().getMemoryModel() == Configuration.MemoryModel.DELAYED) {
                AtomicAllocator.getInstance().getMemoryHandler().alloc(AllocationStatus.DEVICE, point, point.getShape(),
                                false);
            }

            val profD = PerformanceTracker.getInstance().helperStartTransaction();

            if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(point.getPointers().getDevicePointer(), point.getPointers().getHostPointer(),
                    originalBytes, 1, context.getSpecialStream()) == 0) {
                throw new ND4JIllegalStateException("memcpyAsync failed");
            }
            flowController.commitTransfer(context.getSpecialStream());

            PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), profD, point.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

            point.setConstant(true);
            point.tickDeviceWrite();
            point.tickHostRead();
            point.setDeviceId(deviceId);

            protector.persistDataBuffer(dataBuffer);

            return 0;
        }



        NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyConstantAsync(currentOffset, point.getPointers().getHostPointer(), originalBytes, 1,
                        context.getSpecialStream());
        flowController.commitTransfer(context.getSpecialStream());

        long cAddr = deviceAddresses.get(deviceId).address() + currentOffset;

        //if (resetHappened)
        //    logger.info("copying to constant: {}, bufferLength: {}, bufferDtype: {}, currentOffset: {}, currentAddres: {}", requiredMemoryBytes, dataBuffer.length(), dataBuffer.dataType(), currentOffset, cAddr);

        point.setAllocationStatus(AllocationStatus.CONSTANT);
        point.getPointers().setDevicePointer(new CudaPointer(cAddr));
        point.setConstant(true);
        point.tickDeviceWrite();
        point.setDeviceId(deviceId);
        point.tickHostRead();


        protector.persistDataBuffer(dataBuffer);

        return cAddr;
    }

    /**
     * PLEASE NOTE: This method implementation is hardware-dependant.
     * PLEASE NOTE: This method does NOT allow concurrent use of any array
     *
     * @param dataBuffer
     * @return
     */
    @Override
    public DataBuffer relocateConstantSpace(DataBuffer dataBuffer) {
        // we always assume that data is sync, and valid on host side
        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();
        ensureMaps(deviceId);

        if (dataBuffer instanceof CudaIntDataBuffer) {
            int[] data = dataBuffer.asInt();
            return getConstantBuffer(data, DataType.INT);
        } else if (dataBuffer instanceof CudaFloatDataBuffer) {
            float[] data = dataBuffer.asFloat();
            return getConstantBuffer(data, DataType.FLOAT);
        } else if (dataBuffer instanceof CudaDoubleDataBuffer) {
            double[] data = dataBuffer.asDouble();
            return getConstantBuffer(data, DataType.DOUBLE);
        } else if (dataBuffer instanceof CudaHalfDataBuffer) {
            float[] data = dataBuffer.asFloat();
            return getConstantBuffer(data, DataType.HALF);
        } else if (dataBuffer instanceof CudaLongDataBuffer) {
            long[] data = dataBuffer.asLong();
            return getConstantBuffer(data, DataType.LONG);
        }

        throw new IllegalStateException("Unknown CudaDataBuffer opType");
    }

    private void ensureMaps(Integer deviceId) {
        if (!buffersCache.containsKey(deviceId)) {
            if (flowController == null)
                flowController = AtomicAllocator.getInstance().getFlowController();

            try {
                synchronized (this) {
                    if (!buffersCache.containsKey(deviceId)) {

                        // TODO: this op call should be checked
                        //nativeOps.setDevice(new CudaPointer(deviceId));

                        buffersCache.put(deviceId, new ConcurrentHashMap<ArrayDescriptor, DataBuffer>());
                        constantOffsets.put(deviceId, new AtomicLong(0));
                        deviceLocks.put(deviceId, new Semaphore(1));

                        Pointer cAddr = NativeOpsHolder.getInstance().getDeviceNativeOps().getConstantSpace();
                        //                    logger.info("constant pointer: {}", cAddr.address() );

                        deviceAddresses.put(deviceId, cAddr);
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * This method returns DataBuffer with contant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(int[] array, DataType type) {
        return Nd4j.getExecutioner().createConstantBuffer(array, type);
    }

    /**
     * This method returns DataBuffer with contant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(float[] array, DataType type) {
        return Nd4j.getExecutioner().createConstantBuffer(array, type);
    }

    /**
     * This method returns DataBuffer with contant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(double[] array, DataType type) {
        return Nd4j.getExecutioner().createConstantBuffer(array, type);
        /*
        ArrayDescriptor descriptor = new ArrayDescriptor(array, type);

        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();

        ensureMaps(deviceId);

        if (!buffersCache.get(deviceId).containsKey(descriptor)) {
            // we create new databuffer
            //logger.info("Creating new constant buffer...");
            DataBuffer buffer = Nd4j.createTypedBufferDetached(array, type);

            if (constantOffsets.get(deviceId).get() + (array.length * Nd4j.sizeOfDataType()) < MAX_CONSTANT_LENGTH) {
                buffer.setConstant(true);
                // now we move data to constant memory, and keep happy
                moveToConstantSpace(buffer);

                buffersCache.get(deviceId).put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType());
            }
            return buffer;
        } //else logger.info("Reusing constant buffer...");

        return buffersCache.get(deviceId).get(descriptor);
         */
    }

    @Override
    public DataBuffer getConstantBuffer(long[] array, DataType type) {
        return Nd4j.getExecutioner().createConstantBuffer(array, type);
        /*
        //  logger.info("getConstantBuffer(int[]) called");
        ArrayDescriptor descriptor = new ArrayDescriptor(array, type);

        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();

        ensureMaps(deviceId);

        if (!buffersCache.get(deviceId).containsKey(descriptor)) {
            // we create new databuffer
            //logger.info("Creating new constant buffer...");
            DataBuffer buffer = Nd4j.createTypedBufferDetached(array, type);

            if (constantOffsets.get(deviceId).get() + (array.length * 8) < MAX_CONSTANT_LENGTH) {
                buffer.setConstant(true);
                // now we move data to constant memory, and keep happy
                moveToConstantSpace(buffer);

                buffersCache.get(deviceId).put(descriptor, buffer);

                bytes.addAndGet(array.length * 8);
            }
            return buffer;
        } //else logger.info("Reusing constant buffer...");

        return buffersCache.get(deviceId).get(descriptor);
         */
    }

    @Override
    public DataBuffer getConstantBuffer(boolean[] array, DataType dataType) {
        return getConstantBuffer(ArrayUtil.toLongs(array), dataType);
    }

    @Override
    public long getCachedBytes() {
        return bytes.get();
    }
}
