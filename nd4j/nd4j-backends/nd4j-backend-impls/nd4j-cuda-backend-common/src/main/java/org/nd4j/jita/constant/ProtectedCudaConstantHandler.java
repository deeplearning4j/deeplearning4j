/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.jita.constant;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cache.ConstantProtector;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.*;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import lombok.extern.slf4j.Slf4j;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/**
 * Created by raver on 08.06.2016.
 */
@Slf4j
public class ProtectedCudaConstantHandler implements ConstantHandler {
    private static ProtectedCudaConstantHandler ourInstance = new ProtectedCudaConstantHandler();

    protected Map<Integer, AtomicLong> constantOffsets = new HashMap<>();
    protected Map<Integer, Semaphore> deviceLocks = new ConcurrentHashMap<>();

    protected Map<Integer, Map<ArrayDescriptor, DataBuffer>> buffersCache = new ConcurrentHashMap<>();
    protected Map<Integer, Pointer> deviceAddresses = new HashMap<>();
    protected AtomicLong bytes = new AtomicLong(0);
    protected FlowController flowController;

    protected static final ConstantProtector protector = ConstantProtector.getInstance();

    private static Logger logger = LoggerFactory.getLogger(ProtectedCudaConstantHandler.class);

    protected Semaphore lock = new Semaphore(1);
    private boolean resetHappened = false;


    public static ProtectedCudaConstantHandler getInstance() {
        return ourInstance;
    }

    private ProtectedCudaConstantHandler() {}

    /**
     * Resets the constant space. The buffers cached by content stay: each wraps an entry of the native
     * constant cache, which keeps every entry for the life of the process, and each is constant, so its
     * native wrapper is never freed. Dropping them here would leak every wrapper and create it again on
     * the next request.
     */
    @Override
    public void purgeConstants() {
        protector.purgeProtector();

        resetHappened = true;
        logger.info("Resetting Constants...");

        for (Integer device : constantOffsets.keySet()) {
            constantOffsets.get(device).set(0);
        }
    }

    /**
     * Method suited for debug purposes only
     *
     * @return
     */
    protected int amountOfEntries(int deviceId) {
        ensureMaps(deviceId);
        return buffersCache.get(deviceId).size();
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
                        buffersCache.put(deviceId, new ConcurrentHashMap<>());
                        constantOffsets.put(deviceId, new AtomicLong(0));
                        deviceLocks.put(deviceId, new Semaphore(1));

                        Pointer cAddr = NativeOpsHolder.getInstance().getDeviceNativeOps().getConstantSpace();
                        deviceAddresses.put(deviceId, cAddr);
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * This method returns DataBuffer with constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(int[] array, DataType type) {
        return cachedConstant(new ArrayDescriptor(array, type), () -> Nd4j.getExecutioner().createConstantBuffer(array, type));
    }

    /**
     * This method returns DataBuffer with constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(float[] array, DataType type) {
        return cachedConstant(new ArrayDescriptor(array, type), () -> Nd4j.getExecutioner().createConstantBuffer(array, type));
    }

    /**
     * This method returns DataBuffer with constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    @Override
    public DataBuffer getConstantBuffer(double[] array, DataType type) {
        return cachedConstant(new ArrayDescriptor(array, type), () -> Nd4j.getExecutioner().createConstantBuffer(array, type));
    }

    @Override
    public DataBuffer getConstantBuffer(long[] array, DataType type) {
        return cachedConstant(new ArrayDescriptor(array, type), () -> Nd4j.getExecutioner().createConstantBuffer(array, type));
    }

    /**
     * Returns the constant buffer for this content and type on the current device, creating it once.
     * The native constant cache keeps each content for the life of the process, but every Java buffer
     * wrapping it is constant and never released, so wrapping it again on each call leaked one buffer
     * per call (for example, one per op instance with extra arguments).
     *
     * @param descriptor descriptor over the caller's array; a new entry is keyed by a copy of it
     * @param create     creates the buffer on the first request for this content
     */
    private DataBuffer cachedConstant(ArrayDescriptor descriptor, Supplier<DataBuffer> create) {
        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();
        ensureMaps(deviceId);
        Map<ArrayDescriptor, DataBuffer> cache = buffersCache.computeIfAbsent(deviceId, d -> new ConcurrentHashMap<>());
        DataBuffer cached = cache.get(descriptor);
        return cached != null ? cached : cache.computeIfAbsent(descriptor.copy(), key -> create.get());
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
