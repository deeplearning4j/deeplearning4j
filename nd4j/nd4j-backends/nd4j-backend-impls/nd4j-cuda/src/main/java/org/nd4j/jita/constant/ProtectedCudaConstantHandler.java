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

/**
 * Created by raver on 08.06.2016.
 */
@Slf4j
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
            buffersCache.put(device, new ConcurrentHashMap<>());
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
        return Nd4j.getExecutioner().createConstantBuffer(array, type);
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
        return Nd4j.getExecutioner().createConstantBuffer(array, type);
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
        return Nd4j.getExecutioner().createConstantBuffer(array, type);
    }

    @Override
    public DataBuffer getConstantBuffer(long[] array, DataType type) {
        return Nd4j.getExecutioner().createConstantBuffer(array, type);
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
