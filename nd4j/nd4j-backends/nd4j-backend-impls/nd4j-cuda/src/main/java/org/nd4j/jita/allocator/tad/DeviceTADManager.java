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

package org.nd4j.jita.allocator.tad;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.TadDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class DeviceTADManager extends BasicTADManager {
    protected List<Map<TadDescriptor, Pair<DataBuffer, DataBuffer>>> tadCache = new ArrayList<>();
    private Semaphore lock = new Semaphore(1);

    public DeviceTADManager() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        for (int i = 0; i < numDevices; i++) {
            tadCache.add(i, new ConcurrentHashMap<TadDescriptor, Pair<DataBuffer, DataBuffer>>());
        }
    }

    /**
     * This method removes all cached shape buffers
     */
    @Override
    public void purgeBuffers() {
        log.info("Purging TAD buffers...");

        tadCache = new ArrayList<>();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        for (int i = 0; i < numDevices; i++) {
            log.info("Resetting device: [{}]", i);
            tadCache.add(i, new ConcurrentHashMap<TadDescriptor, Pair<DataBuffer, DataBuffer>>());
        }

        super.purgeBuffers();
    }

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        /*
            so, we check, if we have things cached.
            If we don't - we just create new TAD shape, and push it to constant memory
        */
        if (dimension != null && dimension.length > 1)
            Arrays.sort(dimension);

        Integer deviceId = AtomicAllocator.getInstance().getDeviceId();

        //log.info("Requested TAD for device [{}], dimensions: [{}]", deviceId, Arrays.toString(dimension));

        //extract the dimensions and shape buffer for comparison
        TadDescriptor descriptor = new TadDescriptor(array, dimension);

        if (!tadCache.get(deviceId).containsKey(descriptor)) {
            log.trace("Creating new TAD...");
            //create the TAD with the shape information and corresponding offsets
            //note that we use native code to get access to the shape information.
            Pair<DataBuffer, DataBuffer> buffers = super.getTADOnlyShapeInfo(array, dimension);
            /**
             * Store the buffers in constant memory.
             * The main implementation of this is cuda right now.
             *
             * Explanation from: http://cuda-programming.blogspot.jp/2013/01/what-is-constant-memory-in-cuda.html
             * The CUDA language makes available another kind of memory known as constant memory. As the opName may indicate, we use constant memory for data that will not change over the course of a kernel execution.
            
             Why Constant Memory?
            
             NVIDIA hardware provides 64KB of constant memory that
             it treats differently than it treats standard global memory. In some situations,
             using constant memory rather than global memory will reduce the required memory bandwidth.
            
             NOTE HERE FOR US: We use 48kb of it using these methods.
            
             Note also that we use the {@link AtomicAllocator} which is the cuda memory manager
             for moving the current host space data buffer to constant memory.
            
             We do this for device access to shape information.
             */
            if (buffers.getFirst() != array.shapeInfoDataBuffer())
                AtomicAllocator.getInstance().moveToConstant(buffers.getFirst());
            /**
             * @see {@link org.nd4j.jita.constant.ProtectedCudaConstantHandler}
             */
            if (buffers.getSecond() != null)
                AtomicAllocator.getInstance().moveToConstant(buffers.getSecond());

            // so, at this point we have buffer valid on host side.
            // And we just need to replace DevicePointer with constant pointer
            tadCache.get(deviceId).put(descriptor, buffers);

            bytes.addAndGet((buffers.getFirst().length() * 4));

            if (buffers.getSecond() != null)
                bytes.addAndGet(buffers.getSecond().length() * 8);

            log.trace("Using TAD from cache...");
        }

        return tadCache.get(deviceId).get(descriptor);
    }
}
