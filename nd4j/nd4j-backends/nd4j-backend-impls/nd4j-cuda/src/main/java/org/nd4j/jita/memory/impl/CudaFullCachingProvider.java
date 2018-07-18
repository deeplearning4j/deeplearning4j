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
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This MemoryProvider implementation does caching for both host and device memory within predefined limits.
 *
 * @author raver119@gmail.com
 */
public class CudaFullCachingProvider extends CudaCachingZeroProvider {

    //protected final long MAX_GPU_ALLOCATION = configuration.getMaximumSingleDeviceAllocation();

    //protected final long MAX_GPU_CACHE = configuration.getMaximumDeviceCache();


    protected volatile ConcurrentHashMap<Integer, ConcurrentHashMap<AllocationShape, CacheHolder>> deviceCache =
                    new ConcurrentHashMap<>();


    private static Logger log = LoggerFactory.getLogger(CudaFullCachingProvider.class);

    public CudaFullCachingProvider() {

        init();
    }

    public void init() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        deviceCachedAmount = new ArrayList<>();

        for (int i = 0; i < numDevices; i++) {
            deviceCachedAmount.add(new AtomicLong());
        }
    }

    /**
     * This method provides PointersPair to memory chunk specified by AllocationShape
     *
     * PLEASE NOTE: This method can actually ignore malloc request, and give out previously cached free memory chunk with equal shape.
     *
     * @param shape shape of desired memory chunk
     * @param point target AllocationPoint structure
     * @param location either HOST or DEVICE
     * @return
     */
    @Override
    public PointersPair malloc(AllocationShape shape, AllocationPoint point, AllocationStatus location) {
        long reqMemory = AllocationUtils.getRequiredMemory(shape);
        if (location == AllocationStatus.DEVICE && reqMemory < CudaEnvironment.getInstance().getConfiguration().getMaximumDeviceAllocation()) {


            int deviceId = AtomicAllocator.getInstance().getDeviceId();
            ensureDeviceCacheHolder(deviceId, shape);

            CacheHolder cache = deviceCache.get(deviceId).get(shape);
            if (cache != null) {
                Pointer pointer = cache.poll();
                if (pointer != null) {
                    cacheDeviceHit.incrementAndGet();

                    deviceCachedAmount.get(deviceId).addAndGet(-1 * reqMemory);

                    PointersPair pair = new PointersPair();
                    pair.setDevicePointer(pointer);

                    point.setAllocationStatus(AllocationStatus.DEVICE);
                    point.setDeviceId(deviceId);
                    return pair;
                }
            }
            cacheDeviceMiss.incrementAndGet();
            return super.malloc(shape, point, location);
        }
        return super.malloc(shape, point, location);
    }

    /**
     * This method frees specific chunk of memory, described by AllocationPoint passed in
     *
     * PLEASE NOTE: This method can actually ignore free, and keep released memory chunk for future reuse.
     *
     * @param point
     */
    @Override
    public void free(AllocationPoint point) {
        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            if (point.isConstant())
                return;

            AllocationShape shape = point.getShape();
            int deviceId = point.getDeviceId();
            long address = point.getDevicePointer().address();
            long reqMemory = AllocationUtils.getRequiredMemory(shape);
            // we don't cache too big objects

            if (reqMemory > CudaEnvironment.getInstance().getConfiguration().getMaximumDeviceCacheableLength() || deviceCachedAmount.get(deviceId).get() >= CudaEnvironment.getInstance().getConfiguration().getMaximumHostCache()) {
                //log.info("DEVICE_{} memory purging: {} bytes; MS: {}; MT: {}", deviceId, reqMemory, MAX_GPU_ALLOCATION, MAX_GPU_CACHE);
                super.free(point);
                return;
            }

//            log.info("Saving HOST memory into cache...");

            ensureDeviceCacheHolder(deviceId, shape);

            CacheHolder cache = deviceCache.get(deviceId).get(shape);



            if (point.getDeviceId() != deviceId)
                throw new RuntimeException("deviceId changed!");

            // memory chunks < threshold will be cached no matter what
            if (reqMemory <= FORCED_CACHE_THRESHOLD) {
                cache.put(new CudaPointer(point.getDevicePointer().address()));
                return;
            } else {
                long cacheEntries = cache.size();
                long cacheHeight = deviceCache.get(deviceId).size();

                // total memory allocated within this bucket
                long cacheDepth = cacheEntries * reqMemory;

                //if (cacheDepth < MAX_CACHED_MEMORY / cacheHeight) {
                cache.put(new CudaPointer(point.getDevicePointer().address()));
                return;
                //} else {
                //    super.free(point);
                // }
            }
        }
        super.free(point);
    }

    /**
     * This method checks, if storage contains holder for specified shape
     *
     * @param deviceId
     * @param shape
     */
    protected void ensureDeviceCacheHolder(Integer deviceId, AllocationShape shape) {
        if (!deviceCache.containsKey(deviceId)) {
            try {
                synchronized (this) {
                   if (!deviceCache.containsKey(deviceId)) {
                        deviceCache.put(deviceId, new ConcurrentHashMap<AllocationShape, CacheHolder>());
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (!deviceCache.get(deviceId).containsKey(shape)) {
            try {
                singleLock.acquire();

                if (!deviceCache.get(deviceId).containsKey(shape)) {
                    deviceCache.get(deviceId).put(shape, new CacheHolder(shape, deviceCachedAmount.get(deviceId)));
                }
            } catch (Exception e) {

            } finally {
                singleLock.release();
            }
        }
    }

    @Override
    public synchronized void purgeCache() {
        for (Integer device : deviceCache.keySet()) {
            for (AllocationShape shape : deviceCache.get(device).keySet()) {
                Pointer ptr = null;
                while ((ptr = deviceCache.get(device).get(shape).poll()) != null) {
                    freeDevice(ptr, device);
                }
            }

            deviceCachedAmount.get(device).set(0);
        }
        super.purgeCache();
    }
}
