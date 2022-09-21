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

package org.nd4j.autodiff.samediff.internal.memory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicLong;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;
import org.nd4j.shade.guava.primitives.Longs;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

@Getter
@Setter
@Slf4j
public class ArrayCacheMemoryMgr extends AbstractMemoryMgr {

    private static Map<INDArray,INDArray> released = new IdentityHashMap<>();

    @Getter
    @Setter
    private static  AtomicDouble maxMemFrac;
   @Getter
   @Setter
    private static AtomicLong smallArrayThreshold;

    public static Set<Long> getLruCache() {
        return lruCache;
    }

    public static Map<Long, INDArray> getLruCacheValues() {
        return lruCacheValues;
    }

    public static AtomicDouble getMaxMemFrac() {
        return maxMemFrac;
    }

    public static void setMaxMemFrac(AtomicDouble maxMemFrac) {
        ArrayCacheMemoryMgr.maxMemFrac = maxMemFrac;
    }

    public static void setMaxMemFrac(double maxMemFrac) {
        ArrayCacheMemoryMgr.maxMemFrac.set(maxMemFrac);
    }

    public static AtomicDouble getLargerArrayMaxMultiple() {
        return largerArrayMaxMultiple;
    }

    public static void setLargerArrayMaxMultiple(AtomicDouble largerArrayMaxMultiple) {
        ArrayCacheMemoryMgr.largerArrayMaxMultiple = largerArrayMaxMultiple;
    }
    public static void setLargerArrayMaxMultiple(double largerArrayMaxMultiple) {
        ArrayCacheMemoryMgr.largerArrayMaxMultiple.set(largerArrayMaxMultiple);
    }
    public static AtomicLong getMaxCacheBytes() {
        return maxCacheBytes;
    }

    public static void setMaxCacheBytes(AtomicLong maxCacheBytes) {
        ArrayCacheMemoryMgr.maxCacheBytes = maxCacheBytes;
    }

    public static AtomicLong getCurrentCacheSize() {
        return currentCacheSize;
    }

    public static void setCurrentCacheSize(AtomicLong currentCacheSize) {
        ArrayCacheMemoryMgr.currentCacheSize = currentCacheSize;
    }

    private static AtomicDouble largerArrayMaxMultiple;

    private static AtomicLong maxCacheBytes;
    private static AtomicLong totalMemBytes;

    private static AtomicLong currentCacheSize =  new AtomicLong(0);

    private static Set<Long> lruCache = new ConcurrentSkipListSet<>();
    private static Map<Long, INDArray> lruCacheValues = new ConcurrentHashMap<>();

    private static Table<DataType, String, List<INDArray>> arrays = HashBasedTable.create();

    private static boolean enableCache = Boolean
            .parseBoolean(System.getProperty(ND4JSystemProperties.SAMEDIFF_MEMORY_CACHE_DISABLE, "true"));

    /**
     * Create an ArrayCacheMemoryMgr with default settings as per
     * {@link ArrayCacheMemoryMgr}
     */
    public ArrayCacheMemoryMgr() {
        this(0.25, 1024, 2.0);
    }

    /**
     * @param maxMemFrac             Maximum memory fraction to use as cache
     * @param smallArrayThreshold    Below this size (elements), don't apply the
     *                               "largerArrayMaxMultiple" rule
     * @param largerArrayMaxMultiple Maximum multiple of the requested size to
     *                               return from the cache. If an array of size
     *                               1024 is requested, and largerArrayMaxMultiple
     *                               is 2.0, then we'll return from the cache
     *                               the array with the smallest data buffer up to
     *                               2.0*1024 elements; otherwise we'll return
     *                               a new array
     */
    public ArrayCacheMemoryMgr(double maxMemFrac, long smallArrayThreshold, double largerArrayMaxMultiple) {
        Preconditions.checkArgument(maxMemFrac > 0 && maxMemFrac < 1,
                "Maximum memory fraction for cache must be between 0.0 and 1.0, got %s", maxMemFrac);
        Preconditions.checkArgument(smallArrayThreshold >= 0, "Small array threshold must be >= 0, got %s",
                smallArrayThreshold);
        Preconditions.checkArgument(largerArrayMaxMultiple >= 1.0, "Larger array max multiple must be >= 1.0, got %s",
                largerArrayMaxMultiple);
        this.maxMemFrac = new AtomicDouble(maxMemFrac);
        this.smallArrayThreshold = new AtomicLong(smallArrayThreshold);
        this.largerArrayMaxMultiple = new AtomicDouble(largerArrayMaxMultiple);

        if (isCpu()) {
            totalMemBytes = new AtomicLong(Pointer.maxBytes());
        } else {
            Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
            List devList = (List) p.get("cuda.devicesInformation");
            Map m = (Map) devList.get(0);
            totalMemBytes = new AtomicLong((Long) m.get("cuda.totalMemory"));
        }

        maxCacheBytes = new AtomicLong((long) maxMemFrac * totalMemBytes.get());
    }

    private boolean isCpu() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        return !"CUDA".equalsIgnoreCase(backend);
    }

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        String arrayShapeString = Arrays.toString(shape);
        if (arrays.contains(dataType, arrayShapeString) && enableCache) {
            INDArray arr = null;
            boolean arrFound = false;
            while(!arrFound) {
                arr = !arrays.get(dataType, arrayShapeString).isEmpty()
                        ? arrays.get(dataType, arrayShapeString).remove(0)
                        : null;
                if(arr != null && !arr.closeable() || arr.wasClosed() || arr.isView()) {
                    log.trace("Found array closeable, not returning from cache. Only closeable arrays are returnable from the cache.");
                    if(arr.isView())
                        arr.setCloseable(false);
                    log.trace("Found view array with id " + arr.getId() + " in cache. Avoiding return. Allocating new array.");

                    continue;
                }

                if (arr != null) {
                    // Decrement cache size
                    currentCacheSize.set(currentCacheSize.get() - dataType.width() * arr.data().length());
                    lruCache.remove(arr.getId());
                    lruCacheValues.remove(arr.getId());
                    // We need to assign new Id. this way we will break any possible relationship it
                    // had in Tracker.
                    // the old cache was recreating New Array using buffer and thus gaining new
                    // reference . Note that it had IdentityHash with references being keys
                    ((BaseNDArray) arr).assignNewId();
                    return arr; // Allocated from cache
                }
            }

        }

        // Allocation failed, allocate new array
        INDArray ret = Nd4j.createUninitializedDetached(dataType, shape);
        return ret;
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        if (descriptor.isEmpty()) {
            INDArray ret = Nd4j.create(descriptor);
            if (detached) {
                ret = ret.detach();
            }

            return ret;
        }

        DataType dataType = descriptor.dataType();
        long[] shape = descriptor.getShape();
        String arrayShape = Arrays.toString(shape);
        if (arrays.contains(dataType, arrayShape) && enableCache && shape.length > 0 && !Longs.contains(shape, 0)) {
            INDArray arr = null;
            List<INDArray> arrays2 = arrays.get(dataType, arrayShape);

            while (arrays2.size() > 0) {
                arr = arrays2.remove(0);
                if(arr.isView()) {
                    //set closeable to prevent reuse elsewhere
                    arr.setCloseable(false);
                    log.trace("Found view array with id " + arr.getId() + " in cache. Avoiding allocation.");
                } else {
                    break;
                }
            }

            if (arr != null && arr.ordering() != descriptor.getOrder()) {
                arr.setOrder(descriptor.getOrder());
            }

            if (arr != null && !arr.wasClosed()) {
                // Decrement cache size
                currentCacheSize.set(currentCacheSize.get() - dataType.width() * arr.data().length());
                // We need to assign new Id. this way we will break any possible relationship it
                // had in Tracker.
                // the old cache was recreating New Array using buffer and thus gaining new
                // reference . Note that it had IdentityHash with references being keys
                lruCache.remove(arr.getId());
                lruCacheValues.remove(arr.getId());
                ((BaseNDArray) arr).assignNewId();
                return arr; // Allocated from cache
            }
        }

        // Allocation failed, allocate new array
        return Nd4j.createUninitializedDetached(dataType, shape);
    }

    @Override
    public void release(@NonNull INDArray array) {
        // Check for multiple releases of the array
        long id = array.getId();
        Preconditions.checkState(!lruCache.contains(id), "Array was released multiple times: id=%s, shape=%ndShape", id,
                array);

        if (enableCache == false) {
            if (array.closeable()) {
                array.close();
            }
            return;
        }

        DataType dt = array.dataType();
        if (array.data() == null && array.closeable()) {
            array.close();
            return;
        }

        if (array != null && array.data() != null && Nd4j.getExecutioner().useCount(array.data()) > 1) {
            // DataBuffer is used more than once. Close it and return
            if (array.closeable()) {
                array.close();
            }
            return;
        }

        long thisBytes = array.data().length() * dt.width();
        if (array.dataType() == DataType.UTF8) {
            // Don't cache string arrays due to variable length buffers
            if (array.closeable()) {
                array.close();
            }
        } else if (currentCacheSize.get() + thisBytes > maxCacheBytes.get()) {
            if (thisBytes > maxCacheBytes.get()) {

                // Can't store even if we clear everything - too large
                if (array.closeable())
                    array.close();
                return;
            }

            // Need to deallocate some arrays to stay under limit - do in "oldest first"
            // order
            Iterator<Long> iter = lruCache.iterator();
            while (currentCacheSize.get() + thisBytes > maxCacheBytes.get()) {
                long next = iter.next();
                iter.remove();
                INDArray nextOldest = lruCacheValues.remove(next);
                DataType ndt = nextOldest.dataType();
                long nextBytes = ndt.width() * nextOldest.data().length();
                List<INDArray> listx = arrays.get(ndt, Arrays.toString(nextOldest.shape()));
                if (listx != null)
                    listx.remove(nextOldest);
                currentCacheSize.set(currentCacheSize.get() - nextBytes);

                if (nextOldest.closeable()) {
                    nextOldest.close();
                }
            }

            // After clearing space - can now cache
            cacheArray(array);
        } else {
            // OK to cache
            cacheArray(array);
        }

        // Store in LRU cache for "last used" removal if we exceed cache size
        lruCache.add(array.getId());
        lruCacheValues.put(array.getId(), array);
    }

    private void cacheArray(INDArray array) {
        DataType dt = array.dataType();
        String arrayShapeString = Arrays.toString(array.shape());
        if (!arrays.contains(dt, arrayShapeString))
            arrays.put(dt, arrayShapeString, new ArrayList<>());
        arrays.get(dt, arrayShapeString).add(array);
        currentCacheSize.set(currentCacheSize.get() + array.data().length() * dt.width());

        lruCache.add(array.getId());
        lruCacheValues.put(array.getId(), array);

    }

    @Override
    public void close() {
        arrays.values().stream().forEach(input -> input.stream().forEach(arr -> {
            if (arr.closeable())
                arr.close();
        }));
    }

}
