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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
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

    private static ThreadLocal<Map<INDArray,INDArray>> released = new ThreadLocal<>();

    public final static double DEFAULT_MAX_MEM_FRACTION = 0.25;
    public final static long DEFAULT_SMALL_ARRAY_THRESHOLD = 1024;
    public final static double DEFAULT_LARGE_ARRAY_MAX_MULTIPLE = 2.0;
    private static AtomicDouble largerArrayMaxMultiple;

    private static AtomicLong maxCacheBytes;
    private static AtomicLong totalMemBytes;
    @Getter
    @Setter
    private static  AtomicDouble maxMemFrac;
    private static AtomicLong currentCacheSize =  new AtomicLong(0);

    private static ThreadLocal<Set<Long>> lruCache = new ThreadLocal<>();
    private static ThreadLocal<Map<Long, INDArray>> lruCacheValues = new ThreadLocal<>();

    private static ThreadLocal<Table<DataType, String, List<INDArray>>> arrays = new ThreadLocal<>();

    private static boolean enableCache = Boolean
            .parseBoolean(System.getProperty(ND4JSystemProperties.SAMEDIFF_MEMORY_CACHE_ENABLE, "true"));

    static {
        setCacheDefaults();
        released.set(new IdentityHashMap<>());
        arrays.set(HashBasedTable.create());
        lruCacheValues.set(new ConcurrentHashMap<>());
        lruCache.set(new ConcurrentSkipListSet<>());

    }


    private static Set<Long> getLruCacheForThread() {
        if(lruCache.get() != null)
            return lruCache.get();
        else {
            lruCache.set(new ConcurrentSkipListSet<>());
            return lruCache.get();
        }
    }

    private static Table<DataType, String, List<INDArray>> getArraysForThread() {
        if(arrays.get() != null)
            return arrays.get();
        else {
            arrays.set(HashBasedTable.create());
            return arrays.get();
        }
    }
    private static Map<Long, INDArray> getLruCachedValuesForThread() {
        if(lruCacheValues.get() != null)
            return lruCacheValues.get();
        else {
            lruCacheValues.set(new ConcurrentHashMap<>());
            return lruCacheValues.get();
        }
    }

    public static void setCacheDefaults() {
        maxMemFrac = new AtomicDouble(Double.parseDouble(System.getProperty(ND4JSystemProperties.CACHE_MEM_FRACTION,String.valueOf(DEFAULT_MAX_MEM_FRACTION))));
        smallArrayThreshold = new AtomicLong(Long.parseLong(System.getProperty(ND4JSystemProperties.SMALL_ARRAY_THRESHOLD,String.valueOf(DEFAULT_SMALL_ARRAY_THRESHOLD))));
        largerArrayMaxMultiple = new AtomicDouble(Double.parseDouble(System.getProperty(ND4JSystemProperties.LARGE_ARRAY_MAX_MULTIPLE,String.valueOf(DEFAULT_LARGE_ARRAY_MAX_MULTIPLE))));

        if (isCpu()) {
            totalMemBytes = new AtomicLong(Pointer.maxBytes());
        } else {
            Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
            List devList = (List) p.get("cuda.devicesInformation");
            Map m = (Map) devList.get(0);
            totalMemBytes = new AtomicLong((Long) m.get("cuda.totalMemory"));
        }

        long cacheValue = Math.round(maxMemFrac.get() * totalMemBytes.get());
        maxCacheBytes = new AtomicLong(cacheValue);
    }

    @Getter
    @Setter
    private static AtomicLong smallArrayThreshold;

    public static Set<Long> getLruCache() {
        return getLruCacheForThread();
    }

    public static Map<Long, INDArray> getLruCacheValues() {
        return getLruCachedValuesForThread();
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



    /**
     * Create an ArrayCacheMemoryMgr with default settings as per
     * {@link ArrayCacheMemoryMgr}
     */
    public ArrayCacheMemoryMgr() {

    }



    private static boolean isCpu() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        return !"CUDA".equalsIgnoreCase(backend);
    }



    @Override
    public synchronized INDArray allocate(boolean detached, DataType dataType, long... shape) {
        String arrayShapeString = Arrays.toString(shape);
        Table<DataType, String, List<INDArray>> arraysForThread = getArraysForThread();
        Set<Long> lruCacheForThread = getLruCacheForThread();
        Map<Long, INDArray> lruCacheValues = getLruCacheValues();
        if (arraysForThread.contains(dataType, arrayShapeString) && enableCache) {
            INDArray arr = null;
            boolean arrFound = false;
            while(!arrFound) {
                arr = !arraysForThread.get(dataType, arrayShapeString).isEmpty()
                        ? arraysForThread.get(dataType, arrayShapeString).remove(0)
                        : null;
                if(arr != null && (!arr.closeable() || arr.wasClosed() || arr.isView())) {
                    log.trace("Found array closeable, not returning from cache. Only closeable arrays are returnable from the cache.");
                    if(arr.isView())
                        arr.setCloseable(false);
                    log.trace("Found view array with id " + arr.getId() + " in cache. Avoiding return. Allocating new array.");

                    continue;
                } else if(!arraysForThread.contains(dataType, arrayShapeString) || getArraysForThread().get(dataType,arrayShapeString).isEmpty()) {
                    break;
                }

                if (arr != null) {
                    // Decrement cache size
                    currentCacheSize.set(currentCacheSize.get() - dataType.width() * arr.data().length());
                    lruCacheForThread.remove(arr.getId());
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
        //switch to using current workspace rather than detached
        INDArray ret = detached ? Nd4j.createUninitializedDetached(dataType,shape) : Nd4j.create(dataType, shape);
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
        Table<DataType, String, List<INDArray>> arraysForThread = getArraysForThread();
        if (arraysForThread.contains(dataType, arrayShape) && enableCache && shape.length > 0 && !Longs.contains(shape, 0)) {
            INDArray arr = null;
            List<INDArray> arrays2 = arraysForThread.get(dataType, arrayShape);

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
                getLruCache().remove(arr.getId());
                getLruCacheValues().remove(arr.getId());
                ((BaseNDArray) arr).assignNewId();
                return arr; // Allocated from cache
            }
        }

        // Allocation failed, allocate new array
        return Nd4j.createUninitializedDetached(dataType, shape);
    }

    @Override
    public  void release(@NonNull INDArray array) {
        if(!array.closeable())
            return;

        Set<Long> lruCacheForThread = getLruCacheForThread();
        Table<DataType, String, List<INDArray>> arraysForThread = getArraysForThread();
        Map<Long, INDArray> lruCacheValues = getLruCacheValues();
        // Check for multiple releases of the array
        long id = array.getId();
        Preconditions.checkState(!lruCacheForThread.contains(id), "Array was released multiple times: id=%s, shape=%ndShape", id,
                array);

        if (!enableCache) {
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
            Iterator<Long> iter = lruCacheForThread.iterator();
            while (currentCacheSize.get() + thisBytes > maxCacheBytes.get() && iter.hasNext()) {
                long next = iter.next();
                iter.remove();
                INDArray nextOldest = lruCacheValues.remove(next);
                DataType ndt = nextOldest.dataType();
                long nextBytes = ndt.width() * nextOldest.data().length();
                List<INDArray> listx = arraysForThread.get(ndt, Arrays.toString(nextOldest.shape()));
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
        lruCacheForThread.add(array.getId());
        lruCacheValues.put(array.getId(), array);
    }

    private void cacheArray(INDArray array) {
        DataType dt = array.dataType();
        Table<DataType, String, List<INDArray>> arraysForThread = getArraysForThread();
        Set<Long> lruCacheForThread = getLruCacheForThread();
        Map<Long, INDArray> lruCacheValues = getLruCacheValues();
        String arrayShapeString = Arrays.toString(array.shape());
        if (!arraysForThread.contains(dt, arrayShapeString))
            arraysForThread.put(dt, arrayShapeString, new ArrayList<>());
        arraysForThread.get(dt, arrayShapeString).add(array);
        currentCacheSize.set(currentCacheSize.get() + array.data().length() * dt.width());

        lruCacheForThread.add(array.getId());
        lruCacheValues.put(array.getId(), array);

    }

    @Override
    public void close() {
        getArraysForThread().values().stream().forEach(input -> input.stream().forEach(arr -> {
           // if (arr.closeable())
           //     arr.close();
        }));
    }

    @Override
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer) {
        long[] asJava = dataBuffer.asLong();
        if (Shape.isEmpty(asJava)) {
            INDArray ret = Nd4j.createFromDescriptor(dataBuffer);
            if (detached) {
                ret = ret.detach();
            }

            return ret;
        }

        DataType dataType = Shape.dataType(asJava);
        long[] shape = Shape.shape(asJava);
        String arrayShape = Arrays.toString(shape);
        Table<DataType, String, List<INDArray>> arraysForThread = getArraysForThread();
        if (arraysForThread.contains(dataType, arrayShape) && enableCache && shape.length > 0 && !Longs.contains(shape, 0)) {
            INDArray arr = null;
            List<INDArray> arrays2 = arraysForThread.get(dataType, arrayShape);

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

            if (arr != null && arr.ordering() != Shape.order(asJava)) {
                arr.setOrder(Shape.order(asJava));
            }

            if (arr != null && !arr.wasClosed()) {
                // Decrement cache size
                currentCacheSize.set(currentCacheSize.get() - dataType.width() * arr.data().length());
                // We need to assign new Id. this way we will break any possible relationship it
                // had in Tracker.
                // the old cache was recreating New Array using buffer and thus gaining new
                // reference . Note that it had IdentityHash with references being keys
                getLruCache().remove(arr.getId());
                getLruCacheValues().remove(arr.getId());
                ((BaseNDArray) arr).assignNewId();
                return arr; // Allocated from cache
            }
        }

        // Allocation failed, allocate new array
        return Nd4j.createUninitializedDetached(dataType, shape);
    }


}
