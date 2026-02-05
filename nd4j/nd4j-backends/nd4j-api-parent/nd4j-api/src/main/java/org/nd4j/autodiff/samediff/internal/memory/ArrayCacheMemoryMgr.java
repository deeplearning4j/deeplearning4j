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

    // Single LinkedHashMap for LRU tracking: insertion-ordered, provides both Set and Map views
    private static ThreadLocal<LinkedHashMap<Long, INDArray>> lruCacheValues = new ThreadLocal<>();

    private static ThreadLocal<Map<Long, ArrayDeque<INDArray>>> arrays = new ThreadLocal<>();

    private static boolean enableCache = Boolean
            .parseBoolean(System.getProperty(ND4JSystemProperties.SAMEDIFF_MEMORY_CACHE_ENABLE, "false"));

    static {
        setCacheDefaults();
        released.set(new IdentityHashMap<>());
        arrays.set(new HashMap<>());
        lruCacheValues.set(new LinkedHashMap<>());

    }

    /**
     * Compute a hash key from DataType and shape, avoiding String allocation.
     * Collisions are safe because retrieval verifies shape match.
     */
    private static long shapeKey(DataType dt, long[] shape) {
        long h = 17L * 31 + dt.ordinal();
        for (long s : shape) {
            h = h * 31 + s;
        }
        return h;
    }


    private static Map<Long, ArrayDeque<INDArray>> getArraysForThread() {
        if(arrays.get() != null)
            return arrays.get();
        else {
            arrays.set(new HashMap<>());
            return arrays.get();
        }
    }
    private static LinkedHashMap<Long, INDArray> getLruCachedValuesForThread() {
        if(lruCacheValues.get() != null)
            return lruCacheValues.get();
        else {
            lruCacheValues.set(new LinkedHashMap<>());
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
        return getLruCachedValuesForThread().keySet();
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



    public static boolean isCacheEnabled() {
        return enableCache;
    }

    public static void setEnableCache(boolean enable) {
        enableCache = enable;
    }

    // Debug counters for capacity cache effectiveness
    private static final ThreadLocal<long[]> cacheCounters = ThreadLocal.withInitial(() -> new long[6]);
    // [0] = exact hits, [1] = capacity hits, [2] = full misses,
    // [3] = releases skipped (view), [4] = releases cached, [5] = capacity cache size at lookup

    public static void resetCacheCounters() {
        long[] c = cacheCounters.get();
        c[0] = c[1] = c[2] = c[3] = c[4] = c[5] = 0;
    }

    public static long[] getCacheCounters() {
        return cacheCounters.get();
    }

    private static boolean isCpu() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        return !"CUDA".equalsIgnoreCase(backend);
    }



    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        // Handle empty arrays (shape contains 0)
        if (shape != null && shape.length > 0 && org.nd4j.common.util.ArrayUtil.prodLong(shape) == 0) {
            return Nd4j.emptyWithShape(shape, dataType);
        }

        {
            long key = shapeKey(dataType, shape);
            Map<Long, ArrayDeque<INDArray>> arraysForThread = getArraysForThread();
            LinkedHashMap<Long, INDArray> lru = getLruCachedValuesForThread();
            ArrayDeque<INDArray> cached = arraysForThread.get(key);
            if (cached != null && !cached.isEmpty() && enableCache) {
                INDArray arr = null;
                boolean arrFound = false;
                while(!arrFound) {
                    arr = cached.poll();
                    if (arr == null) break;
                    // Verify shape match (hash collision safety)
                    if (arr.dataType() != dataType || !Arrays.equals(arr.shape(), shape)) {
                        // Hash collision - skip this array, close it
                        lru.remove(arr.getId());
                        long skippedBytes = arr.data() != null ? dataType.width() * arr.data().length() : 0;
                        currentCacheSize.addAndGet(-skippedBytes);
                        if (arr.closeable()) arr.close();
                        continue;
                    }
                    if(!arr.closeable() || arr.wasClosed() || arr.isView()) {
                        log.trace("Found array closeable, not returning from cache. Only closeable arrays are returnable from the cache.");
                        if(arr.isView()) {
                            arr.setCloseable(false);
                        }
                        // Remove from LRU tracking since we removed it from the cache list
                        lru.remove(arr.getId());
                        long skippedBytes = arr.data() != null ? dataType.width() * arr.data().length() : 0;
                        currentCacheSize.addAndGet(-skippedBytes);
                        log.trace("Found view array with id " + arr.getId() + " in cache. Avoiding return. Allocating new array.");
                        continue;
                    }

                    // Good array found
                    // Decrement cache size
                    currentCacheSize.addAndGet(-(long)(dataType.width() * arr.data().length()));
                    lru.remove(arr.getId());
                    ((BaseNDArray) arr).assignNewId();
                    // Reset native sync counters for cached buffer reuse.
                    if (arr.data() != null) {
                        Nd4j.getNativeOps().dbSetDeviceId(arr.data().opaqueBuffer(), -1);
                    }
                    // Zero out stale data to match Nd4j.create() behavior
                    arr.assign(0);
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

        {
            long key = shapeKey(dataType, shape);
            Map<Long, ArrayDeque<INDArray>> arraysForThread = getArraysForThread();
            LinkedHashMap<Long, INDArray> lru = getLruCachedValuesForThread();
            ArrayDeque<INDArray> cached = arraysForThread.get(key);
            if (cached != null && !cached.isEmpty() && enableCache && shape.length > 0 && !Longs.contains(shape, 0)) {
                INDArray arr = null;

                while (!cached.isEmpty()) {
                    arr = cached.poll();
                    // Verify shape match (hash collision safety)
                    if (arr.dataType() != dataType || !Arrays.equals(arr.shape(), shape)) {
                        lru.remove(arr.getId());
                        long skippedBytes = arr.data() != null ? dataType.width() * arr.data().length() : 0;
                        currentCacheSize.addAndGet(-skippedBytes);
                        if (arr.closeable()) arr.close();
                        arr = null;
                        continue;
                    }
                    if(arr.isView()) {
                        //set closeable to prevent reuse elsewhere
                        arr.setCloseable(false);
                        // Remove from LRU tracking since we removed it from the cache list
                        lru.remove(arr.getId());
                        long skippedBytes = arr.data() != null ? dataType.width() * arr.data().length() : 0;
                        currentCacheSize.addAndGet(-skippedBytes);
                        log.trace("Found view array with id " + arr.getId() + " in cache. Avoiding allocation.");
                        arr = null;
                    } else {
                        break;
                    }
                }

                if (arr != null && arr.ordering() != descriptor.getOrder()) {
                    arr.setOrder(descriptor.getOrder());
                }

                if (arr != null && !arr.wasClosed() && arr.closeable()) {
                    // Decrement cache size
                    currentCacheSize.addAndGet(-(long)(dataType.width() * arr.data().length()));
                    lru.remove(arr.getId());
                    ((BaseNDArray) arr).assignNewId();
                    // Reset native sync counters for cached buffer reuse.
                    if (arr.data() != null) {
                        Nd4j.getNativeOps().dbSetDeviceId(arr.data().opaqueBuffer(), -1);
                    }
                    // Zero out stale data to match Nd4j.create() behavior
                    arr.assign(0);
                    return arr; // Allocated from cache
                }
            }
        }

        // Allocation failed, allocate new array
        return detached ? Nd4j.create(dataType, shape).detach() : Nd4j.create(dataType, shape);
    }

    @Override
    public  void release(@NonNull INDArray array) {
        {
            Map<Long, ArrayDeque<INDArray>> arraysForThread = getArraysForThread();
            LinkedHashMap<Long, INDArray> lru = getLruCachedValuesForThread();

            // Check for multiple releases of the array (only for closeable arrays that might be cached)
            long id = array.getId();
            if (array.closeable()) {
                Preconditions.checkState(!lru.containsKey(id), "Array was released multiple times: id=%s, shape=%ndShape", id,
                        array);
            }

            // Handle non-closeable arrays (views, constants, etc.)
            if (!array.closeable()) {
                return;
            }

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

            // Note: useCount > 1 check removed - it silently leaked arrays (neither cached nor closed).
            // View arrays are already filtered by the closeable/isView checks above.

            long thisBytes = array.data().length() * dt.width();
            if (array.dataType() == DataType.UTF8) {
                // Don't cache string arrays due to variable length buffers
                if (array.closeable()) {
                    array.close();
                }
                return;
            } else if (currentCacheSize.get() + thisBytes > maxCacheBytes.get()) {
                if (thisBytes > maxCacheBytes.get()) {

                    // Can't store even if we clear everything - too large
                    if (array.closeable())
                        array.close();
                    return;
                }

                // Need to deallocate some arrays to stay under limit - do in "oldest first"
                // order (LinkedHashMap iterates in insertion order)
                Iterator<Map.Entry<Long, INDArray>> iter = lru.entrySet().iterator();
                while (currentCacheSize.get() + thisBytes > maxCacheBytes.get() && iter.hasNext()) {
                    Map.Entry<Long, INDArray> entry = iter.next();
                    iter.remove();
                    INDArray nextOldest = entry.getValue();
                    if (nextOldest == null) continue;
                    DataType ndt = nextOldest.dataType();
                    long nextBytes = ndt.width() * nextOldest.data().length();
                    long evictKey = shapeKey(ndt, nextOldest.shape());
                    ArrayDeque<INDArray> listx = arraysForThread.get(evictKey);
                    if (listx != null)
                        listx.remove(nextOldest);
                    currentCacheSize.addAndGet(-nextBytes);

                    if (nextOldest.closeable()) {
                        nextOldest.close();
                    }
                }

                // After clearing space - can now cache
                cacheArrayInternal(array, arraysForThread, lru);
            } else {
                // OK to cache
                cacheArrayInternal(array, arraysForThread, lru);
            }
        }
    }

    private void cacheArray(INDArray array) {
        {
            cacheArrayInternal(array, getArraysForThread(), getLruCachedValuesForThread());
        }
    }

    // Internal method - no synchronization needed (ThreadLocal data)
    private void cacheArrayInternal(INDArray array, Map<Long, ArrayDeque<INDArray>> arraysForThread,
                                    LinkedHashMap<Long, INDArray> lru) {
        DataType dt = array.dataType();
        long key = shapeKey(dt, array.shape());
        arraysForThread.computeIfAbsent(key, k -> new ArrayDeque<>()).add(array);
        currentCacheSize.addAndGet(array.data().length() * dt.width());

        lru.put(array.getId(), array);

    }

    @Override
    public void close() {
        {
            Map<Long, ArrayDeque<INDArray>> arraysForThread = getArraysForThread();
            LinkedHashMap<Long, INDArray> lru = getLruCachedValuesForThread();

            arraysForThread.values().forEach(deque -> deque.forEach(arr -> {
                if (arr != null && arr.closeable() && !arr.wasClosed()) {
                    arr.close();
                }
            }));

            // Clear the caches
            arraysForThread.clear();
            lru.clear();
            currentCacheSize.set(0);
        }
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

        {
            long key = shapeKey(dataType, shape);
            Map<Long, ArrayDeque<INDArray>> arraysForThread = getArraysForThread();
            LinkedHashMap<Long, INDArray> lru = getLruCachedValuesForThread();
            ArrayDeque<INDArray> cached = arraysForThread.get(key);
            if (cached != null && !cached.isEmpty() && enableCache && shape.length > 0 && !Longs.contains(shape, 0)) {
                INDArray arr = null;

                while (!cached.isEmpty()) {
                    arr = cached.poll();
                    // Verify shape match (hash collision safety)
                    if (arr.dataType() != dataType || !Arrays.equals(arr.shape(), shape)) {
                        lru.remove(arr.getId());
                        long skippedBytes = arr.data() != null ? dataType.width() * arr.data().length() : 0;
                        currentCacheSize.addAndGet(-skippedBytes);
                        if (arr.closeable()) arr.close();
                        arr = null;
                        continue;
                    }
                    if(arr.isView()) {
                        //set closeable to prevent reuse elsewhere
                        arr.setCloseable(false);
                        // Remove from LRU tracking since we removed it from the cache list
                        lru.remove(arr.getId());
                        long skippedBytes = arr.data() != null ? dataType.width() * arr.data().length() : 0;
                        currentCacheSize.addAndGet(-skippedBytes);
                        log.trace("Found view array with id " + arr.getId() + " in cache. Avoiding allocation.");
                        arr = null;
                    } else {
                        break;
                    }
                }

                if (arr != null && arr.ordering() != Shape.order(asJava)) {
                    arr.setOrder(Shape.order(asJava));
                }

                if (arr != null && !arr.wasClosed() && arr.closeable()) {
                    // Decrement cache size
                    currentCacheSize.addAndGet(-(long)(dataType.width() * arr.data().length()));
                    lru.remove(arr.getId());
                    ((BaseNDArray) arr).assignNewId();
                    // Reset native sync counters for cached buffer reuse.
                    if (arr.data() != null) {
                        Nd4j.getNativeOps().dbSetDeviceId(arr.data().opaqueBuffer(), -1);
                    }
                    // Zero out stale data to match Nd4j.create() behavior
                    arr.assign(0);
                    return arr; // Allocated from cache
                }
            }
        }

        // Allocation failed, allocate new array
        return detached ? Nd4j.create(dataType, shape).detach() : Nd4j.create(dataType, shape);
    }


}
