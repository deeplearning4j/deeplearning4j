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
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

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
    // Growth factor for over-allocation on cache miss. Buffers are allocated
    // with this multiplier so that next step's slightly larger request (e.g.
    // growing KV cache) can reuse the buffer via capacity matching.
    // Default 1.05 (5% headroom) for better buffer reuse in autoregressive generation.
    // Disable with org.nd4j.cache.growthFactor=1.0 on memory-constrained systems.
    public final static double DEFAULT_GROWTH_FACTOR = 1.05;

    private static AtomicDouble largerArrayMaxMultiple;
    private static AtomicDouble growthFactor;

    // Thread-local scoped growth factor override.
    // When set (non-null), effectiveGrowthFactor() returns this instead of the global growthFactor.
    // Used by DSP execution to scope growth factor to DSP path only, preventing leakage
    // into standard op-by-op execution when DSP returns null (no plan).
    private static final ThreadLocal<Double> scopedGrowthFactor = new ThreadLocal<>();

    private static AtomicLong maxCacheBytes;
    private static AtomicLong totalMemBytes;
    @Getter
    @Setter
    private static  AtomicDouble maxMemFrac;
    private static AtomicLong currentCacheSize =  new AtomicLong(0);

    // Single LinkedHashMap for LRU tracking: insertion-ordered, provides both Set and Map views
    private static ThreadLocal<LinkedHashMap<Long, INDArray>> lruCacheValues = new ThreadLocal<>();

    // Capacity-based cache: TreeMap per DataType, keyed by buffer element count.
    // TreeMap.ceilingEntry(requiredElements) finds the smallest buffer >= needed in O(log n).
    private static ThreadLocal<Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>>> capacityArrays = new ThreadLocal<>();

    // Deferred close: DataBuffers from non-closeable arrays dropped by release().
    // These can't be closed immediately (views may share the buffer). Close at end of execution.
    private static ThreadLocal<IdentityHashMap<DataBuffer, Boolean>> deferredCloseBuffers = new ThreadLocal<>();

    private static boolean enableCache = Boolean
            .parseBoolean(System.getProperty(ND4JSystemProperties.SAMEDIFF_MEMORY_CACHE_ENABLE, "false"));

    static {
        setCacheDefaults();
        released.set(new IdentityHashMap<>());
        capacityArrays.set(new EnumMap<>(DataType.class));
        lruCacheValues.set(new LinkedHashMap<>());

    }

    private static Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> getCapacityArraysForThread() {
        Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> map = capacityArrays.get();
        if (map != null)
            return map;
        map = new EnumMap<>(DataType.class);
        capacityArrays.set(map);
        return map;
    }

    private static LinkedHashMap<Long, INDArray> getLruCachedValuesForThread() {
        if(lruCacheValues.get() != null)
            return lruCacheValues.get();
        else {
            lruCacheValues.set(new LinkedHashMap<>());
            return lruCacheValues.get();
        }
    }

    public static IdentityHashMap<DataBuffer, Boolean> getDeferredCloseBuffers() {
        IdentityHashMap<DataBuffer, Boolean> map = deferredCloseBuffers.get();
        if (map == null) {
            map = new IdentityHashMap<>();
            deferredCloseBuffers.set(map);
        }
        return map;
    }

    /**
     * Close all deferred DataBuffers (from non-closeable arrays dropped by release()).
     * Call this at the end of execution when all ops are done and no views are active.
     * @param protectedBuffers DataBuffers that must NOT be closed (constants, results, etc.)
     * @return number of buffers closed and total bytes freed
     */
    public static long[] closeDeferredBuffers(IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        IdentityHashMap<DataBuffer, Boolean> deferred = getDeferredCloseBuffers();
        int closed = 0;
        long closedBytes = 0;

        // Also protect ALL DataBuffers currently in the capacity cache.
        // Cached arrays' buffers must never be closed — they will be reused in future steps.
        // Without this, deferred close can destroy a buffer that a cached array references,
        // causing shape=[0,0,0,0,0] (freed shape info) on the next cache retrieval.
        IdentityHashMap<DataBuffer, Boolean> fullProtection = protectedBuffers != null
                ? new IdentityHashMap<>(protectedBuffers) : new IdentityHashMap<>();
        Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> allCapacity = getCapacityArraysForThread();
        for (TreeMap<Long, ArrayDeque<INDArray>> treeMap : allCapacity.values()) {
            for (ArrayDeque<INDArray> deque : treeMap.values()) {
                for (INDArray arr : deque) {
                    if (arr != null && !arr.wasClosed() && arr.data() != null) {
                        fullProtection.put(arr.data(), Boolean.TRUE);
                    }
                }
            }
        }

        for (DataBuffer buf : deferred.keySet()) {
            if (buf.wasClosed()) continue;
            if (fullProtection.containsKey(buf)) continue;
            // Don't skip constant-poisoned buffers: anything not in fullProtection that's
            // marked constant is a poisoned intermediate. Force-close it.
            try {
                if (buf.isConstant()) {
                    buf.setConstant(false);
                }
                closedBytes += buf.length() * buf.getElementSize();
                buf.close();
                closed++;
            } catch (Exception e) {
                // non-fatal
            }
        }
        deferred.clear();
        return new long[]{closed, closedBytes};
    }

    public static void setCacheDefaults() {
        maxMemFrac = new AtomicDouble(Double.parseDouble(System.getProperty(ND4JSystemProperties.CACHE_MEM_FRACTION,String.valueOf(DEFAULT_MAX_MEM_FRACTION))));
        smallArrayThreshold = new AtomicLong(Long.parseLong(System.getProperty(ND4JSystemProperties.SMALL_ARRAY_THRESHOLD,String.valueOf(DEFAULT_SMALL_ARRAY_THRESHOLD))));
        largerArrayMaxMultiple = new AtomicDouble(Double.parseDouble(System.getProperty(ND4JSystemProperties.LARGE_ARRAY_MAX_MULTIPLE,String.valueOf(DEFAULT_LARGE_ARRAY_MAX_MULTIPLE))));
        growthFactor = new AtomicDouble(Double.parseDouble(System.getProperty("org.nd4j.cache.growthFactor", String.valueOf(DEFAULT_GROWTH_FACTOR))));

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

    public static AtomicDouble getGrowthFactor() {
        return growthFactor;
    }

    public static void setGrowthFactor(double growthFactor) {
        ArrayCacheMemoryMgr.growthFactor.set(growthFactor);
    }

    /**
     * Set a thread-local scoped growth factor override.
     * Returns an AutoCloseable that restores the previous value on close.
     *
     * Usage:
     * <pre>
     * try (AutoCloseable scope = ArrayCacheMemoryMgr.withGrowthFactor(1.05)) {
     *     // DSP execution uses 1.05x growth factor
     * }
     * // growth factor restored to previous value (global default or outer scope)
     * </pre>
     *
     * This prevents DSP's growth factor from leaking into standard execution
     * when DSP returns null (no compiled plan).
     */
    public static AutoCloseable withGrowthFactor(double factor) {
        Double previous = scopedGrowthFactor.get();
        scopedGrowthFactor.set(factor);
        return () -> scopedGrowthFactor.set(previous);
    }

    /**
     * Get the effective growth factor for the current thread.
     * Returns the scoped override if set, otherwise the global growthFactor.
     */
    public static double effectiveGrowthFactor() {
        Double scoped = scopedGrowthFactor.get();
        return scoped != null ? scoped : growthFactor.get();
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
    // [0] = exact hits, [1] = capacity hits, [2] = full misses,
    // [3] = releases skipped (view), [4] = releases cached, [5] = overalloc count
    private static final ThreadLocal<long[]> cacheCounters = ThreadLocal.withInitial(() -> new long[6]);

    public static void resetCacheCounters() {
        long[] c = cacheCounters.get();
        c[0] = c[1] = c[2] = c[3] = c[4] = c[5] = 0;
    }

    public static long[] getCacheCounters() {
        return cacheCounters.get();
    }

    /**
     * Zero all arrays in the capacity cache for the current thread.
     * Call between forward passes to prevent stale data from being reused.
     * Arrays remain in the cache and can be reused on the next pass — but their
     * contents are zeroed so no stale intermediate values leak across passes.
     * This is preferred over closing+clearing because some arrays may still be
     * referenced by the SameDiff execution graph cleanup path.
     */
    public static void zeroCapacityCache() {
        Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> allCapacity = getCapacityArraysForThread();
        int zeroedCount = 0;
        for (TreeMap<Long, ArrayDeque<INDArray>> treeMap : allCapacity.values()) {
            for (ArrayDeque<INDArray> deque : treeMap.values()) {
                for (INDArray arr : deque) {
                    if (arr != null && !arr.wasClosed() && arr.data() != null) {
                        arr.assign(0);
                        zeroedCount++;
                    }
                }
            }
        }
        if (zeroedCount > 0) {
            log.debug("zeroCapacityCache: zeroed {} cached arrays", zeroedCount);
        }
    }

    private static boolean isCpu() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        return !"CUDA".equalsIgnoreCase(backend);
    }

    /**
     * Try to find a cached array with a buffer large enough for the requested shape.
     * Uses TreeMap.ceilingEntry to find the smallest buffer >= requiredElements.
     * Returns null if no suitable buffer is found.
     *
     * @param dataType the data type
     * @param shape the requested shape
     * @param requiresZeroed if true, zero the buffer before returning (for ops with sparse output)
     */
    private INDArray tryAllocateFromCapacityCache(DataType dataType, long[] shape, boolean requiresZeroed) {
        if (!enableCache || shape == null || shape.length == 0)
            return null;

        long requiredElements = ArrayUtil.prodLong(shape);
        if (requiredElements <= 0)
            return null;

        Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> allCapacity = getCapacityArraysForThread();
        TreeMap<Long, ArrayDeque<INDArray>> treeMap = allCapacity.get(dataType);
        if (treeMap == null || treeMap.isEmpty())
            return null;

        LinkedHashMap<Long, INDArray> lru = getLruCachedValuesForThread();
        long maxElements = (long)(requiredElements * largerArrayMaxMultiple.get());
        long[] counters = cacheCounters.get();

        // Search for a suitable buffer starting from ceilingEntry
        Map.Entry<Long, ArrayDeque<INDArray>> entry = treeMap.ceilingEntry(requiredElements);
        while (entry != null) {
            long bufferElements = entry.getKey();
            // Check if buffer is within acceptable size (not too wasteful)
            if (bufferElements > maxElements)
                break;

            ArrayDeque<INDArray> deque = entry.getValue();
            // Try to get a valid array from this deque
            while (deque != null && !deque.isEmpty()) {
                INDArray arr = deque.poll();
                if (arr == null) continue;

                // Clean up empty deque
                if (deque.isEmpty()) {
                    treeMap.remove(bufferElements);
                }

                // Skip invalid arrays: check INDArray state AND DataBuffer health.
                // The InferenceSession cleanup force-closes DataBuffers of non-closeable
                // arrays (views). If a cached array shares a DataBuffer with a force-closed
                // view (parent-view relationship), the shared buffer is destroyed but the
                // cached INDArray doesn't know. We must check data().wasClosed() to catch this.
                // Also check the native OpaqueDataBuffer pointer: the GC deallocator
                // (OpaqueDataBufferDeallocator) can free the native memory and call
                // buffer.setNull() while the Java DataBuffer is still alive and released=false.
                // This leaves wasClosed()=false but the native pointer at address 0, causing
                // a "dataBuffer is null" crash in native dbSetDeviceId/dbClose calls.
                DataBuffer arrBuf = arr.data();
                boolean nativePointerInvalid = false;
                if (arrBuf != null && !arrBuf.wasClosed()) {
                    try {
                        var opaque = arrBuf.opaqueBuffer();
                        nativePointerInvalid = (opaque == null || opaque.isNull());
                    } catch (IllegalStateException e) {
                        // opaqueBuffer() throws if released - treat as invalid
                        nativePointerInvalid = true;
                    }
                }
                if (!arr.closeable() || arr.wasClosed() || arr.isView()
                        || arrBuf == null || arrBuf.wasClosed() || nativePointerInvalid) {
                    lru.remove(arr.getId());
                    long skippedBytes = arrBuf != null && !arrBuf.wasClosed() ? dataType.width() * arrBuf.length() : 0;
                    currentCacheSize.addAndGet(-skippedBytes);
                    if (arr.isView()) {
                        arr.setCloseable(false);
                    }
                    continue;
                }

                // DataType safety check
                if (arr.dataType() != dataType) {
                    lru.remove(arr.getId());
                    long skippedBytes = arr.data() != null ? dataType.width() * arr.data().length() : 0;
                    currentCacheSize.addAndGet(-skippedBytes);
                    if (arr.closeable()) arr.close();
                    continue;
                }

                // Found a valid array - decrement cache size using buffer capacity
                long cachedBytes = dataType.width() * arr.data().length();
                currentCacheSize.addAndGet(-cachedBytes);
                lru.remove(arr.getId());

                boolean isExactSize = (arr.data().length() == requiredElements);
                boolean isExactShape = isExactSize && Arrays.equals(arr.shape(), shape);

                // Always reset strides to contiguous layout. Cached arrays may have
                // broadcast strides (e.g. [1,1] for shape [1,1024]) from prior assign(scalar)
                // operations. These non-contiguous strides cause buffer overruns in downstream
                // ops like scatter_nd_update that iterate using physical strides.
                long[] newStrides = Nd4j.getStrides(shape, arr.ordering());
                if (!isExactShape) {
                    int[] intShape = ArrayUtil.toInts(shape);
                    int[] intStrides = ArrayUtil.toInts(newStrides);
                    ((BaseNDArray) arr).setShapeAndStride(intShape, intStrides);
                    if (isExactSize) {
                        counters[0]++;
                    } else {
                        counters[1]++;
                    }
                } else {
                    // Even for exact shape match, verify strides are contiguous
                    long[] currentStrides = arr.stride();
                    boolean stridesMatch = true;
                    for (int s = 0; s < currentStrides.length; s++) {
                        if (currentStrides[s] != newStrides[s]) {
                            stridesMatch = false;
                            break;
                        }
                    }
                    if (!stridesMatch) {
                        int[] intShape = ArrayUtil.toInts(shape);
                        int[] intStrides = ArrayUtil.toInts(newStrides);
                        ((BaseNDArray) arr).setShapeAndStride(intShape, intStrides);
                    }
                    counters[0]++;
                }

                ((BaseNDArray) arr).assignNewId();
                // Reset native sync counters for cached buffer reuse.
                // Guard against null native pointer: the GC deallocator can free the
                // OpaqueDataBuffer (setting its native pointer to 0/null) while the Java
                // DataBuffer object is still alive. Passing a null pointer to dbSetDeviceId
                // crashes with "dataBuffer is null" from the C++ side.
                if (arr.data() != null) {
                    try {
                        var opaque = arr.data().opaqueBuffer();
                        if (opaque != null && !opaque.isNull()) {
                            Nd4j.getNativeOps().dbSetDeviceId(opaque, -1);
                        }
                    } catch (IllegalStateException e) {
                        // opaqueBuffer() throws if DataBuffer was released - skip
                        log.debug("Cached array's DataBuffer was released before dbSetDeviceId - skipping");
                    }
                }
                // Always zero cached arrays to prevent stale data from previous forward
                // passes from leaking into computation. The cache is kept enabled during
                // standard executeOperations to prevent premature DataBuffer closes for
                // view-producing ops, but cached arrays carry stale intermediate values.
                // Without zeroing, ops that don't fully overwrite output (or reuse buffers
                // slightly larger than needed) see stale data that causes degenerate output
                // in multi-step autoregressive decode.
                arr.assign(0);
                return arr;
            }

            // This deque was exhausted, try next larger entry
            entry = treeMap.higherEntry(bufferElements);
        }

        counters[2]++; // miss
        return null;
    }

    /**
     * Allocate a new array with growth headroom so that future slightly-larger
     * requests (e.g. KV cache growing by 1 token per step) can reuse this buffer
     * via capacity matching instead of hitting cudaMalloc again.
     *
     * Creates a 1D buffer with requiredElements * growthFactor, then reshapes
     * to the requested shape. The extra capacity is invisible to ops (they see
     * the shape, not buffer length) but available for future capacity matching.
     */
    private INDArray allocateWithHeadroom(boolean detached, DataType dataType, long[] shape) {
        long requiredElements = ArrayUtil.prodLong(shape);
        double gf = effectiveGrowthFactor();

        // Only over-allocate for larger arrays (> 10K elements) where growing shapes
        // (e.g. KV cache) benefit from headroom. Small arrays don't grow and there
        // are many of them, so over-allocating wastes memory for no cache benefit.
        long overAllocThreshold = Math.max(smallArrayThreshold.get(), 10_000);
        if (!enableCache || gf <= 1.0 || requiredElements <= overAllocThreshold) {
            return detached ? Nd4j.createUninitializedDetached(dataType, shape) : Nd4j.create(dataType, shape);
        }

        long allocElements = (long)(requiredElements * gf);
        // Create oversized 1D array
        INDArray oversized = detached
                ? Nd4j.createUninitializedDetached(dataType, allocElements)
                : Nd4j.createUninitialized(dataType, allocElements);

        // Reshape to the requested shape (buffer retains full capacity)
        long[] newStrides = Nd4j.getStrides(shape, oversized.ordering());
        int[] intShape = ArrayUtil.toInts(shape);
        int[] intStrides = ArrayUtil.toInts(newStrides);
        ((BaseNDArray) oversized).setShapeAndStride(intShape, intStrides);
        ((BaseNDArray) oversized).assignNewId(); // clear OpaqueNDArray after reshape

        cacheCounters.get()[5]++; // overalloc count
        return oversized;
    }


    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        // Default: don't require zeroed output (most ops fully write their output)
        return allocate(detached, dataType, shape, false);
    }

    /**
     * Allocate an array, optionally zeroing it for ops with sparse output patterns.
     *
     * @param detached if true, allocate detached from any workspace
     * @param dataType the data type
     * @param shape the requested shape
     * @param requiresZeroed if true, zero the buffer (for ops like where, scatter_nd, unique)
     */
    public INDArray allocate(boolean detached, DataType dataType, long[] shape, boolean requiresZeroed) {
        // Handle empty arrays (shape contains 0)
        if (shape != null && shape.length > 0 && ArrayUtil.prodLong(shape) == 0) {
            return Nd4j.emptyWithShape(shape, dataType);
        }

        INDArray cached = tryAllocateFromCapacityCache(dataType, shape, requiresZeroed);
        if (cached != null)
            return cached;

        // Cache miss - allocate with headroom for future reuse
        return allocateWithHeadroom(detached, dataType, shape);
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        return allocate(detached, descriptor, false);
    }

    /**
     * Allocate from descriptor, optionally zeroing for ops with sparse output.
     */
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor, boolean requiresZeroed) {
        if (descriptor.isEmpty()) {
            INDArray ret = Nd4j.create(descriptor);
            if (detached) {
                ret = ret.detach();
            }
            return ret;
        }

        return allocate(detached, descriptor.dataType(), descriptor.getShape(), requiresZeroed);
    }

    @Override
    public  void release(@NonNull INDArray array) {
        {
            LinkedHashMap<Long, INDArray> lru = getLruCachedValuesForThread();

            // Check for multiple releases of the array (only for closeable arrays that might be cached)
            long id = array.getId();
            if (array.closeable()) {
                Preconditions.checkState(!lru.containsKey(id), "Array was released multiple times: id=%s, shape=%ndShape", id,
                        array);
            }

            // Handle non-closeable arrays (views, oversized buffers, etc.)
            // closeable() returns false when length() < data().length(). We can't close them
            // here because views may still share the DataBuffer with live arrays. Instead,
            // accumulate them for deferred close at the end of execution when all ops are done.
            if (!array.closeable()) {
                DataBuffer buf = array.data();
                if (buf != null && !buf.isConstant() && !buf.wasClosed()) {
                    getDeferredCloseBuffers().put(buf, Boolean.TRUE);
                }
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
                Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> allCapacity = getCapacityArraysForThread();
                Iterator<Map.Entry<Long, INDArray>> iter = lru.entrySet().iterator();
                while (currentCacheSize.get() + thisBytes > maxCacheBytes.get() && iter.hasNext()) {
                    Map.Entry<Long, INDArray> entry = iter.next();
                    iter.remove();
                    INDArray nextOldest = entry.getValue();
                    if (nextOldest == null) continue;
                    DataType ndt = nextOldest.dataType();
                    long nextBytes = ndt.width() * nextOldest.data().length();
                    long evictKey = nextOldest.data().length(); // buffer element count

                    TreeMap<Long, ArrayDeque<INDArray>> evictTree = allCapacity.get(ndt);
                    if (evictTree != null) {
                        ArrayDeque<INDArray> listx = evictTree.get(evictKey);
                        if (listx != null) {
                            listx.remove(nextOldest);
                            if (listx.isEmpty()) {
                                evictTree.remove(evictKey);
                            }
                        }
                    }
                    currentCacheSize.addAndGet(-nextBytes);

                    // Don't close evicted arrays immediately! Their DataBuffers may be
                    // shared with views (from reshape/permute/transpose ops) that are still
                    // in use by downstream ops during execution. Closing the DataBuffer
                    // here would zero out those views, causing all-zero propagation.
                    // Instead, defer close to end of execution when all ops are done.
                    DataBuffer evictBuf = nextOldest.data();
                    if (evictBuf != null && !evictBuf.isConstant() && !evictBuf.wasClosed()) {
                        getDeferredCloseBuffers().put(evictBuf, Boolean.TRUE);
                    }
                }

                // After clearing space - can now cache
                cacheArrayInternal(array);
            } else {
                // OK to cache
                cacheArrayInternal(array);
            }
        }
    }

    // Internal method - no synchronization needed (ThreadLocal data)
    private void cacheArrayInternal(INDArray array) {
        DataType dt = array.dataType();
        long bufferElements = array.data().length(); // buffer capacity, not shape product
        Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> allCapacity = getCapacityArraysForThread();
        TreeMap<Long, ArrayDeque<INDArray>> treeMap = allCapacity.computeIfAbsent(dt, k -> new TreeMap<>());
        treeMap.computeIfAbsent(bufferElements, k -> new ArrayDeque<>()).add(array);
        currentCacheSize.addAndGet(array.data().length() * dt.width());

        LinkedHashMap<Long, INDArray> lru = getLruCachedValuesForThread();
        lru.put(array.getId(), array);
        cacheCounters.get()[4]++; // releases cached
    }

    @Override
    public void scopeOut() {
        // When the cache is enabled, do NOT destroy cached arrays on scope exit.
        // The whole point of ArrayCacheMemoryMgr is to reuse allocations across
        // successive output() calls (e.g., autoregressive decode steps).
        // The close() method handles final cleanup.
        if (enableCache) {
            return;
        }

        // Cache disabled: close everything immediately (original behavior)
        Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> allCapacity = getCapacityArraysForThread();
        for (TreeMap<Long, ArrayDeque<INDArray>> treeMap : allCapacity.values()) {
            for (ArrayDeque<INDArray> deque : treeMap.values()) {
                for (INDArray arr : deque) {
                    if (arr != null && arr.closeable() && !arr.wasClosed()) {
                        arr.close();
                    }
                }
            }
        }
        allCapacity.clear();
        getLruCachedValuesForThread().clear();
        currentCacheSize.set(0);
    }

    @Override
    public void close() {
        close(null);
    }

    /**
     * Close all cached arrays, freeing their DataBuffers.
     * @param protectedBuffers DataBuffers to skip (e.g., placeholder/static KV buffers that
     *                         are shared with cached views via reshape_no_copy/permute).
     *                         May be null if no protection is needed.
     */
    public void close(IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        // Log cache effectiveness
        long[] counters = cacheCounters.get();
        long total = counters[0] + counters[1] + counters[2];
        if (total > 0) {
            log.info("ArrayCacheMemoryMgr closing - exact hits: {}, capacity hits: {}, misses: {}, cached: {}, overallocs: {} (hit rate: {}/{} = {}%)",
                    counters[0], counters[1], counters[2], counters[4], counters[5],
                    counters[0] + counters[1], total,
                    Math.round(100.0 * (counters[0] + counters[1]) / total));
        }

        long[] deferredStats = closeDeferredBuffers(protectedBuffers);

        // Close unique DataBuffers directly using IdentityHashMap to ensure each
        // physical buffer is closed exactly once.
        Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> allCapacity = getCapacityArraysForThread();
        IdentityHashMap<DataBuffer, Boolean> uniqueBuffers = new IdentityHashMap<>();
        for (TreeMap<Long, ArrayDeque<INDArray>> treeMap : allCapacity.values()) {
            for (ArrayDeque<INDArray> deque : treeMap.values()) {
                for (INDArray arr : deque) {
                    if (arr != null && !arr.wasClosed() && arr.data() != null) {
                        uniqueBuffers.put(arr.data(), Boolean.TRUE);
                    }
                }
            }
        }

        int skippedProtected = 0;
        int forceClosedConstant = 0;
        int closedNormal = 0;
        long closedBytes = 0;
        for (DataBuffer buf : uniqueBuffers.keySet()) {
            if (protectedBuffers != null && protectedBuffers.containsKey(buf)) {
                skippedProtected++;
                continue;
            }
            if (buf.closeable()) {
                try {
                    closedBytes += buf.length() * buf.getElementSize();
                    buf.close();
                    closedNormal++;
                } catch (Exception e) {
                    // Buffer may already be deallocated; ignore
                }
            } else if (buf.isConstant() && !buf.isAttached()) {
                // Constant-poisoned intermediate: the buffer was marked isConstant=true
                // by setCloseable(false) on placeholder arrays, propagated through shared
                // DataBuffers. Since it's not in protectedBuffers, it's safe to force-close.
                try {
                    closedBytes += buf.length() * buf.getElementSize();
                    buf.setConstant(false);
                    buf.close();
                    forceClosedConstant++;
                } catch (Exception e) {
                    // non-fatal
                }
            }
        }
        log.info("ArrayCacheMemoryMgr.close: uniqueBuffers={}, closedNormal={}, forceClosedConstant={}, skippedProtected={}, closedBytes={}MB, deferred={}",
                uniqueBuffers.size(), closedNormal, forceClosedConstant, skippedProtected,
                closedBytes / (1024 * 1024), deferredStats[0]);

        allCapacity.clear();
        getLruCachedValuesForThread().clear();
        currentCacheSize.set(0);
    }

    /**
     * Reset all cache data structures without closing any buffers.
     */
    public static void clearCacheState() {
        getCapacityArraysForThread().clear();
        getLruCachedValuesForThread().clear();
        getDeferredCloseBuffers().clear();
        currentCacheSize.set(0);
        resetCacheCounters();
    }

    /**
     * Collect all unique DataBuffers from the capacity cache and deferred close set.
     */
    public static void collectAllManagedBuffers(IdentityHashMap<DataBuffer, Boolean> target) {
        Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> allCapacity = getCapacityArraysForThread();
        for (TreeMap<Long, ArrayDeque<INDArray>> treeMap : allCapacity.values()) {
            for (ArrayDeque<INDArray> deque : treeMap.values()) {
                for (INDArray arr : deque) {
                    if (arr != null && !arr.wasClosed() && arr.data() != null) {
                        target.put(arr.data(), Boolean.TRUE);
                    }
                }
            }
        }
        IdentityHashMap<DataBuffer, Boolean> deferred = getDeferredCloseBuffers();
        for (DataBuffer buf : deferred.keySet()) {
            if (buf != null && !buf.wasClosed()) {
                target.put(buf, Boolean.TRUE);
            }
        }
    }

    @Override
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer) {
        return allocateFromDescriptor(detached, dataBuffer, false);
    }

    /**
     * Allocate from shape descriptor, optionally zeroing for ops with sparse output.
     */
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer, boolean requiresZeroed) {
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

        INDArray cached = tryAllocateFromCapacityCache(dataType, shape, requiresZeroed);
        if (cached != null) {
            // Fix ordering if needed
            if (cached.ordering() != Shape.order(asJava)) {
                cached.setOrder(Shape.order(asJava));
            }
            return cached;
        }

        // Cache miss - allocate with headroom for future reuse
        return allocateWithHeadroom(detached, dataType, shape);
    }


}
