package org.nd4j.jita.memory.impl;

import jcuda.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.memory.MemoryProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


/**
 * This is MemoryProvider implementation, that adds cache for memory reuse purposes.
 * If some memory chunk gets released via allocator, it'll be probably saved for future reused within same JVM process.
 *
 * @author raver119@gmail.com
 */
public class CudaCachingProvider extends CudaDirectProvider implements MemoryProvider {
    private static Logger log = LoggerFactory.getLogger(CudaCachingProvider.class);

    private volatile ConcurrentHashMap<AllocationShape, CacheHolder> zeroCache = new ConcurrentHashMap<>();
    private volatile ConcurrentHashMap<Integer, ConcurrentHashMap<AllocationShape, CacheHolder>> deviceCache = new ConcurrentHashMap<>();

    private final AtomicLong cacheHit = new AtomicLong(0);
    private final AtomicLong cacheMiss = new AtomicLong(0);

    private final AtomicLong allocRequests = new AtomicLong(0);

    private final AtomicLong zeroCachedAmount = new AtomicLong(0);
    private final AtomicLong deviceCachedAmount = new AtomicLong(0);

    private final Semaphore singleLock = new Semaphore(1);

    // we don't cache allocations greater then this value
    private final long MAX_SINGLE_ALLOCATION = 1000000;

    // maximum cached size of memory
    private final long MAX_CACHED_MEMORY;

    // memory chunks below this threshold will be guaranteed regardless of number of cache entries
    // that especially covers all possible variations of shapeInfoDataBuffers in all possible cases
    private final long FORCED_CACHE_THRESHOLD = 96;

    //  number of preallocation entries for each yet-unknown shape
    private final int PREALLOCATION_LIMIT = 50;

    public CudaCachingProvider() {
        MAX_CACHED_MEMORY = Runtime.getRuntime().maxMemory() / 2;
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

        if (location == AllocationStatus.HOST  && reqMemory < MAX_SINGLE_ALLOCATION) {
            if (allocRequests.incrementAndGet() % 100000 == 0)
                printCacheStats();

            CacheHolder cache = zeroCache.get(shape);
            if (cache != null ) {
                Pointer pointer = cache.poll();
                if (pointer != null) {
                    cacheHit.incrementAndGet();

                    // since this memory chunk is going to be used now, remove it's amount from
                    zeroCachedAmount.addAndGet(-1 * reqMemory);

                    PointersPair pair = new PointersPair();
                    pair.setDevicePointer(new CudaPointer(pointer.getNativePointer()));
                    pair.setHostPointer(new CudaPointer(pointer.getNativePointer()));

                    point.setAllocationStatus(AllocationStatus.HOST);
                    return pair;
                }
            }
            cacheMiss.incrementAndGet();

            if (zeroCachedAmount.get() < MAX_CACHED_MEMORY / 10) {
                CachePreallocator preallocator = new CachePreallocator(shape, location, PREALLOCATION_LIMIT);
                preallocator.start();
            }

            cacheMiss.incrementAndGet();
            return super.malloc(shape, point, location);
        }

        if (location == AllocationStatus.DEVICE && reqMemory < MAX_SINGLE_ALLOCATION) {
            ensureDeviceCacheHolder(point.getDeviceId(), shape);

            CacheHolder cache = deviceCache.get(point.getDeviceId()).get(shape);
            if (cache != null) {
                Pointer pointer = cache.poll();
                if (pointer != null) {
                    cacheHit.incrementAndGet();

                    deviceCachedAmount.addAndGet(-1 * reqMemory);

                    PointersPair pair = new PointersPair();
                    pair.setDevicePointer(new CudaPointer(pointer.getNativePointer()));
                    pair.setHostPointer(new CudaPointer(pointer.getNativePointer()));

                    point.setAllocationStatus(AllocationStatus.DEVICE);
                    return pair;
                }
            }
            cacheMiss.incrementAndGet();
            return super.malloc(shape, point, location);
        }

        return super.malloc(shape, point, location);
    }

    protected  void ensureDeviceCacheHolder(Integer deviceId, AllocationShape shape) {
        if (!deviceCache.containsKey(deviceId)) {
            try {
                singleLock.acquire();

                if (!deviceCache.containsKey(deviceId)) {
                    deviceCache.put(deviceId, new ConcurrentHashMap<AllocationShape, CacheHolder>());
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                singleLock.release();
            }
        }

        if (!deviceCache.get(deviceId).containsKey(shape)) {
            try {
                singleLock.acquire();

                if (!deviceCache.get(deviceId).containsKey(shape)) {
                    deviceCache.get(deviceId).put(shape, new CacheHolder(shape));
                }
            } catch (Exception e) {

            } finally {
                singleLock.release();
            }
        }
    }

    protected void ensureCacheHolder(AllocationShape shape) {
        if (!zeroCache.containsKey(shape)) {
            try {
                singleLock.acquire();
                if (!zeroCache.containsKey(shape)) {
                    zeroCache.put(shape, new CacheHolder(shape));
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                singleLock.release();
            }
        }

    }

    /**
     * This method frees specific chunk of memory, described by AllocationPoint passed in.
     *
     * PLEASE NOTE: This method can actually ignore free, and keep released memory chunk for future reuse.
     *
     * @param point
     */
    @Override
    public void free(AllocationPoint point) {
        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            AllocationShape shape = point.getShape();
            long reqMemory = AllocationUtils.getRequiredMemory(shape);
            // we don't cache too big objects

            if (reqMemory > MAX_SINGLE_ALLOCATION || deviceCachedAmount.get() >= MAX_CACHED_MEMORY) {
                super.free(point);
                return;
            }

            ensureDeviceCacheHolder(point.getDeviceId(), shape);

            /*
                Now we should decide if this object can be cached or not
             */
            CacheHolder cache = deviceCache.get(point.getDeviceId()).get(shape);

            // memory chunks < threshold will be cached no matter what
            if (reqMemory <= FORCED_CACHE_THRESHOLD) {
                cache.put(new Pointer(point.getDevicePointer().address()));
            } else {
                long cacheEntries = cache.size();
                long cacheHeight = deviceCache.get(point.getDeviceId()).size();

                // total memory allocated within this bucket
                long cacheDepth = cacheEntries * reqMemory;

                if (cacheDepth < MAX_CACHED_MEMORY / cacheHeight) {
                    cache.put(new Pointer(point.getDevicePointer().address()));
                } else {
                    super.free(point);
                }
            }
        } else {
            AllocationShape shape = point.getShape();
            long reqMemory = AllocationUtils.getRequiredMemory(shape);

            // we don't cache too big objects
            if (reqMemory > MAX_SINGLE_ALLOCATION || zeroCachedAmount.get() >= MAX_CACHED_MEMORY) {
                super.free(point);
                return;
            }

            ensureCacheHolder(shape);

            /*
                Now we should decide if this object can be cached or not
             */
            CacheHolder cache = zeroCache.get(shape);

            // memory chunks < threshold will be cached no matter what
            if (reqMemory <= FORCED_CACHE_THRESHOLD) {
                cache.put(new Pointer(point.getHostPointer().address()));
            } else {
                long cacheEntries = cache.size();
                long cacheHeight = zeroCache.size();

                // total memory allocated within this bucket
                long cacheDepth = cacheEntries * reqMemory;

                if (cacheDepth < MAX_CACHED_MEMORY / cacheHeight) {
                    cache.put(new Pointer(point.getHostPointer().address()));
                } else {
                    super.free(point);
                }
            }
        }
    }

    private float getCacheHitRatio() {
        long totalHits = cacheHit.get() + cacheMiss.get();
        float cacheRatio = cacheHit.get() * 100 / (float) totalHits;
        return cacheRatio;
    }

    public void printCacheStats() {
        float cacheRatio = getCacheHitRatio();

        log.debug("Cached amount: " + zeroCachedAmount.get());
        log.debug("Total shapes in cache: " + zeroCache.size());
        log.debug("Current hit ratio: " + cacheRatio);
    }

    protected class CacheHolder {
        private Queue<Pointer> queue = new ConcurrentLinkedQueue<>();
        private AtomicInteger counter = new AtomicInteger(0);
        private long reqMem = 0;

        public CacheHolder(AllocationShape shape) {
            this.reqMem = AllocationUtils.getRequiredMemory(shape);
        }

        public int size() {
            return counter.get();
        }

        public Pointer poll() {
            Pointer pointer = queue.poll();
            if (pointer != null)
                counter.decrementAndGet();

            return pointer;
        }

        public void put(Pointer pointer) {
            zeroCachedAmount.addAndGet(reqMem);
            counter.incrementAndGet();
            queue.add(pointer);
        }
    }

    protected class CachePreallocator extends Thread implements Runnable {

        private AllocationShape shape;
        private AllocationStatus location;
        private int target;

        public CachePreallocator(AllocationShape shape, AllocationStatus location, int numberOfEntries) {
            this.shape = shape;
            this.target = numberOfEntries;
            this.location = location;
        }

        @Override
        public void run() {
//            log.info("Precaching ["+target+"] chunks for shape: " + shape);

            ensureCacheHolder(shape);

            for (int i = 0; i < target; i ++) {
                AllocationPoint point = new AllocationPoint();

                PointersPair pair = CudaCachingProvider.super.malloc(shape, point, this.location);
                if (this.location == AllocationStatus.HOST) {
                    Pointer pointer = new Pointer(pair.getHostPointer().address());
                    CudaCachingProvider.this.zeroCache.get(shape).put(pointer);
                }
            }
        }
    }
}
