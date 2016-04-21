package org.nd4j.jita.memory.impl;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This MemoryProvider implementation does caching for both host and device memory within predefined limits.
 *
 * @author raver119@gmail.com
 */
public class CudaFullCachingProvider extends CudaCachingZeroProvider {

    protected final long MAX_GPU_ALLOCATION = 10000000;

    protected final AtomicLong deviceCachedAmount = new AtomicLong(0);

    protected volatile ConcurrentHashMap<Integer, ConcurrentHashMap<AllocationShape, CacheHolder>> deviceCache = new ConcurrentHashMap<>();

    private static Logger log = LoggerFactory.getLogger(CudaFullCachingProvider.class);

    @Override
    public PointersPair malloc(AllocationShape shape, AllocationPoint point, AllocationStatus location) {
        long reqMemory = AllocationUtils.getRequiredMemory(shape);
        if (location == AllocationStatus.DEVICE && reqMemory < MAX_GPU_ALLOCATION) {
            ensureDeviceCacheHolder(point.getDeviceId(), shape);

            CacheHolder cache = deviceCache.get(point.getDeviceId()).get(shape);
            if (cache != null) {
                Pointer pointer = cache.poll();
                if (pointer != null) {
                    cacheHit.incrementAndGet();

                    deviceCachedAmount.addAndGet(-1 * reqMemory);

                    PointersPair pair = new PointersPair();
                    pair.setDevicePointer(pointer);

                    point.setAllocationStatus(AllocationStatus.DEVICE);
                    return pair;
                }
            }
            cacheMiss.incrementAndGet();
            return super.malloc(shape, point, location);
        }
        return super.malloc(shape, point, location);
    }

    @Override
    public void free(AllocationPoint point) {
        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            AllocationShape shape = point.getShape();
            long reqMemory = AllocationUtils.getRequiredMemory(shape);
            // we don't cache too big objects

            if (reqMemory > MAX_GPU_ALLOCATION || deviceCachedAmount.get() >= MAX_CACHED_MEMORY) {
                super.free(point);
                return;
            }

            ensureDeviceCacheHolder(point.getDeviceId(), shape);


            CacheHolder cache = deviceCache.get(point.getDeviceId()).get(shape);

            // memory chunks < threshold will be cached no matter what
            if (reqMemory <= FORCED_CACHE_THRESHOLD) {
                cache.put(new CudaPointer(point.getDevicePointer().address()));
                return;
            } else {
                long cacheEntries = cache.size();
                long cacheHeight = deviceCache.get(point.getDeviceId()).size();

                // total memory allocated within this bucket
                long cacheDepth = cacheEntries * reqMemory;

                if (cacheDepth < MAX_CACHED_MEMORY / cacheHeight) {
                    cache.put(new CudaPointer(point.getDevicePointer().address()));
                    return;
                } else {
                    super.free(point);
                }
            }
        }
        super.free(point);
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
}
