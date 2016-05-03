package org.nd4j.linalg.jcublas;

import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.shape.ShapeDescriptor;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class CachedShapeInfoProvider extends BaseShapeInfoProvider {
    private static Logger logger = LoggerFactory.getLogger(CachedShapeInfoProvider.class);

    private AtomicAllocator allocator = AtomicAllocator.getInstance();

    private AtomicLong cacheHit = new AtomicLong(1);
    private AtomicLong cacheMiss = new AtomicLong(1);

    private Semaphore lock = new Semaphore(1);

    private Map<Integer, Map<ShapeDescriptor, DataBuffer>> deviceCache = new HashMap<>();

    @Override
    public DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order) {
    //    logger.info("CachedShapeInfo request");

        Integer deviceId = allocator.getDeviceId();
        if (!deviceCache.containsKey(deviceId)) {
            try {
                lock.acquire();

                if (!deviceCache.containsKey(deviceId))
                    deviceCache.put(deviceId,  new ConcurrentHashMap<ShapeDescriptor, DataBuffer>());
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                lock.release();
            }
        }

        if (cacheMiss.get() % 100 == 0) {
            printCacheStats();
        }

        ShapeDescriptor descriptor = new ShapeDescriptor(shape, stride, offset, elementWiseStride, order);

        if (!deviceCache.get(deviceId).containsKey(descriptor)) {
            DataBuffer buffer = super.createShapeInformation(shape, stride, offset, elementWiseStride, order);
            deviceCache.get(deviceId).put(descriptor, buffer);
            cacheMiss.incrementAndGet();
            return buffer;
        } else cacheHit.incrementAndGet();

        return deviceCache.get(deviceId).get(descriptor);
    }

    private float getDeviceCacheHitRatio() {
        long totalHits = cacheHit.get() + cacheMiss.get();
        float cacheRatio = cacheHit.get() * 100 / (float) totalHits;
        return cacheRatio;
    }

    public void printCacheStats() {
        logger.debug("Total shapeInfo buffers in cache: " + deviceCache.get(0).size());
        logger.debug("Current shapeInfo hit ratio: " + getDeviceCacheHitRatio());
    }
}
