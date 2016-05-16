package org.nd4j.linalg.jcublas;

import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.ShapeDescriptor;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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

    private Configuration configuration = CudaEnvironment.getInstance().getConfiguration();
    private List<Map<ShapeDescriptor, DataBuffer>> deviceCache = new ArrayList<>();

    public CachedShapeInfoProvider() {
        int numDevices =  configuration.getAvailableDevices().size();

        for (int i = 0; i < numDevices; i++ ) {
            deviceCache.add(i, new ConcurrentHashMap<ShapeDescriptor, DataBuffer>());
        }
    }

    @Override
    // FIXME: this method should NOT be synchronized
    public synchronized DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order) {
        offset = 0;

        Integer deviceId = allocator.getDeviceId();

        if (cacheHit.get() % 100000 == 0) {
            printCacheStats();
        }

        ShapeDescriptor descriptor = new ShapeDescriptor(shape, stride, offset, elementWiseStride, order);

        if (!deviceCache.get(deviceId).containsKey(descriptor)) {
//            logger.info("Cache miss");
            DataBuffer buffer = super.createShapeInformation(shape, stride, offset, elementWiseStride, order);

            AtomicAllocator.getInstance().moveToConstant(buffer);

            deviceCache.get(deviceId).put(descriptor, buffer);

            cacheMiss.incrementAndGet();
            return buffer;
        } else {
            //logger.info("Cache hit");
            cacheHit.incrementAndGet();
        }

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
