package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.shape.ShapeDescriptor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class DirectShapeInfoProvider extends BaseShapeInfoProvider {
    private Map<ShapeDescriptor, DataBuffer> shapeCache = new ConcurrentHashMap<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private static final int MAX_ENTRIES = 100;

    @Override
    public DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order) {

        ShapeDescriptor descriptor = new ShapeDescriptor(shape, stride, offset, elementWiseStride, order);
        if (!shapeCache.containsKey(descriptor)) {
            if (counter.get() < MAX_ENTRIES) {
                synchronized (this) {
                    if (!shapeCache.containsKey(descriptor)) {
                        counter.incrementAndGet();
                        DataBuffer buffer = super.createShapeInformation(shape, stride, offset, elementWiseStride, order);
                        shapeCache.put(descriptor, buffer);

                        return buffer;
                    } else return shapeCache.get(descriptor);
                }
            } else {
                return super.createShapeInformation(shape, stride, offset, elementWiseStride, order);
            }
        }

        return shapeCache.get(descriptor);
    }

    @Override
    public void purgeCache() {
        shapeCache = new ConcurrentHashMap<>();
    }
}
