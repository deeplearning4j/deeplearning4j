package org.nd4j.linalg.cpu.nativecpu;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.shape.ShapeDescriptor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class DirectShapeInfoProvider extends BaseShapeInfoProvider {
    private Map<ShapeDescriptor, Pair<DataBuffer, int[]>> shapeCache = new ConcurrentHashMap<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private static final int MAX_ENTRIES = 1000;

    @Override
    public Pair<DataBuffer, int[]> createShapeInformation(int[] shape, int[] stride, long offset, int elementWiseStride, char order) {

        // We enforce offset to 0 in shapeBuffer, since we need it for cache efficiency + we don't actually use offset value @ native side
        offset = 0;

        ShapeDescriptor descriptor = new ShapeDescriptor(shape, stride, offset, elementWiseStride, order);
        if (!shapeCache.containsKey(descriptor)) {
            if (counter.get() < MAX_ENTRIES) {
                synchronized (this) {
                    if (!shapeCache.containsKey(descriptor)) {
                        counter.incrementAndGet();
                        Pair<DataBuffer, int[]> buffer =
                                        super.createShapeInformation(shape, stride, offset, elementWiseStride, order);
                        shapeCache.put(descriptor, buffer);

                        bytes.addAndGet(buffer.getFirst().length() * 4 * 2);

                        return buffer;
                    } else
                        return shapeCache.get(descriptor);
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
