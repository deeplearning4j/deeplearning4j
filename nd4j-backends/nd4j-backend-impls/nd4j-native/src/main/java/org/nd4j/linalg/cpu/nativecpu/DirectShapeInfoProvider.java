package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.shape.ShapeDescriptor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
public class DirectShapeInfoProvider extends BaseShapeInfoProvider {
    private Map<ShapeDescriptor, DataBuffer> shapeCache = new ConcurrentHashMap<>();

    @Override
    public DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order) {

        ShapeDescriptor descriptor = new ShapeDescriptor(shape, stride, offset, elementWiseStride, order);
        if (!shapeCache.containsKey(descriptor)) {
            synchronized (this) {
                if (!shapeCache.containsKey(descriptor)) {
                    DataBuffer buffer = super.createShapeInformation(shape, stride, offset, elementWiseStride, order);
                    shapeCache.put(descriptor, buffer);

                    return buffer;
                } else return shapeCache.get(descriptor);
            }
        }

        return shapeCache.get(descriptor);
    }
}
