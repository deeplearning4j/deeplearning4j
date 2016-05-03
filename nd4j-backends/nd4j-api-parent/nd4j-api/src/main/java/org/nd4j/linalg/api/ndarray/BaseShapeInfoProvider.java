package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.Shape;

/**
 * @author raver119@gmail.com
 */
public class BaseShapeInfoProvider implements ShapeInfoProvider {
    @Override
    public DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order) {
        return Shape.createShapeInformation(shape, stride, offset, elementWiseStride, order);
    }
}
