package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.Shape;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseShapeInfoProvider implements ShapeInfoProvider {
    @Override
    public DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order) {
        DataBuffer buffer = Shape.createShapeInformation(shape, stride, offset, elementWiseStride, order);
        buffer.setConstant(true);
        return buffer;
    }
}
