package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseShapeInfoProvider implements ShapeInfoProvider {
    /**
     * This method creates shapeInformation buffer, based on shape being passed in
     *
     * @param shape
     * @return
     */
    @Override
    public DataBuffer createShapeInformation(int[] shape) {
        char order = Nd4j.order();

        return createShapeInformation(shape, order);
    }

    /**
     * This method creates shapeInformation buffer, based on shape & order being passed in
     *
     * @param shape
     * @param order
     * @return
     */
    @Override
    public DataBuffer createShapeInformation(int[] shape, char order) {
        int[] stride = Nd4j.getStrides(shape, order);

        // this won't be view, so ews is 1
        int ews = 1;

        return createShapeInformation(shape, stride, 0, ews, order);
    }

    @Override
    public DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order) {
        DataBuffer buffer = Shape.createShapeInformation(shape, stride, offset, elementWiseStride, order);
        buffer.setConstant(true);
        return buffer;
    }
}
