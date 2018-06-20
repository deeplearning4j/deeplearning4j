package org.nd4j.linalg.api.ndarray;

import lombok.val;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseShapeInfoProvider implements ShapeInfoProvider {
    protected AtomicLong bytes = new AtomicLong(0);

    /**
     * This method creates shapeInformation buffer, based on shape being passed in
     *
     * @param shape
     * @return
     */
    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(int[] shape) {
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
    public Pair<DataBuffer, long[]> createShapeInformation(int[] shape, char order) {
        int[] stride = Nd4j.getStrides(shape, order);

        // this won't be view, so ews is 1
        int ews = 1;

        return createShapeInformation(shape, stride, 0, ews, order);
    }

    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(int[] shape, int[] stride, long offset, int elementWiseStride,
                    char order) {
        DataBuffer buffer = Shape.createShapeInformation(ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride), offset, (long) elementWiseStride, order);
        buffer.setConstant(true);
        return Pair.create(buffer, buffer.asLong());
    }

    /**
     * This method creates shapeInformation buffer, based on detailed shape information being passed in
     *
     * @param shape
     * @param stride
     * @param offset
     * @param elementWiseStride
     * @param order
     * @param extras
     * @return
     */
    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(int[] shape, int[] stride, long offset, int elementWiseStride, char order, long extras) {
        val result = createShapeInformation(shape, stride, offset, elementWiseStride, order);

        val jvm = result.getSecond();

        result.getFirst().put(jvm.length - 3, extras);
        result.getSecond()[jvm.length - 1] = extras;

        return result;
    }

    /**
     * This method creates shapeInformation buffer, based on shape being passed in
     *
     * @param shape
     * @return
     */
    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape) {
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
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, char order) {
        long[] stride = Nd4j.getStrides(shape, order);

        // this won't be view, so ews is 1
        int ews = 1;

        return createShapeInformation(shape, stride, 0, ews, order);
    }

    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long offset, long elementWiseStride, char order) {
        DataBuffer buffer = Shape.createShapeInformation(shape, stride, offset, elementWiseStride, order);
        buffer.setConstant(true);
        return Pair.create(buffer, buffer.asLong());
    }


    @Override
    public long getCachedBytes() {
        return bytes.get();
    }
}
