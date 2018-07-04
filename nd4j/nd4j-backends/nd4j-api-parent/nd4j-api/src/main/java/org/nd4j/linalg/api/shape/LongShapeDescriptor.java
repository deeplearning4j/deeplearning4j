package org.nd4j.linalg.api.shape;

import lombok.NonNull;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
public class LongShapeDescriptor {

    private char order;
    private long offset;
    private long ews;
    private long hashShape = 0;
    private long hashStride = 0;

    private long[] shape;
    private long[] stride;

    private long extras;

    public LongShapeDescriptor(long[] shape, long[] stride, long offset, long ews, char order, long extras) {
        /*
        if (shape != null) {
            hashShape = shape[0];
            for (int i = 1; i < shape.length; i++)
                hashShape = 31 * hashShape + shape[i];
        }
        
        if (stride != null) {
            hashStride = stride[0];
            for (int i = 1; i < stride.length; i++)
                hashStride = 31 * hashStride + stride[i];
        }
        */
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = Arrays.copyOf(stride, stride.length);

        this.offset = offset;
        this.ews = ews;
        this.order = order;

        this.extras = extras;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        LongShapeDescriptor that = (LongShapeDescriptor) o;

        if (extras != that.extras)
            return false;
        if (order != that.order)
            return false;
        if (offset != that.offset)
            return false;
        if (ews != that.ews)
            return false;
        if (!Arrays.equals(shape, that.shape))
            return false;
        return Arrays.equals(stride, that.stride);

    }

    @Override
    public int hashCode() {
        int result = (int) order;

        result = 31 * result + longHashCode(offset);
        result = 31 * result + longHashCode(ews);
        result = 31 * result + longHashCode(extras);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(stride);
        return result;
    }

    @Override
    public String toString() {

        StringBuilder builder = new StringBuilder();

        builder.append(shape.length).append(",").append(Arrays.toString(shape)).append(",")
                        .append(Arrays.toString(stride)).append(",").append(offset).append(",").append(ews).append(",")
                        .append(order);

        String result = builder.toString().replaceAll("\\]", "").replaceAll("\\[", "");
        result = "[" + result + "]";

        return result;
    }

    private int longHashCode(long v) {
        // impl from j8
        return (int)(v ^ (v >>> 32));
    }

    public static LongShapeDescriptor fromShapeDescriptor(@NonNull ShapeDescriptor descriptor) {
        return new LongShapeDescriptor(ArrayUtil.toLongArray(descriptor.getShape()), ArrayUtil.toLongArray(descriptor.getStride()), descriptor.getOffset(), descriptor.getEws(), descriptor.getOrder(), descriptor.getExtras());
    }
}
