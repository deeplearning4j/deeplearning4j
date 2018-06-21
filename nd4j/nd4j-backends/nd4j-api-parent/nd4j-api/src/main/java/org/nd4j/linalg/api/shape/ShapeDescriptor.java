package org.nd4j.linalg.api.shape;

import lombok.Getter;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
public class ShapeDescriptor {

    @Getter private char order;
    @Getter private long offset;
    @Getter private int ews;
    private long hashShape = 0;
    private long hashStride = 0;

    @Getter private int[] shape;
    @Getter private int[] stride;
    @Getter private long extras;

    public ShapeDescriptor(int[] shape, int[] stride, long offset, int ews, char order, long extras) {
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

        ShapeDescriptor that = (ShapeDescriptor) o;

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
        // FIXME: LONG
        result = 31 * result + Long.hashCode(offset);
        result = 31 * result + Long.hashCode(extras);
        result = 31 * result + ews;
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
}
