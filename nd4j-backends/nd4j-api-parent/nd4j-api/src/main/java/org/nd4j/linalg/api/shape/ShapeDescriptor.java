package org.nd4j.linalg.api.shape;

/**
 * @author raver119@gmail.com
 */
public class ShapeDescriptor {

    private char order;
    private int offset;
    private int ews;
    private long hashShape = 0;
    private long hashStride = 0;

    public ShapeDescriptor(int[] shape, int[] stride, int offset, int ews, char order) {
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

        this.offset = offset;
        this.ews = ews;
        this.order = order;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ShapeDescriptor that = (ShapeDescriptor) o;

        if (order != that.order) return false;
        if (offset != that.offset) return false;
        if (ews != that.ews) return false;
        if (hashShape != that.hashShape) return false;
        return hashStride == that.hashStride;

    }

    @Override
    public int hashCode() {
        int result = (int) order;
        result = 31 * result + offset;
        result = 31 * result + ews;
        result = 31 * result + (int) (hashShape ^ (hashShape >>> 32));
        result = 31 * result + (int) (hashStride ^ (hashStride >>> 32));
        return result;
    }
}
