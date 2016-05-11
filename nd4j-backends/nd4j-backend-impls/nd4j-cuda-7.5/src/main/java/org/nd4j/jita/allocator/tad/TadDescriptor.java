package org.nd4j.jita.allocator.tad;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
public class TadDescriptor {
    private int dimensionLength;
    private int[] dimension;
    private int[] shape;

    public TadDescriptor(INDArray array, int[] dimension, int dimensionLength) {
        this.dimensionLength = dimensionLength;
        this.dimension = dimension;

        // TODO: change this to fill shapeInfo
        this.shape = array.shape();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TadDescriptor that = (TadDescriptor) o;

        if (dimensionLength != that.dimensionLength) return false;
        if (!Arrays.equals(dimension, that.dimension)) return false;
        return Arrays.equals(shape, that.shape);

    }

    @Override
    public int hashCode() {
        int result = dimensionLength;
        result = 31 * result + Arrays.hashCode(dimension);
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }
}
