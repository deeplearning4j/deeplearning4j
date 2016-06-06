package org.nd4j.linalg.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * This is utility class, made to compare TADs for caching purposes.
 *
 * Idea: for any given INDArray with any specific shape, TAD for specific dimension will always be the same. So it can be reused as much as we want.
 *
 * @author raver119@gmail.com
 */
public class TadDescriptor {
    private static Logger logger = LoggerFactory.getLogger(TadDescriptor.class);
    private int dimensionLength;
    private int[] dimension;
    private int[] shape;

    public TadDescriptor(INDArray array, int[] dimension) {
        this.dimensionLength = dimension == null ? 0 : dimension.length;
        this.dimension = dimension;

        // TODO: change this to fill shapeInfo
        this.shape = dataBufferToArray(array.shapeInfoDataBuffer());
    }


    public static int[] dataBufferToArray(DataBuffer buffer) {
        int rank = buffer.getInt(0);
        int ret[] = new int[rank * 2 + 4];
        ret[0] = rank;
        for (int e = 1; e < rank * 2 + 4; e++) {
            ret[e] = buffer.getInt(e);
        }

        return ret;
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
