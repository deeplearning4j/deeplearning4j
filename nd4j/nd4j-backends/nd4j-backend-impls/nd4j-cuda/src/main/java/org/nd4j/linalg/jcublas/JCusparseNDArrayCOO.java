package org.nd4j.linalg.jcublas;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCOO;

/**
 * @author Audrey Loeffel
 */
public class JCusparseNDArrayCOO extends BaseSparseNDArrayCOO {

    public JCusparseNDArrayCOO(double[] values, int[][] indices, long[] shape) {
        super(values, indices, shape);
    }

    public JCusparseNDArrayCOO(DataBuffer values, DataBuffer indices, long[] shape) {
        super(values, indices, shape);
    }

    public JCusparseNDArrayCOO(float[] values, int[][] indices, long[] shape) {
        super(values, indices, shape);
    }

    public JCusparseNDArrayCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation, long[] shape) {
        super(values, indices, sparseInformation, shape);
    }

    public JCusparseNDArrayCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags, int[] hiddenDimensions, int underlyingRank, long[] shape) {
        super(values, indices, sparseOffsets, flags, hiddenDimensions, underlyingRank, shape);
    }
}
