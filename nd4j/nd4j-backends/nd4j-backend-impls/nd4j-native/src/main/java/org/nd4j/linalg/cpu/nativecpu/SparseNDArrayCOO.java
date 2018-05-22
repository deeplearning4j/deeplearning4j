package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCOO;

/**
 * @author Audrey Loeffel
 */
public class SparseNDArrayCOO extends BaseSparseNDArrayCOO {
    public SparseNDArrayCOO(double[] values, int[][] indices, int[] shape){
        super(values, indices, shape);
    }

    public SparseNDArrayCOO(DataBuffer values, DataBuffer indices, int[] shape){
        super(values, indices, shape);
    }

    public SparseNDArrayCOO(float[] values, int[][] indices, int[] shape) {
        super(values, indices, shape);
    }

    public SparseNDArrayCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags, int[] hiddenDimensions, int underlyingRank, int[] shape){
        super(values, indices, sparseOffsets, flags, hiddenDimensions, underlyingRank, shape);
    }

    public SparseNDArrayCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation, int[] shape) {
        super(values, indices, sparseInformation, shape);
    }
}
