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

    public SparseNDArrayCOO(DataBuffer values, DataBuffer indices, int[] sparseOffset, int[] fixed, int[] shape, char ordering){
        super(values, indices, sparseOffset, fixed, shape, ordering);
    }

}
