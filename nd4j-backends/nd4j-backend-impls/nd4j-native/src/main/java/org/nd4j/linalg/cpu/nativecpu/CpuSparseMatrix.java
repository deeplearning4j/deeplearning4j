package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.api.ndarray.ISparseMatrix;

/**
 * @author Audrey Loeffel
 */
public class CpuSparseMatrix extends ISparseMatrix {

    public CpuSparseMatrix(double[] data, int[] columns, int[] pointerB, int[] pointerE, int[] shape) {
        super(data, columns, pointerB, pointerE, shape);
    }
}
