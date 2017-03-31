package org.nd4j.linalg.jcublas;

import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCSR;

/**
 * @author Audrey Loeffel
 */
public class JcusparseNDArrayCSR extends BaseSparseNDArrayCSR {

    public JcusparseNDArrayCSR(double[] data, int[] columns, int[] pointerB, int[] pointerE, int[] shape) {
        super(data, columns, pointerB, pointerE, shape);
    }
}
