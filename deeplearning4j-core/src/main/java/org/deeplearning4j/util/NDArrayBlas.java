package org.deeplearning4j.util;

import org.deeplearning4j.nn.linalg.NDArray;
import org.jblas.DoubleMatrix;
import org.jblas.NativeBlas;

/**
 * Meant to be an interpretation of SimpleBlas
 * for NDArrays.
 *
 * Based on the work by the jblas creators
 *
 * @author Adam Gibson
 */
public class NDArrayBlas {

    /**
     * Compute y <- alpha * x + y (elementwise addition)
     */
    public static DoubleMatrix axpy(double da, NDArray dx, NDArray dy) {
        NativeBlas.daxpy(dx.length, da, dx.data, 0, 1, dy.data, 0, 1);
        //JavaBlas.raxpy(dx.length, da, dx.data, 0, 1, dy.data, 0, 1);

        return dy;
    }

}
