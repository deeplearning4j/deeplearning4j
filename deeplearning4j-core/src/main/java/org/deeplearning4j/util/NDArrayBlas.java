package org.deeplearning4j.util;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.jblas.*;

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
    public static NDArray axpy(double da, NDArray dx, NDArray dy) {
        NativeBlas.daxpy(dx.length, da, dx.data, dx.offset(), 1, dy.data, dy.offset(), 1);
        //JavaBlas.raxpy(dx.length, da, dx.data, 0, 1, dy.data, 0, 1);

        return dy;
    }

    /**
     * Compute c <- a*b + beta * c (general matrix matrix
     * multiplication)
     */
    public static NDArray gemm(double alpha, NDArray a,
                                    NDArray b, double beta, NDArray c) {
        NativeBlas.dgemm('N', 'N', c.rows(), c.columns(), a.columns(), alpha, a.data, a.offset(),
                a.rows(), b.data, b.offset(), b.rows(), beta, c.data, c.offset(), c.rows());
        return c;
    }



    public static ComplexNDArray gemm(ComplexDouble alpha,
                                      ComplexNDArray a,
                                      ComplexNDArray b,
                                      ComplexDouble beta,
                                      ComplexNDArray c) {

        NativeBlas.zgemm('N', 'N', c.rows(), c.columns(), a.columns(), alpha, a.data, a.offset(),
                a.rows(), b.data, b.offset(), b.rows(), beta, c.data, c.offset(), c.rows);
        return c;
    }


    public static ComplexNDArray copy(ComplexNDArray x, ComplexNDArray y) {
        NativeBlas.zcopy(x.length, x.data, x.offset(), 1, y.data, y.offset(), 1);
        return y;
    }




}
