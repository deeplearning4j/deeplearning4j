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


    /***************************************************************************
     * BLAS Level 3
     */

    /**
     * Compute c <- a*b + beta * c (general matrix matrix
     * multiplication)
     */
    public static DoubleMatrix gemm(double alpha, DoubleMatrix a,
                                    DoubleMatrix b, double beta, DoubleMatrix c) {
        NativeBlas.dgemm('V', 'V', c.rows, c.columns, a.columns, alpha, a.data, 0,
                a.rows, b.data, 0, b.rows, beta, c.data, 0, c.rows);
        return c;
    }


    public static ComplexNDArray gemm(ComplexDouble alpha,
                                      ComplexNDArray a,
                                      ComplexNDArray b,
                                      ComplexDouble beta,
                                      ComplexNDArray c) {

        int aOffset = a.offset();
        int bOffset = b.offset();
        int cOffset = c.offset();
        double[] aData = a.data;
        double[] bData = b.data;
        double[] cData = c.data;


        NativeBlas.zgemm(
                'N', 'N',
                c.rows(), c.columns(),
                a.columns(), alpha, aData, aOffset, a.rows(),
                bData, bOffset, b.rows(), beta,
                cData, cOffset, c.rows());

        return c;
    }


    /**
     * Copy data from x to y
     * @param x
     * @param y
     * @return
     */
    public static NDArray copy(NDArray x, NDArray y) {
        JavaBlas.rcopy(x.length, x.data, 0, 1, y.data, 0, 1);
        return y;
    }

    public static ComplexNDArray copy(ComplexNDArray x, ComplexNDArray y) {
        NativeBlas.zcopy(x.length, x.data, x.offset(), 1, y.data, y.offset(), 1);
        return y;
    }


    /***************************************************************************
     * BLAS Level 2
     */

    /**
     * Compute y <- alpha*op(a)*x + beta * y (general matrix vector
     * multiplication)
     */
    public static NDArray gemv(double alpha, NDArray a,
                               NDArray x, double beta, NDArray y) {
        if (false) {
            NativeBlas.dgemv('N', a.rows, a.columns, alpha, a.data, a.offset(), a.rows, x.data, 0,
                    1, beta, y.data, y.offset(), 1);
        } else {
            if (beta == 0.0) {
                for (int i = 0; i < y.length; i++)
                    y.data[i] = 0.0;

                for (int j = 0; j < a.columns; j++) {
                    double xj = x.get(j);
                    if (xj != 0.0) {
                        for (int i = 0; i < a.rows; i++)
                            y.data[i] += a.get(i, j) * xj;
                    }
                }
            } else {
                for (int j = 0; j < a.columns; j++) {
                    double byj = beta * y.data[j];
                    double xj = x.get(j);
                    for (int i = 0; i < a.rows; i++)
                        y.data[j] = a.get(i, j) * xj + byj;
                }
            }
        }
        return y;
    }

}
