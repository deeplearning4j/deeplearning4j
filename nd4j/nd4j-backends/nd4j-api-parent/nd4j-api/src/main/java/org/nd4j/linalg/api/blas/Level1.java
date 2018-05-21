package org.nd4j.linalg.api.blas;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * Level 1 blas implementations.
 * Incx and other parameters are inferred
 * from the given ndarrays.
 *
 * To avoid boxing, doubles are used in place of normal numbers.
 * The underlying implementation will call the proper data opType.
 *
 * This is a fortran 95 style api that gives us the efficiency
 * and flexibility of the fortran 77 api
 *
 * Credit to:
 * https://www.ualberta.ca/AICT/RESEARCH/LinuxClusters/doc/mkl81/mklqref/blaslev1.htm
 *
 * for the descriptions
 *
 * @author Adam Gibson
 */
public interface Level1 {
    /**
     * computes a vector-vector dot product.
     * @param n
     * @param alpha
     * @param X
     * @param Y
     * @return
     */
    double dot(long N, double alpha, INDArray X, INDArray Y);

    /** Vector-vector dot product */
    double dot(long N, DataBuffer dx, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY);

    /**
     * computes a vector-vector dot product.
     * @param n
     * @param alpha
     * @param X
     * @param Y
     * @return
     */
    IComplexNumber dot(long N, IComplexNumber alpha, IComplexNDArray X, IComplexNDArray Y);

    /**
     * computes the Euclidean norm of a vector.
     * @param arr
     * @return
     */
    double nrm2(INDArray arr);

    /**
     * computes the Euclidean norm of a vector.
     * @param arr
     * @return
     */
    IComplexNumber nrm2(IComplexNDArray arr);

    /**
     * computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
     * @param arr
     * @return
     */
    double asum(INDArray arr);

    /** sum of magnitudes of all elements */
    double asum(long N, DataBuffer x, int offsetX, int incrX);

    /**
     * computes the sum of magnitudes
     * of all vector elements or, for a complex vector x, the sum
     * @param arr
     * @return
     */
    IComplexNumber asum(IComplexNDArray arr);

    /**
     * finds the element of a
     * vector that has the largest absolute value.
     * @param arr
     * @return
     */
    int iamax(INDArray arr);

    /**
     * finds the element of a
     * vector that has the largest absolute value.
     * @param n the length to iterate for
     * @param arr the array to get the max
     *            index for
     * @param stride  the stride for the array
     * @return
     */
    int iamax(long N, INDArray arr, int stride);

    /** Index of largest absolute value */
    int iamax(long N, DataBuffer x, int offsetX, int incrX);

    /**
     * finds the element of a vector that has the largest absolute value.
     * @param arr
     * @return
     */
    int iamax(IComplexNDArray arr);

    /**
     * finds the element of a vector that has the minimum absolute value.
     * @param arr
     * @return
     */
    int iamin(INDArray arr);

    /**
     * finds the element of a vector that has the minimum absolute value.
     * @param arr
     * @return
     */
    int iamin(IComplexNDArray arr);

    /**
     * swaps a vector with another vector.
     * @param x
     * @param y
     */
    void swap(INDArray x, INDArray y);

    void swap(IComplexNDArray x, IComplexNDArray y);

    /**
     * copy a vector to another vector.
     * @param x
     * @param y
     */
    void copy(INDArray x, INDArray y);

    /**copy a vector to another vector.
     * @param x
     * @param y
     */
    void copy(long N, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY);

    void copy(IComplexNDArray x, IComplexNDArray y);

    /**
     *  computes a vector-scalar product and adds the result to a vector.
     * @param n
     * @param alpha
     * @param x
     * @param y
     */
    void axpy(long N, double alpha, INDArray x, INDArray y);

    /**
     * computes a vector-scalar product and adds the result to a vector.
     * y = a*x + y
     * @param n number of operations
     * @param alpha
     * @param x X
     * @param offsetX offset of first element of X in buffer
     * @param incrX increment/stride between elements of X in buffer
     * @param y Y
     * @param offsetY offset of first element of Y in buffer
     * @param incrY increment/stride between elements of Y in buffer
     */
    void axpy(long N, double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY);

    /**
     *  computes a vector-scalar product and adds the result to a vector.
     * @param n
     * @param alpha
     * @param x
     * @param y
     */
    void axpy(long N, IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y);

    /**
     * computes parameters for a Givens rotation.
     * @param a
     * @param b
     * @param c
     * @param s
     */
    void rotg(INDArray a, INDArray b, INDArray c, INDArray s);

    /**
     * performs rotation of points in the plane.
     * @param N
     * @param X
     * @param Y
     * @param c
     * @param s
     */
    void rot(long N, INDArray X, INDArray Y, double c, double s);

    /**
     * performs rotation of points in the plane.
     * @param N
     * @param X
     * @param Y
     * @param c
     * @param s
     */
    void rot(long N, IComplexNDArray X, IComplexNDArray Y, IComplexNumber c, IComplexNumber s);

    /**
     * computes the modified parameters for a Givens rotation.
     * @param d1
     * @param d2
     * @param b1
     * @param b2
     * @param P
     */
    void rotmg(INDArray d1, INDArray d2, INDArray b1, double b2, INDArray P);

    /**
     * computes the modified parameters for a Givens rotation.
     * @param d1
     * @param d2
     * @param b1
     * @param b2
     * @param P
     */
    void rotmg(IComplexNDArray d1, IComplexNDArray d2, IComplexNDArray b1, IComplexNumber b2, IComplexNDArray P);

    /**
     *  computes a vector by a scalar product.
     * @param N
     * @param alpha
     * @param X
     */
    void scal(long N, double alpha, INDArray X);

    /**
     *  computes a vector by a scalar product.
     * @param N
     * @param alpha
     * @param X
     */
    void scal(long N, IComplexNumber alpha, IComplexNDArray X);


    /** Can we use the axpy and copy methods that take a DataBuffer instead of an INDArray with this backend? */
    boolean supportsDataBufferL1Ops();
}
