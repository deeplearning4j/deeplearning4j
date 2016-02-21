package org.nd4j.nativeblas;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Native bindings for level 1
 * @author Adam Gibson
 */
@Platform(include="NativeLevel1.h",link = "libnd4j")
public class NativeLevel1 extends Pointer {

    /**
     * computes a vector-vector dot product.
     * @param n
     * @param alpha
     * @param X
     * @param Y
     * @return
     */
    public native double dot(long[] extraPointers,int n,double alpha,  long X,long Y);

    /**
     * computes a vector-vector dot product.
     * @param n
     * @param alpha
     * @param X
     * @param Y
     * @return
     */
    public native  IComplexNumber dot(long[] extraPointers,int n,IComplexNumber alpha,  long X,long Y);

    /**
     * computes the Euclidean norm of a vector.
     * @param arr
     * @return
     */
    public native  double nrm2(long[] extraPointers,long arr);
    /**
     * computes the Euclidean norm of a vector.
     * @param arr
     * @return
     */
    public native IComplexNumber nrm2Complex(long[] extraPointers,long arr);

    /**
     * computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
     * @param arr
     * @return
     */
    public native  double asumComplex(long[] extraPointers,long arr);


    /**
     * computes the sum of magnitudes
     * of all vector elements or, for a complex vector x, the sum
     * @param arr
     * @return
     */
    public native  IComplexNumber asum(long[] extraPointers,long arr);

    /**
     * finds the element of a
     * vector that has the largest absolute value.
     * @param arr
     * @return
     */
    public native  int iamax(long[] extraPointers,long arr);

    /**
     * finds the element of a
     * vector that has the largest absolute value.
     * @param n the length to iterate for
     * @param arr the array to get the max
     *            index for
     * @param stride  the stride for the array
     * @return
     */
    public native  int iamax(long[] extraPointers,int n,long arr,int stride);

    /** Index of largest absolute value */
    public native   int iamax(long[] extraPointers,int n,DataBuffer x, int offsetX, int incrX);
    /**
     * finds the element of a vector that has the largest absolute value.
     * @param arr
     * @return
     */
    public native   int iamaxComplex(long[] extraPointers,long arr);

    /**
     * finds the element of a vector that has the minimum absolute value.
     * @param arr
     * @return
     */
    public native  int iamin(long[] extraPointers,long arr);
    /**
     * finds the element of a vector that has the minimum absolute value.
     * @param arr
     * @return
     */
    public native   int iaminComplex(long[] extraPointers,long arr);

    /**
     * swaps a vector with another vector.
     * @param x
     * @param y
     */
    public native void  swap(long[] extraPointers,long x,long y);

    /**
     *
     * @param extraPointers
     * @param x
     * @param y
     */
    public native void  swapComplex(long[] extraPointers,long x, long y);

    /**
     * copy a vector to another vector.
     * @param x
     * @param y
     */
    public native void  copy(long[] extraPointers,long x,long y);

    /**copy a vector to another vector.
     * @param x
     * @param y
     */
    public native void  copyComplex(long[] extraPointers,long x, long y);

    /**
     *  computes a vector-scalar product and adds the result to a vector.
     * @param n
     * @param alpha
     * @param x
     * @param y
     */
    public native void  axpy(long[] extraPointers,int n,double alpha,long x,long y);

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
    public native void  axpy(long[] extraPointers,int n,double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY );

    /**
     *  computes a vector-scalar product and adds the result to a vector.
     * @param n
     * @param alpha
     * @param x
     * @param y
     */
    public native void  axpy(long[] extraPointers,int n,IComplexNumber alpha,long x,long y);

    /**
     * computes parameters for a Givens rotation.
     * @param a
     * @param b
     * @param c
     * @param s
     */
    public native void  rotg(long[] extraPointers,long a,long b,long c,long s);

    /**
     * performs rotation of points in the plane.
     * @param N
     * @param X
     * @param Y
     * @param c
     * @param s
     */
    public native void  rot(long[] extraPointers, int N, long X,
              long Y,  double c,  double s);

    /**
     * performs rotation of points in the plane.
     * @param N
     * @param X
     * @param Y
     * @param c
     * @param s
     */
    public native void  rot(long[] extraPointers, int N, long X,
              long Y,  IComplexNumber c,  IComplexNumber s);

    /**
     * computes the modified parameters for a Givens rotation.
     * @param d1
     * @param d2
     * @param b1
     * @param b2
     * @param P
     */
    public native void  rotmg(long[] extraPointers,long d1, long d2, long b1,  double b2, long P);
    /**
     * computes the modified parameters for a Givens rotation.
     * @param d1
     * @param d2
     * @param b1
     * @param b2
     * @param P
     */
    public native void  rotmg(long[] extraPointers,long d1, long d2, long b1,  IComplexNumber b2, long P);

    /**
     *  computes a vector by a scalar product.
     * @param N
     * @param alpha
     * @param X
     */
    public native void  scal(long[] extraPointers,int N, double alpha, long X);

    /**
     *  computes a vector by a scalar product.
     * @param N
     * @param alpha
     * @param X
     */
    public native void  scal(long[] extraPointers,int N,  IComplexNumber alpha, long X);


    /** Can we use the axpy and copy methods that take a DataBuffer instead of an long with this backend? */
    public native  boolean supportsDataBufferL1Ops(long[] extraPointers,);
}
