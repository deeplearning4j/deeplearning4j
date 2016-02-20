package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Base class for level 1 functions, abstract headers pulled from:
 * http://www.netlib.org/blas/blast-forum/cblas.h
 *
 * @author Adam Gibson
 */
public abstract  class BaseLevel1 extends BaseLevel implements Level1 {
    /**
     * computes a vector-vector dot product.
     *
     * @param n
     * @param alpha
     * @param X
     * @param Y
     * @return
     */
    @Override
    public double dot(int n, double alpha, INDArray X, INDArray Y) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            return ddot(n,X,BlasBufferUtil.getBlasStride(X),Y,BlasBufferUtil.getBlasStride(X));
        return sdot(n,X,BlasBufferUtil.getBlasStride(X),Y,BlasBufferUtil.getBlasStride(X));
    }

    @Override
    public double dot(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        if(supportsDataBufferL1Ops()){
            if(x.dataType() == DataBuffer.Type.FLOAT){
                return sdot(n,x,offsetX,incrX,y,offsetY,incrY);
            } else {
                return ddot(n, x, offsetX, incrX, y, offsetY, incrY);
            }
        } else {
            int[] shapex = {1,n};
            int[] shapey = {1,n};
            int[] stridex = {incrX,incrX};
            int[] stridey = {incrY,incrY};
            INDArray arrX = Nd4j.create(x,shapex,stridex,offsetX,'c');
            INDArray arrY = Nd4j.create(x,shapey,stridey,offsetY,'c');
            return dot(n,0.0,arrX,arrY);
        }
    }

    /**
     * computes a vector-vector dot product.
     *
     * @param n
     * @param alpha
     * @param X
     * @param Y
     * @return
     */
    @Override
    public IComplexNumber dot(int n, IComplexNumber alpha, IComplexNDArray X, IComplexNDArray Y) {
        throw new UnsupportedOperationException();
    }

    /**
     * computes the Euclidean norm of a vector.
     *
     * @param arr
     * @return
     */
    @Override
    public double nrm2(INDArray arr) {
        if(arr.data().dataType() == DataBuffer.Type.DOUBLE)
            return dnrm2(arr.length(),arr,BlasBufferUtil.getBlasStride(arr));
        return snrm2(arr.length(),arr,BlasBufferUtil.getBlasStride(arr));
    }

    /**
     * computes the Euclidean norm of a vector.
     *
     * @param arr
     * @return
     */
    @Override
    public IComplexNumber nrm2(IComplexNDArray arr) {
        throw new UnsupportedOperationException();
    }

    /**
     * computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
     *
     * @param arr
     * @return
     */
    @Override
    public double asum(INDArray arr) {
        if(arr.data().dataType() == DataBuffer.Type.DOUBLE)
            return dasum(arr.length(),arr,BlasBufferUtil.getBlasStride(arr));
        return sasum(arr.length(),arr, BlasBufferUtil.getBlasStride(arr));
    }

    @Override
    public double asum(int n, DataBuffer x, int offsetX, int incrX){
        if(supportsDataBufferL1Ops()){
            if(x.dataType() == DataBuffer.Type.FLOAT){
                return sasum(n,x,offsetX,incrX);
            } else {
                return dasum(n,x,offsetX,incrX);
            }
        } else {
            int[] shapex = {1,n};
            int[] stridex = {incrX,incrX};
            INDArray arrX = Nd4j.create(x,shapex,stridex,offsetX,'c');
            return asum(arrX);
        }
    }

    /**
     * computes the sum of magnitudes
     * of all vector elements or,
     * for a complex vector x, the sum
     *
     * @param arr the array to get the sum for
     * @return
     */
    @Override
    public IComplexNumber asum(IComplexNDArray arr) {
        throw new UnsupportedOperationException();

    }

    @Override
    public int iamax(int n, INDArray arr, int stride) {
        if(arr.data().dataType() == DataBuffer.Type.DOUBLE)
            return idamax(n,arr,stride);
        return isamax(n,arr,stride);
    }

    @Override
    public int iamax(int n,DataBuffer x, int offsetX, int incrX){
        if(supportsDataBufferL1Ops()){
            if(x.dataType() == DataBuffer.Type.FLOAT){
                return isamax(n,x,offsetX,incrX);
            } else {
                return isamax(n,x,offsetX,incrX);
            }
        } else {
            int[] shapex = {1,n};
            int[] stridex = {incrX,incrX};
            INDArray arrX = Nd4j.create(x,shapex,stridex,offsetX,'c');
            return iamax(n, arrX, incrX);
        }
    }

    /**
     * finds the element of a
     * vector that has the largest absolute value.
     *
     * @param arr
     * @return
     */
    @Override
    public int iamax(INDArray arr) {
        if(arr.data().dataType() == DataBuffer.Type.DOUBLE)
            return idamax(arr.length(), arr, BlasBufferUtil.getBlasStride(arr));
        return isamax(arr.length(), arr, BlasBufferUtil.getBlasStride(arr));
    }

    /**
     * finds the element of a vector that has the largest absolute value.
     *
     * @param arr
     * @return
     */
    @Override
    public int iamax(IComplexNDArray arr) {
        if(arr.data().dataType() == DataBuffer.Type.DOUBLE)
            return izamax(arr.length(), arr, BlasBufferUtil.getBlasStride(arr));
        return icamax(arr.length(), arr, BlasBufferUtil.getBlasStride(arr));
    }

    /**
     * finds the element of a vector that has the minimum absolute value.
     *
     * @param arr
     * @return
     */
    @Override
    public int iamin(INDArray arr) {
        throw new UnsupportedOperationException();
    }

    /**
     * finds the element of
     * a vector that has the minimum absolute value.
     *
     * @param arr
     * @return
     */
    @Override
    public int iamin(IComplexNDArray arr) {
        throw new UnsupportedOperationException();
    }

    /**
     * swaps a vector with another vector.
     *
     * @param x
     * @param y
     */
    @Override
    public void swap(INDArray x, INDArray y) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            dswap(x.length(), x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));
        else
            sswap(x.length(), x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));
    }

    @Override
    public void swap(IComplexNDArray x, IComplexNDArray y) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            zswap(x.length(), x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));

        else
            cswap(x.length(), x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));


    }


    /**
     * swaps a vector with another vector.
     *
     * @param x
     * @param y
     */
    @Override
    public void copy(INDArray x, INDArray y) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            dcopy(x.length(), x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));
        else
            scopy(x.length(), x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));
    }

    /**copy a vector to another vector.
     */
    @Override
    public void copy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ) {
        if(supportsDataBufferL1Ops()) {
            if (x.dataType() == DataBuffer.Type.DOUBLE) {
                dcopy(n, x, offsetX, incrX, y, offsetY, incrY);
            } else {
                scopy(n,x,offsetX,incrX,y,offsetY,incrY);
            }
        } else {
            int[] shapex = {1,n};
            int[] shapey = {1,n};
            int[] stridex = {incrX,incrX};
            int[] stridey = {incrY,incrY};
            INDArray arrX = Nd4j.create(x,shapex,stridex,offsetX,'c');
            INDArray arrY = Nd4j.create(x,shapey,stridey,offsetY,'c');
            copy(arrX,arrY);
        }
    }



    /**
     * copy a vector to another vector.
     *
     * @param x
     * @param y
     */
    @Override
    public void copy(IComplexNDArray x, IComplexNDArray y) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            zcopy(x.length(), x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));
        else
            ccopy(x.length(), x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));
    }



    /**
     * computes a vector-scalar product and adds the result to a vector.
     *
     * @param n
     * @param alpha
     * @param x
     * @param y
     */
    @Override
    public void axpy(int n, double alpha, INDArray x, INDArray y) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            daxpy(n, alpha, x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));
        else
            saxpy(n, (float) alpha, x, BlasBufferUtil.getBlasStride(x), y, BlasBufferUtil.getBlasStride(y));
    }

    @Override
    public void axpy(int n,double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        if(supportsDataBufferL1Ops()) {
            if (x.dataType() == DataBuffer.Type.DOUBLE) {
                daxpy(n, alpha, x, offsetX, incrX, y, offsetY, incrY);
            } else {
                saxpy(n, (float) alpha, x, offsetX, incrX, y, offsetY, incrY);
            }
        } else {
            int[] shapex = {1,n};
            int[] shapey = {1,n};
            int[] stridex = {incrX,incrX};
            int[] stridey = {incrY,incrY};
            INDArray arrX = Nd4j.create(x,shapex,stridex,offsetX,'c');
            INDArray arrY = Nd4j.create(x,shapey,stridey,offsetY,'c');
            axpy(n,alpha,arrX,arrY);
        }
    }

    /**
     * computes a vector-scalar product and adds the result to a vector.
     *
     * @param n
     * @param alpha
     * @param x
     * @param y
     */
    @Override
    public void axpy(int n, IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y) {
        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            zaxpy(n,alpha.asDouble(),x,BlasBufferUtil.getBlasStride(x),y,BlasBufferUtil.getBlasStride(y));
        else
            caxpy(n,alpha.asFloat(),x,BlasBufferUtil.getBlasStride(x),y,BlasBufferUtil.getBlasStride(y));
    }

    /**
     * computes parameters for a Givens rotation.
     *
     * @param a
     * @param b
     * @param c
     * @param s
     */
    @Override
    public void rotg(INDArray a, INDArray b, INDArray c, INDArray s) {
        throw new UnsupportedOperationException();
    }

    /**
     * performs rotation of points in the plane.
     *
     * @param N
     * @param X
     * @param Y
     * @param c
     * @param s
     */
    @Override
    public void rot(int N, INDArray X, INDArray Y, double c, double s) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            drot(N, X, BlasBufferUtil.getBlasStride(X), Y, BlasBufferUtil.getBlasStride(X), c, s);
        else
            srot(N, X, BlasBufferUtil.getBlasStride(X), Y, BlasBufferUtil.getBlasStride(X), (float) c, (float) s);
    }

    /**
     * performs rotation of points in the plane.
     *
     * @param N
     * @param X
     * @param Y
     * @param c
     * @param s
     */
    @Override
    public void rot(int N, IComplexNDArray X, IComplexNDArray Y, IComplexNumber c, IComplexNumber s) {
        throw new UnsupportedOperationException();
    }

    /**
     * computes the modified parameters for a Givens rotation.
     *
     * @param d1
     * @param d2
     * @param b1
     * @param b2
     * @param P
     */
    @Override
    public void rotmg(INDArray d1, INDArray d2, INDArray b1, double b2, INDArray P) {
     throw new UnsupportedOperationException();
    }

    /**
     * computes the modified parameters for a Givens rotation.
     *
     * @param d1
     * @param d2
     * @param b1
     * @param b2
     * @param P
     */
    @Override
    public void rotmg(IComplexNDArray d1, IComplexNDArray d2, IComplexNDArray b1, IComplexNumber b2, IComplexNDArray P) {
        throw new UnsupportedOperationException();
    }

    /**
     * computes a vector by a scalar product.
     *
     * @param N
     * @param alpha
     * @param X
     */
    @Override
    public void scal(int N, double alpha, INDArray X) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dscal(N, alpha, X, BlasBufferUtil.getBlasStride(X));
        else
            sscal(N, (float) alpha, X, BlasBufferUtil.getBlasStride(X));
    }

    /**
     * computes a vector by a scalar product.
     *
     * @param N
     * @param alpha
     * @param X
     */
    @Override
    public void scal(int N, IComplexNumber alpha, IComplexNDArray X) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            zscal(N, alpha.asDouble(), X, BlasBufferUtil.getBlasStride(X));
        else
            cscal(N, alpha.asFloat(), X, BlasBufferUtil.getBlasStride(X));

    }



    /*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
    protected abstract  float  sdsdot( int N,  float alpha,  INDArray X,
                                       int incX,  INDArray Y,  int incY);
    protected abstract    double dsdot( int N,  INDArray X,  int incX,  INDArray Y,
                                        int incY);
    protected abstract  float  sdot( int N,  INDArray X,  int incX,
                                     INDArray Y,  int incY);
    protected abstract  float  sdot( int N,  DataBuffer X, int offsetX, int incX,
                                     DataBuffer Y,  int offsetY, int incY);
    protected abstract    double ddot( int N, INDArray X,  int incX,
                                       INDArray Y,  int incY);
    protected abstract    double ddot( int N, DataBuffer X, int offsetX, int incX,
                                       DataBuffer Y, int offsetY, int incY);

    /*
     * Functions having prefixes Z and C only
     */
    protected abstract void   cdotu_sub( int N,  IComplexNDArray X,  int incX,
                                         IComplexNDArray Y,  int incY, IComplexNDArray dotu);
    protected abstract  void   cdotc_sub( int N,  IComplexNDArray X,  int incX,
                                          IComplexNDArray Y,  int incY, IComplexNDArray dotc);

    protected abstract   void   zdotu_sub( int N,  IComplexNDArray X,  int incX,
                                           IComplexNDArray Y,  int incY, IComplexNDArray dotu);
    protected abstract  void   zdotc_sub( int N,  IComplexNDArray X,  int incX,
                                          IComplexNDArray Y,  int incY, IComplexNDArray dotc);


    /*
     * Functions having prefixes S D SC DZ
     */
    protected abstract   float  snrm2( int N,  INDArray X,  int incX);
    protected abstract  float  sasum( int N,  INDArray X,  int incX);
    protected abstract  float  sasum( int N,  DataBuffer X,  int offsetX, int incX);

    protected abstract  double dnrm2( int N,  INDArray X,  int incX);
    protected abstract  double dasum( int N,  INDArray X,  int incX);
    protected abstract  double dasum( int N,  DataBuffer X,  int offsetX, int incX);

    protected abstract float  scnrm2( int N,  IComplexNDArray X,  int incX);
    protected abstract   float  scasum( int N,  IComplexNDArray X,  int incX);

    protected abstract double dznrm2( int N,  IComplexNDArray X,  int incX);
    protected abstract  double dzasum( int N,  IComplexNDArray X,  int incX);


    /*
     * Functions having standard 4 prefixes (S D C Z)
     */
    protected abstract int isamax( int N,  INDArray X,  int incX);
    protected abstract int isamax( int N,  DataBuffer X,  int offsetX, int incX);
    protected abstract int idamax( int N,  INDArray X,  int incX);
    protected abstract int idamax( int N,  DataBuffer X,  int offsetX, int incX);
    protected abstract int icamax( int N,  IComplexNDArray X,  int incX);
    protected abstract int izamax( int N,  IComplexNDArray X,  int incX);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

    /*
     * Routines with standard 4 prefixes (s, d, c, z)
     */
    protected abstract  void sswap( int N, INDArray X,  int incX,
                                    INDArray Y,  int incY);
    protected abstract void scopy( int N,  INDArray X,  int incX,
                                   INDArray Y,  int incY);
    protected abstract void scopy(int n, DataBuffer x, int offsetX, int incrX,
                                  DataBuffer y, int offsetY, int incrY );
    protected abstract  void saxpy( int N,  float alpha,  INDArray X,
                                    int incX, INDArray Y,  int incY);
    protected abstract void saxpy( int N, float alpha, DataBuffer x, int offsetX, int incrX,
                                   DataBuffer y, int offsetY, int incrY );

    protected abstract  void dswap( int N, INDArray X,  int incX,
                                    INDArray Y,  int incY);
    protected abstract  void dcopy( int N,  INDArray X,  int incX,
                                    INDArray Y,  int incY);
    protected abstract void dcopy(int n, DataBuffer x, int offsetX, int incrX,
                                  DataBuffer y, int offsetY, int incrY );
    protected abstract  void daxpy( int N,  double alpha,  INDArray X,
                                    int incX, INDArray Y,  int incY);
    protected abstract void daxpy( int N, double alpha, DataBuffer x, int offsetX, int incrX,
                                   DataBuffer y, int offsetY, int incrY );

    protected abstract  void cswap( int N, IComplexNDArray X,  int incX,
                                    IComplexNDArray Y,  int incY);
    protected abstract  void ccopy( int N,  IComplexNDArray X,  int incX,
                                    IComplexNDArray Y,  int incY);
    protected abstract void caxpy( int N,  IComplexFloat alpha,  IComplexNDArray X,
                                   int incX, IComplexNDArray Y,  int incY);

    protected abstract  void zswap( int N, IComplexNDArray X,  int incX,
                                    IComplexNDArray Y,  int incY);
    protected abstract  void zcopy( int N,  IComplexNDArray X,  int incX,
                                    IComplexNDArray Y,  int incY);
    protected abstract void zaxpy( int N,  IComplexDouble alpha,  IComplexNDArray X,
                                   int incX, IComplexNDArray Y,  int incY);


    /*
     * Routines with S and D prefix only
     */
    protected abstract void srotg(float a, float b, float c, float s);
    protected abstract void srotmg(float d1, float d2, float b1,  float b2, INDArray P);
    protected abstract  void srot( int N, INDArray X,  int incX,
                                   INDArray Y,  int incY,  float c,  float s);
    protected abstract  void srotm( int N, INDArray X,  int incX,
                                    INDArray Y,  int incY,  INDArray P);

    protected abstract   void drotg(double a, double b, double c, double s);
    protected abstract  void drotmg(double d1, double d2, double b1,  double b2, INDArray P);
    protected abstract  void drot( int N, INDArray X,  int incX,
                                   INDArray Y,  int incY,  double c,  double s);


    protected abstract void drotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P);

    /*
         * Routines with S D C Z CS and ZD prefixes
         */
    protected abstract void sscal( int N,  float alpha, INDArray X,  int incX);
    protected abstract void dscal( int N,  double alpha, INDArray X,  int incX);
    protected abstract void cscal( int N,  IComplexFloat alpha, IComplexNDArray X,  int incX);
    protected abstract void zscal( int N,  IComplexDouble alpha, IComplexNDArray X,  int incX);
    protected abstract void csscal( int N,  float alpha, IComplexNDArray X,  int incX);
    protected abstract void zdscal( int N,  double alpha, IComplexNDArray X,  int incX);

    @Override
    public boolean supportsDataBufferL1Ops(){
        return true;
    }

}
