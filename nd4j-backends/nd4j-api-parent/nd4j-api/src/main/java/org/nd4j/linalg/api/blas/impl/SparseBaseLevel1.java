package org.nd4j.linalg.api.blas.impl;

import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.factory.BaseSparseBlasWrapper;

/**
 * @author Audrey Loeffel
 */
public abstract class SparseBaseLevel1 extends SparseBaseLevel implements Level1{

    /**
     * computes a vector-vector dot product.
     *
     * @param n number of accessed element
     * @param alpha
     * @param X an INDArray
     * @param Y an INDArray
     * @return the vector-vector dot product of X and Y
     */
    @Override
    public double dot(int n, double alpha, INDArray X, INDArray Y) {

        if(X instanceof BaseSparseNDArray) {
            BaseSparseNDArray sparseX = (BaseSparseNDArray) X;
            DataBuffer pointers = sparseX.getMinorPointer();
            switch(X.data().dataType()){
                case DOUBLE:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, X, Y);
                    return ddoti(n, X, pointers, Y);
                case FLOAT:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, X, Y);
                    return sdoti(n, X, pointers, Y);
                case HALF:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, X, Y);
                    return hdoti(n, X, pointers, Y);
                default:
            }
         }
        throw new UnsupportedOperationException();
    }




    @Override
    public double dot(int n, DataBuffer dx, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber dot(int n, IComplexNumber alpha, IComplexNDArray X, IComplexNDArray Y) {
        throw new UnsupportedOperationException();
    }

    /**
     * Computes the Euclidean norm of a vector.
     *
     * @param arr a vector
     * @return the Euclidean norm of the vector
     */
    @Override
    public double nrm2(INDArray arr) {

        switch(arr.data().dataType()){
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, arr);
                return dnrm2(arr.length(), arr, 1);
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, arr);
                return snrm2(arr.length(), arr, 1);
            case HALF:
                return hnrm2(arr.length(), arr, 1);
            default:
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber nrm2(IComplexNDArray arr) {
        throw new UnsupportedOperationException();
    }

    /**
     * Compute the sum of magnitude of the vector elements
     *
     * @param arr a vector
     * @return the sum of magnitude of the vector elements
     * */
    @Override
    public double asum(INDArray arr) {

        switch(arr.data().dataType()){
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, arr);
                return dasum(arr.length(), arr, 1);
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, arr);
                return sasum(arr.length(), arr, 1);
            case HALF:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, arr);
                return hasum(arr.length(), arr, 1);
            default:
        }
        throw new UnsupportedOperationException();
    }


    @Override
    public double asum(int n, DataBuffer x, int offsetX, int incrX) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber asum(IComplexNDArray arr) {
        throw new UnsupportedOperationException();
    }

    /**
     * Find the index of the element with maximum absolute value
     *
     * @param arr a vector
     * @return the index of the element with maximum absolute value
     * */
    @Override
    public int iamax(INDArray arr) {
        switch(arr.data().dataType()){
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, arr);
                return idamax(arr.length(), arr, 1);
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, arr);
                return isamax(arr.length(), arr, 1);
            case HALF:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, arr);
                return ihamax(arr.length(), arr, 1);
            default:
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public int iamax(int n, INDArray arr, int stride) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int iamax(int n, DataBuffer x, int offsetX, int incrX) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int iamax(IComplexNDArray arr) {
        throw new UnsupportedOperationException();
    }

    /**
     * Find the index of the element with minimum absolute value
     *
     * @param arr a vector
     * @return the index of the element with maximum absolute value
     * */    /**
     * Find the index of the element with maximum absolute value
     *
     * @param arr a vector
     * @return the index of the element with minimum absolute value
     * */
    @Override
    public int iamin(INDArray arr) {
        switch(arr.data().dataType()){
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, arr);
                return idamin(arr.length(), arr, 1);
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, arr);
                return isamin(arr.length(), arr, 1);
            case HALF:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, arr);
                return ihamin(arr.length(), arr, 1);
            default:
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public int iamin(IComplexNDArray arr) {
        throw new UnsupportedOperationException();
    }


    /**
     * Swap a vector with another vector
     *
     * @param x a vector
     * @param y a vector
     * */
    /*
    @Override
    public void swap(INDArray x, INDArray y) {
        // not sure if the routines are useable with sparse ndarray
        // -> TODO test if it works with sparse array
        // also swap minorPointers ?
        if(x.isSparse() && y.isSparse()){
            BaseSparseNDArray xSparse = (BaseSparseNDArray) x;
            BaseSparseNDArray ySparse = (BaseSparseNDArray) y;
            switch(x.data().dataType()){
                case DOUBLE:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, x, y);
                    dswap(x.length(), x, 1, y, 1);
                    // swap pointers?
                case FLOAT:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, x, y);
                    sswap(x.length(), x, 1, y, 1);
                case HALF:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, x, y);
                    hswap(x.length(), x, 1, y, 1);
                default:
            }
            throw new UnsupportedOperationException();

        } else {
            throw new UnsupportedOperationException();
        }
    }*/

    @Override
    public void swap(IComplexNDArray x, IComplexNDArray y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copy(INDArray x, INDArray y) {

    }

    @Override
    public void copy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {

    }

    @Override
    public void copy(IComplexNDArray x, IComplexNDArray y) {

    }

    @Override
    public void axpy(int n, double alpha, INDArray x, INDArray y) {

    }

    @Override
    public void axpy(int n, double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {

    }

    @Override
    public void axpy(int n, IComplexNumber alpha, IComplexNDArray x, IComplexNDArray y) {

    }

    @Override
    public void rotg(INDArray a, INDArray b, INDArray c, INDArray s) {

    }

    @Override
    public void rot(int N, INDArray X, INDArray Y, double c, double s) {

    }

    @Override
    public void rot(int N, IComplexNDArray X, IComplexNDArray Y, IComplexNumber c, IComplexNumber s) {

    }

    @Override
    public void rotmg(INDArray d1, INDArray d2, INDArray b1, double b2, INDArray P) {

    }

    @Override
    public void rotmg(IComplexNDArray d1, IComplexNDArray d2, IComplexNDArray b1, IComplexNumber b2, IComplexNDArray P) {

    }

    @Override
    public void scal(int N, double alpha, INDArray X) {

    }

    @Override
    public void scal(int N, IComplexNumber alpha, IComplexNDArray X) {

    }

    @Override
    public boolean supportsDataBufferL1Ops() {
        return false;
    }


    /*
    * ===========================================================================
    * Prototypes for level 1 BLAS functions (complex are recast as routines)
    * ===========================================================================
    */

    protected abstract double ddoti(int N, INDArray X, DataBuffer indx, INDArray Y);

    protected abstract double sdoti(int N, INDArray X, DataBuffer indx, INDArray Y);

    protected abstract double hdoti(int N, INDArray X, DataBuffer indx, INDArray Y);

    protected abstract double snrm2(int N, INDArray X, int incx);

    protected abstract double dnrm2(int N, INDArray X, int incx);

    protected abstract double hnrm2(int N, INDArray X, int incx);

    protected abstract double dasum(int N, INDArray X, int incx);

    protected abstract double sasum(int N, INDArray X, int incx);

    protected abstract double hasum(int N, INDArray X, int incx);

    protected abstract int isamax(int N, INDArray X, int incx);

    protected abstract int idamax(int N, INDArray X, int incx);

    protected abstract int ihamax(int N, INDArray X, int incx);

    protected abstract int isamin(int N, INDArray X, int incx);

    protected abstract int idamin(int N, INDArray X, int incx);

    protected abstract int ihamin(int N, INDArray X, int incx);

    //protected abstract void dswap(int N, INDArray X, int incrx, INDArray Y, int incry);

    //protected abstract void sswap(int N, INDArray X, int incrx, INDArray Y, int incry);

    //protected abstract void hswap(int N, INDArray X, int incrx, INDArray Y, int incry);

    protected abstract int scopy(int N, INDArray X, int incrx, INDArray Y, int incry);

    protected abstract int dcopy(int N, INDArray X, int incrx, INDArray Y, int incry);

    protected abstract int hcopy(int N, INDArray X, int incrx, INDArray Y, int incry);

    /*
     * Functions having prefixes Z and C only
     */
    /*


    /*
     * Functions having prefixes S D SC DZ
     */

    /*
     * Functions having standard 4 prefixes (S D C Z)
     */

    /*
     * ===========================================================================
     * Prototypes for level 1 BLAS routines
     * ===========================================================================
     */

    /*
     * Routines with standard 4 prefixes (s, d, c, z)
     */


    /*
     * Routines with S and D prefix only
     */

    /*
     * Routines with S D C Z CS and ZD prefixes
     */

    }