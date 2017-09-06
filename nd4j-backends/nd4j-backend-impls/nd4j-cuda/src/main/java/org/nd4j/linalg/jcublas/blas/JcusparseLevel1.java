package org.nd4j.linalg.jcublas.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Audrey Loeffel
 */
public class JcusparseLevel1 extends SparseBaseLevel1 {

    @Override
    protected double ddoti(int N, INDArray X, DataBuffer indx, INDArray Y) {
        return 0;
    }

    @Override
    protected double sdoti(int N, INDArray X, DataBuffer indx, INDArray Y) {
        return 0;
    }

    @Override
    protected double hdoti(int N, INDArray X, DataBuffer indx, INDArray Y) {
        return 0;
    }

    @Override
    protected double snrm2(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double hnrm2(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double dasum(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double sasum(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double hasum(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int isamax(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int idamax(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int ihamax(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int isamin(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int idamin(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int ihamin(int N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected void daxpyi(int N, double alpha, INDArray X, DataBuffer pointers, INDArray Y) {

    }

    @Override
    protected void saxpyi(int N, double alpha, INDArray X, DataBuffer pointers, INDArray Y) {

    }

    @Override
    protected void haxpyi(int N, double alpha, INDArray X, DataBuffer pointers, INDArray Y) {

    }

    @Override
    protected void droti(int N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {

    }

    @Override
    protected void sroti(int N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {

    }

    @Override
    protected void hroti(int N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {

    }

    @Override
    protected void dscal(int N, double a, INDArray X, int incx) {

    }

    @Override
    protected void sscal(int N, double a, INDArray X, int incx) {

    }

    @Override
    protected void hscal(int N, double a, INDArray X, int incx) {

    }
}
