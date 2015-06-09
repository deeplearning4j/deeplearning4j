package org.nd4j.linalg.jblas.blas;

import org.jblas.JavaBlas;
import org.jblas.NativeBlas;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.nd4j.linalg.api.blas.BlasBufferUtil.*;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.getDoubleData;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.getFloatData;

/**
 * @author Adam Gibson
 */
public class JblasLevel1 extends BaseLevel1 {
    @Override
    protected float sdsdot(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        return 0;
    }

    @Override
    protected double dsdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return 0;
    }

    @Override
    protected float sdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        float[] xData = getFloatData(X);
        float[] yData = getFloatData(Y);
        int xOffset = getBlasOffset(X);
        int yOffset = getBlasOffset(Y);
        return JavaBlas.rdot(xData.length,xData,xOffset,incX,yData,yOffset,incY);
    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        double[] xData = getDoubleData(X);
        double[] yData = getDoubleData(Y);
        int xOffset = getBlasOffset(X);
        int yOffset = getBlasOffset(Y);
        return JavaBlas.rdot(xData.length,xData,xOffset,incX,yData,yOffset,incY);
    }

    @Override
    protected void cdotu_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotu) {

    }

    @Override
    protected void cdotc_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotc) {

    }

    @Override
    protected void zdotu_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotu) {

    }

    @Override
    protected void zdotc_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotc) {

    }

    @Override
    protected float snrm2(int N, INDArray X, int incX) {
        float[] data = getFloatData(X);
        return NativeBlas.snrm2(N,data,getBlasOffset(X),incX);
    }

    @Override
    protected float sasum(int N, INDArray X, int incX) {
        float[] data = getFloatData(X);
        return NativeBlas.sasum(N,data,getBlasOffset(X),incX);
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        double[] data = getDoubleData(X);
        return NativeBlas.dnrm2(N,data,getBlasOffset(X),incX);
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        double[] data = getDoubleData(X);
        return NativeBlas.dasum(N,data,getBlasOffset(X),incX);
    }

    @Override
    protected float scnrm2(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float scasum(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected double dznrm2(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected double dzasum(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected int isamax(int N, INDArray X, int incX) {
        return NativeBlas.isamax(N,getFloatData(X),getBlasOffset(X),incX);
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return NativeBlas.idamax(N, getDoubleData(X), getBlasOffset(X), incX);
    }

    @Override
    protected int icamax(int N, IComplexNDArray X, int incX) {
        return NativeBlas.icamax(N,getFloatData(X),getBlasOffset(X),incX);
    }

    @Override
    protected int izamax(int N, IComplexNDArray X, int incX) {
        return NativeBlas.izamax(N,getDoubleData(X),getBlasOffset(X),incX);
    }

    @Override
    protected void sswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        NativeBlas.sswap(N,getFloatData(X),getBlasOffset(X),incX,getFloatData(Y),getBlasOffset(Y),incY);
    }

    @Override
    protected void scopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        NativeBlas.scopy(N,getFloatData(X),getBlasOffset(X),incX,getFloatData(Y),getBlasOffset(Y),incY);
    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        float[] dataToSret = getFloatData(Y);
        JavaBlas.raxpy(N,alpha,getFloatData(X),getBlasOffset(X),incX,dataToSret,getBlasOffset(Y),incY);

    }

    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        NativeBlas.dswap(N,getDoubleData(X),getBlasOffset(X),incX,getDoubleData(Y),getBlasOffset(Y),incY);
    }

    @Override
    protected void dcopy(int N, INDArray X, int incX, INDArray Y, int incY) {

    }

    @Override
    protected void daxpy(int N, double alpha, INDArray X, int incX, INDArray Y, int incY) {

    }

    @Override
    protected void cswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {

    }

    @Override
    protected void ccopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {

    }

    @Override
    protected void caxpy(int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {

    }

    @Override
    protected void zswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {

    }

    @Override
    protected void zcopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {

    }

    @Override
    protected void zaxpy(int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {

    }

    @Override
    protected void srotg(INDArray a, INDArray b, INDArray c, INDArray s) {

    }

    @Override
    protected void srotmg(INDArray d1, INDArray d2, INDArray b1, float b2, INDArray P) {

    }

    @Override
    protected void srot(int N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {

    }

    @Override
    protected void srotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {

    }

    @Override
    protected void drotg(INDArray a, INDArray b, INDArray c, INDArray s) {

    }

    @Override
    protected void drotmg(INDArray d1, INDArray d2, INDArray b1, double b2, INDArray P) {

    }

    @Override
    protected void drot(int N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {

    }


    @Override
    protected void drotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {

    }

    @Override
    protected void sscal(int N, float alpha, INDArray X, int incX) {

    }

    @Override
    protected void dscal(int N, double alpha, INDArray X, int incX) {

    }

    @Override
    protected void cscal(int N, IComplexFloat alpha, IComplexNDArray X, int incX) {

    }

    @Override
    protected void zscal(int N, IComplexDouble alpha, IComplexNDArray X, int incX) {

    }

    @Override
    protected void csscal(int N, float alpha, IComplexNDArray X, int incX) {

    }

    @Override
    protected void zdscal(int N, double alpha, IComplexNDArray X, int incX) {

    }
}
