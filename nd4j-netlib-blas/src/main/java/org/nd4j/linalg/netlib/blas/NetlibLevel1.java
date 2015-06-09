package org.nd4j.linalg.netlib.blas;

import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class NetlibLevel1 extends BaseLevel1 {
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
        return 0;
    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return 0;
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
        return 0;
    }

    @Override
    protected float sasum(int N, INDArray X, int incX) {
        return 0;
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        return 0;
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        return 0;
    }

    @Override
    protected float scnrm2(int N, IComplexNDArray X, int incX) {
        return 0;
    }

    @Override
    protected float scasum(int N, IComplexNDArray X, int incX) {
        return 0;
    }

    @Override
    protected double dznrm2(int N, IComplexNDArray X, int incX) {
        return 0;
    }

    @Override
    protected double dzasum(int N, IComplexNDArray X, int incX) {
        return 0;
    }

    @Override
    protected int isamax(int N, INDArray X, int incX) {
        return 0;
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return 0;
    }

    @Override
    protected int icamax(int N, IComplexNDArray X, int incX) {
        return 0;
    }

    @Override
    protected int izamax(int N, IComplexNDArray X, int incX) {
        return 0;
    }

    @Override
    protected void sswap(int N, INDArray X, int incX, INDArray Y, int incY) {

    }

    @Override
    protected void scopy(int N, INDArray X, int incX, INDArray Y, int incY) {

    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {

    }

    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {

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
    protected void srotg(float a, float b, float c, float s) {

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
    protected void drotg(double a, double b, double c, double s) {

    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {

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
