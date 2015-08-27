package org.nd4j.linalg.java.blas;

import com.github.fommil.netlib.BLAS;
import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.netlib.util.doubleW;
import org.netlib.util.floatW;

import static org.nd4j.linalg.api.blas.BlasBufferUtil.*;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.getDoubleData;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.getFloatData;


/**
 * @author Adam Gibson
 */
public class JavaLevel1 extends BaseLevel1 {
    @Override
    protected float sdsdot(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected double dsdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected float sdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return BLAS.getInstance().sdot(N, getFloatData(X), getBlasOffset(X), incX, getFloatData(Y), getBlasOffset(Y), incY);

    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return BLAS.getInstance().ddot(N, getDoubleData(X), getBlasOffset(X), incX, getDoubleData(Y), getBlasOffset(Y), incY);

    }

    @Override
    protected void cdotu_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotu) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cdotc_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zdotu_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotu) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zdotc_sub(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray dotc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected float snrm2(int N, INDArray X, int incX) {
        return BLAS.getInstance().snrm2(N,getFloatData(X),getBlasOffset(X),incX);
    }

    @Override
    protected float sasum(int N, INDArray X, int incX) {
        return BLAS.getInstance().sasum(N,getFloatData(X),getBlasOffset(X),incX);
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        return BLAS.getInstance().dnrm2(N,getDoubleData(X),getBlasOffset(X),incX);
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        return BLAS.getInstance().dasum(N,getDoubleData(X),getBlasOffset(X),incX);
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
        return BLAS.getInstance().isamax(N,getFloatData(X),getBlasOffset(X),incX) - 1;
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return BLAS.getInstance().idamax(N,getDoubleData(X),getBlasOffset(X),incX) - 1;
    }

    @Override
    protected int icamax(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected int izamax(int N, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void sswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().sswap(N,getFloatData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void scopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().scopy(N,getFloatData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().saxpy(N,alpha,getFloatData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY);
        setData(yData, Y);
    }

    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().dswap(N,getDoubleData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void dcopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().dcopy(N,getDoubleData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void daxpy(int N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().daxpy(N,alpha,getDoubleData(X),getBlasOffset(X),getBlasOffset(X),getDoubleData(Y),getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void cswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ccopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void caxpy(int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zcopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zaxpy(int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotg(float a, float b, float c, float s) {
        BLAS.getInstance().srotg(new floatW(a),new floatW(b),new floatW(c),new floatW(s));
    }

    @Override
    protected void srotmg(float d1, float d2, float b1, float b2, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srot(int N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().srot(N,getFloatData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY,c,s);
        setData(yData,Y);
    }

    @Override
    protected void srotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        float[] pData = getFloatData(P);
        BLAS.getInstance().srotm(N,getFloatData(X),getBlasOffset(X),incX,getFloatData(Y),getBlasOffset(P),incY,pData,getBlasOffset(P));
        setData(pData,P);
    }

    @Override
    protected void drotg(double a, double b, double c, double s) {
        BLAS.getInstance().drotg(new doubleW(a),new doubleW(b),new doubleW(c),new doubleW(s));

    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {
        double[] data = getDoubleData(P);
        doubleW dd1 = new doubleW(d1);
        doubleW dd2 = new doubleW(d2);
        BLAS.getInstance().drotmg(dd1,dd2,new doubleW(b1),b2,data);
        setData(data,P);
    }

    @Override
    protected void drot(int N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().drot(N,getDoubleData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY,c,s);
        setData(yData,Y);
    }

    @Override
    protected void drotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        double[] pData = getDoubleData(P);
        BLAS.getInstance().drotm(N,getDoubleData(X),getBlasOffset(X),incX,getDoubleData(Y),getBlasOffset(Y),incY,pData,getBlasOffset(P));
        setData(pData,P);
    }

    @Override
    protected void sscal(int N, float alpha, INDArray X, int incX) {
        float[] xData = getFloatData(X);
        BLAS.getInstance().sscal(N, alpha, xData, getBlasOffset(X), incX);
        setData(xData,X);
    }

    @Override
    protected void dscal(int N, double alpha, INDArray X, int incX) {
        double[] xData = getDoubleData(X);
        BLAS.getInstance().dscal(N,alpha,xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void cscal(int N, IComplexFloat alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zscal(int N, IComplexDouble alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csscal(int N, float alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zdscal(int N, double alpha, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }
}
