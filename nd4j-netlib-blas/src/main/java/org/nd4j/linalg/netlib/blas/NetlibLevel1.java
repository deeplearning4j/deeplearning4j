package org.nd4j.linalg.netlib.blas;

import com.github.fommil.netlib.BLAS;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
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
public class NetlibLevel1 extends BaseLevel1 {
    @Override
    protected float sdsdot(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        return BLAS.getInstance().sdsdot(N,alpha,getFloatData(X),getBlasOffset(X),incX,getFloatData(Y),getBlasOffset(Y),incY);
    }

    @Override
    protected double dsdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected float sdot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return BLAS.getInstance().sdot(N,getFloatData(X),getBlasOffset(X),incX,getFloatData(Y),getBlasOffset(Y),incY);
    }

    @Override
    protected float sdot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        return BLAS.getInstance().sdot(N, getFloatData(X),offsetX,incX, getFloatData(Y), offsetY, incY );
    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        return BLAS.getInstance().ddot(N, getDoubleData(X), getBlasOffset(X), incX, getDoubleData(Y), getBlasOffset(Y), incY);
    }

    @Override
    protected double ddot( int N, DataBuffer X, int offsetX, int incX, DataBuffer Y,  int offsetY, int incY){
        return BLAS.getInstance().ddot(N, getDoubleData(X), offsetX, incX, getDoubleData(Y), offsetY, incY);
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
        return BLAS.getInstance().sasum(N, getFloatData(X), getBlasOffset(X), incX);
    }

    @Override
    protected float sasum(int N, DataBuffer X, int offsetX, int incX){
        return BLAS.getInstance().sasum(N,getFloatData(X), offsetX, incX);
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        return BLAS.getInstance().dnrm2(N, getDoubleData(X), getBlasOffset(X), incX);
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        return BLAS.getInstance().dasum(N, getDoubleData(X), getBlasOffset(X), incX);
    }

    @Override
    protected double dasum(int N, DataBuffer X, int offsetX, int incX){
        return BLAS.getInstance().dasum(N, getDoubleData(X), offsetX, incX);
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
    protected int isamax(int N, DataBuffer X, int offsetX, int incX){
        return BLAS.getInstance().isamax(N,getFloatData(X), offsetX, incX);
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return BLAS.getInstance().idamax(N, getDoubleData(X), getBlasOffset(X), incX) - 1;
    }

    @Override
    protected int idamax(int N, DataBuffer X, int offsetX, int incX){
        return BLAS.getInstance().idamax(N, getDoubleData(X), offsetX, incX);
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
        float[] xData = getFloatData(X);
        BLAS.getInstance().sswap(N, xData, getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(xData,X);
        setData(yData,Y);
    }

    @Override
    protected void scopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().scopy(N, getFloatData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void scopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        BLAS.getInstance().scopy(n, getFloatData(x), offsetX, incrX, getFloatData(y), offsetY, incrY);
    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().saxpy(N, alpha, getFloatData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    public void saxpy(int n,float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        BLAS.getInstance().saxpy(n, alpha, getFloatData(x), offsetX, incrX, getFloatData(y), offsetY, incrY);
    }

    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        double[] xData = getDoubleData(X);
        BLAS.getInstance().dswap(N,xData,getBlasOffset(X),incX,yData,getBlasOffset(Y),incY);
        setData(xData, X);
        setData(yData, Y);
    }

    @Override
    protected void dcopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().dcopy(N, getDoubleData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void dcopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        BLAS.getInstance().dcopy(n, getDoubleData(x), offsetX, incrX, getDoubleData(y), offsetY, incrY);
    }

    @Override
    protected void daxpy(int N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().daxpy(N, alpha, getDoubleData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    public void daxpy(int n,double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        BLAS.getInstance().daxpy(n, alpha, getDoubleData(x), offsetX, incrX, getDoubleData(y), offsetY, incrY);
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
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotmg(float d1, float d2, float b1, float b2, INDArray P) {
        float[] pData = getFloatData(P);
        BLAS.getInstance().srotmg(new floatW(d1), new floatW(d2), new floatW(b1), b2, pData, getBlasOffset(P));
        setData(pData,P);
    }

    @Override
    protected void srot(int N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
         throw new UnsupportedOperationException();
    }

    @Override
    protected void srotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        float[] pData = getFloatData(P);
        BLAS.getInstance().srotm(N,getFloatData(X),getBlasOffset(X),incX,getFloatData(Y),getBlasOffset(Y),incY,pData,getBlasOffset(P));
        setData(pData,P);
    }

    @Override
    protected void drotg(double a, double b, double c, double s) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {
        double[] pData = getDoubleData(P);
        BLAS.getInstance().drotmg(new doubleW(d1),new doubleW(d2),new doubleW(b1),b2,pData,getBlasOffset(P));
        setData(pData,P);
    }

    @Override
    protected void drot(int N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().drot(N,getDoubleData(X),getBlasOffset(X),incX,getDoubleData(Y),getBlasOffset(Y),incY,c,s);
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
        float[] data = getFloatData(X);
        BLAS.getInstance().sscal(N,alpha,data,getBlasOffset(X),incX);
        setData(data,X);
    }

    @Override
    protected void dscal(int N, double alpha, INDArray X, int incX) {
        double[] data = getDoubleData(X);
        BLAS.getInstance().dscal(N,alpha,data, BlasBufferUtil.getBlasOffset(X),incX);
        setData(data,X);
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
