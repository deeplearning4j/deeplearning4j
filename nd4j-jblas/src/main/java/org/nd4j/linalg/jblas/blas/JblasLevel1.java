package org.nd4j.linalg.jblas.blas;

import org.jblas.JavaBlas;
import org.jblas.NativeBlas;
import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jblas.util.JblasComplex;
import org.nd4j.linalg.api.shape.Shape;

import static org.nd4j.linalg.api.blas.BlasBufferUtil.getBlasOffset;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.getDoubleData;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.setData;


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
        return JavaBlas.rdot(N,xData,xOffset,incX,yData,yOffset,incY);
    }

    @Override
    protected double ddot(int N, INDArray X, int incX, INDArray Y, int incY) {
        double[] xData = getDoubleData(X);
        double[] yData = getDoubleData(Y);
        int xOffset = getBlasOffset(X);
        int yOffset = getBlasOffset(Y);
        return JavaBlas.rdot(N,xData,xOffset,incX,yData,yOffset,incY);
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
        X = Shape.toOffsetZero(X);

        float[] data = getFloatData(X);
        return NativeBlas.snrm2(N,data,getBlasOffset(X),incX);
    }

    @Override
    protected float sasum(int N, INDArray X, int incX) {
        X = Shape.toOffsetZero(X);

        float[] data = getFloatData(X);
        return NativeBlas.sasum(N, data, getBlasOffset(X), incX);
    }

    @Override
    protected double dnrm2(int N, INDArray X, int incX) {
        X = Shape.toOffsetZero(X);

        double[] data = getDoubleData(X);
        return NativeBlas.dnrm2(N, data, getBlasOffset(X), incX);
    }

    @Override
    protected double dasum(int N, INDArray X, int incX) {
        X = Shape.toOffsetZero(X);

        double[] data = getDoubleData(X);
        return NativeBlas.dasum(N, data, getBlasOffset(X), incX);
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
        return NativeBlas.dznrm2(N,getDoubleData(X),getBlasOffset(X),incX);

    }

    @Override
    protected double dzasum(int N, IComplexNDArray X, int incX) {
       return NativeBlas.dzasum(N,getDoubleData(X),getBlasOffset(X),incX) - 1;
    }

    @Override
    protected int isamax(int N, INDArray X, int incX) {
        return NativeBlas.isamax(N, getFloatData(X), getBlasOffset(X), incX) - 1;
    }

    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return NativeBlas.idamax(N, getDoubleData(X), getBlasOffset(X), incX) -1;
    }

    @Override
    protected int icamax(int N, IComplexNDArray X, int incX) {
        return NativeBlas.icamax(N, getFloatData(X), getBlasOffset(X), incX) - 1;
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
        float[] dataToSret = getFloatData(Y);
        NativeBlas.scopy(N,getFloatData(X),getBlasOffset(X),incX,dataToSret,getBlasOffset(Y),incY);
        setData(dataToSret, Y);
    }

    @Override
    protected void scopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        NativeBlas.scopy(n,getFloatData(x),offsetX,incrX,getFloatData(y),offsetY,incrY);
    }

    @Override
    protected void saxpy(int N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        float[] dataToSret = getFloatData(Y);
        JavaBlas.raxpy(N, alpha, getFloatData(X), getBlasOffset(X), incX, dataToSret, getBlasOffset(Y), incY);
        setData(dataToSret,Y);
    }

    @Override
    protected void saxpy( int N, float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException("offset operations not supported");
    }

    @Override
    protected void dswap(int N, INDArray X, int incX, INDArray Y, int incY) {
        X = Shape.toOffsetZero(X);
        NativeBlas.dswap(N, getDoubleData(X), getBlasOffset(X), incX, getDoubleData(Y), getBlasOffset(Y), incY);
    }

    @Override
    protected void dcopy(int N, INDArray X, int incX, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        JavaBlas.rcopy(N,getDoubleData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY);
        setData(yData, Y);
    }

    @Override
    protected void dcopy(int n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        JavaBlas.rcopy(n,getDoubleData(x),offsetX,incrX,getDoubleData(y),offsetY,incrY);
    }

    @Override
    protected void daxpy(int N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        X = Shape.toOffsetZero(X);
        double[] yData = getDoubleData(Y);
        JavaBlas.raxpy(N, alpha, getDoubleData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void daxpy( int N, double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY ){
        throw new UnsupportedOperationException("offset operations not supported");
    }

    @Override
    protected void cswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        NativeBlas.cswap(N,getFloatData(X),getBlasOffset(X),incX,yData,getBlasOffset(Y),incY);
        setData(yData,X);
    }

    @Override
    protected void ccopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        NativeBlas.ccopy(N, getFloatData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void caxpy(int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        X = (IComplexNDArray) Shape.toOffsetZero(X);
        float[] yData = getFloatData(Y);
        NativeBlas.caxpy(N, JblasComplex.getComplexFloat(alpha), getFloatData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void zswap(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        NativeBlas.zswap(N, getDoubleData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData, Y);
    }

    @Override
    protected void zcopy(int N, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        NativeBlas.zcopy(N, getDoubleData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void zaxpy(int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY) {
        X = (IComplexNDArray) Shape.toOffsetZero(X);
        double[] yData = getDoubleData(Y);
        NativeBlas.zaxpy(N, JblasComplex.getComplexDouble(alpha), getDoubleData(X), getBlasOffset(X), incX, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void srotg(float a, float b, float c, float s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotmg(float d1, float d2, float b1, float b2, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srot(int N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void srotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drotg(double a, double b, double c, double s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drot(int N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        throw new UnsupportedOperationException();

    }


    @Override
    protected void drotm(int N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sscal(int N, float alpha, INDArray X, int incX) {
        float[] xData = getFloatData(X);
        NativeBlas.sscal(N,alpha,xData,getBlasOffset(X),incX);
        setData(xData, X);
    }

    @Override
    protected void dscal(int N, double alpha, INDArray X, int incX) {
        double[] xData = getDoubleData(X);
        NativeBlas.dscal(N, alpha, xData, getBlasOffset(X), incX);
        setData(xData,X);
    }

    @Override
    protected void cscal(int N, IComplexFloat alpha, IComplexNDArray X, int incX) {
        float[] xData = getFloatData(X);
        NativeBlas.cscal(N, JblasComplex.getComplexFloat(alpha), xData, getBlasOffset(X), incX);
        setData(xData, X);
    }

    @Override
    protected void zscal(int N, IComplexDouble alpha, IComplexNDArray X, int incX) {
        double[] xData = getDoubleData(X);
        NativeBlas.zscal(N, JblasComplex.getComplexDouble(alpha), xData, getBlasOffset(X), incX);
        setData(xData, X);
    }

    @Override
    protected void csscal(int N, float alpha, IComplexNDArray X, int incX) {
        float[] xData = getFloatData(X);
        NativeBlas.csscal(N, alpha, xData, getBlasOffset(X), incX);
        setData(xData, X);
    }

    @Override
    protected void zdscal(int N, double alpha, IComplexNDArray X, int incX) {
        double[] xData = getDoubleData(X);
        NativeBlas.zdscal(N, alpha, xData, getBlasOffset(X), incX);
        setData(xData, X);
    }

    @Override
    public boolean supportsDataBufferL1Ops(){
        return false;
    }
}
