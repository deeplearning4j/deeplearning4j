package org.nd4j.linalg.cpu.blas;

import com.github.fommil.netlib.BLAS;
import org.jblas.NativeBlas;
import org.nd4j.linalg.api.blas.impl.BaseLevel2;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.util.JblasComplex;
import org.nd4j.linalg.api.shape.Shape;

import static org.nd4j.linalg.api.blas.BlasBufferUtil.getBlasOffset;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.setData;

/**
 * @author Adam Gibson
 */
public class CpuLevel2 extends BaseLevel2 {
    @Override
    protected void sgemv(char order, char TransA, int M, int N, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().sgemv(String.valueOf(TransA)
                , M, N, alpha, getFloatData(A)
                , getBlasOffset(A), lda
                , getFloatData(X)
                , getBlasOffset(X)
                , incX
                , beta,
                yData
                , getBlasOffset(Y)
                , incY);
        setData(yData,Y);
    }

    @Override
    protected void sgbmv(char order, char TransA, int M, int N, int KL, int KU, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().sgbmv(String.valueOf(TransA),M,N,KL,KU,alpha,getFloatData(A),getBlasOffset(A),lda,getFloatData(X),getBlasOffset(X),incX,beta,yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void strmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        float[] xData = getFloatData(X);
        BLAS.getInstance().strmv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,getFloatData(A),getBlasOffset(A),lda,xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void stbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        float[] xData = getFloatData(X);
        BLAS.getInstance().stbmv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,K,getFloatData(A),getBlasOffset(A),lda,xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void stpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        float[] xData = getFloatData(X);
        BLAS.getInstance().stpmv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,getFloatData(Ap),getBlasOffset(Ap),xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void strsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        float[] xData = getFloatData(X);
        BLAS.getInstance().strsv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,getFloatData(A),getBlasOffset(A),lda,xData,getBlasOffset(X),incX);
        setData(xData, X);
    }

    @Override
    protected void stbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        float[] xData = getFloatData(X);
        BLAS.getInstance().stbsv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,K,getFloatData(A),getBlasOffset(A),lda,xData,getBlasOffset(X),incX);
        setData(xData,X);

    }

    @Override
    protected void stpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        float[] xData = getFloatData(X);
        BLAS.getInstance().stpsv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,getFloatData(Ap),getBlasOffset(Ap),xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void dgemv(char order, char TransA, int M, int N, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().dgemv(String.valueOf(TransA), M, N, alpha, getDoubleData(A), getBlasOffset(A), lda, getDoubleData(X), getBlasOffset(X), incX, beta, yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void dgbmv(char order, char TransA, int M, int N, int KL, int KU, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().dgbmv(String.valueOf(TransA),M,N,KL,KU,alpha,getDoubleData(A),getBlasOffset(A),lda,getDoubleData(X),getBlasOffset(X),incX,beta,yData,getBlasOffset(Y),incY);
        setData(yData,Y);

    }

    @Override
    protected void dtrmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        double[] xData = getDoubleData(X);
        BLAS.getInstance().dtrmv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N, getDoubleData(A),getBlasOffset(A),lda,xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void dtbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        double[] xData = getDoubleData(X);
        BLAS.getInstance().dtbmv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,K,getDoubleData(A),getBlasOffset(A),lda,xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void dtpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        double[] xData = getDoubleData(X);
        BLAS.getInstance().dtpmv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,getDoubleData(Ap),getBlasOffset(Ap),xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void dtrsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        double[] xData = getDoubleData(X);
        BLAS.getInstance().dtrsv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,getDoubleData(A),getBlasOffset(A),lda,xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void dtbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        double[] xData = getDoubleData(X);
        BLAS.getInstance().dtbsv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,K,getDoubleData(A),getBlasOffset(A),lda,xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void dtpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        double[] xData = getDoubleData(X);
        BLAS.getInstance().dtpsv(String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),N,getDoubleData(Ap),getBlasOffset(Ap),xData,getBlasOffset(X),incX);
        setData(xData,X);
    }

    @Override
    protected void cgemv(char order, char TransA, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        X = (IComplexNDArray) Shape.toOffsetZero(X);
        Y = (IComplexNDArray) Shape.toOffsetZero(Y);
        A = (IComplexNDArray) Shape.toOffsetZero(A);

        float[] yData = getFloatData(Y);
        NativeBlas.cgemv(TransA, M, N, JblasComplex.getComplexFloat(alpha), getFloatData(A), getBlasOffset(A), A.size(0), getFloatData(X), getBlasOffset(X), incX, JblasComplex.getComplexFloat(beta), yData, getBlasOffset(Y), incY);
        setData(yData,Y);
    }

    @Override
    protected void cgbmv(char order, char TransA, int M, int N, int KL, int KU, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctbmv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctpmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctbsv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctpsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgemv(char order, char TransA, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        X = (IComplexNDArray) Shape.toOffsetZero(X);
        Y = (IComplexNDArray) Shape.toOffsetZero(Y);
        A = (IComplexNDArray) Shape.toOffsetZero(A);

        double[] yData = getDoubleData(Y);
        NativeBlas.zgemv(TransA,M,N, JblasComplex.getComplexDouble(alpha),getDoubleData(A),getBlasOffset(A),A.size(0),getDoubleData(X),getBlasOffset(X),incX,JblasComplex.getComplexDouble(beta),yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void zgbmv(char order, char TransA, int M, int N, int KL, int KU, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztbmv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztpmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztbsv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztpsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssymv(char order, char Uplo, int N, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().ssymv(String.valueOf(Uplo),N,alpha,getFloatData(A),getBlasOffset(A),lda,getFloatData(X),getBlasOffset(X),incX,beta,yData,getBlasOffset(Y),incY);
        setData(yData,Y);

    }

    @Override
    protected void ssbmv(char order, char Uplo, int N, int K, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().ssbmv(String.valueOf(Uplo),N,K,alpha,getFloatData(A),getBlasOffset(A),lda,getFloatData(X),getBlasOffset(X),incX,beta,yData,getBlasOffset(Y),incY);
        setData(yData,Y);

    }

    @Override
    protected void sspmv(char order, char Uplo, int N, float alpha, INDArray Ap, INDArray X, int incX, float beta, INDArray Y, int incY) {
        float[] yData = getFloatData(Y);
        BLAS.getInstance().sspmv(String.valueOf(Uplo),N,alpha,getFloatData(Ap),getBlasOffset(Ap),getFloatData(X),getBlasOffset(X),incX,beta,yData,getBlasOffset(Y),incY);
        setData(yData,Y);

    }

    @Override
    protected void sger(char order, int M, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        float[] aData = getFloatData(A);
        BLAS.getInstance().sger(M, N, alpha, getFloatData(X), getBlasOffset(X), incX, getFloatData(Y), getBlasOffset(Y), incY, aData, getBlasOffset(A), lda);
        setData(aData,A);
    }

    @Override
    protected void ssyr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray A, int lda) {
        float[] aData = getFloatData(A);
        BLAS.getInstance().ssyr(String.valueOf(Uplo),N,alpha,getFloatData(X),getBlasOffset(X),incX,aData,getBlasOffset(A),lda);
        setData(aData,A);
    }

    @Override
    protected void sspr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Ap) {
        float[] apData = getFloatData(Ap);
        BLAS.getInstance().sspr(String.valueOf(Uplo),N,alpha,getFloatData(X),getBlasOffset(X),incX,apData,getBlasOffset(Ap));
        setData(apData,Ap);
    }

    @Override
    protected void ssyr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        float[] aData = getFloatData(A);
        BLAS.getInstance().ssyr2(String.valueOf(Uplo),N,alpha,getFloatData(X),getBlasOffset(X),incY,getFloatData(Y),getBlasOffset(Y),incY,aData,getBlasOffset(A),lda);
        setData(aData,A);
    }

    @Override
    protected void sspr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A) {
        float[] aData = getFloatData(A);
        BLAS.getInstance().sspr2(String.valueOf(Uplo),N,alpha,getFloatData(X),getBlasOffset(X),incX,getFloatData(Y),getBlasOffset(Y),incY,aData,getBlasOffset(A));
        setData(aData,A);
    }

    @Override
    protected void dsymv(char order, char Uplo, int N, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().dsymv(String.valueOf(Uplo),N,alpha,getDoubleData(A),getBlasOffset(A),lda,getDoubleData(X),getBlasOffset(X),incX,beta,yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void dsbmv(char order, char Uplo, int N, int K, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().dsbmv(String.valueOf(Uplo),N,K,alpha,getDoubleData(A),getBlasOffset(A),lda,getDoubleData(X),getBlasOffset(X),incX,beta,yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void dspmv(char order, char Uplo, int N, double alpha, INDArray Ap, INDArray X, int incX, double beta, INDArray Y, int incY) {
        double[] yData = getDoubleData(Y);
        BLAS.getInstance().dspmv(String.valueOf(Uplo),N,alpha,getDoubleData(Ap),getBlasOffset(Ap),getDoubleData(X),getBlasOffset(X),incX,beta,yData,getBlasOffset(Y),incY);
        setData(yData,Y);
    }

    @Override
    protected void dger(char order, int M, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        double[] aData = getDoubleData(A);
        BLAS.getInstance().dger(M, N, alpha, getDoubleData(X), getBlasOffset(X), incX, getDoubleData(Y), getBlasOffset(Y), incY, aData, getBlasOffset(A), lda);
        setData(aData,A);
    }

    @Override
    protected void dsyr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray A, int lda) {
        double[] aData = getDoubleData(A);
        BLAS.getInstance().dsyr(String.valueOf(Uplo),N,alpha,getDoubleData(X),getBlasOffset(X),incX,aData,getBlasOffset(A),lda);
        setData(aData,A);
    }

    @Override
    protected void dspr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Ap) {
        double[] apData = getDoubleData(Ap);
        BLAS.getInstance().dspr(String.valueOf(Uplo),N,alpha,getDoubleData(X),getBlasOffset(X),incX,apData,getBlasOffset(Ap));
        setData(apData,Ap);
    }

    @Override
    protected void dsyr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        double[] aData = getDoubleData(A);
        BLAS.getInstance().dsyr2(String.valueOf(Uplo),N,alpha,getDoubleData(X),getBlasOffset(X),incX,getDoubleData(Y),getBlasOffset(Y),incY,aData,getBlasOffset(A),lda);
        setData(aData,A);
    }

    @Override
    protected void dspr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A) {
        double[] aData = getDoubleData(A);
        BLAS.getInstance().dspr2(String.valueOf(Uplo),N,alpha,getDoubleData(X),getBlasOffset(X),incX,getDoubleData(Y),getBlasOffset(Y),incY,aData,getBlasOffset(A));
        setData(aData,A);
    }

    @Override
    protected void chemv(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chbmv(char order, char Uplo, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpmv(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray Ap, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgeru(char order, int M, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        X = (IComplexNDArray) Shape.toOffsetZero(X);
        Y = (IComplexNDArray) Shape.toOffsetZero(Y);
        A = (IComplexNDArray) Shape.toOffsetZero(A);

        float[] aData = getFloatData(A);
        NativeBlas.cgeru(M, N, JblasComplex.getComplexFloat(alpha), getFloatData(X), getBlasOffset(X), incX, getFloatData(Y), getBlasOffset(Y), incY, aData, getBlasOffset(A), lda);
        setData(aData,A);
    }

    @Override
    protected void cgerc(char order, int M, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher(char order, char Uplo, int N, float alpha, IComplexNDArray X, int incX, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpr(char order, char Uplo, int N, INDArray alpha, IComplexNDArray X, int incX, IComplexNDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher2(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpr2(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray Ap) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhemv(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhbmv(char order, char Uplo, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpmv(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray Ap, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgeru(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        X = (IComplexNDArray) Shape.toOffsetZero(X);
        Y = (IComplexNDArray) Shape.toOffsetZero(Y);
        A = (IComplexNDArray) Shape.toOffsetZero(A);

        double[] aData = getDoubleData(A);
        NativeBlas.zgeru(M,N,JblasComplex.getComplexDouble(alpha),getDoubleData(X),getBlasOffset(X),incX,getDoubleData(Y),getBlasOffset(Y),incY,aData,getBlasOffset(A),A.size(0));
        setData(aData,A);
    }

    @Override
    protected void zgerc(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        X = (IComplexNDArray) Shape.toOffsetZero(X);
        Y = (IComplexNDArray) Shape.toOffsetZero(Y);
        A = (IComplexNDArray) Shape.toOffsetZero(A);

        double[] aData = getDoubleData(A);
        NativeBlas.zgerc(M,N,JblasComplex.getComplexDouble(alpha),getDoubleData(X),getBlasOffset(X),incX,getDoubleData(Y),getBlasOffset(Y),incY,aData,getBlasOffset(A),lda);
        setData(aData, A);
    }

    @Override
    protected void zher(char order, char Uplo, int N, double alpha, IComplexNDArray X, int incX, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpr(char order, char Uplo, int N, INDArray alpha, IComplexNDArray X, int incX, IComplexNDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher2(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpr2(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray Ap) {
        throw new UnsupportedOperationException();

    }
}
