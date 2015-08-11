package org.nd4j.linalg.netlib.blas;

import com.github.fommil.netlib.BLAS;
import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

import static org.nd4j.linalg.api.blas.BlasBufferUtil.*;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.getDoubleData;
import static org.nd4j.linalg.api.blas.BlasBufferUtil.getFloatData;


/**
 * @author Adam Gibson
 */
public class NetlibLevel3 extends BaseLevel3 {
    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        A = Shape.toOffsetZero(A);
        B = Shape.toOffsetZero(B);

        float[] cData = getFloatData(C);
        BLAS.getInstance().sgemm(String.valueOf(TransA),String.valueOf(TransB),M,N,K,alpha,getFloatData(A),getBlasOffset(A),lda,getFloatData(B),getBlasOffset(B),ldb,beta,cData,getBlasOffset(C),ldc);
        setData(cData,C);
    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        float[] cData = getFloatData(C);
        BLAS.getInstance().ssymm(String.valueOf(Side),String.valueOf(Uplo),M,N,alpha,getFloatData(A),getBlasOffset(A),lda,getFloatData(B),getBlasOffset(B),ldb,beta,cData,getBlasOffset(C),ldc);
        setData(cData,C);
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta, INDArray C, int ldc) {
        float[] cData = getFloatData(C);
        BLAS.getInstance().ssyrk(String.valueOf(Uplo),String.valueOf(Trans),N,K,alpha,getFloatData(A),getBlasOffset(A),lda,beta,cData,getBlasOffset(C),ldc);
        setData(cData,C);
    }

    @Override
    protected void ssyr2k(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        float[] cData = getFloatData(C);
        BLAS.getInstance().ssyr2k(String.valueOf(Uplo),String.valueOf(Trans),N,K,alpha,getFloatData(A),getBlasOffset(A),lda,getFloatData(B),getBlasOffset(B),ldb,beta,cData,getBlasOffset(C),ldc);;
        setData(cData, C);
    }

    @Override
    protected void strmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb) {
        float[] bData = getFloatData(B);
        BLAS.getInstance().strmm(String.valueOf(Side),String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),M,N,alpha,getFloatData(A),getBlasOffset(A),lda,getFloatData(B),getBlasOffset(B),ldb);
        setData(bData,B);
    }

    @Override
    protected void strsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb) {
        float[] bData = getFloatData(B);
        BLAS.getInstance().strsm(String.valueOf(Side),String.valueOf(Uplo),String.valueOf(TransA),String.valueOf(Diag),M,N,alpha,getFloatData(A),getBlasOffset(A),lda,bData,getBlasOffset(B),ldb);
        setData(bData,B);
    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        A = Shape.toOffsetZero(A);
        B = Shape.toOffsetZero(B);

        double[] cData = getDoubleData(C);
        BLAS.getInstance().dgemm(
                String.valueOf(TransA)
                , String.valueOf(TransB)
                , M
                , N
                , K
                , alpha
                , getDoubleData(A)
                , getBlasOffset(A)
                , lda
                , getDoubleData(B)
                , getBlasOffset(B)
                , ldb
                , beta
                , cData
                , getBlasOffset(C),
                ldc);
        setData(cData,C);
    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        double[] cData = getDoubleData(C);
        BLAS.getInstance().dsymm(String.valueOf(Side), String.valueOf(Uplo), M, N, alpha, getDoubleData(A), getBlasOffset(A), lda, getDoubleData(B), getBlasOffset(B), ldb, beta, cData, getBlasOffset(C), ldc);
        setData(cData,C);
    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, double beta, INDArray C, int ldc) {
        double[] cData = getDoubleData(C);
        BLAS.getInstance().dsyrk(String.valueOf(Uplo), String.valueOf(Trans), N, K, alpha, getDoubleData(A), getBlasOffset(A), lda, beta, cData, getBlasOffset(C), ldc);
        setData(cData,C);
    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        double[] cData = getDoubleData(C);
        BLAS.getInstance().dsyr2k(String.valueOf(Uplo), String.valueOf(Trans), N, K, alpha, getDoubleData(A), getBlasOffset(A), lda, getDoubleData(B), getBlasOffset(B), ldb, beta, cData, getBlasOffset(C), ldc);;
        setData(cData, C);
    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        double[] bData = getDoubleData(B);
        BLAS.getInstance().dtrmm(String.valueOf(Side), String.valueOf(Uplo), String.valueOf(TransA), String.valueOf(Diag), M, N, alpha, getDoubleData(A), getBlasOffset(A), lda, getDoubleData(B), getBlasOffset(B), ldb);
        setData(bData,B);
    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        double[] bData = getDoubleData(B);
        BLAS.getInstance().dtrsm(String.valueOf(Side), String.valueOf(Uplo), String.valueOf(TransA), String.valueOf(Diag), M, N, alpha, getDoubleData(A), getBlasOffset(A), lda, bData, getBlasOffset(B), ldb);
        setData(bData,B);
    }

    @Override
    protected void cgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csymm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csyrk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zsymm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zsyrk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zsyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chemm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cherk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhemm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zherk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();
    }
}
