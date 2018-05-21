package org.nd4j.linalg.cpu.nativecpu.blas;


import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.blas.impl.BaseLevel2;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jBlas;

import static org.bytedeco.javacpp.openblas.*;
import static org.nd4j.linalg.cpu.nativecpu.blas.CpuBlas.*;


/**
 * @author Adam Gibson
 */
public class CpuLevel2 extends BaseLevel2 {
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();

    @Override
    protected void sgemv(char order, char TransA, int M, int N, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        cblas_sgemv(convertOrder('f'), convertTranspose(TransA), M, N, alpha, (FloatPointer) A.data().addressPointer(),
                        lda, (FloatPointer) X.data().addressPointer(), incX, beta,
                        (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void sgbmv(char order, char TransA, int M, int N, int KL, int KU, float alpha, INDArray A, int lda,
                    INDArray X, int incX, float beta, INDArray Y, int incY) {
        cblas_sgbmv(convertOrder('f'), convertTranspose(TransA), M, N, KL, KU, alpha,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) X.data().addressPointer(), incX,
                        beta, (FloatPointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void strmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void stbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        cblas_stbmv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void stpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        cblas_stpmv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N,
                        (FloatPointer) Ap.data().addressPointer(), (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void strsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        cblas_strsv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void stbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        cblas_stbsv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) X.data().addressPointer(), incX);

    }

    @Override
    protected void stpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        cblas_stpsv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N,
                        (FloatPointer) Ap.data().addressPointer(), (FloatPointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void dgemv(char order, char TransA, int M, int N, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        cblas_dgemv(convertOrder('f'), convertTranspose(TransA), M, N, alpha, (DoublePointer) A.data().addressPointer(),
                        lda, (DoublePointer) X.data().addressPointer(), incX, beta,
                        (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dgbmv(char order, char TransA, int M, int N, int KL, int KU, double alpha, INDArray A, int lda,
                    INDArray X, int incX, double beta, INDArray Y, int incY) {
        cblas_dgbmv(convertOrder('f'), convertTranspose(TransA), M, N, KL, KU, alpha,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(), incX,
                        beta, (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dtrmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        cblas_dtrmv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(),
                        incX);
    }

    @Override
    protected void dtbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        cblas_dtbmv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(),
                        incX);
    }

    @Override
    protected void dtpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        cblas_dtpmv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N,
                        (DoublePointer) Ap.data().addressPointer(), (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void dtrsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X,
                    int incX) {
        cblas_dtrsv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(),
                        incX);
    }

    @Override
    protected void dtbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X,
                    int incX) {
        cblas_dtbsv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) X.data().addressPointer(),
                        incX);
    }

    @Override
    protected void dtpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        cblas_dtpsv(convertOrder('f'), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N,
                        (DoublePointer) Ap.data().addressPointer(), (DoublePointer) X.data().addressPointer(), incX);
    }

    @Override
    protected void cgemv(char order, char TransA, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void cgbmv(char order, char TransA, int M, int N, int KL, int KU, IComplexFloat alpha, IComplexNDArray A,
                    int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctbmv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctpmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctbsv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctpsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgemv(char order, char TransA, int M, int N, IComplexDouble alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void zgbmv(char order, char TransA, int M, int N, int KL, int KU, IComplexDouble alpha, IComplexNDArray A,
                    int lda, IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztbmv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztpmv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztbsv(char order, char Uplo, char TransA, char Diag, int N, int K, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztpsv(char order, char Uplo, char TransA, char Diag, int N, IComplexNDArray Ap, IComplexNDArray X,
                    int incX) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ssymv(char order, char Uplo, int N, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        cblas_ssymv(convertOrder('f'), convertUplo(Uplo), N, alpha, (FloatPointer) A.data().addressPointer(), lda,
                        (FloatPointer) X.data().addressPointer(), incX, beta, (FloatPointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected void ssbmv(char order, char Uplo, int N, int K, float alpha, INDArray A, int lda, INDArray X, int incX,
                    float beta, INDArray Y, int incY) {
        cblas_ssbmv(convertOrder('f'), convertUplo(Uplo), N, K, alpha, (FloatPointer) A.data().addressPointer(), lda,
                        (FloatPointer) X.data().addressPointer(), incX, beta, (FloatPointer) Y.data().addressPointer(),
                        incY);
    }

    @Override
    protected void sspmv(char order, char Uplo, int N, float alpha, INDArray Ap, INDArray X, int incX, float beta,
                    INDArray Y, int incY) {
        cblas_sspmv(convertOrder('f'), convertUplo(Uplo), N, alpha, (FloatPointer) Ap.data().addressPointer(),
                        (FloatPointer) X.data().addressPointer(), incX, beta, (FloatPointer) Y.data().addressPointer(),
                        incY);

    }

    @Override
    protected void sger(char order, int M, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A,
                    int lda) {
        cblas_sger(convertOrder('f'), M, N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Y.data().addressPointer(), incY, (FloatPointer) A.data().addressPointer(), lda);
    }

    @Override
    protected void ssyr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray A, int lda) {
        cblas_ssyr(convertOrder('f'), convertUplo(Uplo), N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) A.data().addressPointer(), lda);
    }

    @Override
    protected void sspr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Ap) {
        cblas_sspr(convertOrder('f'), convertUplo(Uplo), N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Ap.data().addressPointer());
    }

    @Override
    protected void ssyr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda) {
        cblas_ssyr2(convertOrder('f'), convertUplo(Uplo), N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Y.data().addressPointer(), incY, (FloatPointer) A.data().addressPointer(), lda);
    }

    @Override
    protected void sspr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A) {
        cblas_sspr2(convertOrder('f'), convertUplo(Uplo), N, alpha, (FloatPointer) X.data().addressPointer(), incX,
                        (FloatPointer) Y.data().addressPointer(), incY, (FloatPointer) A.data().addressPointer());
    }

    @Override
    protected void dsymv(char order, char Uplo, int N, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        cblas_dsymv(convertOrder('f'), convertUplo(Uplo), N, alpha, (DoublePointer) A.data().addressPointer(), lda,
                        (DoublePointer) X.data().addressPointer(), incX, beta,
                        (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dsbmv(char order, char Uplo, int N, int K, double alpha, INDArray A, int lda, INDArray X, int incX,
                    double beta, INDArray Y, int incY) {
        cblas_dsbmv(convertOrder('f'), convertUplo(Uplo), N, K, alpha, (DoublePointer) A.data().addressPointer(), lda,
                        (DoublePointer) X.data().addressPointer(), incX, beta,
                        (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dspmv(char order, char Uplo, int N, double alpha, INDArray Ap, INDArray X, int incX, double beta,
                    INDArray Y, int incY) {
        cblas_dspmv(convertOrder('f'), convertUplo(Uplo), N, alpha, (DoublePointer) Ap.data().addressPointer(),
                        (DoublePointer) X.data().addressPointer(), incX, beta,
                        (DoublePointer) Y.data().addressPointer(), incY);
    }

    @Override
    protected void dger(char order, int M, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A,
                    int lda) {
        cblas_dger(convertOrder('f'), M, N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Y.data().addressPointer(), incY, (DoublePointer) A.data().addressPointer(),
                        lda);
    }

    @Override
    protected void dsyr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray A, int lda) {
        cblas_dsyr(convertOrder('f'), convertUplo(Uplo), N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) A.data().addressPointer(), lda);
    }

    @Override
    protected void dspr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Ap) {
        cblas_dspr(convertOrder('f'), convertUplo(Uplo), N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Ap.data().addressPointer());
    }

    @Override
    protected void dsyr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A, int lda) {
        cblas_dsyr2(convertOrder('f'), convertUplo(Uplo), N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Y.data().addressPointer(), incY, (DoublePointer) A.data().addressPointer(),
                        lda);
    }

    @Override
    protected void dspr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY,
                    INDArray A) {
        cblas_dspr2(convertOrder('f'), convertUplo(Uplo), N, alpha, (DoublePointer) X.data().addressPointer(), incX,
                        (DoublePointer) Y.data().addressPointer(), incY, (DoublePointer) A.data().addressPointer());
    }

    @Override
    protected void chemv(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chbmv(char order, char Uplo, int N, int K, IComplexFloat alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpmv(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray Ap, IComplexNDArray X,
                    int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cgeru(char order, int M, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y,
                    int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void cgerc(char order, int M, int N, IComplexFloat alpha, IComplexNDArray X, int incX, IComplexNDArray Y,
                    int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher(char order, char Uplo, int N, float alpha, IComplexNDArray X, int incX, IComplexNDArray A,
                    int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpr(char order, char Uplo, int N, INDArray alpha, IComplexNDArray X, int incX, IComplexNDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher2(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray X, int incX,
                    IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chpr2(char order, char Uplo, int N, IComplexFloat alpha, IComplexNDArray X, int incX,
                    IComplexNDArray Y, int incY, IComplexNDArray Ap) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhemv(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhbmv(char order, char Uplo, int N, int K, IComplexDouble alpha, IComplexNDArray A, int lda,
                    IComplexNDArray X, int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpmv(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray Ap, IComplexNDArray X,
                    int incX, IComplexDouble beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgeru(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y,
                    int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgerc(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y,
                    int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher(char order, char Uplo, int N, double alpha, IComplexNDArray X, int incX, IComplexNDArray A,
                    int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpr(char order, char Uplo, int N, INDArray alpha, IComplexNDArray X, int incX, IComplexNDArray A) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher2(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray X, int incX,
                    IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhpr2(char order, char Uplo, int N, IComplexDouble alpha, IComplexNDArray X, int incX,
                    IComplexNDArray Y, int incY, IComplexNDArray Ap) {
        throw new UnsupportedOperationException();

    }
}
