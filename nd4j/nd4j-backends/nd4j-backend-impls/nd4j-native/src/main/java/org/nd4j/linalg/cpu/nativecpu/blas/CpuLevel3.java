package org.nd4j.linalg.cpu.nativecpu.blas;


import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.impl.AggregateGEMM;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jBlas;

import static org.bytedeco.javacpp.openblas.*;
import static org.nd4j.linalg.cpu.nativecpu.blas.CpuBlas.*;


/**
 *
 * A jblas delgation for level 3 routines
 *
 * @author Adam Gibson
 */
public class CpuLevel3 extends BaseLevel3 {
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();

    @Override
    protected void hgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda,
                    INDArray B, int ldb, float beta, INDArray C, int ldc) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda,
                    INDArray B, int ldb, float beta, INDArray C, int ldc) {
        if (!Nd4j.isFallbackModeEnabled()) {
            cblas_sgemm(convertOrder('f'), convertTranspose(TransA), convertTranspose(TransB), M, N, K, alpha,
                            (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(),
                            ldb, beta, (FloatPointer) C.data().addressPointer(), ldc);
        } else {
            Nd4j.getExecutioner()
                            .exec(new AggregateGEMM('f', TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
        }
    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B,
                    int ldb, float beta, INDArray C, int ldc) {
        cblas_ssymm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), M, N, alpha,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(), ldb,
                        beta, (FloatPointer) C.data().addressPointer(), ldc);
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta,
                    INDArray C, int ldc) {
        cblas_ssyrk(convertOrder('f'), convertUplo(Uplo), convertTranspose(Trans), N, K, alpha,
                        (FloatPointer) A.data().addressPointer(), lda, beta, (FloatPointer) C.data().addressPointer(),
                        ldc);
    }

    @Override
    protected void ssyr2k(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, INDArray B,
                    int ldb, float beta, INDArray C, int ldc) {
        cblas_ssyr2k(convertOrder('f'), convertUplo(Uplo), convertTranspose(Trans), N, K, alpha,
                        (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(), ldb,
                        beta, (FloatPointer) C.data().addressPointer(), ldc);
    }

    @Override
    protected void strmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        cblas_strmm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), convertTranspose(TransA), Diag, M, N,
                        alpha, (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(),
                        ldb);
    }

    @Override
    protected void strsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        cblas_strsm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), convertTranspose(TransA), Diag, M, N,
                        alpha, (FloatPointer) A.data().addressPointer(), lda, (FloatPointer) B.data().addressPointer(),
                        ldb);
    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda,
                    INDArray B, int ldb, double beta, INDArray C, int ldc) {
        if (!Nd4j.isFallbackModeEnabled()) {
            cblas_dgemm(convertOrder('f'), convertTranspose(TransA), convertTranspose(TransB), M, N, K, alpha,
                            (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) B.data().addressPointer(),
                            ldb, beta, (DoublePointer) C.data().addressPointer(), ldc);
        } else {
            Nd4j.getExecutioner()
                            .exec(new AggregateGEMM('f', TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
        }
    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B,
                    int ldb, double beta, INDArray C, int ldc) {
        cblas_dsymm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), M, N, alpha,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) B.data().addressPointer(), ldb,
                        beta, (DoublePointer) C.data().addressPointer(), ldc);
    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda,
                    double beta, INDArray C, int ldc) {
        cblas_dsyrk(convertOrder('f'), convertUplo(Uplo), convertTranspose(Trans), N, K, alpha,
                        (DoublePointer) A.data().addressPointer(), lda, beta, (DoublePointer) C.data().addressPointer(),
                        ldc);
    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda,
                    INDArray B, int ldb, double beta, INDArray C, int ldc) {
        cblas_dsyr2k(convertOrder('f'), convertUplo(Uplo), convertTranspose(Trans), N, K, alpha,
                        (DoublePointer) A.data().addressPointer(), lda, (DoublePointer) B.data().addressPointer(), ldb,
                        beta, (DoublePointer) C.data().addressPointer(), ldc);
    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        cblas_dtrmm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), convertTranspose(TransA), Diag, M, N,
                        alpha, (DoublePointer) A.data().addressPointer(), lda,
                        (DoublePointer) B.data().addressPointer(), ldb);
    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        cblas_dtrsm(convertOrder('f'), convertSide(Side), convertUplo(Uplo), convertTranspose(TransA), Diag, M, N,
                        alpha, (DoublePointer) A.data().addressPointer(), lda,
                        (DoublePointer) B.data().addressPointer(), ldb);
    }

    @Override
    protected void cgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C,
                    int ldc) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void csymm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A,
                    int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csyrk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A,
                    int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void csyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A,
                    int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ctrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexFloat alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgemm(char Order, char TransA, char TransB, int M, int N, int K, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C,
                    int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zsymm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A,
                    int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zsyrk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A,
                    int lda, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zsyr2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A,
                    int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void ztrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, IComplexDouble alpha,
                    IComplexNDArray A, int lda, IComplexNDArray B, int ldb) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void chemm(char Order, char Side, char Uplo, int M, int N, IComplexFloat alpha, IComplexNDArray A,
                    int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cherk(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A,
                    int lda, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void cher2k(char Order, char Uplo, char Trans, int N, int K, IComplexFloat alpha, IComplexNDArray A,
                    int lda, IComplexNDArray B, int ldb, IComplexFloat beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zhemm(char Order, char Side, char Uplo, int M, int N, IComplexDouble alpha, IComplexNDArray A,
                    int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zherk(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A,
                    int lda, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zher2k(char Order, char Uplo, char Trans, int N, int K, IComplexDouble alpha, IComplexNDArray A,
                    int lda, IComplexNDArray B, int ldb, IComplexDouble beta, IComplexNDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }
}
