package org.nd4j.linalg.cpu.nativecpu.blas;


import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.nativeblas.Nd4jBlas;


/**
 *
 * A jblas delgation for level 3 routines
 *
 * @author Adam Gibson
 */
public class CpuLevel3 extends BaseLevel3 {
    private Nd4jBlas nd4jBlas = new Nd4jBlas();
    private static long[] DUMMY = new long[1];

    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        nd4jBlas.sgemm(DUMMY,'f',TransA,TransB,M,N,K,alpha,A.data().address(),lda,B.data().address(),ldb,beta,C.data().address(),ldc);
    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        nd4jBlas.ssymm(DUMMY,'f',Side,Uplo,M,N,alpha,A.data().address(),lda,B.data().address(),ldb,beta,C.data().address(),ldc);
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta, INDArray C, int ldc) {
        nd4jBlas.ssyrk(DUMMY,'f',Uplo,Trans,N,K,alpha,A.data().address(),lda,beta,C.data().address(),ldc);

    }

    @Override
    protected void ssyr2k(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, INDArray B, int ldb, float beta, INDArray C, int ldc) {
        nd4jBlas.ssyr2k(DUMMY,'f',Uplo,Trans,N,K,alpha,A.data().address(),lda,B.data().address(),ldb,beta,C.data().address(),ldc);

    }

    @Override
    protected void strmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb) {
        nd4jBlas.strmm(DUMMY,'f',Side,Uplo,TransA,Diag,M,N,alpha,A.data().address(),lda,B.data().address(),ldb);
    }

    @Override
    protected void strsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha, INDArray A, int lda, INDArray B, int ldb) {
        nd4jBlas.strsm(DUMMY,'f',Side,Uplo,TransA,Diag,M,N,alpha,A.data().address(),lda,B.data().address(),ldb);

    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        nd4jBlas.dgemm(DUMMY,'f',TransA,TransB,M,N,K,alpha,A.data().address(),lda,B.data().address(),ldb,beta,C.data().address(),ldc);
    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        nd4jBlas.dsymm(DUMMY,'f',Side,Uplo,M,N,alpha,A.data().address(),lda,B.data().address(),ldb,beta,C.data().address(),ldc);

    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, double beta, INDArray C, int ldc) {
        nd4jBlas.dsyrk(DUMMY,'f',Uplo,Trans,N,K,alpha,A.data().address(),lda,beta,C.data().address(),ldc);

    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda, INDArray B, int ldb, double beta, INDArray C, int ldc) {
        nd4jBlas.dsyr2k(DUMMY,'f',Uplo,Trans,N,K,alpha,A.data().address(),lda,B.data().address(),ldb,beta,C.data().address(),ldc);
    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        nd4jBlas.dtrmm(DUMMY,'f',Side,Uplo,TransA,Diag,M,N,alpha,A.data().address(),lda,B.data().address(),ldb);

    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha, INDArray A, int lda, INDArray B, int ldb) {
        nd4jBlas.dtrsm(DUMMY,'f',Side,Uplo,TransA,Diag,M,N,alpha,A.data().address(),lda,B.data().address(),ldb);
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
