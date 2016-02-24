package org.nd4j.linalg.cpu.nativecpu.blas;


import org.nd4j.linalg.api.blas.impl.BaseLevel2;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.nativeblas.Nd4jBlas;


/**
 * @author Adam Gibson
 */
public class CpuLevel2 extends BaseLevel2 {
    private Nd4jBlas nd4jBlas = new Nd4jBlas();
    private static long[] DUMMY = new long[1];

    @Override
    protected void sgemv(char order, char TransA, int M, int N, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        nd4jBlas.sgemv(DUMMY,'f',TransA,M,N,alpha,A.data().address(),lda,X.data().address(),incX,beta,Y.data().address(),incY);
    }

    @Override
    protected void sgbmv(char order, char TransA, int M, int N, int KL, int KU, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
        nd4jBlas.sgbmv(DUMMY,'f',TransA,M,N,KL,KU,alpha,A.data().address(),lda,X.data().address(),incX,beta,Y.data().address(),incY);
    }

    @Override
    protected void strmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void stbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        nd4jBlas.stbmv(DUMMY,'f',Uplo,TransA,Diag,N,K,A.data().address(),lda,X.data().address(),incX);
    }

    @Override
    protected void stpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        nd4jBlas.stpmv(DUMMY,'f',Uplo,TransA,Diag,N,Ap.data().address(),X.data().address(),incX);
    }

    @Override
    protected void strsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        nd4jBlas.strsv(DUMMY,'f',Uplo,TransA,Diag,N,A.data().address(),lda,X.data().address(),incX);
    }

    @Override
    protected void stbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        nd4jBlas.stbsv(DUMMY,'f',Uplo,TransA,Diag,N,K,A.data().address(),lda,X.data().address(),incX);

    }

    @Override
    protected void stpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        nd4jBlas.stpsv(DUMMY,'f',Uplo,TransA,Diag,N, Ap.data().address(),X.data().address(),incX);
    }

    @Override
    protected void dgemv(char order, char TransA, int M, int N, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        nd4jBlas.dgemv(DUMMY,'f',TransA,M,N,alpha,A.data().address(),lda,X.data().address(),incX,beta,Y.data().address(),incY);
    }

    @Override
    protected void dgbmv(char order, char TransA, int M, int N, int KL, int KU, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        nd4jBlas.dgbmv(DUMMY,'f',TransA,M,N,KL,KU,alpha,A.data().address(),lda,X.data().address(),incX,beta,Y.data().address(),incY);

    }

    @Override
    protected void dtrmv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        nd4jBlas.dtrmv(DUMMY,'f',Uplo,TransA,Diag,N,0.0,A.data().address(),lda,X.data().address(),incX);
    }

    @Override
    protected void dtbmv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        nd4jBlas.dtbmv(DUMMY,'f',Uplo,TransA,Diag,N,K,A.data().address(),lda,X.data().address(),incX);
    }

    @Override
    protected void dtpmv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        nd4jBlas.dtpmv(DUMMY,'f',Uplo,TransA,Diag,N,Ap.data().address(),X.data().address(),incX);
    }

    @Override
    protected void dtrsv(char order, char Uplo, char TransA, char Diag, int N, INDArray A, int lda, INDArray X, int incX) {
        nd4jBlas.dtrsv(DUMMY,'f',Uplo,TransA,Diag,N,A.data().address(),lda,X.data().address(),incX);
    }

    @Override
    protected void dtbsv(char order, char Uplo, char TransA, char Diag, int N, int K, INDArray A, int lda, INDArray X, int incX) {
        nd4jBlas.dtbsv(DUMMY,'f',Uplo,TransA,Diag,N,K,A.data().address(),lda,X.data().address(),incX);
    }

    @Override
    protected void dtpsv(char order, char Uplo, char TransA, char Diag, int N, INDArray Ap, INDArray X, int incX) {
        nd4jBlas.dtpsv(DUMMY,'f',Uplo,TransA,Diag,N,Ap.data().address(),X.data().address(),incX);
    }

    @Override
    protected void cgemv(char order, char TransA, int M, int N, IComplexFloat alpha, IComplexNDArray A, int lda, IComplexNDArray X, int incX, IComplexFloat beta, IComplexNDArray Y, int incY) {
        throw new UnsupportedOperationException();
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
        throw new UnsupportedOperationException();
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
        nd4jBlas.ssymv(DUMMY,'f',Uplo,N,alpha,A.data().address(),lda,X.data().address(),incX,beta,Y.data().address(),incY);
    }

    @Override
    protected void ssbmv(char order, char Uplo, int N, int K, float alpha, INDArray A, int lda, INDArray X, int incX, float beta, INDArray Y, int incY) {
      nd4jBlas.ssbmv(DUMMY,'f',Uplo,N,K,alpha,A.data().address(),lda,X.data().address(),incX,beta,Y.data().address(),incY);
    }

    @Override
    protected void sspmv(char order, char Uplo, int N, float alpha, INDArray Ap, INDArray X, int incX, float beta, INDArray Y, int incY) {
        nd4jBlas.sspmv(DUMMY,'f',Uplo,N,alpha,Ap.data().address(),X.data().address(),incX,beta,Y.data().address(),incY);

    }

    @Override
    protected void sger(char order, int M, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        nd4jBlas.sger(DUMMY,'f',M,N,alpha,X.data().address(),incX,Y.data().address(),incY,A.data().address(),lda);
    }

    @Override
    protected void ssyr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray A, int lda) {
        nd4jBlas.ssyr(DUMMY,'f',Uplo,N,alpha,X.data().address(),incX,A.data().address(),lda);
    }

    @Override
    protected void sspr(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Ap) {
        nd4jBlas.sspr(DUMMY,'f',Uplo,N,alpha,X.data().address(),incX,Ap.data().address());
    }

    @Override
    protected void ssyr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        nd4jBlas.ssyr2(DUMMY,'f',Uplo,N,alpha,X.data().address(),incX,Y.data().address(),incY,A.data().address(),lda);
    }

    @Override
    protected void sspr2(char order, char Uplo, int N, float alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A) {
        nd4jBlas.sspr2(DUMMY,'f',Uplo,N,alpha,X.data().address(),incX,Y.data().address(),incY,A.data().address());
    }

    @Override
    protected void dsymv(char order, char Uplo, int N, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        nd4jBlas.dsymv(DUMMY,'f',Uplo,N,alpha,A.data().address(),lda,X.data().address(),incX,beta,Y.data().address(),incY);
    }

    @Override
    protected void dsbmv(char order, char Uplo, int N, int K, double alpha, INDArray A, int lda, INDArray X, int incX, double beta, INDArray Y, int incY) {
        nd4jBlas.dsbmv(DUMMY,'f',Uplo,N,K,alpha,A.data().address(),lda,X.data().address(),incX,beta,Y.data().address(),incY);
    }

    @Override
    protected void dspmv(char order, char Uplo, int N, double alpha, INDArray Ap, INDArray X, int incX, double beta, INDArray Y, int incY) {
        nd4jBlas.dspmv(DUMMY,'f',Uplo,N,alpha,Ap.data().address(),X.data().address(),incX,beta,Y.data().address(),incY);
    }

    @Override
    protected void dger(char order, int M, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        nd4jBlas.dger(DUMMY,'f',M,N,alpha,X.data().address(),incX,Y.data().address(),incY,A.data().address(),lda);
        
    }

    @Override
    protected void dsyr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray A, int lda) {
        nd4jBlas.dsyr(DUMMY,'f',Uplo,N,alpha,X.data().address(),incX,A.data().address(),lda);
    }

    @Override
    protected void dspr(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Ap) {
        nd4jBlas.dspr(DUMMY,'f',Uplo,N,alpha,X.data().address(),incX,Ap.data().address());
    }

    @Override
    protected void dsyr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A, int lda) {
        nd4jBlas.dsyr2(DUMMY,'f',Uplo,N,alpha,X.data().address(),incX,Y.data().address(),incY,A.data().address(),lda);
    }

    @Override
    protected void dspr2(char order, char Uplo, int N, double alpha, INDArray X, int incX, INDArray Y, int incY, INDArray A) {
        nd4jBlas.dspr2(DUMMY,'f',Uplo,N,alpha,X.data().address(),incX,Y.data().address(),incY,A.data().address());
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
        throw new UnsupportedOperationException();
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
        throw new UnsupportedOperationException();

    }

    @Override
    protected void zgerc(char order, int M, int N, IComplexDouble alpha, IComplexNDArray X, int incX, IComplexNDArray Y, int incY, IComplexNDArray A, int lda) {
        throw new UnsupportedOperationException();

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
