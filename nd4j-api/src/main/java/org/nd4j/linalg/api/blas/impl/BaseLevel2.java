package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.Level2;
import org.nd4j.linalg.api.blas.params.GemvParameters;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Base class for level 2 functions, abstract headers pulled from:
 * http://www.netlib.org/blas/blast-forum/cblas.h
 *
 * @author Adam Gibson
 */
public abstract class BaseLevel2 extends BaseLevel implements Level2 {
    /**
     * gemv computes a matrix-vector product using a general matrix and performs one of the following matrix-vector operations:
     * y := alpha*a*x + beta*y  for trans = 'N'or'n';
     * y := alpha*a'*x + beta*y  for trans = 'T'or't';
     * y := alpha*conjg(a')*x + beta*y  for trans = 'C'or'c'.
     * Here a is an m-by-n band matrix, x and y are vectors, alpha and beta are scalars.
     *
     * @param order
     * @param transA
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void gemv(char order, char transA, double alpha, INDArray A, INDArray X, double beta, INDArray Y) {
        GemvParameters parameters = new GemvParameters(A,X,Y);
        if(A.data().dataType() == DataBuffer.Type.DOUBLE)
            dgemv(order
                    , parameters.getAOrdering()
                    , parameters.getM()
                    , parameters.getN()
                    , alpha
                    , parameters.getA()
                    , parameters.getLda()
                    , parameters.getX()
                    , parameters.getIncx()
                    , beta
                    , parameters.getY()
                    , parameters.getIncy());
        else
            sgemv(order
                    , parameters.getAOrdering()
                    , parameters.getM()
                    , parameters.getN()
                    , (float) alpha
                    , parameters.getA()
                    ,parameters.getLda()
                    , parameters.getX()
                    , parameters.getIncx()
                    , (float) beta
                    , parameters.getY()
                    , parameters.getIncy());
    }

    /**
     * gemv computes a matrix-vector product using a general matrix and performs one of the following matrix-vector operations:
     * y := alpha*a*x + beta*y  for trans = 'N'or'n';
     * y := alpha*a'*x + beta*y  for trans = 'T'or't';
     * y := alpha*conjg(a')*x + beta*y  for trans = 'C'or'c'.
     * Here a is an m-by-n band matrix, x and y are vectors, alpha and beta are scalars.
     *
     * @param order
     * @param transA
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void gemv(char order, char transA, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray X, IComplexNumber beta, IComplexNDArray Y) {
        GemvParameters parameters = new GemvParameters(A,X,Y);

        if(A.data().dataType() == DataBuffer.Type.DOUBLE)
            zgemv(
                    order
                    , transA
                    , parameters.getM()
                    , parameters.getN()
                    , alpha.asDouble()
                    , A
                    , parameters.getLda()
                    , X
                    , parameters.getIncx()
                    , beta.asDouble()
                    , Y
                    , parameters.getIncy());
        else
            cgemv(order
                    , transA
                    ,parameters.getM()
                    , parameters.getN()
                    , alpha.asFloat()
                    , A
                    , parameters.getLda()
                    , X
                    , parameters.getIncx()
                    , beta.asFloat()
                    , Y
                    , parameters.getIncy());

    }

    /**
     * gbmv computes a matrix-vector product using a general band matrix and performs one of the following matrix-vector operations:
     * y := alpha*a*x + beta*y  for trans = 'N'or'n';
     * y := alpha*a'*x + beta*y  for trans = 'T'or't';
     * y := alpha*conjg(a')*x + beta*y  for trans = 'C'or'c'.
     * Here a is an m-by-n band matrix with ku superdiagonals and kl subdiagonals, x and y are vectors, alpha and beta are scalars.
     *
     * @param order
     * @param TransA
     * @param KL
     * @param KU
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void gbmv(char order, char TransA, int KL, int KU, double alpha, INDArray A, INDArray X, double beta, INDArray Y) {
        if(A.data().dataType() == DataBuffer.Type.DOUBLE)
            dgbmv(order, TransA, A.rows(), A.columns(), KL, KU, alpha, A, A.size(0), X, X.majorStride(), beta, Y, Y.majorStride());
        else
            sgbmv(order, TransA, A.rows(), A.columns(), KL, KU, (float) alpha, A, A.size(0), X, X.majorStride(), (float) beta, Y, Y.majorStride());

    }

    /**
     * gbmv computes a matrix-vector product using a general band matrix and performs one of the following matrix-vector operations:
     * y := alpha*a*x + beta*y  for trans = 'N'or'n';
     * y := alpha*a'*x + beta*y  for trans = 'T'or't';
     * y := alpha*conjg(a')*x + beta*y  for trans = 'C'or'c'.
     * Here a is an m-by-n band matrix with ku superdiagonals and kl subdiagonals, x and y are vectors, alpha and beta are scalars.
     *
     * @param order
     * @param TransA
     * @param KL
     * @param KU
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void gbmv(char order, char TransA, int KL, int KU, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray X, IComplexNumber beta, IComplexNDArray Y) {
        if(A.data().dataType() == DataBuffer.Type.DOUBLE)
            zgbmv(order, TransA, A.rows(), A.columns(), KL, KU, alpha.asDouble(), A, A.size(0), X, X.majorStride() / 2, beta.asDouble(), Y, Y.majorStride() / 2);
        else
            cgbmv(order, TransA, A.rows(), A.columns(), KL, KU, alpha.asFloat(), A, A.size(0), X, X.majorStride() / 2, beta.asFloat(), Y, Y.majorStride() / 2);

    }

    /**
     * performs a rank-1 update of a general m-by-n matrix a:
     * a := alpha*x*y' + a.
     *
     * @param order
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    @Override
    public void ger(char order, double alpha, INDArray X, INDArray Y, INDArray A) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dger(order,A.rows(),A.columns(),alpha,X,X.majorStride(),Y,Y.majorStride(),A,A.size(0));
        else
            sger(order,A.rows(),A.columns(),(float) alpha,X,X.majorStride(),Y,Y.majorStride(),A,A.size(0));

    }


    /**
     * performs a rank-1 update of a general m-by-n matrix a, without conjugation:
     * a := alpha*x*y' + a.
     *  @param order
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    @Override
    public void geru(char order, IComplexNumber alpha, IComplexNDArray X, IComplexNDArray Y, IComplexNDArray A) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            zgeru(order,A.rows(),A.columns(),alpha.asDouble(),X,X.majorStride() / 2,Y,Y.majorStride() / 2,A,A.size(0));
        else
            cgeru(order, A.rows(), A.columns(), alpha.asFloat(), X, X.majorStride() / 2, Y, Y.majorStride() / 2, A, A.size(0));

    }

    /**
     * performs a rank-1 update of a general m-by-n matrix a, without conjugation:
     * a := alpha*x*y' + a.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void hbmv(char order, char Uplo, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray X, IComplexNumber beta, IComplexNDArray Y) {
        if(A.data().dataType() == DataBuffer.Type.DOUBLE)
            zhbmv(order,Uplo,X.length(),A.columns(),alpha.asDouble(),A,A.size(0),X,X.majorStride() / 2,beta.asDouble(),Y,Y.majorStride() / 2);
        else
            chbmv(order, Uplo, X.length(), A.columns(), alpha.asFloat(), A, A.size(0), X, X.majorStride() / 2, beta.asFloat(), Y, Y.majorStride() / 2);

    }

    /**
     * hemv computes a matrix-vector product using a Hermitian matrix:
     * y := alpha*a*x + beta*y.
     * Here a is an n-by-n Hermitian band matrix with k superdiagonals, x and y are n-element vectors, alpha and beta are scalars.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void hemv(char order, char Uplo, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray X, IComplexNumber beta, IComplexNDArray Y) {
        if(A.data().dataType() == DataBuffer.Type.DOUBLE)
            zhemv(order,Uplo,A.rows(),alpha.asDouble(),A,A.size(0),X,X.majorStride() / 2,beta.asDouble(),Y,Y.majorStride() / 2);
        else
            chemv(order, Uplo, A.rows(), alpha.asFloat(), A, A.size(0), X, X.majorStride() / 2, beta.asFloat(), Y, Y.majorStride() / 2);

    }

    /**
     * ?her2 performs a rank-2 update of an n-by-n Hermitian matrix a:
     * a := alpha*x*conjg(y') + conjg(alpha)*y*conjg(x') + a.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    @Override
    public void her2(char order, char Uplo, IComplexNumber alpha, IComplexNDArray X, IComplexNDArray Y, IComplexNDArray A) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            zher2(order,Uplo,A.rows(),alpha.asDouble(),X,X.majorStride() / 2,Y,Y.majorStride() / 2,A,A.size(0));
        else
            cher2(order, Uplo, A.rows(), alpha.asFloat(), X, X.majorStride() / 2, Y, Y.majorStride() / 2, A, A.size(0));

    }

    /**
     * ?hpmv computes a matrix-vector product using a Hermitian packed matrix:
     * y := alpha*a*x + beta*y.
     * Here a is an n-by-n packed Hermitian matrix, x and y are n-element vectors, alpha and beta are scalars.
     *
     * @param order
     * @param Uplo
     * @param N
     * @param alpha
     * @param Ap
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void hpmv(char order, char Uplo, int N, IComplexNumber alpha, IComplexNDArray Ap, IComplexNDArray X, IComplexNumber beta, IComplexNDArray Y) {
        if(Ap.data().dataType() == DataBuffer.Type.DOUBLE)
            zhpmv(order,Uplo, Ap.rows(),alpha.asDouble(),Ap,X,X.majorStride() / 2,beta.asDouble(),Y,Y.majorStride() / 2);
        else
            chpmv(order, Uplo, Ap.rows(), alpha.asFloat(), Ap, X, X.majorStride() / 2, beta.asFloat(), Y, Y.majorStride() / 2);

    }

    /**
     * hpr2 performs a rank-2 update of an n-by-n packed Hermitian matrix a:
     * a := alpha*x*conjg(y') + conjg(alpha)*y*conjg(x') + a.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param Ap
     */
    @Override
    public void hpr2(char order, char Uplo, IComplexNumber alpha, IComplexNDArray X, IComplexNDArray Y, IComplexNDArray Ap) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            zhpr2(order,Uplo,Ap.rows(),alpha.asDouble(),X,X.majorStride() / 2,Y,Y.majorStride() / 2,Ap);
        else
            chpr2(order, Uplo, Ap.rows(), alpha.asFloat(), X, X.majorStride() / 2, Y, Y.majorStride() / 2, Ap);

    }

    /**
     * sbmv computes a matrix-vector product using a symmetric band matrix:
     * y := alpha*a*x + beta*y.
     * Here a is an n-by-n symmetric band matrix with k superdiagonals, x and y are n-element vectors, alpha and beta are scalars.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void sbmv(char order, char Uplo, double alpha, INDArray A, INDArray X, double beta, INDArray Y) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dsbmv(order,Uplo,X.length(),A.columns(),alpha,A,A.size(0),X,X.majorStride(),beta,Y,Y.majorStride());
        else
            ssbmv(order, Uplo, X.length(), A.columns(), (float) alpha, A, A.size(0), X, X.majorStride(), (float) beta, Y, Y.majorStride());

    }

    /**
     * @param order
     * @param Uplo
     * @param alpha
     * @param Ap
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void spmv(char order, char Uplo, double alpha, INDArray Ap, INDArray X, double beta, INDArray Y) {
        if(Ap.data().dataType() == DataBuffer.Type.DOUBLE)
            dspmv(order,Uplo,X.length(),alpha,Ap,X, Ap.majorStride(),beta,Y,Y.majorStride());
        else
            sspmv(order, Uplo, X.length(), (float) alpha, Ap, X, Ap.majorStride(), (float) beta, Y, Y.majorStride());

    }

    /**
     * spr performs a rank-1 update of an n-by-n packed symmetric matrix a:
     * a := alpha*x*x' + a.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Ap
     */
    @Override
    public void spr(char order, char Uplo, double alpha, INDArray X, INDArray Ap) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dspr(order,Uplo,X.length(),alpha,X,X.majorStride(),Ap);
        else
            sspr(order, Uplo, X.length(), (float) alpha, X, X.majorStride(), Ap);

    }

    /**
     * ?spr2 performs a rank-2 update of an n-by-n packed symmetric matrix a:
     * a := alpha*x*y' + alpha*y*x' + a.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    @Override
    public void spr2(char order, char Uplo, double alpha, INDArray X, INDArray Y, INDArray A) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dspr2(order,Uplo,X.length(),alpha,X,X.majorStride(),Y,Y.majorStride(),A);
        else
            sspr2(order, Uplo, X.length(), (float) alpha, X, X.majorStride(), Y, Y.majorStride(), A);

    }

    /**
     * symv computes a matrix-vector product for a symmetric matrix:
     * y := alpha*a*x + beta*y.
     * Here a is an n-by-n symmetric matrix; x and y are n-element vectors, alpha and beta are scalars.
     *
     * @param order
     * @param Uplo
     * @param alpha
     * @param A
     * @param X
     * @param beta
     * @param Y
     */
    @Override
    public void symv(char order, char Uplo, double alpha, INDArray A, INDArray X, double beta, INDArray Y) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dsymv(order,Uplo,X.length(),alpha,A,A.size(0),X,X.majorStride(),beta,Y,Y.majorStride());
        else
            ssymv(order, Uplo, X.length(), (float) alpha, A, A.size(0), X, X.majorStride(), (float) beta, Y, Y.majorStride());

    }

    /**
     * syr performs a rank-1 update of an n-by-n symmetric matrix a:
     * a := alpha*x*x' + a.
     *
     * @param order
     * @param Uplo
     * @param N
     * @param alpha
     * @param X
     * @param A
     */
    @Override
    public void syr(char order, char Uplo, int N, double alpha, INDArray X, INDArray A) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dsyr(order,Uplo,X.length(),alpha,X,X.majorStride(),A,A.size(0));
        else
            ssyr(order, Uplo, X.length(), (float) alpha, X, X.majorStride(), A, A.size(0));

    }

    /**
     * @param order
     * @param Uplo
     * @param alpha
     * @param X
     * @param Y
     * @param A
     */
    @Override
    public void syr2(char order, char Uplo, double alpha, INDArray X, INDArray Y, INDArray A) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dsyr2(order,Uplo,X.length(),alpha,X,X.majorStride(),Y,Y.majorStride(),A,A.size(0));
        else
            ssyr2(order, Uplo, X.length(), (float) alpha, X, X.majorStride(), Y, Y.majorStride(), A, A.size(0));

    }

    /**
     * syr2 performs a rank-2 update of an n-by-n symmetric matrix a:
     * a := alpha*x*y' + alpha*y*x' + a.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    @Override
    public void tbmv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dtbmv(order,Uplo,TransA,Diag,X.length(),A.columns(),A,A.size(0),X,X.majorStride());
        else
            stbmv(order, Uplo, TransA, Diag, X.length(), A.columns(), A, A.size(0), X, X.majorStride());

    }

    /**
     * ?tbsv solves a system of linear equations whose coefficients are in a triangular band matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    @Override
    public void tbsv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dtbsv(order,Uplo,TransA,Diag,X.length(),A.columns(),A,A.size(0),X,X.majorStride());
        else
            stbsv(order, Uplo, TransA, Diag, X.length(), A.columns(), A, A.size(0), X, X.majorStride());

    }

    /**
     * tpmv computes a matrix-vector product using a triangular packed matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param Ap
     * @param X
     */
    @Override
    public void tpmv(char order, char Uplo, char TransA, char Diag, INDArray Ap, INDArray X) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dtpmv(order,Uplo,TransA,Diag,Ap.length(),Ap,X,X.majorStride());
        else
            stpmv(order, Uplo, TransA, Diag, Ap.length(), Ap, X, X.majorStride());

    }

    /**
     * tpsv solves a system of linear equations whose coefficients are in a triangular packed matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param Ap
     * @param X
     */
    @Override
    public void tpsv(char order, char Uplo, char TransA, char Diag, INDArray Ap, INDArray X) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dtpsv(order,Uplo,TransA,Diag,X.length(),Ap,X,X.majorStride());
        else
            stpsv(order, Uplo, TransA, Diag, X.length(), Ap, X, X.majorStride());

    }

    /**
     * trmv computes a matrix-vector product using a triangular matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    @Override
    public void trmv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X) {
        if(A.data().dataType() == DataBuffer.Type.DOUBLE)
            dtrmv(order,Uplo,TransA,Diag,X.length(),A,A.size(0),X,X.majorStride());
        else
            strmv(order, Uplo, TransA, Diag, X.length(), A, A.size(0), X, X.majorStride());

    }

    /**
     * trsv solves a system of linear equations whose coefficients are in a triangular matrix.
     *
     * @param order
     * @param Uplo
     * @param TransA
     * @param Diag
     * @param A
     * @param X
     */
    @Override
    public void trsv(char order, char Uplo, char TransA, char Diag, INDArray A, INDArray X) {
        if(X.data().dataType() == DataBuffer.Type.DOUBLE)
            dtrsv(order,Uplo,TransA,Diag,A.length(),A,A.size(0),X,X.majorStride());
        else
            strsv(order, Uplo, TransA, Diag, A.length(), A, A.size(0), X, X.majorStride());

    }

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

    /* 
     * Routines with standard 4 prefixes (S, D, C, Z)
     */
    protected abstract void sgemv( char order,
                                   char TransA,  int M,  int N,
                                   float alpha,  INDArray A,  int lda,
                                   INDArray X,  int incX,  float beta,
                                   INDArray Y,  int incY);
    protected abstract  void sgbmv( char order,
                                    char TransA,  int M,  int N,
                                    int KL,  int KU,  float alpha,
                                    INDArray A,  int lda,  INDArray X,
                                    int incX,  float beta, INDArray Y,  int incY);
    protected abstract void strmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  INDArray A,  int lda,
                                   INDArray X,  int incX);
    protected abstract void stbmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  int K,  INDArray A,  int lda,
                                   INDArray X,  int incX);
    protected abstract void stpmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  INDArray Ap, INDArray X,  int incX);
    protected abstract void strsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  INDArray A,  int lda, INDArray X,
                                   int incX);
    protected abstract void stbsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  int K,  INDArray A,  int lda,
                                   INDArray X,  int incX);
    protected abstract void stpsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  INDArray Ap, INDArray X,  int incX);

    protected abstract void dgemv( char order,
                                   char TransA,  int M,  int N,
                                   double alpha,  INDArray A,  int lda,
                                   INDArray X,  int incX,  double beta,
                                   INDArray Y,  int incY);
    protected abstract void dgbmv( char order,
                                   char TransA,  int M,  int N,
                                   int KL,  int KU,  double alpha,
                                   INDArray A,  int lda,  INDArray X,
                                   int incX,  double beta, INDArray Y,  int incY);
    protected abstract void dtrmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  INDArray A,  int lda,
                                   INDArray X,  int incX);
    protected abstract void dtbmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  int K,  INDArray A,  int lda,
                                   INDArray X,  int incX);
    protected abstract void dtpmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  INDArray Ap, INDArray X,  int incX);
    protected abstract void dtrsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  INDArray A,  int lda, INDArray X,
                                   int incX);
    protected abstract void dtbsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  int K,  INDArray A,  int lda,
                                   INDArray X,  int incX);
    protected abstract void dtpsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  INDArray Ap, INDArray X,  int incX);

    protected abstract void cgemv( char order,
                                   char TransA,  int M,  int N,
                                   IComplexFloat alpha,  IComplexNDArray A,  int lda,
                                   IComplexNDArray X,  int incX,  IComplexFloat beta,
                                   IComplexNDArray Y,  int incY);
    protected abstract void cgbmv( char order,
                                   char TransA,  int M,  int N,
                                   int KL,  int KU,  IComplexFloat alpha,
                                   IComplexNDArray A,  int lda,  IComplexNDArray X,
                                   int incX,  IComplexFloat beta, IComplexNDArray Y,  int incY);
    protected abstract void ctrmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  IComplexNDArray A,  int lda,
                                   IComplexNDArray X,  int incX);
    protected abstract void ctbmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  int K,  IComplexNDArray A,  int lda,
                                   IComplexNDArray X,  int incX);
    protected abstract void ctpmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  IComplexNDArray Ap, IComplexNDArray X,  int incX);
    protected abstract void ctrsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  IComplexNDArray A,  int lda, IComplexNDArray X,
                                   int incX);
    protected abstract void ctbsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  int K,  IComplexNDArray A,  int lda,
                                   IComplexNDArray X,  int incX);
    protected abstract void ctpsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  IComplexNDArray Ap, IComplexNDArray X,  int incX);

    protected abstract void zgemv( char order,
                                   char TransA,  int M,  int N,
                                   IComplexDouble alpha,  IComplexNDArray A,  int lda,
                                   IComplexNDArray X,  int incX,  IComplexDouble beta,
                                   IComplexNDArray Y,  int incY);
    protected abstract void zgbmv( char order,
                                   char TransA,  int M,  int N,
                                   int KL,  int KU,  IComplexDouble alpha,
                                   IComplexNDArray A,  int lda,  IComplexNDArray X,
                                   int incX,  IComplexDouble beta, IComplexNDArray Y,  int incY);
    protected abstract void ztrmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  IComplexNDArray A,  int lda,
                                   IComplexNDArray X,  int incX);
    protected abstract void ztbmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  int K,  IComplexNDArray A,  int lda,
                                   IComplexNDArray X,  int incX);
    protected abstract void ztpmv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  IComplexNDArray Ap, IComplexNDArray X,  int incX);
    protected abstract void ztrsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  IComplexNDArray A,  int lda, IComplexNDArray X,
                                   int incX);
    protected abstract void ztbsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  int K,  IComplexNDArray A,  int lda,
                                   IComplexNDArray X,  int incX);
    protected abstract void ztpsv( char order,  char Uplo,
                                   char TransA,  char Diag,
                                   int N,  IComplexNDArray Ap, IComplexNDArray X,  int incX);


    /* 
     * Routines with S and D prefixes only
     */
    protected abstract  void ssymv( char order,  char Uplo,
                                    int N,  float alpha,  INDArray A,
                                    int lda,  INDArray X,  int incX,
                                    float beta, INDArray Y,  int incY);
    protected abstract  void ssbmv( char order,  char Uplo,
                                    int N,  int K,  float alpha,  INDArray A,
                                    int lda,  INDArray X,  int incX,
                                    float beta, INDArray Y,  int incY);
    protected abstract  void sspmv( char order,  char Uplo,
                                    int N,  float alpha,  INDArray Ap,
                                    INDArray X,  int incX,
                                    float beta, INDArray Y,  int incY);
    protected abstract void sger( char order,  int M,  int N,
                                  float alpha,  INDArray X,  int incX,
                                  INDArray Y,  int incY, INDArray A,  int lda);
    protected abstract void ssyr( char order,  char Uplo,
                                  int N,  float alpha,  INDArray X,
                                  int incX, INDArray A,  int lda);
    protected abstract void sspr( char order,  char Uplo,
                                  int N,  float alpha,  INDArray X,
                                  int incX, INDArray Ap);
    protected abstract void ssyr2( char order,  char Uplo,
                                   int N,  float alpha,  INDArray X,
                                   int incX,  INDArray Y,  int incY, INDArray A,
                                   int lda);
    protected abstract void sspr2( char order,  char Uplo,
                                   int N,  float alpha,  INDArray X,
                                   int incX,  INDArray Y,  int incY, INDArray A);

    protected abstract void dsymv( char order,  char Uplo,
                                   int N,  double alpha,  INDArray A,
                                   int lda,  INDArray X,  int incX,
                                   double beta, INDArray Y,  int incY);
    protected abstract void dsbmv( char order,  char Uplo,
                                   int N,  int K,  double alpha,  INDArray A,
                                   int lda,  INDArray X,  int incX,
                                   double beta, INDArray Y,  int incY);
    protected abstract  void dspmv( char order,  char Uplo,
                                    int N,  double alpha,  INDArray Ap,
                                    INDArray X,  int incX,
                                    double beta, INDArray Y,  int incY);
    protected abstract void dger( char order,  int M,  int N,
                                  double alpha,  INDArray X,  int incX,
                                  INDArray Y,  int incY, INDArray A,  int lda);
    protected abstract void dsyr( char order,  char Uplo,
                                  int N,  double alpha,  INDArray X,
                                  int incX, INDArray A,  int lda);
    protected abstract void dspr( char order,  char Uplo,
                                  int N,  double alpha,  INDArray X,
                                  int incX, INDArray Ap);
    protected abstract void dsyr2( char order,  char Uplo,
                                   int N,  double alpha,  INDArray X,
                                   int incX,  INDArray Y,  int incY, INDArray A,
                                   int lda);
    protected abstract void dspr2( char order,  char Uplo,
                                   int N,  double alpha,  INDArray X,
                                   int incX,  INDArray Y,  int incY, INDArray A);


    /* 
     * Routines with C and Z prefixes only
     */
    protected abstract  void chemv( char order,  char Uplo,
                                    int N,  IComplexFloat alpha,  IComplexNDArray A,
                                    int lda,  IComplexNDArray X,  int incX,
                                    IComplexFloat beta, IComplexNDArray Y,  int incY);
    protected abstract void chbmv( char order,  char Uplo,
                                   int N,  int K,  IComplexFloat alpha,  IComplexNDArray A,
                                   int lda,  IComplexNDArray X,  int incX,
                                   IComplexFloat beta, IComplexNDArray Y,  int incY);
    protected abstract void chpmv( char order,  char Uplo,
                                   int N,  IComplexFloat alpha,  IComplexNDArray Ap,
                                   IComplexNDArray X,  int incX,
                                   IComplexFloat beta, IComplexNDArray Y,  int incY);
    protected abstract void cgeru( char order,  int M,  int N,
                                   IComplexFloat alpha,  IComplexNDArray X,  int incX,
                                   IComplexNDArray Y,  int incY, IComplexNDArray A,  int lda);
    protected abstract void cgerc( char order,  int M,  int N,
                                   IComplexFloat alpha,  IComplexNDArray X,  int incX,
                                   IComplexNDArray Y,  int incY, IComplexNDArray A,  int lda);
    protected abstract void cher( char order,  char Uplo,
                                  int N,  float alpha,  IComplexNDArray X,  int incX,
                                  IComplexNDArray A,  int lda);
    protected abstract void chpr( char order,  char Uplo,
                                  int N,  INDArray alpha,  IComplexNDArray X,
                                  int incX, IComplexNDArray A);
    protected abstract void cher2( char order,  char Uplo,  int N,
                                   IComplexFloat alpha,  IComplexNDArray X,  int incX,
                                   IComplexNDArray Y,  int incY, IComplexNDArray A,  int lda);
    protected abstract void chpr2( char order,  char Uplo,  int N,
                                   IComplexFloat alpha,  IComplexNDArray X,  int incX,
                                   IComplexNDArray Y,  int incY, IComplexNDArray Ap);

    protected abstract void zhemv( char order,  char Uplo,
                                   int N,  IComplexDouble alpha,  IComplexNDArray A,
                                   int lda,  IComplexNDArray X,  int incX,
                                   IComplexDouble beta, IComplexNDArray Y,  int incY);
    protected abstract void zhbmv( char order,  char Uplo,
                                   int N,  int K,  IComplexDouble alpha,  IComplexNDArray A,
                                   int lda,  IComplexNDArray X,  int incX,
                                   IComplexDouble beta, IComplexNDArray Y,  int incY);
    protected abstract void zhpmv( char order,  char Uplo,
                                   int N,  IComplexDouble alpha,  IComplexNDArray Ap,
                                   IComplexNDArray X,  int incX,
                                   IComplexDouble beta, IComplexNDArray Y,  int incY);
    protected abstract  void zgeru( char order,  int M,  int N,
                                    IComplexDouble alpha,  IComplexNDArray X,  int incX,
                                    IComplexNDArray Y,  int incY, IComplexNDArray A,  int lda);
    protected abstract void zgerc( char order,  int M,  int N,
                                   IComplexDouble alpha,  IComplexNDArray X,  int incX,
                                   IComplexNDArray Y,  int incY, IComplexNDArray A,  int lda);
    protected abstract void zher( char order,  char Uplo,
                                  int N,  double alpha,  IComplexNDArray X,  int incX,
                                  IComplexNDArray A,  int lda);
    protected abstract void zhpr( char order,  char Uplo,
                                  int N,  INDArray alpha,  IComplexNDArray X,
                                  int incX, IComplexNDArray A);
    protected abstract void zher2( char order,  char Uplo,  int N,
                                   IComplexDouble alpha,  IComplexNDArray X,  int incX,
                                   IComplexNDArray Y,  int incY, IComplexNDArray A,  int lda);
    protected abstract void zhpr2( char order,  char Uplo,  int N,
                                   IComplexDouble alpha,  IComplexNDArray X,  int incX,
                                   IComplexNDArray Y,  int incY, IComplexNDArray Ap);


}


