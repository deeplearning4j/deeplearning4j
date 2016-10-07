//
// Created by agibsonccc on 2/20/16.
//

#ifndef NATIVEOPERATIONS_NATIVEBLAS_H
#define NATIVEOPERATIONS_NATIVEBLAS_H

#include <pointercast.h>


//DO NOT REMOVE: THIS IS AN EDITOR SEMANTICS THING FOR CLION
//IT DEFINES THE EXPORT MACRO FOR THE EDITOR AND THEN
//RE ADDS THE DEFINITION VIA dll.h
#ifdef  _WIN32
#define ND4J_EXPORT __declspec(dllexport)
#else
#define ND4J_EXPORT
#endif
#include <dll.h>
#include <types/float16.h>
#include <stdio.h>
#include <stdlib.h>


class ND4J_EXPORT Nd4jBlas {
public:

    /*
     * ======================================================
     * ND4j service functions
     * ======================================================
     */

    void setMaxThreads(int num);

    int getMaxThreads();

    int getVendor();

    /*
     * ======================================================
     * Level 1 BLAS functions
     * ======================================================
     */


    /*
     * ------------------------------------------------------
     * DOT
     * ------------------------------------------------------
     */

    float sdsdot(Nd4jPointer *extraParams,int N, float alpha,
                 float *X, int incX,
                 float *Y, int incY);

    double dsdot(Nd4jPointer *extraParams,int N,
                 float *X, int incX,
                 float *Y, int incY);

    double ddot(Nd4jPointer *extraParams,int N,
                double *X, int incX,
                double *Y, int incY);

    float sdot(Nd4jPointer *extraParams,int N,
               float *X, int incX,
               float *Y, int incY);

    /*
     * ------------------------------------------------------
     * NRM2
     * ------------------------------------------------------
     */

    float snrm2(Nd4jPointer *extraParams,int N, float *X, int incX);

    double dnrm2(Nd4jPointer *extraParams,int N, double *X, int incX);

    /*
     * ------------------------------------------------------
     * ASUM
     * ------------------------------------------------------
     */

    float sasum(Nd4jPointer *extraParams,int N, float *X, int incX);

    double dasum(Nd4jPointer *extraParams,int N, double *X, int incX);

    /*
     * ------------------------------------------------------
     * IAMAX
     * ------------------------------------------------------
     */

    int isamax(Nd4jPointer *extraParams,int N, float *X, int incX);

    int idamax(Nd4jPointer *extraParams,int N, double *X, int incX);

    /*
     * ======================================================
     * Level 1 BLAS procedures
     * ======================================================
     */

    /*
     * ------------------------------------------------------
     * ROT
     * ------------------------------------------------------
     */

    void srot(Nd4jPointer *extraParams,int N,
              float *X, int incX,
              float *Y, int incY,
              float c, float s);

    void drot(Nd4jPointer *extraParams,int N,
              double *X, int incX,
              double *Y, int incY,
              double c, double s);

    /*
     * ------------------------------------------------------
     * ROTG
     * ------------------------------------------------------
     */

    void srotg(Nd4jPointer *extraParams,float *args);

    void drotg(Nd4jPointer *extraParams,double *args);

    /*
     * ------------------------------------------------------
     * ROTMG
     * ------------------------------------------------------
     */

    void srotmg(Nd4jPointer *extraParams,float *args,
                float *P);

    void drotmg(Nd4jPointer *extraParams,double *args,
                double *P);

    /*
     * ------------------------------------------------------
     * ROTM
     * ------------------------------------------------------
     */

    void srotm(Nd4jPointer *extraParams,int N,
               float *X, int incX,
               float *Y, int incY,
               float *P);

    void drotm(Nd4jPointer *extraParams,int N,
               double *X, int incX,
               double *Y, int incY,
               double *P);

    /*
     * ------------------------------------------------------
     * SWAP
     * ------------------------------------------------------
     */

    void sswap(Nd4jPointer *extraParams,int N,
               float *X, int incX,
               float *Y, int incY);

    void dswap(Nd4jPointer *extraParams,int N,
               double *X, int incX,
               double *Y, int incY);

    /*
     * ------------------------------------------------------
     * SCAL
     * ------------------------------------------------------
     */

    void sscal(Nd4jPointer *extraParams,int N, float alpha,
               float *X, int incX);

    void dscal(Nd4jPointer *extraParams,int N, double alpha,
               double *X, int incX);

    /*
     * ------------------------------------------------------
     * SCOPY
     * ------------------------------------------------------
     */

    void scopy(Nd4jPointer *extraParams,int N,
               float *X, int incX,
               float *Y, int incY);

    void dcopy(Nd4jPointer *extraParams,int N,
               double *X, int incX,
               double *Y, int incY);

    /*
     * ------------------------------------------------------
     * AXPY
     * ------------------------------------------------------
     */

    void saxpy(Nd4jPointer *extraParams,int N, float alpha,
               float *X, int incX,
               float *Y, int incY);

    void daxpy(Nd4jPointer *extraParams,int N, double alpha,
               double *X, int incX,
               double *Y, int incY);

    /*
     * ======================================================
     * Level 2 BLAS procedures
     * ======================================================
     */


    /*
     * ------------------------------------------------------
     * GEMV
     * ------------------------------------------------------
     */

    void sgemv(Nd4jPointer *extraParams,int Order, int TransA,
               int M, int N,
               float alpha,
               float *A, int lda,
               float *X, int incX,
               float beta,
               float *Y, int incY);

    void dgemv(Nd4jPointer *extraParams,int Order, int TransA,
               int M, int N,
               double alpha,
               double *A, int lda,
               double *X, int incX,
               double beta,
               double *Y, int incY);

    /*
     * ------------------------------------------------------
     * GBMV
     * ------------------------------------------------------
     */

    void sgbmv(Nd4jPointer *extraParams,int Order, int TransA,
               int M, int N,
               int KL, int KU,
               float alpha,
               float *A, int lda,
               float *X, int incX,
               float beta,
               float *Y, int incY);

    void dgbmv(Nd4jPointer *extraParams,int Order, int TransA,
               int M, int N,
               int KL, int KU,
               double alpha,
               double *A, int lda,
               double *X, int incX,
               double beta,
               double *Y, int incY);

    /*
     * ------------------------------------------------------
     * SYMV
     * ------------------------------------------------------
     */

    void ssymv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               float *A, int lda,
               float *X, int incX,
               float beta,
               float *Y, int incY);

    void dsymv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               double *A, int lda,
               double *X, int incX,
               double beta,
               double *Y, int incY);

    /*
     * ------------------------------------------------------
     * SBMV
     * ------------------------------------------------------
     */

    void ssbmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N, int K,
               float alpha,
               float *A, int lda,
               float *X, int incX,
               float beta,
               float *Y, int incY);

    void dsbmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N, int K,
               double alpha,
               double *A, int lda,
               double *X, int incX,
               double beta,
               double *Y, int incY);

    /*
     * ------------------------------------------------------
     * SPMV
     * ------------------------------------------------------
     */

    void sspmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               float *Ap,
               float *X, int incX,
               float beta,
               float *Y, int incY);

    void dspmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               double *Ap,
               double *X, int incX,
               double beta,
               double *Y, int incY);

    /*
     * ------------------------------------------------------
     * TRMV
     * ------------------------------------------------------
     */

    void strmv(Nd4jPointer *extraParams,int Order, int Uplo, int TransA,
               int Diag,
               int N, float alpha,
               float *A, int lda,
               float *X, int incX);
    void dtrmv(Nd4jPointer *extraParams,int Order, int Uplo, int TransA,
               int Diag,
               int N, double alpha,
               double *A, int lda,
               double *X, int incX);

    /*
     * ------------------------------------------------------
     * TBMV
     * ------------------------------------------------------
     */

    void stbmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               float *A, int lda,
               float *X, int incX);

    void dtbmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               double *A, int lda,
               double *X, int incX);

    /*
     * ------------------------------------------------------
     * TPMV
     * ------------------------------------------------------
     */

    void stpmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               float *Ap,
               float *X, int incX);

    void dtpmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               double *Ap,
               double *X, int incX);

    /*
     * ------------------------------------------------------
     * TRSV
     * ------------------------------------------------------
     */

    void strsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               float *A, int lda,
               float *X, int incX);

    void dtrsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               double *A, int lda,
               double *X, int incX);

    /*
     * ------------------------------------------------------
     * TBSV
     * ------------------------------------------------------
     */

    void stbsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               float *A, int lda,
               float *X, int incX);

    void dtbsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               double *A, int lda,
               double *X, int incX);

    /*
     * ------------------------------------------------------
     * TPSV
     * ------------------------------------------------------
     */

    void stpsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               float *Ap,
               float *X, int incX);

    void dtpsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               double *Ap,
               double *X, int incX);
    /*
     * ------------------------------------------------------
     * GER
     * ------------------------------------------------------
     */

    void sger(Nd4jPointer *extraParams,int Order,
              int M, int N,
              float alpha,
              float *X, int incX,
              float *Y, int incY,
              float *A, int lda);

    void dger(Nd4jPointer *extraParams,int Order,
              int M, int N,
              double alpha,
              double *X, int incX,
              double *Y, int incY,
              double *A, int lda);

    /*
     * ------------------------------------------------------
     * SYR
     * ------------------------------------------------------
     */

    void ssyr(Nd4jPointer *extraParams,int Order, int Uplo,
              int N,
              float alpha,
              float *X, int incX,
              float *A, int lda);

    void dsyr(Nd4jPointer *extraParams,int Order, int Uplo,
              int N,
              double alpha,
              double *X, int incX,
              double *A, int lda);

    /*
     * ------------------------------------------------------
     * SPR
     * ------------------------------------------------------
     */

    void sspr(Nd4jPointer *extraParams,int Order, int Uplo,
              int N,
              float alpha,
              float *X, int incX,
              float *Ap);

    void dspr(Nd4jPointer *extraParams,int Order, int Uplo,
              int N,
              double alpha,
              double *X, int incX,
              double *Ap);

    /*
     * ------------------------------------------------------
     * SYR2
     * ------------------------------------------------------
     */

    void ssyr2(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               float *X, int incX,
               float *Y, int incY,
               float *A, int lda);

    void dsyr2(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               double *X, int incX,
               double *Y, int incY,
               double *A, int lda);

    /*
     * ------------------------------------------------------
     * SPR2
     * ------------------------------------------------------
     */

    void sspr2(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               float *X, int incX,
               float *Y, int incY,
               float *Ap);

    void dspr2(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               double *X, int incX,
               double *Y, int incY,
               double *Ap);

    /*
     * ======================================================
     * Level 3 BLAS procedures
     * ======================================================
     */


    /*
     * ------------------------------------------------------
     * GEMM
     * ------------------------------------------------------
     */
    void hgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               float alpha,
               float16 *A, int lda,
               float16 *B, int ldb,
               float beta,
               float16 *C, int ldc);


    void sgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               float alpha,
               float *A, int lda,
               float *B, int ldb,
               float beta,
               float *C, int ldc);

    void dgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               double alpha,
               double *A, int lda,
               double *B, int ldb,
               double beta,
               double *C, int ldc);

    /*
     * ------------------------------------------------------
     * SYMM
     * ------------------------------------------------------
     */

    void ssymm(Nd4jPointer *extraParams,int Order, int Side, int Uplo,
               int M, int N,
               float alpha,
               float *A, int lda,
               float *B, int ldb,
               float beta,
               float *C, int ldc);

    void dsymm(Nd4jPointer *extraParams,int Order, int Side, int Uplo,
               int M, int N,
               double alpha,
               double *A, int lda,
               double *B, int ldb,
               double beta,
               double *C, int ldc);

    /*
     * ------------------------------------------------------
     * SYRK
     * ------------------------------------------------------
     */

    void ssyrk(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
               int N, int K,
               float alpha,
               float *A, int lda,
               float beta,
               float *C, int ldc);

    void dsyrk(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
               int N, int K,
               double alpha,
               double *A, int lda,
               double beta,
               double *C, int ldc);

    /*
     * ------------------------------------------------------
     * SYR2K
     * ------------------------------------------------------
     */

    void ssyr2k(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                int N, int K,
                float alpha,
                float *A, int lda,
                float *B, int ldb,
                float beta,
                float *C, int ldc);

    void dsyr2k(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                int N, int K,
                double alpha,
                double *A, int lda,
                double *B, int ldb,
                double beta,
                double *C, int ldc);

    /*
     * ------------------------------------------------------
     * TRMM
     * ------------------------------------------------------
     */

    void strmm(Nd4jPointer *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               float alpha,
               float *A, int lda,
               float *B, int ldb);

    void dtrmm(Nd4jPointer *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               double alpha,
               double *A, int lda,
               double *B, int ldb);

    /*
     * ------------------------------------------------------
     * TRSM
     * ------------------------------------------------------
     */

    void strsm(Nd4jPointer *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               float alpha,
               float *A, int lda,
               float *B, int ldb);

    void dtrsm(Nd4jPointer *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               double alpha,
               double *A, int lda,
               double *B, int ldb);

};

#endif //NATIVEOPERATIONS_NATIVEBLAS_H
