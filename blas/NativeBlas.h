//
// Created by agibsonccc on 2/20/16.
//

#ifndef NATIVEOPERATIONS_NATIVEBLAS_H
#define NATIVEOPERATIONS_NATIVEBLAS_H





class Nd4jBlas {
public:

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

    float sdsdot(long *extraParams,int N, float alpha,
                 long X, int incX,
                 long Y, int incY);

    double dsdot(long *extraParams,int N,
                 long X, int incX,
                 long Y, int incY);

    double ddot(long *extraParams,int N,
                long X, int incX,
                long Y, int incY);

    float sdot(long *extraParams,int N,
               long X, int incX,
               long Y, int incY);

    /*
     * ------------------------------------------------------
     * NRM2
     * ------------------------------------------------------
     */

    float snrm2(long *extraParams,int N, long X, int incX);

    double dnrm2(long *extraParams,int N, long X, int incX);

    /*
     * ------------------------------------------------------
     * ASUM
     * ------------------------------------------------------
     */

    float sasum(long *extraParams,int N, long X, int incX);

    double dasum(long *extraParams,int N, long X, int incX);

    /*
     * ------------------------------------------------------
     * IAMAX
     * ------------------------------------------------------
     */

    int isamax(long *extraParams,int N, long X, int incX);

    int idamax(long *extraParams,int N, long X, int incX);

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

    void srot(long *extraParams,int N,
              long X, int incX,
              long Y, int incY,
              float c, float s);

    void drot(long *extraParams,int N,
              long X, int incX,
              long Y, int incY,
              double c, double s);

    /*
     * ------------------------------------------------------
     * ROTG
     * ------------------------------------------------------
     */

    void srotg(long *extraParams,long args);

    void drotg(long *extraParams,long args);

    /*
     * ------------------------------------------------------
     * ROTMG
     * ------------------------------------------------------
     */

    void srotmg(long *extraParams,long args,
                long P);

    void drotmg(long *extraParams,long args,
                long P);

    /*
     * ------------------------------------------------------
     * ROTM
     * ------------------------------------------------------
     */

    void srotm(long *extraParams,int N,
               long X, int incX,
               long Y, int incY,
               long P);

    void drotm(long *extraParams,int N,
               long X, int incX,
               long Y, int incY,
               long P);

    /*
     * ------------------------------------------------------
     * SWAP
     * ------------------------------------------------------
     */

    void sswap(long *extraParams,int N,
               long X, int incX,
               long Y, int incY);

    void dswap(long *extraParams,int N,
               long X, int incX,
               long Y, int incY);

    /*
     * ------------------------------------------------------
     * SCAL
     * ------------------------------------------------------
     */

    void sscal(long *extraParams,int N, float alpha,
               long X, int incX);

    void dscal(long *extraParams,int N, double alpha,
               long X, int incX);

    /*
     * ------------------------------------------------------
     * SCOPY
     * ------------------------------------------------------
     */

    void scopy(long *extraParams,int N,
               long X, int incX,
               long Y, int incY);

    void dcopy(long *extraParams,int N,
               long X, int incX,
               long Y, int incY);

    /*
     * ------------------------------------------------------
     * AXPY
     * ------------------------------------------------------
     */

    void saxpy(long *extraParams,int N, float alpha,
               long X, int incX,
               long Y, int incY);

    void daxpy(long *extraParams,int N, double alpha,
               long X, int incX,
               long Y, int incY);

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

    void sgemv(long *extraParams,int Order, int TransA,
               int M, int N,
               float alpha,
               long A, int lda,
               long X, int incX,
               float beta,
               long Y, int incY);

    void dgemv(long *extraParams,int Order, int TransA,
               int M, int N,
               double alpha,
               long A, int lda,
               long X, int incX,
               double beta,
               long Y, int incY);

    /*
     * ------------------------------------------------------
     * GBMV
     * ------------------------------------------------------
     */

    void sgbmv(long *extraParams,int Order, int TransA,
               int M, int N,
               int KL, int KU,
               float alpha,
               long A, int lda,
               long X, int incX,
               float beta,
               long Y, int incY);

    void dgbmv(long *extraParams,int Order, int TransA,
               int M, int N,
               int KL, int KU,
               double alpha,
               long A, int lda,
               long X, int incX,
               double beta,
               long Y, int incY);

    /*
     * ------------------------------------------------------
     * SYMV
     * ------------------------------------------------------
     */

    void ssymv(long *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               long A, int lda,
               long X, int incX,
               float beta,
               long Y, int incY);

    void dsymv(long *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               long A, int lda,
               long X, int incX,
               double beta,
               long Y, int incY);

    /*
     * ------------------------------------------------------
     * SBMV
     * ------------------------------------------------------
     */

    void ssbmv(long *extraParams,int Order, int Uplo,
               int N, int K,
               float alpha,
               long A, int lda,
               long X, int incX,
               float beta,
               long Y, int incY);

    void dsbmv(long *extraParams,int Order, int Uplo,
               int N, int K,
               double alpha,
               long A, int lda,
               long X, int incX,
               double beta,
               long Y, int incY);

    /*
     * ------------------------------------------------------
     * SPMV
     * ------------------------------------------------------
     */

    void sspmv(long *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               long Ap,
               long X, int incX,
               float beta,
               long Y, int incY);

    void dspmv(long *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               long Ap,
               long X, int incX,
               double beta,
               long Y, int incY);

    /*
     * ------------------------------------------------------
     * TRMV
     * ------------------------------------------------------
     */

    void strmv(long *extraParams,int Order, int Uplo, int TransA,
               int Diag,
               int N, float alpha,
               long A, int lda,
               long X, int incX);
    void dtrmv(long *extraParams,int Order, int Uplo, int TransA,
               int Diag,
               int N, double alpha,
               long A, int lda,
               long X, int incX);

    /*
     * ------------------------------------------------------
     * TBMV
     * ------------------------------------------------------
     */

    void stbmv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               long A, int lda,
               long X, int incX);

    void dtbmv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               long A, int lda,
               long X, int incX);

    /*
     * ------------------------------------------------------
     * TPMV
     * ------------------------------------------------------
     */

    void stpmv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long Ap,
               long X, int incX);

    void dtpmv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long Ap,
               long X, int incX);

    /*
     * ------------------------------------------------------
     * TRSV
     * ------------------------------------------------------
     */

    void strsv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long A, int lda,
               long X, int incX);

    void dtrsv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long A, int lda,
               long X, int incX);

    /*
     * ------------------------------------------------------
     * TBSV
     * ------------------------------------------------------
     */

    void stbsv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               long A, int lda,
               long X, int incX);

    void dtbsv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               long A, int lda,
               long X, int incX);

    /*
     * ------------------------------------------------------
     * TPSV
     * ------------------------------------------------------
     */

    void stpsv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long Ap,
               long X, int incX);

    void dtpsv(long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long Ap,
               long X, int incX);
    /*
     * ------------------------------------------------------
     * GER
     * ------------------------------------------------------
     */

    void sger(long *extraParams,int Order,
              int M, int N,
              float alpha,
              long X, int incX,
              long Y, int incY,
              long A, int lda);

    void dger(long *extraParams,int Order,
              int M, int N,
              double alpha,
              long X, int incX,
              long Y, int incY,
              long A, int lda);

    /*
     * ------------------------------------------------------
     * SYR
     * ------------------------------------------------------
     */

    void ssyr(long *extraParams,int Order, int Uplo,
              int N,
              float alpha,
              long X, int incX,
              long A, int lda);

    void dsyr(long *extraParams,int Order, int Uplo,
              int N,
              double alpha,
              long X, int incX,
              long A, int lda);

    /*
     * ------------------------------------------------------
     * SPR
     * ------------------------------------------------------
     */

    void sspr(long *extraParams,int Order, int Uplo,
              int N,
              float alpha,
              long X, int incX,
              long Ap);

    void dspr(long *extraParams,int Order, int Uplo,
              int N,
              double alpha,
              long X, int incX,
              long Ap);

    /*
     * ------------------------------------------------------
     * SYR2
     * ------------------------------------------------------
     */

    void ssyr2(long *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               long X, int incX,
               long Y, int incY,
               long A, int lda);

    void dsyr2(long *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               long X, int incX,
               long Y, int incY,
               long A, int lda);

    /*
     * ------------------------------------------------------
     * SPR2
     * ------------------------------------------------------
     */

    void sspr2(long *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               long X, int incX,
               long Y, int incY,
               long Ap);

    void dspr2(long *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               long X, int incX,
               long Y, int incY,
               long Ap);

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

    void sgemm(long *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               float alpha,
               long A, int lda,
               long B, int ldb,
               float beta,
               long C, int ldc);

    void dgemm(long *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               double alpha,
               long A, int lda,
               long B, int ldb,
               double beta,
               long C, int ldc);

    /*
     * ------------------------------------------------------
     * SYMM
     * ------------------------------------------------------
     */

    void ssymm(long *extraParams,int Order, int Side, int Uplo,
               int M, int N,
               float alpha,
               long A, int lda,
               long B, int ldb,
               float beta,
               long C, int ldc);

    void dsymm(long *extraParams,int Order, int Side, int Uplo,
               int M, int N,
               double alpha,
               long A, int lda,
               long B, int ldb,
               double beta,
               long C, int ldc);

    /*
     * ------------------------------------------------------
     * SYRK
     * ------------------------------------------------------
     */

    void ssyrk(long *extraParams,int Order, int Uplo, int Trans,
               int N, int K,
               float alpha,
               long A, int lda,
               float beta,
               long C, int ldc);

    void dsyrk(long *extraParams,int Order, int Uplo, int Trans,
               int N, int K,
               double alpha,
               long A, int lda,
               double beta,
               long C, int ldc);

    /*
     * ------------------------------------------------------
     * SYR2K
     * ------------------------------------------------------
     */

    void ssyr2k(long *extraParams,int Order, int Uplo, int Trans,
                int N, int K,
                float alpha,
                long A, int lda,
                long B, int ldb,
                float beta,
                long C, int ldc);

    void dsyr2k(long *extraParams,int Order, int Uplo, int Trans,
                int N, int K,
                double alpha,
                long A, int lda,
                long B, int ldb,
                double beta,
                long C, int ldc);

    /*
     * ------------------------------------------------------
     * TRMM
     * ------------------------------------------------------
     */

    void strmm(long *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               float alpha,
               long A, int lda,
               long B, int ldb);

    void dtrmm(long *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               double alpha,
               long A, int lda,
               long B, int ldb);

    /*
     * ------------------------------------------------------
     * TRSM
     * ------------------------------------------------------
     */

    void strsm(long *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               float alpha,
               long A, int lda,
               long B, int ldb);

    void dtrsm(long *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               double alpha,
               long A, int lda,
               long B, int ldb);

};

#endif //NATIVEOPERATIONS_NATIVEBLAS_H
