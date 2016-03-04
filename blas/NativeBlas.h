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

    float sdsdot(long long *extraParams,int N, float alpha,
                 long long X, int incX,
                 long long Y, int incY);

    double dsdot(long long *extraParams,int N,
                 long long X, int incX,
                 long long Y, int incY);

    double ddot(long long *extraParams,int N,
                long long X, int incX,
                long long Y, int incY);

    float sdot(long long *extraParams,int N,
               long long X, int incX,
               long long Y, int incY);

    /*
     * ------------------------------------------------------
     * NRM2
     * ------------------------------------------------------
     */

    float snrm2(long long *extraParams,int N, long long X, int incX);

    double dnrm2(long long *extraParams,int N, long long X, int incX);

    /*
     * ------------------------------------------------------
     * ASUM
     * ------------------------------------------------------
     */

    float sasum(long long *extraParams,int N, long long X, int incX);

    double dasum(long long *extraParams,int N, long long X, int incX);

    /*
     * ------------------------------------------------------
     * IAMAX
     * ------------------------------------------------------
     */

    int isamax(long long *extraParams,int N, long long X, int incX);

    int idamax(long long *extraParams,int N, long long X, int incX);

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

    void srot(long long *extraParams,int N,
              long long X, int incX,
              long long Y, int incY,
              float c, float s);

    void drot(long long *extraParams,int N,
              long long X, int incX,
              long long Y, int incY,
              double c, double s);

    /*
     * ------------------------------------------------------
     * ROTG
     * ------------------------------------------------------
     */

    void srotg(long long *extraParams,long long args);

    void drotg(long long *extraParams,long long args);

    /*
     * ------------------------------------------------------
     * ROTMG
     * ------------------------------------------------------
     */

    void srotmg(long long *extraParams,long long args,
                long long P);

    void drotmg(long long *extraParams,long long args,
                long long P);

    /*
     * ------------------------------------------------------
     * ROTM
     * ------------------------------------------------------
     */

    void srotm(long long *extraParams,int N,
               long long X, int incX,
               long long Y, int incY,
               long long P);

    void drotm(long long *extraParams,int N,
               long long X, int incX,
               long long Y, int incY,
               long long P);

    /*
     * ------------------------------------------------------
     * SWAP
     * ------------------------------------------------------
     */

    void sswap(long long *extraParams,int N,
               long long X, int incX,
               long long Y, int incY);

    void dswap(long long *extraParams,int N,
               long long X, int incX,
               long long Y, int incY);

    /*
     * ------------------------------------------------------
     * SCAL
     * ------------------------------------------------------
     */

    void sscal(long long *extraParams,int N, float alpha,
               long long X, int incX);

    void dscal(long long *extraParams,int N, double alpha,
               long long X, int incX);

    /*
     * ------------------------------------------------------
     * SCOPY
     * ------------------------------------------------------
     */

    void scopy(long long *extraParams,int N,
               long long X, int incX,
               long long Y, int incY);

    void dcopy(long long *extraParams,int N,
               long long X, int incX,
               long long Y, int incY);

    /*
     * ------------------------------------------------------
     * AXPY
     * ------------------------------------------------------
     */

    void saxpy(long long *extraParams,int N, float alpha,
               long long X, int incX,
               long long Y, int incY);

    void daxpy(long long *extraParams,int N, double alpha,
               long long X, int incX,
               long long Y, int incY);

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

    void sgemv(long long *extraParams,int Order, int TransA,
               int M, int N,
               float alpha,
               long long A, int lda,
               long long X, int incX,
               float beta,
               long long Y, int incY);

    void dgemv(long long *extraParams,int Order, int TransA,
               int M, int N,
               double alpha,
               long long A, int lda,
               long long X, int incX,
               double beta,
               long long Y, int incY);

    /*
     * ------------------------------------------------------
     * GBMV
     * ------------------------------------------------------
     */

    void sgbmv(long long *extraParams,int Order, int TransA,
               int M, int N,
               int KL, int KU,
               float alpha,
               long long A, int lda,
               long long X, int incX,
               float beta,
               long long Y, int incY);

    void dgbmv(long long *extraParams,int Order, int TransA,
               int M, int N,
               int KL, int KU,
               double alpha,
               long long A, int lda,
               long long X, int incX,
               double beta,
               long long Y, int incY);

    /*
     * ------------------------------------------------------
     * SYMV
     * ------------------------------------------------------
     */

    void ssymv(long long *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               long long A, int lda,
               long long X, int incX,
               float beta,
               long long Y, int incY);

    void dsymv(long long *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               long long A, int lda,
               long long X, int incX,
               double beta,
               long long Y, int incY);

    /*
     * ------------------------------------------------------
     * SBMV
     * ------------------------------------------------------
     */

    void ssbmv(long long *extraParams,int Order, int Uplo,
               int N, int K,
               float alpha,
               long long A, int lda,
               long long X, int incX,
               float beta,
               long long Y, int incY);

    void dsbmv(long long *extraParams,int Order, int Uplo,
               int N, int K,
               double alpha,
               long long A, int lda,
               long long X, int incX,
               double beta,
               long long Y, int incY);

    /*
     * ------------------------------------------------------
     * SPMV
     * ------------------------------------------------------
     */

    void sspmv(long long *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               long long Ap,
               long long X, int incX,
               float beta,
               long long Y, int incY);

    void dspmv(long long *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               long long Ap,
               long long X, int incX,
               double beta,
               long long Y, int incY);

    /*
     * ------------------------------------------------------
     * TRMV
     * ------------------------------------------------------
     */

    void strmv(long long *extraParams,int Order, int Uplo, int TransA,
               int Diag,
               int N, float alpha,
               long long A, int lda,
               long long X, int incX);
    void dtrmv(long long *extraParams,int Order, int Uplo, int TransA,
               int Diag,
               int N, double alpha,
               long long A, int lda,
               long long X, int incX);

    /*
     * ------------------------------------------------------
     * TBMV
     * ------------------------------------------------------
     */

    void stbmv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               long long A, int lda,
               long long X, int incX);

    void dtbmv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               long long A, int lda,
               long long X, int incX);

    /*
     * ------------------------------------------------------
     * TPMV
     * ------------------------------------------------------
     */

    void stpmv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long long Ap,
               long long X, int incX);

    void dtpmv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long long Ap,
               long long X, int incX);

    /*
     * ------------------------------------------------------
     * TRSV
     * ------------------------------------------------------
     */

    void strsv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long long A, int lda,
               long long X, int incX);

    void dtrsv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long long A, int lda,
               long long X, int incX);

    /*
     * ------------------------------------------------------
     * TBSV
     * ------------------------------------------------------
     */

    void stbsv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               long long A, int lda,
               long long X, int incX);

    void dtbsv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               long long A, int lda,
               long long X, int incX);

    /*
     * ------------------------------------------------------
     * TPSV
     * ------------------------------------------------------
     */

    void stpsv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long long Ap,
               long long X, int incX);

    void dtpsv(long long *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               long long Ap,
               long long X, int incX);
    /*
     * ------------------------------------------------------
     * GER
     * ------------------------------------------------------
     */

    void sger(long long *extraParams,int Order,
              int M, int N,
              float alpha,
              long long X, int incX,
              long long Y, int incY,
              long long A, int lda);

    void dger(long long *extraParams,int Order,
              int M, int N,
              double alpha,
              long long X, int incX,
              long long Y, int incY,
              long long A, int lda);

    /*
     * ------------------------------------------------------
     * SYR
     * ------------------------------------------------------
     */

    void ssyr(long long *extraParams,int Order, int Uplo,
              int N,
              float alpha,
              long long X, int incX,
              long long A, int lda);

    void dsyr(long long *extraParams,int Order, int Uplo,
              int N,
              double alpha,
              long long X, int incX,
              long long A, int lda);

    /*
     * ------------------------------------------------------
     * SPR
     * ------------------------------------------------------
     */

    void sspr(long long *extraParams,int Order, int Uplo,
              int N,
              float alpha,
              long long X, int incX,
              long long Ap);

    void dspr(long long *extraParams,int Order, int Uplo,
              int N,
              double alpha,
              long long X, int incX,
              long long Ap);

    /*
     * ------------------------------------------------------
     * SYR2
     * ------------------------------------------------------
     */

    void ssyr2(long long *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               long long X, int incX,
               long long Y, int incY,
               long long A, int lda);

    void dsyr2(long long *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               long long X, int incX,
               long long Y, int incY,
               long long A, int lda);

    /*
     * ------------------------------------------------------
     * SPR2
     * ------------------------------------------------------
     */

    void sspr2(long long *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               long long X, int incX,
               long long Y, int incY,
               long long Ap);

    void dspr2(long long *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               long long X, int incX,
               long long Y, int incY,
               long long Ap);

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

    void sgemm(long long *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               float alpha,
               long long A, int lda,
               long long B, int ldb,
               float beta,
               long long C, int ldc);

    void dgemm(long long *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               double alpha,
               long long A, int lda,
               long long B, int ldb,
               double beta,
               long long C, int ldc);

    /*
     * ------------------------------------------------------
     * SYMM
     * ------------------------------------------------------
     */

    void ssymm(long long *extraParams,int Order, int Side, int Uplo,
               int M, int N,
               float alpha,
               long long A, int lda,
               long long B, int ldb,
               float beta,
               long long C, int ldc);

    void dsymm(long long *extraParams,int Order, int Side, int Uplo,
               int M, int N,
               double alpha,
               long long A, int lda,
               long long B, int ldb,
               double beta,
               long long C, int ldc);

    /*
     * ------------------------------------------------------
     * SYRK
     * ------------------------------------------------------
     */

    void ssyrk(long long *extraParams,int Order, int Uplo, int Trans,
               int N, int K,
               float alpha,
               long long A, int lda,
               float beta,
               long long C, int ldc);

    void dsyrk(long long *extraParams,int Order, int Uplo, int Trans,
               int N, int K,
               double alpha,
               long long A, int lda,
               double beta,
               long long C, int ldc);

    /*
     * ------------------------------------------------------
     * SYR2K
     * ------------------------------------------------------
     */

    void ssyr2k(long long *extraParams,int Order, int Uplo, int Trans,
                int N, int K,
                float alpha,
                long long A, int lda,
                long long B, int ldb,
                float beta,
                long long C, int ldc);

    void dsyr2k(long long *extraParams,int Order, int Uplo, int Trans,
                int N, int K,
                double alpha,
                long long A, int lda,
                long long B, int ldb,
                double beta,
                long long C, int ldc);

    /*
     * ------------------------------------------------------
     * TRMM
     * ------------------------------------------------------
     */

    void strmm(long long *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               float alpha,
               long long A, int lda,
               long long B, int ldb);

    void dtrmm(long long *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               double alpha,
               long long A, int lda,
               long long B, int ldb);

    /*
     * ------------------------------------------------------
     * TRSM
     * ------------------------------------------------------
     */

    void strsm(long long *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               float alpha,
               long long A, int lda,
               long long B, int ldb);

    void dtrsm(long long *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               double alpha,
               long long A, int lda,
               long long B, int ldb);

};

#endif //NATIVEOPERATIONS_NATIVEBLAS_H
