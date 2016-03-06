//
// Created by agibsonccc on 2/20/16.
//

#ifndef NATIVEOPERATIONS_NATIVEBLAS_H
#define NATIVEOPERATIONS_NATIVEBLAS_H

#include <pointercast.h>


#ifdef _WIN32
#define __declspec(dllexport)
#endif

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
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    float sdsdot(Nd4jPointer *extraParams,int N, float alpha,
                 Nd4jPointer X, int incX,
                 Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    double dsdot(Nd4jPointer *extraParams,int N,
                 Nd4jPointer X, int incX,
                 Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    double ddot(Nd4jPointer *extraParams,int N,
                Nd4jPointer X, int incX,
                Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    float sdot(Nd4jPointer *extraParams,int N,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * NRM2
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    float snrm2(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    double dnrm2(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX);

    /*
     * ------------------------------------------------------
     * ASUM
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    float sasum(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    double dasum(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX);

    /*
     * ------------------------------------------------------
     * IAMAX
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    int isamax(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    int idamax(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX);

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
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void srot(Nd4jPointer *extraParams,int N,
              Nd4jPointer X, int incX,
              Nd4jPointer Y, int incY,
              float c, float s);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void drot(Nd4jPointer *extraParams,int N,
              Nd4jPointer X, int incX,
              Nd4jPointer Y, int incY,
              double c, double s);

    /*
     * ------------------------------------------------------
     * ROTG
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void srotg(Nd4jPointer *extraParams,Nd4jPointer args);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void drotg(Nd4jPointer *extraParams,Nd4jPointer args);

    /*
     * ------------------------------------------------------
     * ROTMG
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void srotmg(Nd4jPointer *extraParams,Nd4jPointer args,
                Nd4jPointer P);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void drotmg(Nd4jPointer *extraParams,Nd4jPointer args,
                Nd4jPointer P);

    /*
     * ------------------------------------------------------
     * ROTM
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void srotm(Nd4jPointer *extraParams,int N,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY,
               Nd4jPointer P);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void drotm(Nd4jPointer *extraParams,int N,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY,
               Nd4jPointer P);

    /*
     * ------------------------------------------------------
     * SWAP
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sswap(Nd4jPointer *extraParams,int N,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dswap(Nd4jPointer *extraParams,int N,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SCAL
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sscal(Nd4jPointer *extraParams,int N, float alpha,
               Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dscal(Nd4jPointer *extraParams,int N, double alpha,
               Nd4jPointer X, int incX);

    /*
     * ------------------------------------------------------
     * SCOPY
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void scopy(Nd4jPointer *extraParams,int N,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dcopy(Nd4jPointer *extraParams,int N,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * AXPY
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void saxpy(Nd4jPointer *extraParams,int N, float alpha,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void daxpy(Nd4jPointer *extraParams,int N, double alpha,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY);

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
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sgemv(Nd4jPointer *extraParams,int Order, int TransA,
               int M, int N,
               float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX,
               float beta,
               Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dgemv(Nd4jPointer *extraParams,int Order, int TransA,
               int M, int N,
               double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX,
               double beta,
               Nd4jPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * GBMV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sgbmv(Nd4jPointer *extraParams,int Order, int TransA,
               int M, int N,
               int KL, int KU,
               float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX,
               float beta,
               Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dgbmv(Nd4jPointer *extraParams,int Order, int TransA,
               int M, int N,
               int KL, int KU,
               double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX,
               double beta,
               Nd4jPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SYMV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void ssymv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX,
               float beta,
               Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dsymv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX,
               double beta,
               Nd4jPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SBMV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void ssbmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N, int K,
               float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX,
               float beta,
               Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dsbmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N, int K,
               double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX,
               double beta,
               Nd4jPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SPMV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sspmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               Nd4jPointer Ap,
               Nd4jPointer X, int incX,
               float beta,
               Nd4jPointer Y, int incY);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dspmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               Nd4jPointer Ap,
               Nd4jPointer X, int incX,
               double beta,
               Nd4jPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * TRMV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void strmv(Nd4jPointer *extraParams,int Order, int Uplo, int TransA,
               int Diag,
               int N, float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dtrmv(Nd4jPointer *extraParams,int Order, int Uplo, int TransA,
               int Diag,
               int N, double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX);

    /*
     * ------------------------------------------------------
     * TBMV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void stbmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dtbmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX);

    /*
     * ------------------------------------------------------
     * TPMV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void stpmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               Nd4jPointer Ap,
               Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dtpmv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               Nd4jPointer Ap,
               Nd4jPointer X, int incX);

    /*
     * ------------------------------------------------------
     * TRSV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void strsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dtrsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX);

    /*
     * ------------------------------------------------------
     * TBSV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void stbsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dtbsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N, int K,
               Nd4jPointer A, int lda,
               Nd4jPointer X, int incX);

    /*
     * ------------------------------------------------------
     * TPSV
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void stpsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               Nd4jPointer Ap,
               Nd4jPointer X, int incX);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dtpsv(Nd4jPointer *extraParams,int Order, int Uplo,
               int TransA, int Diag,
               int N,
               Nd4jPointer Ap,
               Nd4jPointer X, int incX);
    /*
     * ------------------------------------------------------
     * GER
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sger(Nd4jPointer *extraParams,int Order,
              int M, int N,
              float alpha,
              Nd4jPointer X, int incX,
              Nd4jPointer Y, int incY,
              Nd4jPointer A, int lda);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dger(Nd4jPointer *extraParams,int Order,
              int M, int N,
              double alpha,
              Nd4jPointer X, int incX,
              Nd4jPointer Y, int incY,
              Nd4jPointer A, int lda);

    /*
     * ------------------------------------------------------
     * SYR
     * ------------------------------------------------------
     */

#ifdef _WIN32
#define __declspec(dllexport)
#endif
    void ssyr(Nd4jPointer *extraParams,int Order, int Uplo,
              int N,
              float alpha,
              Nd4jPointer X, int incX,
              Nd4jPointer A, int lda);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dsyr(Nd4jPointer *extraParams,int Order, int Uplo,
              int N,
              double alpha,
              Nd4jPointer X, int incX,
              Nd4jPointer A, int lda);

    /*
     * ------------------------------------------------------
     * SPR
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sspr(Nd4jPointer *extraParams,int Order, int Uplo,
              int N,
              float alpha,
              Nd4jPointer X, int incX,
              Nd4jPointer Ap);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dspr(Nd4jPointer *extraParams,int Order, int Uplo,
              int N,
              double alpha,
              Nd4jPointer X, int incX,
              Nd4jPointer Ap);

    /*
     * ------------------------------------------------------
     * SYR2
     * ------------------------------------------------------
     */

#ifdef _WIN32
#define __declspec(dllexport)
#endif
    void ssyr2(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY,
               Nd4jPointer A, int lda);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dsyr2(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY,
               Nd4jPointer A, int lda);

    /*
     * ------------------------------------------------------
     * SPR2
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sspr2(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               float alpha,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY,
               Nd4jPointer Ap);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dspr2(Nd4jPointer *extraParams,int Order, int Uplo,
               int N,
               double alpha,
               Nd4jPointer X, int incX,
               Nd4jPointer Y, int incY,
               Nd4jPointer Ap);

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
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void sgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer B, int ldb,
               float beta,
               Nd4jPointer C, int ldc);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
               int M, int N, int K,
               double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer B, int ldb,
               double beta,
               Nd4jPointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYMM
     * ------------------------------------------------------
     */

#ifdef _WIN32
#define __declspec(dllexport)
#endif
    void ssymm(Nd4jPointer *extraParams,int Order, int Side, int Uplo,
               int M, int N,
               float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer B, int ldb,
               float beta,
               Nd4jPointer C, int ldc);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dsymm(Nd4jPointer *extraParams,int Order, int Side, int Uplo,
               int M, int N,
               double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer B, int ldb,
               double beta,
               Nd4jPointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYRK
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void ssyrk(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
               int N, int K,
               float alpha,
               Nd4jPointer A, int lda,
               float beta,
               Nd4jPointer C, int ldc);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dsyrk(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
               int N, int K,
               double alpha,
               Nd4jPointer A, int lda,
               double beta,
               Nd4jPointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYR2K
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void ssyr2k(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                int N, int K,
                float alpha,
                Nd4jPointer A, int lda,
                Nd4jPointer B, int ldb,
                float beta,
                Nd4jPointer C, int ldc);

#ifdef _WIN32
#define __declspec(dllexport)
#endif
    void dsyr2k(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                int N, int K,
                double alpha,
                Nd4jPointer A, int lda,
                Nd4jPointer B, int ldb,
                double beta,
                Nd4jPointer C, int ldc);

    /*
     * ------------------------------------------------------
     * TRMM
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void strmm(Nd4jPointer *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer B, int ldb);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dtrmm(Nd4jPointer *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer B, int ldb);

    /*
     * ------------------------------------------------------
     * TRSM
     * ------------------------------------------------------
     */
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void strsm(Nd4jPointer *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               float alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer B, int ldb);
#ifdef _WIN32
#define __declspec(dllexport)
#endif

    void dtrsm(Nd4jPointer *extraParams,int Order, int Side,
               int Uplo, int TransA, int Diag,
               int M, int N,
               double alpha,
               Nd4jPointer A, int lda,
               Nd4jPointer B, int ldb);

};

#endif //NATIVEOPERATIONS_NATIVEBLAS_H
