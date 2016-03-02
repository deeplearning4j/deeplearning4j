package org.nd4j.linalg.cpu.nativecpu;



import org.nd4j.linalg.cpu.nativecpu.util.LibUtils;

import java.nio.Buffer;

/**
 * CBlas bindings
 *
 * Original credit:
 * https://github.com/uncomplicate/neanderthal-atlas
 *
 *
 */

public class CBLAS {
    static {
        LibUtils.loadLibrary("libnd4j");
    }

    public static final int ORDER_ROW_MAJOR = 101;
    public static final int ORDER_COLUMN_MAJOR = 102;
    public static final int TRANSPOSE_NO_TRANS = 111;
    public static final int TRANSPOSE_TRANS = 112;
    public static final int TRANSPOSE_CONJ_TRANS = 113;
    public static final int TRANSPOSE_ATLAS_CONJ = 114;
    public static final int UPLO_UPPER = 121;
    public static final int UPLO_LOWER = 122;
    public static final int DIAG_NON_UNIT = 131;
    public static final int DIAG_UNIT = 132;
    public static final int SIDE_LEFT = 141;
    public static final int SIDE_RIGHT = 142;



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

    public static native float sdsdot(int N, float alpha,
                                      Buffer X, int incX,
                                      Buffer Y, int incY);

    public static native double dsdot(int N,
                                      Buffer X, int incX,
                                      Buffer Y, int incY);

    public static native double ddot(int N,
                                     Buffer X, int incX,
                                     Buffer Y, int incY);

    public static native float sdot(int N,
                                    Buffer X, int incX,
                                    Buffer Y, int incY);

    /*
     * ------------------------------------------------------
     * NRM2
     * ------------------------------------------------------
     */

    public static native float snrm2(int N, Buffer X, int incX);

    public static native double dnrm2(int N, Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * ASUM
     * ------------------------------------------------------
     */

    public static native float sasum(int N, Buffer X, int incX);

    public static native double dasum(int N, Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * IAMAX
     * ------------------------------------------------------
     */

    public static native int isamax(int N, Buffer X, int incX);

    public static native int idamax(int N, Buffer X, int incX);

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

    public static native void srot(int N,
                                   Buffer X, int incX,
                                   Buffer Y, int incY,
                                   float c, float s);

    public static native void drot(int N,
                                   Buffer X, int incX,
                                   Buffer Y, int incY,
                                   double c, double s);

    /*
     * ------------------------------------------------------
     * ROTG
     * ------------------------------------------------------
     */

    public static native void srotg(Buffer args);

    public static native void drotg(Buffer args);

    /*
     * ------------------------------------------------------
     * ROTMG
     * ------------------------------------------------------
     */

    public static native void srotmg(Buffer args,
                                     Buffer P);

    public static native void drotmg(Buffer args,
                                     Buffer P);

    /*
     * ------------------------------------------------------
     * ROTM
     * ------------------------------------------------------
     */

    public static native void srotm(int N,
                                    Buffer X, int incX,
                                    Buffer Y, int incY,
                                    Buffer P);

    public static native void drotm(int N,
                                    Buffer X, int incX,
                                    Buffer Y, int incY,
                                    Buffer P);

    /*
     * ------------------------------------------------------
     * SWAP
     * ------------------------------------------------------
     */

    public static native void sswap(int N,
                                    Buffer X, int incX,
                                    Buffer Y, int incY);

    public static native void dswap(int N,
                                    Buffer X, int incX,
                                    Buffer Y, int incY);

    /*
     * ------------------------------------------------------
     * SCAL
     * ------------------------------------------------------
     */

    public static native void sscal(int N, float alpha,
                                    Buffer X, int incX);

    public static native void dscal(int N, double alpha,
                                    Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * SCOPY
     * ------------------------------------------------------
     */

    public static native void scopy(int N,
                                    Buffer X, int incX,
                                    Buffer Y, int incY);

    public static native void dcopy(int N,
                                    Buffer X, int incX,
                                    Buffer Y, int incY);

    /*
     * ------------------------------------------------------
     * AXPY
     * ------------------------------------------------------
     */

    public static native void saxpy(int N, float alpha,
                                    Buffer X, int incX,
                                    Buffer Y, int incY);

    public static native void daxpy(int N, double alpha,
                                    Buffer X, int incX,
                                    Buffer Y, int incY);

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

    public static native void sgemv(int Order, int TransA,
                                    int M, int N,
                                    float alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX,
                                    float beta,
                                    Buffer Y, int incY);

    public static native void dgemv(int Order, int TransA,
                                    int M, int N,
                                    double alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX,
                                    double beta,
                                    Buffer Y, int incY);

    /*
     * ------------------------------------------------------
     * GBMV
     * ------------------------------------------------------
     */

    public static native void sgbmv(int Order, int TransA,
                                    int M, int N,
                                    int KL, int KU,
                                    float alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX,
                                    float beta,
                                    Buffer Y, int incY);

    public static native void dgbmv(int Order, int TransA,
                                    int M, int N,
                                    int KL, int KU,
                                    double alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX,
                                    double beta,
                                    Buffer Y, int incY);

    /*
     * ------------------------------------------------------
     * SYMV
     * ------------------------------------------------------
     */

    public static native void ssymv(int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX,
                                    float beta,
                                    Buffer Y, int incY);

    public static native void dsymv(int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX,
                                    double beta,
                                    Buffer Y, int incY);

    /*
     * ------------------------------------------------------
     * SBMV
     * ------------------------------------------------------
     */

    public static native void ssbmv(int Order, int Uplo,
                                    int N, int K,
                                    float alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX,
                                    float beta,
                                    Buffer Y, int incY);

    public static native void dsbmv(int Order, int Uplo,
                                    int N, int K,
                                    double alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX,
                                    double beta,
                                    Buffer Y, int incY);

    /*
     * ------------------------------------------------------
     * SPMV
     * ------------------------------------------------------
     */

    public static native void sspmv(int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    Buffer Ap,
                                    Buffer X, int incX,
                                    float beta,
                                    Buffer Y, int incY);

    public static native void dspmv(int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    Buffer Ap,
                                    Buffer X, int incX,
                                    double beta,
                                    Buffer Y, int incY);

    /*
     * ------------------------------------------------------
     * TRMV
     * ------------------------------------------------------
     */

    public static native void strmv(int Order, int Uplo, int TransA,
                                    int N, float alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX);

    public static native void dtrmv(int Order, int Uplo, int TransA,
                                    int N, double alpha,
                                    Buffer A, int lda,
                                    Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * TBMV
     * ------------------------------------------------------
     */

    public static native void stbmv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    Buffer A, int lda,
                                    Buffer X, int incX);

    public static native void dtbmv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    Buffer A, int lda,
                                    Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * TPMV
     * ------------------------------------------------------
     */

    public static native void stpmv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Buffer Ap,
                                    Buffer X, int incX);

    public static native void dtpmv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Buffer Ap,
                                    Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * TRSV
     * ------------------------------------------------------
     */

    public static native void strsv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Buffer A, int lda,
                                    Buffer X, int incX);

    public static native void dtrsv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Buffer A, int lda,
                                    Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * TBSV
     * ------------------------------------------------------
     */

    public static native void stbsv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    Buffer A, int lda,
                                    Buffer X, int incX);

    public static native void dtbsv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    Buffer A, int lda,
                                    Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * TPSV
     * ------------------------------------------------------
     */

    public static native void stpsv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Buffer Ap,
                                    Buffer X, int incX);

    public static native void dtpsv(int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Buffer Ap,
                                    Buffer X, int incX);

    /*
     * ------------------------------------------------------
     * GER
     * ------------------------------------------------------
     */

    public static native void sger(int Order,
                                   int M, int N,
                                   float alpha,
                                   Buffer X, int incX,
                                   Buffer Y, int incY,
                                   Buffer A, int lda);

    public static native void dger(int Order,
                                   int M, int N,
                                   double alpha,
                                   Buffer X, int incX,
                                   Buffer Y, int incY,
                                   Buffer A, int lda);

    /*
     * ------------------------------------------------------
     * SYR
     * ------------------------------------------------------
     */

    public static native void ssyr(int Order, int Uplo,
                                   int N,
                                   float alpha,
                                   Buffer X, int incX,
                                   Buffer A, int lda);

    public static native void dsyr(int Order, int Uplo,
                                   int N,
                                   double alpha,
                                   Buffer X, int incX,
                                   Buffer A, int lda);

    /*
     * ------------------------------------------------------
     * SPR
     * ------------------------------------------------------
     */

    public static native void sspr(int Order, int Uplo,
                                   int N,
                                   float alpha,
                                   Buffer X, int incX,
                                   Buffer Ap);

    public static native void dspr(int Order, int Uplo,
                                   int N,
                                   double alpha,
                                   Buffer X, int incX,
                                   Buffer Ap);

    /*
     * ------------------------------------------------------
     * SYR2
     * ------------------------------------------------------
     */

    public static native void ssyr2(int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    Buffer X, int incX,
                                    Buffer Y, int incY,
                                    Buffer A, int lda);

    public static native void dsyr2(int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    Buffer X, int incX,
                                    Buffer Y, int incY,
                                    Buffer A, int lda);

    /*
     * ------------------------------------------------------
     * SPR2
     * ------------------------------------------------------
     */

    public static native void sspr2(int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    Buffer X, int incX,
                                    Buffer Y, int incY,
                                    Buffer Ap);

    public static native void dspr2(int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    Buffer X, int incX,
                                    Buffer Y, int incY,
                                    Buffer Ap);

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

    public static native void sgemm(int Order, int TransA, int TransB,
                                    int M, int N, int K,
                                    float alpha,
                                    Buffer A, int lda,
                                    Buffer B, int ldb,
                                    float beta,
                                    Buffer C, int ldc);

    public static native void dgemm(int Order, int TransA, int TransB,
                                    int M, int N, int K,
                                    double alpha,
                                    Buffer A, int lda,
                                    Buffer B, int ldb,
                                    double beta,
                                    Buffer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYMM
     * ------------------------------------------------------
     */

    public static native void ssymm(int Order, int Side, int Uplo,
                                    int M, int N,
                                    float alpha,
                                    Buffer A, int lda,
                                    Buffer B, int ldb,
                                    float beta,
                                    Buffer C, int ldc);

    public static native void dsymm(int Order, int Side, int Uplo,
                                    int M, int N,
                                    double alpha,
                                    Buffer A, int lda,
                                    Buffer B, int ldb,
                                    double beta,
                                    Buffer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYRK
     * ------------------------------------------------------
     */

    public static native void ssyrk(int Order, int Uplo, int Trans,
                                    int N, int K,
                                    float alpha,
                                    Buffer A, int lda,
                                    float beta,
                                    Buffer C, int ldc);

    public static native void dsyrk(int Order, int Uplo, int Trans,
                                    int N, int K,
                                    double alpha,
                                    Buffer A, int lda,
                                    double beta,
                                    Buffer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYR2K
     * ------------------------------------------------------
     */

    public static native void ssyr2k(int Order, int Uplo, int Trans,
                                     int N, int K,
                                     float alpha,
                                     Buffer A, int lda,
                                     Buffer B, int ldb,
                                     float beta,
                                     Buffer C, int ldc);

    public static native void dsyr2k(int Order, int Uplo, int Trans,
                                     int N, int K,
                                     double alpha,
                                     Buffer A, int lda,
                                     Buffer B, int ldb,
                                     double beta,
                                     Buffer C, int ldc);

    /*
     * ------------------------------------------------------
     * TRMM
     * ------------------------------------------------------
     */

    public static native void strmm(int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    float alpha,
                                    Buffer A, int lda,
                                    Buffer B, int ldb);

    public static native void dtrmm(int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    double alpha,
                                    Buffer A, int lda,
                                    Buffer B, int ldb);

    /*
     * ------------------------------------------------------
     * TRSM
     * ------------------------------------------------------
     */

    public static native void strsm(int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    float alpha,
                                    Buffer A, int lda,
                                    Buffer B, int ldb);

    public static native void dtrsm(int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    double alpha,
                                    Buffer A, int lda,
                                    Buffer B, int ldb);

}
