package org.nd4j.nativeblas;


import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.buffer.util.LibUtils;

/**
 * CBlas bindings
 *
 * Original credit:
 * https://github.com/uncomplicate/neanderthal-atlas
 *
 *
 */
@Platform(include="NativeBlas.h",link = "nd4j")
public class Nd4jBlas extends Pointer {
    static {
        Loader.load();
    }

    public Nd4jBlas() {
        allocate();
    }

    private native void allocate();

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

    public native float sdsdot(long[] extraPointers,int N, float alpha,
                                      long X, int incX,
                                      long Y, int incY);

    public native double dsdot(long[] extraPointers,int N,
                                      long X, int incX,
                                      long Y, int incY);

    public native double ddot(long[] extraPointers,int N,
                                     long X, int incX,
                                     long Y, int incY);

    public native float sdot(long[] extraPointers,int N,
                                    long X, int incX,
                                    long Y, int incY);

    /*
     * ------------------------------------------------------
     * NRM2
     * ------------------------------------------------------
     */

    public native float snrm2(long[] extraPointers,int N, long X, int incX);

    public native double dnrm2(long[] extraPointers,int N, long X, int incX);

    /*
     * ------------------------------------------------------
     * ASUM
     * ------------------------------------------------------
     */

    public native float sasum(long[] extraPointers,int N, long X, int incX);

    public native double dasum(long[] extraPointers,int N, long X, int incX);

    /*
     * ------------------------------------------------------
     * IAMAX
     * ------------------------------------------------------
     */

    public native int isamax(long[] extraPointers,int N, long X, int incX);

    public native int idamax(long[] extraPointers,int N, long X, int incX);

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

    public native void srot(long[] extraPointers,int N,
                                   long X, int incX,
                                   long Y, int incY,
                                   float c, float s);

    public native void drot(long[] extraPointers,int N,
                                   long X, int incX,
                                   long Y, int incY,
                                   double c, double s);

    /*
     * ------------------------------------------------------
     * ROTG
     * ------------------------------------------------------
     */

    public native void srotg(long[] extraPointers,long args);

    public native void drotg(long[] extraPointers,long args);

    /*
     * ------------------------------------------------------
     * ROTMG
     * ------------------------------------------------------
     */

    public native void srotmg(long[] extraPointers,long args,
                                     long P);

    public native void drotmg(long[] extraPointers,long args,
                                     long P);

    /*
     * ------------------------------------------------------
     * ROTM
     * ------------------------------------------------------
     */

    public native void srotm(long[] extraPointers,int N,
                                    long X, int incX,
                                    long Y, int incY,
                                    long P);

    public native void drotm(long[] extraPointers,int N,
                                    long X, int incX,
                                    long Y, int incY,
                                    long P);

    /*
     * ------------------------------------------------------
     * SWAP
     * ------------------------------------------------------
     */

    public native void sswap(long[] extraPointers,int N,
                                    long X, int incX,
                                    long Y, int incY);

    public native void dswap(long[] extraPointers,int N,
                                    long X, int incX,
                                    long Y, int incY);

    /*
     * ------------------------------------------------------
     * SCAL
     * ------------------------------------------------------
     */

    public native void sscal(long[] extraPointers,int N, float alpha,
                                    long X, int incX);

    public native void dscal(long[] extraPointers,int N, double alpha,
                                    long X, int incX);

    /*
     * ------------------------------------------------------
     * SCOPY
     * ------------------------------------------------------
     */

    public native void scopy(long[] extraPointers,int N,
                                    long X, int incX,
                                    long Y, int incY);

    public native void dcopy(long[] extraPointers,int N,
                                    long X, int incX,
                                    long Y, int incY);

    /*
     * ------------------------------------------------------
     * AXPY
     * ------------------------------------------------------
     */

    public native void saxpy(long[] extraPointers,int N, float alpha,
                                    long X, int incX,
                                    long Y, int incY);

    public native void daxpy(long[] extraPointers,int N, double alpha,
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

    public native void sgemv(long[] extraPointers,int Order, int TransA,
                                    int M, int N,
                                    float alpha,
                                    long A, int lda,
                                    long X, int incX,
                                    float beta,
                                    long Y, int incY);

    public native void dgemv(long[] extraPointers,int Order, int TransA,
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

    public native void sgbmv(long[] extraPointers,int Order, int TransA,
                                    int M, int N,
                                    int KL, int KU,
                                    float alpha,
                                    long A, int lda,
                                    long X, int incX,
                                    float beta,
                                    long Y, int incY);

    public native void dgbmv(long[] extraPointers,int Order, int TransA,
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

    public native void ssymv(long[] extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    long A, int lda,
                                    long X, int incX,
                                    float beta,
                                    long Y, int incY);

    public native void dsymv(long[] extraPointers,int Order, int Uplo,
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

    public native void ssbmv(long[] extraPointers,int Order, int Uplo,
                                    int N, int K,
                                    float alpha,
                                    long A, int lda,
                                    long X, int incX,
                                    float beta,
                                    long Y, int incY);

    public native void dsbmv(long[] extraPointers,int Order, int Uplo,
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

    public native void sspmv(long[] extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    long Ap,
                                    long X, int incX,
                                    float beta,
                                    long Y, int incY);

    public native void dspmv(long[] extraPointers,int Order, int Uplo,
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

    public native void strmv(long[] extraPointers,int Order, int Uplo, int TransA,
                             int Diag,
                                    int N, float alpha,
                                    long A, int lda,
                                    long X, int incX);

    public native void dtrmv(long[] extraPointers,int Order, int Uplo, int TransA,
                             int Diag,
                                    int N, double alpha,
                                    long A, int lda,
                                    long X, int incX);

    /*
     * ------------------------------------------------------
     * TBMV
     * ------------------------------------------------------
     */

    public native void stbmv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    long A, int lda,
                                    long X, int incX);

    public native void dtbmv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    long A, int lda,
                                    long X, int incX);

    /*
     * ------------------------------------------------------
     * TPMV
     * ------------------------------------------------------
     */

    public native void stpmv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    long Ap,
                                    long X, int incX);

    public native void dtpmv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    long Ap,
                                    long X, int incX);

    /*
     * ------------------------------------------------------
     * TRSV
     * ------------------------------------------------------
     */

    public native void strsv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    long A, int lda,
                                    long X, int incX);

    public native void dtrsv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    long A, int lda,
                                    long X, int incX);

    /*
     * ------------------------------------------------------
     * TBSV
     * ------------------------------------------------------
     */

    public native void stbsv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    long A, int lda,
                                    long X, int incX);

    public native void dtbsv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    long A, int lda,
                                    long X, int incX);

    /*
     * ------------------------------------------------------
     * TPSV
     * ------------------------------------------------------
     */

    public native void stpsv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    long Ap,
                                    long X, int incX);

    public native void dtpsv(long[] extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    long Ap,
                                    long X, int incX);

    /*
     * ------------------------------------------------------
     * GER
     * ------------------------------------------------------
     */

    public native void sger(long[] extraPointers,int Order,
                                   int M, int N,
                                   float alpha,
                                   long X, int incX,
                                   long Y, int incY,
                                   long A, int lda);

    public native void dger(long[] extraPointers,int Order,
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

    public native void ssyr(long[] extraPointers,int Order, int Uplo,
                                   int N,
                                   float alpha,
                                   long X, int incX,
                                   long A, int lda);

    public native void dsyr(long[] extraPointers,int Order, int Uplo,
                                   int N,
                                   double alpha,
                                   long X, int incX,
                                   long A, int lda);

    /*
     * ------------------------------------------------------
     * SPR
     * ------------------------------------------------------
     */

    public native void sspr(long[] extraPointers,int Order, int Uplo,
                                   int N,
                                   float alpha,
                                   long X, int incX,
                                   long Ap);

    public native void dspr(long[] extraPointers,int Order, int Uplo,
                                   int N,
                                   double alpha,
                                   long X, int incX,
                                   long Ap);

    /*
     * ------------------------------------------------------
     * SYR2
     * ------------------------------------------------------
     */

    public native void ssyr2(long[] extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    long X, int incX,
                                    long Y, int incY,
                                    long A, int lda);

    public native void dsyr2(long[] extraPointers,int Order, int Uplo,
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

    public native void sspr2(long[] extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    long X, int incX,
                                    long Y, int incY,
                                    long Ap);

    public native void dspr2(long[] extraPointers,int Order, int Uplo,
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

    public native void sgemm(long[] extraPointers,int Order, int TransA, int TransB,
                                    int M, int N, int K,
                                    float alpha,
                                    long A, int lda,
                                    long B, int ldb,
                                    float beta,
                                    long C, int ldc);

    public native void dgemm(long[] extraPointers,int Order, int TransA, int TransB,
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

    public native void ssymm(long[] extraPointers,int Order, int Side, int Uplo,
                                    int M, int N,
                                    float alpha,
                                    long A, int lda,
                                    long B, int ldb,
                                    float beta,
                                    long C, int ldc);

    public native void dsymm(long[] extraPointers,int Order, int Side, int Uplo,
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

    public native void ssyrk(long[] extraPointers,int Order, int Uplo, int Trans,
                                    int N, int K,
                                    float alpha,
                                    long A, int lda,
                                    float beta,
                                    long C, int ldc);

    public native void dsyrk(long[] extraPointers,int Order, int Uplo, int Trans,
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

    public native void ssyr2k(long[] extraPointers,int Order, int Uplo, int Trans,
                                     int N, int K,
                                     float alpha,
                                     long A, int lda,
                                     long B, int ldb,
                                     float beta,
                                     long C, int ldc);

    public native void dsyr2k(long[] extraPointers,int Order, int Uplo, int Trans,
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

    public native void strmm(long[] extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    float alpha,
                                    long A, int lda,
                                    long B, int ldb);

    public native void dtrmm(long[] extraPointers,int Order, int Side,
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

    public native void strsm(long[] extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    float alpha,
                                    long A, int lda,
                                    long B, int ldb);

    public native void dtrsm(long[] extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    double alpha,
                                    long A, int lda,
                                    long B, int ldb);

}
