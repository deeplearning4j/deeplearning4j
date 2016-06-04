package org.nd4j.nativeblas;


import java.util.Properties;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.annotation.Platform;

/**
 * CBlas bindings
 *
 * Original credit:
 * https://github.com/uncomplicate/neanderthal-atlas
 */
@Platform(include = "NativeBlas.h", compiler = "cpp11", link = "nd4j", library = "jnind4j")
public class Nd4jBlas extends Pointer {
    static {
        // using our custom platform properties from resources, and on user request,
        // load in priority libraries found in the library path over bundled ones
        String platform = Loader.getPlatform();
        Properties properties = Loader.loadProperties(platform + "-nd4j", platform);
        properties.remove("platform.preloadpath");
        String s = System.getProperty("org.nd4j.nativeblas.pathsfirst", "false").toLowerCase();
        boolean pathsFirst = s.equals("true") || s.equals("t") || s.equals("");
        Loader.load(Nd4jBlas.class, properties, pathsFirst);
    }

    public Nd4jBlas() {
        allocate();
    }

    private native void allocate();

/*
     * ======================================================
     * Level 1 BLAS functions
     * ===========================in==========================
     */


    /*
     * ------------------------------------------------------
     * DOT
     * ------------------------------------------------------
     */

    public native float sdsdot(PointerPointer extraPointers,int N, float alpha,
                                      Pointer X, int incX,
                                      Pointer Y, int incY);

    public native double dsdot(PointerPointer extraPointers,int N,
                                      Pointer X, int incX,
                                      Pointer Y, int incY);

    public native double ddot(PointerPointer extraPointers,int N,
                                     Pointer X, int incX,
                                     Pointer Y, int incY);

    public native float sdot(PointerPointer extraPointers,int N,
                                    Pointer X, int incX,
                                    Pointer Y, int incY);

    /*
     * ------------------------------------------------------
     * NRM2
     * ------------------------------------------------------
     */

    public native float snrm2(PointerPointer extraPointers,int N, Pointer X, int incX);

    public native double dnrm2(PointerPointer extraPointers,int N, Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * ASUM
     * ------------------------------------------------------
     */

    public native float sasum(PointerPointer extraPointers,int N, Pointer X, int incX);

    public native double dasum(PointerPointer extraPointers,int N, Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * IAMAX
     * ------------------------------------------------------
     */

    public native int isamax(PointerPointer extraPointers,int N, Pointer X, int incX);

    public native int idamax(PointerPointer extraPointers,int N, Pointer X, int incX);

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

    public native void srot(PointerPointer extraPointers,int N,
                                   Pointer X, int incX,
                                   Pointer Y, int incY,
                                   float c, float s);

    public native void drot(PointerPointer extraPointers,int N,
                                   Pointer X, int incX,
                                   Pointer Y, int incY,
                                   double c, double s);

    /*
     * ------------------------------------------------------
     * ROTG
     * ------------------------------------------------------
     */

    public native void srotg(PointerPointer extraPointers,Pointer args);

    public native void drotg(PointerPointer extraPointers,Pointer args);

    /*
     * ------------------------------------------------------
     * ROTMG
     * ------------------------------------------------------
     */

    public native void srotmg(PointerPointer extraPointers,Pointer args,
                                     Pointer P);

    public native void drotmg(PointerPointer extraPointers,Pointer args,
                                     Pointer P);

    /*
     * ------------------------------------------------------
     * ROTM
     * ------------------------------------------------------
     */

    public native void srotm(PointerPointer extraPointers,int N,
                                    Pointer X, int incX,
                                    Pointer Y, int incY,
                                    Pointer P);

    public native void drotm(PointerPointer extraPointers,int N,
                                    Pointer X, int incX,
                                    Pointer Y, int incY,
                                    Pointer P);

    /*
     * ------------------------------------------------------
     * SWAP
     * ------------------------------------------------------
     */

    public native void sswap(PointerPointer extraPointers,int N,
                                    Pointer X, int incX,
                                    Pointer Y, int incY);

    public native void dswap(PointerPointer extraPointers,int N,
                                    Pointer X, int incX,
                                    Pointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SCAL
     * ------------------------------------------------------
     */

    public native void sscal(PointerPointer extraPointers,int N, float alpha,
                                    Pointer X, int incX);

    public native void dscal(PointerPointer extraPointers,int N, double alpha,
                                    Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * SCOPY
     * ------------------------------------------------------
     */

    public native void scopy(PointerPointer extraPointers,int N,
                                    Pointer X, int incX,
                                    Pointer Y, int incY);

    public native void dcopy(PointerPointer extraPointers,int N,
                                    Pointer X, int incX,
                                    Pointer Y, int incY);

    /*
     * ------------------------------------------------------
     * AXPY
     * ------------------------------------------------------
     */

    public native void saxpy(PointerPointer extraPointers,int N, float alpha,
                                    Pointer X, int incX,
                                    Pointer Y, int incY);

    public native void daxpy(PointerPointer extraPointers,int N, double alpha,
                                    Pointer X, int incX,
                                    Pointer Y, int incY);

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

    public native void sgemv(PointerPointer extraPointers,int Order, int TransA,
                                    int M, int N,
                                    float alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX,
                                    float beta,
                                    Pointer Y, int incY);

    public native void dgemv(PointerPointer extraPointers,int Order, int TransA,
                                    int M, int N,
                                    double alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX,
                                    double beta,
                                    Pointer Y, int incY);

    /*
     * ------------------------------------------------------
     * GBMV
     * ------------------------------------------------------
     */

    public native void sgbmv(PointerPointer extraPointers,int Order, int TransA,
                                    int M, int N,
                                    int KL, int KU,
                                    float alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX,
                                    float beta,
                                    Pointer Y, int incY);

    public native void dgbmv(PointerPointer extraPointers,int Order, int TransA,
                                    int M, int N,
                                    int KL, int KU,
                                    double alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX,
                                    double beta,
                                    Pointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SYMV
     * ------------------------------------------------------
     */

    public native void ssymv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX,
                                    float beta,
                                    Pointer Y, int incY);

    public native void dsymv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX,
                                    double beta,
                                    Pointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SBMV
     * ------------------------------------------------------
     */

    public native void ssbmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N, int K,
                                    float alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX,
                                    float beta,
                                    Pointer Y, int incY);

    public native void dsbmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N, int K,
                                    double alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX,
                                    double beta,
                                    Pointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SPMV
     * ------------------------------------------------------
     */

    public native void sspmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    Pointer Ap,
                                    Pointer X, int incX,
                                    float beta,
                                    Pointer Y, int incY);

    public native void dspmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    Pointer Ap,
                                    Pointer X, int incX,
                                    double beta,
                                    Pointer Y, int incY);

    /*
     * ------------------------------------------------------
     * TRMV
     * ------------------------------------------------------
     */

    public native void strmv(PointerPointer extraPointers,int Order, int Uplo, int TransA,
                             int Diag,
                                    int N, float alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX);

    public native void dtrmv(PointerPointer extraPointers,int Order, int Uplo, int TransA,
                             int Diag,
                                    int N, double alpha,
                                    Pointer A, int lda,
                                    Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * TBMV
     * ------------------------------------------------------
     */

    public native void stbmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    Pointer A, int lda,
                                    Pointer X, int incX);

    public native void dtbmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    Pointer A, int lda,
                                    Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * TPMV
     * ------------------------------------------------------
     */

    public native void stpmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Pointer Ap,
                                    Pointer X, int incX);

    public native void dtpmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Pointer Ap,
                                    Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * TRSV
     * ------------------------------------------------------
     */

    public native void strsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Pointer A, int lda,
                                    Pointer X, int incX);

    public native void dtrsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Pointer A, int lda,
                                    Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * TBSV
     * ------------------------------------------------------
     */

    public native void stbsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    Pointer A, int lda,
                                    Pointer X, int incX);

    public native void dtbsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    Pointer A, int lda,
                                    Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * TPSV
     * ------------------------------------------------------
     */

    public native void stpsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Pointer Ap,
                                    Pointer X, int incX);

    public native void dtpsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    Pointer Ap,
                                    Pointer X, int incX);

    /*
     * ------------------------------------------------------
     * GER
     * ------------------------------------------------------
     */

    public native void sger(PointerPointer extraPointers,int Order,
                                   int M, int N,
                                   float alpha,
                                   Pointer X, int incX,
                                   Pointer Y, int incY,
                                   Pointer A, int lda);

    public native void dger(PointerPointer extraPointers,int Order,
                                   int M, int N,
                                   double alpha,
                                   Pointer X, int incX,
                                   Pointer Y, int incY,
                                   Pointer A, int lda);

    /*
     * ------------------------------------------------------
     * SYR
     * ------------------------------------------------------
     */

    public native void ssyr(PointerPointer extraPointers,int Order, int Uplo,
                                   int N,
                                   float alpha,
                                   Pointer X, int incX,
                                   Pointer A, int lda);

    public native void dsyr(PointerPointer extraPointers,int Order, int Uplo,
                                   int N,
                                   double alpha,
                                   Pointer X, int incX,
                                   Pointer A, int lda);

    /*
     * ------------------------------------------------------
     * SPR
     * ------------------------------------------------------
     */

    public native void sspr(PointerPointer extraPointers,int Order, int Uplo,
                                   int N,
                                   float alpha,
                                   Pointer X, int incX,
                                   Pointer Ap);

    public native void dspr(PointerPointer extraPointers,int Order, int Uplo,
                                   int N,
                                   double alpha,
                                   Pointer X, int incX,
                                   Pointer Ap);

    /*
     * ------------------------------------------------------
     * SYR2
     * ------------------------------------------------------
     */

    public native void ssyr2(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    Pointer X, int incX,
                                    Pointer Y, int incY,
                                    Pointer A, int lda);

    public native void dsyr2(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    Pointer X, int incX,
                                    Pointer Y, int incY,
                                    Pointer A, int lda);

    /*
     * ------------------------------------------------------
     * SPR2
     * ------------------------------------------------------
     */

    public native void sspr2(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    Pointer X, int incX,
                                    Pointer Y, int incY,
                                    Pointer Ap);

    public native void dspr2(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    Pointer X, int incX,
                                    Pointer Y, int incY,
                                    Pointer Ap);

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

    public native void sgemm(PointerPointer extraPointers,int Order, int TransA, int TransB,
                                    int M, int N, int K,
                                    float alpha,
                                    Pointer A, int lda,
                                    Pointer B, int ldb,
                                    float beta,
                                    Pointer C, int ldc);

    public native void dgemm(PointerPointer extraPointers,int Order, int TransA, int TransB,
                                    int M, int N, int K,
                                    double alpha,
                                    Pointer A, int lda,
                                    Pointer B, int ldb,
                                    double beta,
                                    Pointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYMM
     * ------------------------------------------------------
     */

    public native void ssymm(PointerPointer extraPointers,int Order, int Side, int Uplo,
                                    int M, int N,
                                    float alpha,
                                    Pointer A, int lda,
                                    Pointer B, int ldb,
                                    float beta,
                                    Pointer C, int ldc);

    public native void dsymm(PointerPointer extraPointers,int Order, int Side, int Uplo,
                                    int M, int N,
                                    double alpha,
                                    Pointer A, int lda,
                                    Pointer B, int ldb,
                                    double beta,
                                    Pointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYRK
     * ------------------------------------------------------
     */

    public native void ssyrk(PointerPointer extraPointers,int Order, int Uplo, int Trans,
                                    int N, int K,
                                    float alpha,
                                    Pointer A, int lda,
                                    float beta,
                                    Pointer C, int ldc);

    public native void dsyrk(PointerPointer extraPointers,int Order, int Uplo, int Trans,
                                    int N, int K,
                                    double alpha,
                                    Pointer A, int lda,
                                    double beta,
                                    Pointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYR2K
     * ------------------------------------------------------
     */

    public native void ssyr2k(PointerPointer extraPointers,int Order, int Uplo, int Trans,
                                     int N, int K,
                                     float alpha,
                                     Pointer A, int lda,
                                     Pointer B, int ldb,
                                     float beta,
                                     Pointer C, int ldc);

    public native void dsyr2k(PointerPointer extraPointers,int Order, int Uplo, int Trans,
                                     int N, int K,
                                     double alpha,
                                     Pointer A, int lda,
                                     Pointer B, int ldb,
                                     double beta,
                                     Pointer C, int ldc);

    /*
     * ------------------------------------------------------
     * TRMM
     * ------------------------------------------------------
     */

    public native void strmm(PointerPointer extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    float alpha,
                                    Pointer A, int lda,
                                    Pointer B, int ldb);

    public native void dtrmm(PointerPointer extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    double alpha,
                                    Pointer A, int lda,
                                    Pointer B, int ldb);

    /*
     * ------------------------------------------------------
     * TRSM
     * ------------------------------------------------------
     */

    public native void strsm(PointerPointer extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    float alpha,
                                    Pointer A, int lda,
                                    Pointer B, int ldb);

    public native void dtrsm(PointerPointer extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    double alpha,
                                    Pointer A, int lda,
                                    Pointer B, int ldb);

}
