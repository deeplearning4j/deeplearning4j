package org.nd4j.nativeblas;


import java.util.Properties;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CBlas bindings
 *
 * Original credit:
 * https://github.com/uncomplicate/neanderthal-atlas
 */
@Platform(include = "NativeBlas.h", compiler = "cpp11", link = "nd4j", library = "jnind4j")
public class Nd4jBlas extends Pointer {

    public enum Vendor {
        UNKNOWN,
        CUBLAS,
        OPENBLAS,
        MKL,
    }


    private static Logger logger = LoggerFactory.getLogger(Nd4jBlas.class);
    static {
        // using our custom platform properties from resources, and on user request,
        // load in priority libraries found in the library path over bundled ones
        String platform = Loader.getPlatform();
        Properties properties = Loader.loadProperties(platform + "-nd4j", platform);
        properties.remove("platform.preloadpath");
        String s = System.getProperty("org.nd4j.nativeblas.pathsfirst", "false").toLowerCase();
        boolean pathsFirst = s.equals("true") || s.equals("t") || s.equals("");
        try {
            Loader.load(Nd4jBlas.class, properties, pathsFirst);
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("ND4J is probably missing dependencies. For more information, please refer to: http://nd4j.org/getstarted.html", e);
        }
    }

    public Nd4jBlas() {
        allocate();

        int numThreads;
        String skipper = System.getenv("ND4J_SKIP_BLAS_THREADS");
        if (skipper == null || skipper.isEmpty()) {
            String numThreadsString = System.getenv("OMP_NUM_THREADS");
            if (numThreadsString != null && !numThreadsString.isEmpty()) {
                numThreads = Integer.parseInt(numThreadsString);
                setMaxThreads(numThreads);
            } else {
                numThreads = getCores(Runtime.getRuntime().availableProcessors());
                setMaxThreads(numThreads);
            }
            logger.info("Number of threads used for BLAS: {}", numThreads);
        }
    }

    private int getCores(int totals) {
        // that's special case for Xeon Phi
        if (totals >= 256) return  64;

        int ht_off = totals / 2; // we count off HyperThreading without any excuses
        if (ht_off <= 4) return 4; // special case for Intel i5. and nobody likes i3 anyway

        if (ht_off > 24) {
            int rounds = 0;
            while (ht_off > 24) { // we loop until final value gets below 24 cores, since that's reasonable threshold as of 2016
                if (ht_off > 24) {
                    ht_off /= 2; // we dont' have any cpus that has higher number then 24 physical cores
                    rounds++;
                }
            }
            // 20 threads is special case in this branch
            if (ht_off == 20 && rounds < 2)
                ht_off /= 2;
        } else { // low-core models are known, but there's a gap, between consumer cpus and xeons
            if (ht_off <= 6) {
                // that's more likely consumer-grade cpu, so leave this value alone
                return ht_off;
            } else {
                if (isOdd(ht_off)) // if that's odd number, it's final result
                    return ht_off;

                // 20 threads & 16 threads are special case in this branch, where we go min value
                if (ht_off == 20 || ht_off == 16)
                    ht_off /= 2;
            }
        }
        return ht_off;
    }

    /**
     * This method returns BLAS library vendor
     *
     * @return
     */
    public Vendor getBlasVendor() {
        if (getVendor() > 3)
            return Vendor.UNKNOWN;

        return Vendor.values()[getVendor()];
    }

    private boolean isOdd(int value) {
        return (value % 2 != 0);
    }

    private native void allocate();

    public native void setMaxThreads(int num);

    public native int getMaxThreads();

    protected native int getVendor();

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
                               FloatPointer X, int incX,
                               FloatPointer Y, int incY);

    public native double dsdot(PointerPointer extraPointers,int N,
                               FloatPointer X, int incX,
                               FloatPointer Y, int incY);

    public native double ddot(PointerPointer extraPointers,int N,
                              DoublePointer X, int incX,
                              DoublePointer Y, int incY);

    public native float sdot(PointerPointer extraPointers,int N,
                             FloatPointer X, int incX,
                             FloatPointer Y, int incY);

    /*
     * ------------------------------------------------------
     * NRM2
     * ------------------------------------------------------
     */

    public native float snrm2(PointerPointer extraPointers,int N, FloatPointer X, int incX);

    public native double dnrm2(PointerPointer extraPointers,int N, DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * ASUM
     * ------------------------------------------------------
     */

    public native float sasum(PointerPointer extraPointers,int N, FloatPointer X, int incX);

    public native double dasum(PointerPointer extraPointers,int N, DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * IAMAX
     * ------------------------------------------------------
     */

    public native int isamax(PointerPointer extraPointers,int N, FloatPointer X, int incX);

    public native int idamax(PointerPointer extraPointers,int N, DoublePointer X, int incX);

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
                                   FloatPointer X, int incX,
                                   FloatPointer Y, int incY,
                                   float c, float s);

    public native void drot(PointerPointer extraPointers,int N,
                                   DoublePointer X, int incX,
                                   DoublePointer Y, int incY,
                                   double c, double s);

    /*
     * ------------------------------------------------------
     * ROTG
     * ------------------------------------------------------
     */

    public native void srotg(PointerPointer extraPointers,FloatPointer args);

    public native void drotg(PointerPointer extraPointers,DoublePointer args);

    /*
     * ------------------------------------------------------
     * ROTMG
     * ------------------------------------------------------
     */

    public native void srotmg(PointerPointer extraPointers,FloatPointer args,
                                     FloatPointer P);

    public native void drotmg(PointerPointer extraPointers,DoublePointer args,
                                     DoublePointer P);

    /*
     * ------------------------------------------------------
     * ROTM
     * ------------------------------------------------------
     */

    public native void srotm(PointerPointer extraPointers,int N,
                                    FloatPointer X, int incX,
                                    FloatPointer Y, int incY,
                                    FloatPointer P);

    public native void drotm(PointerPointer extraPointers,int N,
                                    DoublePointer X, int incX,
                                    DoublePointer Y, int incY,
                                    DoublePointer P);

    /*
     * ------------------------------------------------------
     * SWAP
     * ------------------------------------------------------
     */

    public native void sswap(PointerPointer extraPointers,int N,
                                    FloatPointer X, int incX,
                                    FloatPointer Y, int incY);

    public native void dswap(PointerPointer extraPointers,int N,
                                    DoublePointer X, int incX,
                                    DoublePointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SCAL
     * ------------------------------------------------------
     */

    public native void sscal(PointerPointer extraPointers,int N, float alpha,
                                    FloatPointer X, int incX);

    public native void dscal(PointerPointer extraPointers,int N, double alpha,
                                    DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * SCOPY
     * ------------------------------------------------------
     */

    public native void scopy(PointerPointer extraPointers,int N,
                                    FloatPointer X, int incX,
                                    FloatPointer Y, int incY);

    public native void dcopy(PointerPointer extraPointers,int N,
                                    DoublePointer X, int incX,
                                    DoublePointer Y, int incY);

    /*
     * ------------------------------------------------------
     * AXPY
     * ------------------------------------------------------
     */

    public native void saxpy(PointerPointer extraPointers,int N, float alpha,
                                    FloatPointer X, int incX,
                                    FloatPointer Y, int incY);

    public native void daxpy(PointerPointer extraPointers,int N, double alpha,
                                    DoublePointer X, int incX,
                                    DoublePointer Y, int incY);

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
                                    FloatPointer A, int lda,
                                    FloatPointer X, int incX,
                                    float beta,
                                    FloatPointer Y, int incY);

    public native void dgemv(PointerPointer extraPointers,int Order, int TransA,
                                    int M, int N,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer X, int incX,
                                    double beta,
                                    DoublePointer Y, int incY);

    /*
     * ------------------------------------------------------
     * GBMV
     * ------------------------------------------------------
     */

    public native void sgbmv(PointerPointer extraPointers,int Order, int TransA,
                                    int M, int N,
                                    int KL, int KU,
                                    float alpha,
                                    FloatPointer A, int lda,
                                    FloatPointer X, int incX,
                                    float beta,
                                    FloatPointer Y, int incY);

    public native void dgbmv(PointerPointer extraPointers,int Order, int TransA,
                                    int M, int N,
                                    int KL, int KU,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer X, int incX,
                                    double beta,
                                    DoublePointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SYMV
     * ------------------------------------------------------
     */

    public native void ssymv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    FloatPointer A, int lda,
                                    FloatPointer X, int incX,
                                    float beta,
                                    FloatPointer Y, int incY);

    public native void dsymv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer X, int incX,
                                    double beta,
                                    DoublePointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SBMV
     * ------------------------------------------------------
     */

    public native void ssbmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N, int K,
                                    float alpha,
                                    FloatPointer A, int lda,
                                    FloatPointer X, int incX,
                                    float beta,
                                    FloatPointer Y, int incY);

    public native void dsbmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N, int K,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer X, int incX,
                                    double beta,
                                    DoublePointer Y, int incY);

    /*
     * ------------------------------------------------------
     * SPMV
     * ------------------------------------------------------
     */

    public native void sspmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    FloatPointer Ap,
                                    FloatPointer X, int incX,
                                    float beta,
                                    FloatPointer Y, int incY);

    public native void dspmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    DoublePointer Ap,
                                    DoublePointer X, int incX,
                                    double beta,
                                    DoublePointer Y, int incY);

    /*
     * ------------------------------------------------------
     * TRMV
     * ------------------------------------------------------
     */

    public native void strmv(PointerPointer extraPointers,int Order, int Uplo, int TransA,
                             int Diag,
                                    int N, float alpha,
                                    FloatPointer A, int lda,
                                    FloatPointer X, int incX);

    public native void dtrmv(PointerPointer extraPointers,int Order, int Uplo, int TransA,
                             int Diag,
                                    int N, double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * TBMV
     * ------------------------------------------------------
     */

    public native void stbmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    FloatPointer A, int lda,
                                    FloatPointer X, int incX);

    public native void dtbmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    DoublePointer A, int lda,
                                    DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * TPMV
     * ------------------------------------------------------
     */

    public native void stpmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    FloatPointer Ap,
                                    FloatPointer X, int incX);

    public native void dtpmv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    DoublePointer Ap,
                                    DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * TRSV
     * ------------------------------------------------------
     */

    public native void strsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    FloatPointer A, int lda,
                                    FloatPointer X, int incX);

    public native void dtrsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    DoublePointer A, int lda,
                                    DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * TBSV
     * ------------------------------------------------------
     */

    public native void stbsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    FloatPointer A, int lda,
                                    FloatPointer X, int incX);

    public native void dtbsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N, int K,
                                    DoublePointer A, int lda,
                                    DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * TPSV
     * ------------------------------------------------------
     */

    public native void stpsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    FloatPointer Ap,
                                    FloatPointer X, int incX);

    public native void dtpsv(PointerPointer extraPointers,int Order, int Uplo,
                                    int TransA, int Diag,
                                    int N,
                                    DoublePointer Ap,
                                    DoublePointer X, int incX);

    /*
     * ------------------------------------------------------
     * GER
     * ------------------------------------------------------
     */

    public native void sger(PointerPointer extraPointers,int Order,
                                   int M, int N,
                                   float alpha,
                                   FloatPointer X, int incX,
                                   FloatPointer Y, int incY,
                                   FloatPointer A, int lda);

    public native void dger(PointerPointer extraPointers,int Order,
                                   int M, int N,
                                   double alpha,
                                   DoublePointer X, int incX,
                                   DoublePointer Y, int incY,
                                   DoublePointer A, int lda);

    /*
     * ------------------------------------------------------
     * SYR
     * ------------------------------------------------------
     */

    public native void ssyr(PointerPointer extraPointers,int Order, int Uplo,
                                   int N,
                                   float alpha,
                                   FloatPointer X, int incX,
                                   FloatPointer A, int lda);

    public native void dsyr(PointerPointer extraPointers,int Order, int Uplo,
                                   int N,
                                   double alpha,
                                   DoublePointer X, int incX,
                                   DoublePointer A, int lda);

    /*
     * ------------------------------------------------------
     * SPR
     * ------------------------------------------------------
     */

    public native void sspr(PointerPointer extraPointers,int Order, int Uplo,
                                   int N,
                                   float alpha,
                                   FloatPointer X, int incX,
                                   FloatPointer Ap);

    public native void dspr(PointerPointer extraPointers,int Order, int Uplo,
                                   int N,
                                   double alpha,
                                   DoublePointer X, int incX,
                                   DoublePointer Ap);

    /*
     * ------------------------------------------------------
     * SYR2
     * ------------------------------------------------------
     */

    public native void ssyr2(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    FloatPointer X, int incX,
                                    FloatPointer Y, int incY,
                                    FloatPointer A, int lda);

    public native void dsyr2(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    DoublePointer X, int incX,
                                    DoublePointer Y, int incY,
                                    DoublePointer A, int lda);

    /*
     * ------------------------------------------------------
     * SPR2
     * ------------------------------------------------------
     */

    public native void sspr2(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    float alpha,
                                    FloatPointer X, int incX,
                                    FloatPointer Y, int incY,
                                    FloatPointer Ap);

    public native void dspr2(PointerPointer extraPointers,int Order, int Uplo,
                                    int N,
                                    double alpha,
                                    DoublePointer X, int incX,
                                    DoublePointer Y, int incY,
                                    DoublePointer Ap);

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

    public native void hgemm(PointerPointer extraPointers,int Order, int TransA, int TransB,
                             int M, int N, int K,
                             float alpha,
                             @Cast("float16*") ShortPointer A, int lda,
                             @Cast("float16*") ShortPointer B, int ldb,
                             float beta,
                             @Cast("float16*") ShortPointer C, int ldc);


    public native void sgemm(PointerPointer extraPointers,int Order, int TransA, int TransB,
                                    int M, int N, int K,
                                    float alpha,
                                    FloatPointer A, int lda,
                                    FloatPointer B, int ldb,
                                    float beta,
                                    FloatPointer C, int ldc);

    public native void dgemm(PointerPointer extraPointers,int Order, int TransA, int TransB,
                                    int M, int N, int K,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer B, int ldb,
                                    double beta,
                                    DoublePointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYMM
     * ------------------------------------------------------
     */

    public native void ssymm(PointerPointer extraPointers,int Order, int Side, int Uplo,
                                    int M, int N,
                                    float alpha,
                                    FloatPointer A, int lda,
                                    FloatPointer B, int ldb,
                                    float beta,
                                    FloatPointer C, int ldc);

    public native void dsymm(PointerPointer extraPointers,int Order, int Side, int Uplo,
                                    int M, int N,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer B, int ldb,
                                    double beta,
                                    DoublePointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYRK
     * ------------------------------------------------------
     */

    public native void ssyrk(PointerPointer extraPointers,int Order, int Uplo, int Trans,
                                    int N, int K,
                                    float alpha,
                                    FloatPointer A, int lda,
                                    float beta,
                                    FloatPointer C, int ldc);

    public native void dsyrk(PointerPointer extraPointers,int Order, int Uplo, int Trans,
                                    int N, int K,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    double beta,
                                    DoublePointer C, int ldc);

    /*
     * ------------------------------------------------------
     * SYR2K
     * ------------------------------------------------------
     */

    public native void ssyr2k(PointerPointer extraPointers,int Order, int Uplo, int Trans,
                                     int N, int K,
                                     float alpha,
                                     FloatPointer A, int lda,
                                     FloatPointer B, int ldb,
                                     float beta,
                                     FloatPointer C, int ldc);

    public native void dsyr2k(PointerPointer extraPointers,int Order, int Uplo, int Trans,
                                     int N, int K,
                                     double alpha,
                                     DoublePointer A, int lda,
                                     DoublePointer B, int ldb,
                                     double beta,
                                     DoublePointer C, int ldc);

    /*
     * ------------------------------------------------------
     * TRMM
     * ------------------------------------------------------
     */

    public native void strmm(PointerPointer extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    float alpha,
                                    FloatPointer A, int lda,
                                    FloatPointer B, int ldb);

    public native void dtrmm(PointerPointer extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer B, int ldb);

    /*
     * ------------------------------------------------------
     * TRSM
     * ------------------------------------------------------
     */

    public native void strsm(PointerPointer extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    float alpha,
                                    FloatPointer A, int lda,
                                    FloatPointer B, int ldb);

    public native void dtrsm(PointerPointer extraPointers,int Order, int Side,
                                    int Uplo, int TransA, int Diag,
                                    int M, int N,
                                    double alpha,
                                    DoublePointer A, int lda,
                                    DoublePointer B, int ldb);

}
