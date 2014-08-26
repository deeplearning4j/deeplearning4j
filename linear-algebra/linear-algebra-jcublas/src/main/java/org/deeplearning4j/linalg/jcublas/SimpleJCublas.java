package org.deeplearning4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas;
import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.jcublas.complex.ComplexDouble;
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexNDArray;

/**
 * Created by mjk on 8/20/14.
 */
public class SimpleJCublas {
    private static void ThreePointerM(Pointer d_A, Pointer d_B, Pointer d_C,
                                      JCublasNDArray A, JCublasNDArray B, JCublasNDArray C) {

        JCublas.cublasAlloc(A.rows()*A.columns(), Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(B.rows()*B.columns(), Sizeof.DOUBLE, d_B);
        JCublas.cublasAlloc(A.rows()*B.columns(), Sizeof.DOUBLE, d_C);

        int ret;
        ret = JCublas.cublasSetMatrix(
                A.rows(),
                A.columns(),
                Sizeof.DOUBLE,
                Pointer.to(A.data()),
                A.rows(),
                d_A,
                A.rows()
        );
        ret = JCublas.cublasSetMatrix(
                B.rows(),
                B.columns(),
                Sizeof.DOUBLE,
                Pointer.to(B.data()),
                B.rows(),
                d_B,
                B.rows()
        );
    }

    private static void ThreePointerMi(Pointer d_A, Pointer d_B, Pointer d_C,
                                      JCublasComplexNDArray A, JCublasComplexNDArray B, JCublasComplexNDArray C) {

        JCublas.cublasAlloc(A.rows()*A.columns(), Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(B.rows()*B.columns(), Sizeof.DOUBLE, d_B);
        JCublas.cublasAlloc(A.rows()*B.columns(), Sizeof.DOUBLE, d_C);

        int ret;
        ret = JCublas.cublasSetMatrix(
                A.rows(),
                A.columns(),
                Sizeof.DOUBLE,
                Pointer.to(A.data()),
                A.rows(),
                d_A,
                A.rows()
        );
        ret = JCublas.cublasSetMatrix(
                B.rows(),
                B.columns(),
                Sizeof.DOUBLE,
                Pointer.to(B.data()),
                B.rows(),
                d_B,
                B.rows()
        );
    }
    private static void ThreePointersV(Pointer d_A, Pointer d_B, Pointer d_C,
                                       JCublasNDArray A, JCublasNDArray B) {
        JCublas.cublasAlloc(A.rows()*A.columns(), Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(B.rows()*B.columns(), Sizeof.DOUBLE, d_B);
        JCublas.cublasAlloc(A.rows()*B.columns(), Sizeof.DOUBLE, d_C);

        JCublas.cublasSetVector(
                A.length(),
                Sizeof.DOUBLE,
                Pointer.to(A.data()),
                1,
                d_A,
                1);
        JCublas.cublasSetVector(
                B.length(),
                Sizeof.DOUBLE,
                Pointer.to(B.data()),
                1,
                d_B,
                1);
    }
    private static void TwoPointersV(Pointer d_A, Pointer d_B, JCublasNDArray A, JCublasNDArray B) {

        JCublas.cublasAlloc(A.length(), Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(B.length(), Sizeof.DOUBLE, d_B);

        JCublas.cublasSetVector(
                A.length(),
                Sizeof.DOUBLE,
                Pointer.to(A.data()),
                1,
                d_A,
                1);

        JCublas.cublasSetVector(
                B.length(),
                Sizeof.DOUBLE,
                Pointer.to(B.data()),
                1,
                d_B,
                1);
    }

    private static void TwoPointersVi(Pointer d_A, Pointer d_B, JCublasComplexNDArray A, JCublasComplexNDArray B) {

        JCublas.cublasAlloc(A.length(), Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(B.length(), Sizeof.DOUBLE, d_B);

        JCublas.cublasSetVector(
                A.length(),
                Sizeof.DOUBLE,
                Pointer.to(A.data()),
                1,
                d_A,
                1);

        JCublas.cublasSetVector(
                B.length(),
                Sizeof.DOUBLE,
                Pointer.to(B.data()),
                1,
                d_B,
                1);
    }
    private static void OnePointerV(Pointer d_A, JCublasNDArray A) {
        JCublas.cublasAlloc(A.length(), Sizeof.DOUBLE, d_A);
        JCublas.cublasSetVector(
                A.length(),
                Sizeof.DOUBLE,
                Pointer.to(A.data()),
                1,
                d_A,
                1);
    }

    private static void OnePointerVi(Pointer d_A, JCublasComplexNDArray A) {
        JCublas.cublasAlloc(A.length(), Sizeof.DOUBLE, d_A);
        JCublas.cublasSetVector(
                A.length(),
                Sizeof.DOUBLE,
                Pointer.to(A.data()),
                1,
                d_A,
                1);
    }

    private static void gv(Pointer d_A, JCublasNDArray B) {
        JCublas.cublasGetVector(
                B.length(),
                Sizeof.DOUBLE,
                d_A,
                1,
                Pointer.to(B.data()),
                1);
    }

    private static void gvi(Pointer d_A, JCublasComplexNDArray B) {
        JCublas.cublasGetVector(
                B.length(),
                Sizeof.DOUBLE,
                d_A,
                1,
                Pointer.to(B.data()),
                1);
    }

    private static void gm(Pointer d_C, JCublasNDArray C) {
        int ret;
        ret = JCublas.cublasGetMatrix(
                C.rows(),
                C.columns(),
                Sizeof.DOUBLE,
                d_C,
                C.rows(),
                Pointer.to(C.data()),
                C.rows());
    }

    private static void gmi(Pointer d_C, JCublasComplexNDArray C) {
        int ret;
        ret = JCublas.cublasGetMatrix(
                C.rows(),
                C.columns(),
                Sizeof.DOUBLE,
                d_C,
                C.rows(),
                Pointer.to(C.data()),
                C.rows());
    }

    private static void init() {
        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);
    }

    public static JCublasNDArray gemv(JCublasNDArray A, JCublasNDArray B, JCublasNDArray C, double alpha, double beta) {

        init();

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        ThreePointerM(d_A, d_B, d_C, A, B,C);
        char trans = 'n';
        if (A.rows() == B.columns()) {
            trans = 'T';
        }
        JCublas.cublasDgemv(
                'n', //trans
                A.rows(),  // m
                A.columns(), // n
                alpha, //alpha
                d_A, // A
                A.rows(),  // lda
                d_B, // x
                B.rows(), // incx
                beta,  // beta
                d_C, // y
                1); // incy

        gv(d_C, C);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;
    }
    public static JCublasComplexNDArray gemm(JCublasComplexNDArray A, JCublasComplexNDArray B, JCublasComplexNDArray C,
                                             double Alpha, double Beta) {

        init();

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        ThreePointerMi(d_A,d_B,d_C,A,B,C);
        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha,0);
        cuDoubleComplex beta = cuDoubleComplex.cuCmplx(Beta,0);

        JCublas.cublasZgemm(
                'n', //trans
                'n',
                A.rows(),  // m
                B.columns(), // n
                B.rows(), //k,
                alpha,
                d_A, // A
                A.rows(),  // lda
                d_B, // x
                B.rows(), // incx
                beta,  // beta
                d_C, // y
                C.rows()); // incy

        gvi(d_C, C);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;

    }
    public static JCublasNDArray gemm(JCublasNDArray A, JCublasNDArray B, JCublasNDArray C,
                                      double alpha, double beta) {

        init();

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        ThreePointerM(d_A,d_B,d_C,A,B,C);

        JCublas.cublasDgemm(
                'n', //trans
                'n',
                A.rows(),  // m
                B.columns(), // n
                B.rows(), //k,
                alpha,
                d_A, // A
                A.rows(),  // lda
                d_B, // x
                B.rows(), // incx
                beta,  // beta
                d_C, // y
                C.rows()); // incy


        gm(d_C, C);

        JCublas.cublasShutdown();

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        return C;

    }

    public static void dcopy(int length, double[] data, int offset, int i, double[] data1, int i1, int i2) {

    }

    public static double nrm2(JCublasComplexNDArray A) {
        init();

        Pointer d_A = new Pointer();

        OnePointerVi(d_A, A);

        double s = JCublas.cublasDnrm2(A.length(),d_A,2);

        JCublas.cublasShutdown();
        JCublas.cublasFree(d_A);

        return s;
    }

    public static void copy(JCublasComplexNDArray x, JCublasComplexNDArray y) {
        Pointer X = new Pointer();
        Pointer Y = new Pointer();

        init();

        TwoPointersVi(X,Y,x,y);

        JCublas.cublasZcopy(x.length(),
                X,
                1,
                Y,
                1);


        gvi(Y, y);

        JCublas.cublasShutdown();

        JCublas.cublasFree(X);
        JCublas.cublasFree(Y);

    }

    public static int iamax(JCublasComplexNDArray x) {
        Pointer X = new Pointer();

        int max;

        init();

        OnePointerVi(X, x);

        max = JCublas.cublasIzamax(x.length(), X, 1);

        JCublas.cublasShutdown();
        JCublas.cublasFree(X);
        return max;
    }

    public static double asum(JCublasComplexNDArray x) {
        init();

        Pointer X = new Pointer();

        OnePointerVi(X, x);

        double sum = 0;
        sum = JCublas.cublasDzasum(x.length(),X,1);

        JCublas.cublasShutdown();
        JCublas.cublasFree(X);

        return sum;
    }

    public static int dznrm2(int length, float[] data, int offset, int i) {

        return 0;
    }

    public static int dzasum(int length, float[] data, int offset, int i) {
        return 0;
    }

    public static int izamax(int length, float[] data, int offset, int i) {
        return 0;
    }

    public static void swap(JCublasNDArray x, JCublasNDArray y) {

        init();

        Pointer X = new Pointer();
        Pointer Y = new Pointer();

        int length = x.length();
        int length_o = y.length();

        if (length != length_o)
            return;

        TwoPointersV(X, Y, x, y);

        JCublas.cublasDswap(length,
                X,
                1,
                Y,
                1);

        gv(Y, y);

        JCublas.cublasShutdown();

        JCublas.cublasFree(X);
        JCublas.cublasFree(Y);

    }

    public static double asum(JCublasNDArray x) {

        init();

        Pointer X = new Pointer();

        OnePointerV(X,x);

        double sum = 0;
        sum = JCublas.cublasDasum(x.length(),X,1);

        JCublas.cublasShutdown();
        JCublas.cublasFree(X);

        return sum;
    }

    public static double nrm2(JCublasNDArray x) {
        Pointer X = new Pointer();

        double normal2;

        init();

        OnePointerV(X,x);

        normal2 = JCublas.cublasDnrm2(x.length(), X, 1);

        JCublas.cublasShutdown();
        JCublas.cublasFree(X);

        return normal2;
    }

    public static int iamax(JCublasNDArray x) {

        Pointer X = new Pointer();

        int max;

        init();

        OnePointerV(X,x);

        max = JCublas.cublasIdamax(x.length(), X, 1);

        JCublas.cublasShutdown();
        JCublas.cublasFree(X);
        return max;

    }

    public static void axpy(double da, JCublasNDArray A, JCublasNDArray B) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        int length = A.length();
        int length_o = B.length();

        if (length != length_o)
            return;

        init();

        TwoPointersV(d_A, d_B, A, B);

        JCublas.cublasDaxpy(length, da, d_A, 1, d_B, 1);


        gv(d_B, B);

        JCublas.cublasShutdown();
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);

    }
    public static void axpy(IComplexDouble da, JCublasComplexNDArray A, JCublasComplexNDArray B) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        int length = A.length();
        int length_o = B.length();

        if (length != length_o)
            return;

        init();

        TwoPointersVi(d_A, d_B, A, B);


        JCublas.cublasZaxpy(
                length,
                jcuda.cuDoubleComplex.cuCmplx(da.realComponent(),da.imaginaryComponent()),
                d_A,
                1,
                d_B,
                1
        );

        gvi(d_B, B);

        JCublas.cublasShutdown();
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
    }

    public static JCublasNDArray scal(double alpha, JCublasNDArray x) {
        Pointer d_A = new Pointer();
        int length = x.length();

        init();

        JCublas.cublasAlloc(length, Sizeof.DOUBLE, d_A);


        OnePointerV(d_A, x);

        JCublas.cublasDscal(length,alpha,d_A,1);


        gv(d_A, x);

        JCublas.cublasShutdown();
        JCublas.cublasFree(d_A);

        return x;

    }

    public static void copy(JCublasNDArray x, JCublasNDArray y) {
        Pointer X = new Pointer();
        Pointer Y = new Pointer();

        init();

        TwoPointersV(X,Y,x,y);

        JCublas.cublasDcopy(x.length(),
                X,
                1,
                Y,
                1);


        gv(Y, y);

        JCublas.cublasFree(X);
        JCublas.cublasFree(Y);

        JCublas.cublasShutdown();
    }

    public static double dot(JCublasNDArray x, JCublasNDArray y) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        init();

        TwoPointersV(d_A,d_B,x,y);

        double dott = 0;
        dott = JCublas.cublasDdot(x.length(),d_A,1,d_B,1);

        JCublas.cublasShutdown();
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        return dott;
    }
    public static ComplexDouble dot(JCublasComplexNDArray x, JCublasComplexNDArray y) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        init();

        TwoPointersVi(d_A, d_B, x, y);

        jcuda.cuDoubleComplex dott = jcuda.cuDoubleComplex.cuCmplx(0,0);
        dott = JCublas.cublasZdotc(x.length(),d_A,1,d_B,1);

        JCublas.cublasShutdown();
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);

        return new ComplexDouble(dott.x,dott.y);
    }
    public static JCublasNDArray ger(JCublasNDArray A, JCublasNDArray B, JCublasNDArray C, double alpha) {
        // = alpha * A * tranpose(B) + C

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        init();

        ThreePointerM(d_A,d_B,d_C,A,B,C);

        JCublas.cublasDger(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                d_A,        // d_A or x
                A.rows(),   // incx
                d_B,        // d_B or y
                B.rows(),   // incy
                d_C,        // d_C or A
                C.rows()    // lda
        );

        gm(d_C,C);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;
    }


    public static JCublasComplexNDArray zscal(IComplexDouble alpha, JCublasComplexNDArray x) {
        Pointer d_A = new Pointer();

        init();

        OnePointerVi(d_A,x);

        JCublas.cublasZscal(
                x.length(),
                jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(),alpha.imaginaryComponent()),
                d_A,
                2
                );
        gvi(d_A,x);
        JCublas.cublasFree(d_A);

        JCublas.cublasShutdown();
        return x;
    }

    public static IComplexDouble dotu(JCublasComplexNDArray x, JCublasComplexNDArray y) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        init();

        TwoPointersVi(d_A, d_B, x, y);

        jcuda.cuDoubleComplex dott = jcuda.cuDoubleComplex.cuCmplx(0,0);
        dott = JCublas.cublasZdotu(x.length(), d_A, 1, d_B, 1);

        JCublas.cublasShutdown();
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);

        return new ComplexDouble(dott.x,dott.y);
    }

    public static JCublasComplexNDArray geru(JCublasComplexNDArray A,
                                             JCublasComplexNDArray B,
                                             JCublasComplexNDArray C, IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        init();

        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());

        ThreePointerMi(d_A, d_B, d_C, A, B, C);
        JCublas.cublasZgeru(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                d_A,        // d_A or x
                A.rows(),   // incx
                d_B,        // d_B or y
                B.rows(),   // incy
                d_C,        // d_C or A
                C.rows()    // lda
        );

        gmi(d_C, C);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;
    }

    public static JCublasComplexNDArray gerc(JCublasComplexNDArray A, JCublasComplexNDArray B, JCublasComplexNDArray C,
                                             IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        init();

        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());

        ThreePointerMi(d_A, d_B, d_C, A, B, C);

        JCublas.cublasZgerc(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                d_A,        // d_A or x
                A.rows(),   // incx
                d_B,        // d_B or y
                B.rows(),   // incy
                d_C,        // d_C or A
                C.rows()    // lda
        );

        gmi(d_C, C);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;
    }
}
