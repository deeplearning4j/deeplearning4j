package org.deeplearning4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
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
                d_A,//.withByteOffset(A.offset()),
                A.rows(),
                d_A,
                A.rows()
        );
        ret = JCublas.cublasSetMatrix(
                B.rows(),
                B.columns(),
                Sizeof.DOUBLE,
                d_B,//.withByteOffset(B.offset()),
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

    private static void gv(Pointer d_A, JCublasNDArray B) {
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
    public static JCublasNDArray gemv(JCublasNDArray A, JCublasNDArray B, double alpha, double beta) {

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        JCublasNDArray C = new JCublasNDArray(A.rows(), B.columns());

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        ThreePointersV(d_A, d_B, d_C, A, B);

        JCublas.cublasDgemv(
                'n', //trans
                A.rows(),  // m
                B.columns(), // n
                alpha, //alpha
                d_A, // A
                A.rows(),  // lda
                d_B, // x
                1, // incx
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
    public static JCublasComplexNDArray gemm(IComplexNDArray A_, IComplexNDArray B_, double alpha, double beta) {
    //cublasDnrm2(int n, jcuda.Pointer x, int incx)

        return null;
    }
    public static JCublasNDArray gemm(JCublasNDArray A, JCublasNDArray B, double alpha, double beta) {

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        JCublasNDArray C = new JCublasNDArray(A.rows(), B.columns());

        ThreePointerM(d_A,d_B,d_C,A,B,C);

        int ret;


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

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;

    }

    public static void dcopy(int length, double[] data, int offset, int i, double[] data1, int i1, int i2) {
    }

    public static double nrm2(JCublasComplexNDArray jCublasComplexNDArray) {
        return 0;
    }

    public static void copy(JCublasComplexNDArray a, JCublasComplexNDArray jCublasComplexNDArray) {

    }

    public static int iamax(JCublasComplexNDArray jCublasComplexNDArray) {
        return 0;
    }

    public static int asum(JCublasComplexNDArray arr) {
        return 0;
    }

    public static int dznrm2(int length, double[] data, int offset, int i) {

        return 0;
    }

    public static int dzasum(int length, double[] data, int offset, int i) {
        return 0;
    }

    public static int izamax(int length, double[] data, int offset, int i) {
        return 0;
    }

    public static void swap(JCublasNDArray x, JCublasNDArray y) {
        Pointer X = new Pointer();
        Pointer Y = new Pointer();

        int length = x.length();
        int length_o = y.length();

        if (length != length_o)
            return;


        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        TwoPointersV(X, Y, x, y);

        JCublas.cublasDswap(length,
                X,
                1,
                Y,
                1);

        gv(Y, y);

        JCublas.cublasFree(X);
        JCublas.cublasFree(Y);

        JCublas.cublasShutdown();
    }

    public static double asum(JCublasNDArray x) {

        Pointer X = new Pointer();

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        OnePointerV(X,x);

        double sum = 0;
        sum = JCublas.cublasDasum(x.length(),X,1);

        JCublas.cublasFree(X);
        JCublas.cublasShutdown();

        return sum;
    }

    public static double nrm2(JCublasNDArray x) {
        Pointer X = new Pointer();

        double normal2;

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        OnePointerV(X,x);

        normal2 = JCublas.cublasDnrm2(x.length(), X, 1);

        return normal2;
    }

    public static int iamax(JCublasNDArray x) {
        Pointer X = new Pointer();

        int max;

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        OnePointerV(X,x);

        max = JCublas.cublasIdamax(x.length(), X, 1);

        return max;

    }

    public static void axpy(double da, JCublasNDArray A, JCublasNDArray B) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        int length = A.length();
        int length_o = B.length();

        if (length != length_o)
            return;

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        TwoPointersV(d_A, d_B, A, B);

        JCublas.cublasDaxpy(length, da, d_A, 1, d_B, 1);


        gv(d_B, B);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);

    }

    public static JCublasNDArray scal(double alpha, JCublasNDArray x) {
        Pointer d_A = new Pointer();
        int length = x.length();
        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        JCublas.cublasAlloc(length, Sizeof.DOUBLE, d_A);


        OnePointerV(d_A,x);

        JCublas.cublasDscal(length,alpha,d_A,1);


        gv(d_A, x);

        JCublas.cublasFree(d_A);

        return x;

    }

    public static void copy(JCublasNDArray x, JCublasNDArray y) {
        Pointer X = new Pointer();
        Pointer Y = new Pointer();


        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

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

        TwoPointersV(d_A,d_B,x,y);

        double dott = 0;
        dott = JCublas.cublasDdot(x.length(),d_A,1,d_B,1);

        JCublas.cublasShutdown();

        return dott;
    }
}
