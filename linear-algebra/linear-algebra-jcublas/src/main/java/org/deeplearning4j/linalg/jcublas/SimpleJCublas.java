package org.deeplearning4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas;
import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.api.complex.IComplexFloat;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;

/**
 * Created by mjk on 8/20/14.
 * @author mjk
 * @author Adam Gibson
 *
 */
public class SimpleJCublas {
    static {
        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        final Thread mainThread = Thread.currentThread();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run() {
                JCublas.cublasShutdown();
            }
        });
    }


    public static Pointer pointerFor(INDArray arr) {
        return Pointer.to(arr.data()).withByteOffset(arr.offset());
    }


    private static void ThreePointerM(Pointer dA, Pointer dB, Pointer dC,
                                      INDArray A, INDArray B, INDArray C, int sze) {

        JCublas.cublasAlloc(A.rows()*A.columns(), sze, dA);
        JCublas.cublasAlloc(B.rows()*B.columns(), sze, dB);
        JCublas.cublasAlloc(A.rows()*B.columns(), sze, dC);


        JCublas.cublasSetMatrix(
                A.rows(),
                A.columns(),
                sze,
                Pointer.to(A.data()),
                A.rows(),
                dA,
                A.rows()
        );
       JCublas.cublasSetMatrix(
                B.rows(),
                B.columns(),
                sze,
                Pointer.to(B.data()),
                B.rows(),
                dB,
                B.rows()
        );
    }

    private static void ThreePointerMi(Pointer d_A, Pointer d_B, Pointer d_C,
                                      IComplexNDArray A, IComplexNDArray B, IComplexNDArray C, int sze) {

        JCublas.cublasAlloc(A.rows()*A.columns(), sze, d_A);
        JCublas.cublasAlloc(B.rows()*B.columns(), sze, d_B);
        JCublas.cublasAlloc(A.rows()*B.columns(), sze, d_C);

        int ret;
        ret = JCublas.cublasSetMatrix(
                A.rows(),
                A.columns(),
                sze,
                Pointer.to(A.data()).withByteOffset((A.offset())),
                A.rows(),
                d_A,
                A.rows()
        );
        ret = JCublas.cublasSetMatrix(
                B.rows(),
                B.columns(),
                sze,
                Pointer.to(B.data()).withByteOffset((B.offset())),
                B.rows(),
                d_B,
                B.rows()
        );
    }
    private static void ThreePointersV(Pointer d_A, Pointer d_B, Pointer d_C,
                                       INDArray A, INDArray B, int sze) {
        JCublas.cublasAlloc(A.rows()*A.columns(), sze, d_A);
        JCublas.cublasAlloc(B.rows()*B.columns(), sze, d_B);
        JCublas.cublasAlloc(A.rows()*B.columns(), sze, d_C);

        JCublas.cublasSetVector(
                A.length(),
                sze,
                Pointer.to(A.data()).withByteOffset((A.offset())),
                1,
                d_A,
                1);
        JCublas.cublasSetVector(
                B.length(),
                sze,
                Pointer.to(B.data()).withByteOffset((B.offset())),
                1,
                d_B,
                1);
    }
    private static void twoPointers(Pointer d_A, Pointer d_B, INDArray A, INDArray B, int sze) {

        JCublas.cublasAlloc(A.length(), sze, d_A);
        JCublas.cublasAlloc(B.length(), sze, d_B);

        JCublas.cublasSetVector(
                A.length(),
                sze,
                Pointer.to(A.data()).withByteOffset((A.offset())),
                1,
                d_A,
                1);

        JCublas.cublasSetVector(
                B.length(),
                sze,
                Pointer.to(B.data()).withByteOffset((B.offset())),
                1,
                d_B,
                1);
    }

    private static void twoPointersV(Pointer d_A, Pointer d_B, IComplexNDArray A, IComplexNDArray B, int sze) {

        JCublas.cublasAlloc(A.length(), sze, d_A);
        JCublas.cublasAlloc(B.length(), sze, d_B);

        JCublas.cublasSetVector(
                A.length(),
                sze,
                Pointer.to(A.data()).withByteOffset((A.offset())),
                1,
                d_A,
                1);

        JCublas.cublasSetVector(
                B.length(),
                sze,
                Pointer.to(B.data()).withByteOffset((B.offset())),
                1,
                d_B,
                1);
    }
    private static void onePointerV(Pointer d_A, INDArray A, int sze) {
        JCublas.cublasAlloc(A.length(), sze, d_A);
        JCublas.cublasSetVector(
                A.length(),
                sze,
                Pointer.to(A.data()),
                1,
                d_A,
                1);
    }

    private static void onePointerVi(Pointer dA, IComplexNDArray A, int sze) {
        JCublas.cublasAlloc(A.length(), sze, dA);
        JCublas.cublasSetVector(
                A.length(),
                sze,
                pointerFor(A),
                1,
                dA,
                1);
    }

    private static void gv(Pointer dA, INDArray A, int sze) {
        JCublas.cublasGetVector(
                A.length(),
                sze,
                dA,
                A.stride()[0],
                Pointer.to(A.data()),
                1);
    }

    private static void gvi(Pointer d_A, IComplexNDArray B, int sze) {
        JCublas.cublasGetVector(
                B.length(),
                sze,
                d_A,
                1,
                Pointer.to(B.data()),
                1);
    }

    private static void gm(Pointer d_C, INDArray C, int sze) {
        int ret;
        ret = JCublas.cublasGetMatrix(
                C.rows(),
                C.columns(),
                sze,
                d_C,
                C.rows(),
                Pointer.to(C.data()),
                C.rows());
    }

    private static void gmi(Pointer d_C, IComplexNDArray C, int sze) {
        int ret;
        ret = JCublas.cublasGetMatrix(
                C.rows(),
                C.columns(),
                sze,
                d_C,
                C.rows(),
                Pointer.to(C.data()).withByteOffset((C.offset())),
                C.rows());
    }


    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) {

        

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        ThreePointerM(d_A, d_B, d_C, A, B,C, Sizeof.DOUBLE);
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

        gv(d_C, C, Sizeof.DOUBLE);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);



        return C;
    }


    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexNDArray C,
                                             float Alpha, float Beta) {

        

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        ThreePointerMi(d_A,d_B,d_C,A,B,C, Sizeof.DOUBLE);
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

        gvi(d_C, C, Sizeof.DOUBLE);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);



        return C;

    }
    public static INDArray gemm(INDArray A, INDArray B, INDArray C,
                                      float alpha, float beta) {

        

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        ThreePointerM(d_A,d_B,d_C,A,B,C, Sizeof.FLOAT);

        JCublas.cublasDgemm(
                'n', //trans
                'n',
                A.rows(),  // m
                B.columns(), // n
                B.rows(), //k,
                (float)alpha,
                d_A, // A
                A.rows(),  // lda
                d_B, // x
                B.rows(), // ldb
                (float)beta,  // beta
                d_C, // y
                C.rows()); // incy


        gm(d_C, C, Sizeof.FLOAT);



        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        return C;

    }

    public static void dcopy(int length, float[] data, int offset, int i, float[] data1, int i1, int i2) {

    }

    public static float nrm2(IComplexNDArray A) {

        Pointer d_A = new Pointer();

        onePointerVi(d_A, A, Sizeof.FLOAT);

        float s = JCublas.cublasSnrm2(A.length(), d_A, 2);


        JCublas.cublasFree(d_A);

        return s;
    }

    public static void copy(IComplexNDArray x, IComplexNDArray y) {
        Pointer X = new Pointer();
        Pointer Y = new Pointer();

        ;

        twoPointersV(X, Y, x, y, Sizeof.FLOAT);

        JCublas.cublasZcopy(x.length(),
                X,
                1,
                Y,
                1);


        gvi(Y, y, Sizeof.FLOAT);



        JCublas.cublasFree(X);
        JCublas.cublasFree(Y);

    }

    public static int iamax(IComplexNDArray x) {
        Pointer X = new Pointer();

        int max;

        ;

        onePointerVi(X, x, Sizeof.FLOAT);

        max = JCublas.cublasIzamax(x.length(), X, 1);


        JCublas.cublasFree(X);
        return max;
    }

    public static float asum(IComplexNDArray x) {
        

        Pointer X = new Pointer();

        onePointerVi(X, x, Sizeof.FLOAT);

        float sum = JCublas.cublasScasum(x.length(), X, 1);
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

    public static void swap(INDArray x, INDArray y) {

        

        Pointer X = new Pointer();
        Pointer Y = new Pointer();

        int length = x.length();
        int length_o = y.length();

        if (length != length_o)
            return;

        twoPointers(X, Y, x, y, Sizeof.FLOAT);

        JCublas.cublasDswap(length,
                X,
                1,
                Y,
                1);

        gv(Y, y, Sizeof.FLOAT);



        JCublas.cublasFree(X);
        JCublas.cublasFree(Y);

    }

    public static float asum(INDArray x) {

        

        Pointer X = new Pointer();

        onePointerV(X, x, Sizeof.FLOAT);

        float sum = 0;
        sum = JCublas.cublasSasum(x.length(), X, x.stride()[0]);


        JCublas.cublasFree(X);

        return sum;
    }

    public static float nrm2(INDArray x) {
        Pointer X = new Pointer();

        float normal2;

        

        onePointerV(X, x, Sizeof.FLOAT);

        normal2 = JCublas.cublasSnrm2(x.length(), X, 1);


        JCublas.cublasFree(X);

        return normal2;
    }

    public static int iamax(INDArray x) {

        Pointer X = new Pointer();

        int max;

        

        onePointerV(X, x, Sizeof.FLOAT);

        max = JCublas.cublasIdamax(x.length(), X, 1);


        JCublas.cublasFree(X);
        return max;

    }

    public static void axpy(float da, INDArray A, INDArray B) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        int length = A.length();
        int length_o = B.length();

        if (length != length_o)
            return;

        

        twoPointers(d_A, d_B, A, B, Sizeof.FLOAT);

        JCublas.cublasDaxpy(length, da, d_A, 1, d_B, 1);


        gv(d_B, B, Sizeof.FLOAT);


        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);

    }
    public static void axpy(IComplexNumber da, IComplexNDArray A, IComplexNDArray B) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        int length = A.length();
        int length_o = B.length();

        if (length != length_o)
            return;

        

        twoPointersV(d_A, d_B, A, B, Sizeof.FLOAT);


        JCublas.cublasCaxpy(
                length,
                jcuda.cuComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                d_A,
                1,
                d_B,
                1
        );

        gvi(d_B, B, Sizeof.FLOAT);


        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
    }

    public static INDArray scal(float alpha, INDArray x) {
        Pointer d_A = new Pointer();
        int length = x.length();

        

        JCublas.cublasAlloc(length, Sizeof.FLOAT, d_A);


        onePointerV(d_A, x, Sizeof.FLOAT);

        JCublas.cublasDscal(length,alpha,d_A,1);


        gv(d_A, x, Sizeof.FLOAT);


        JCublas.cublasFree(d_A);

        return x;

    }

    public static void copy(INDArray x, INDArray y) {
        Pointer X = pointerFor(x);
        Pointer Y = pointerFor(y);

        

        twoPointers(X, Y, x, y, Sizeof.FLOAT);

        JCublas.cublasDcopy(x.length(),
                X,
                1,
                Y,
                1);


        gv(Y, y, Sizeof.FLOAT);

        JCublas.cublasFree(X);
        JCublas.cublasFree(Y);


    }

    public static float dot(INDArray x, INDArray y) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        

        twoPointers(d_A, d_B, x, y, Sizeof.FLOAT);

        float dott = 0;
        dott = JCublas.cublasSdot(x.length(), d_A, 1, d_B, 1);


        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        return dott;
    }


    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        

        twoPointersV(d_A, d_B, x, y, Sizeof.FLOAT);

        jcuda.cuDoubleComplex dott = jcuda.cuDoubleComplex.cuCmplx(0,0);
        dott = JCublas.cublasZdotc(x.length(),d_A,1,d_B,1);


        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);

        return  NDArrays.createDouble(dott.x,dott.y);
    }
    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) {
        // = alpha * A * tranpose(B) + C

        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();
        

        ThreePointerM(dA,dB,dC,A,B,C, Sizeof.FLOAT);

        JCublas.cublasSger(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                dA,        // d_A or x
                A.rows(),   // incx
                dB,        // dB or y
                B.rows(),   // incy
                dC,        // dC or A
                C.rows()    // lda
        );

        gm(dC,C, Sizeof.FLOAT);

        JCublas.cublasFree(dA);
        JCublas.cublasFree(dB);
        JCublas.cublasFree(dC);



        return C;
    }





    public static IComplexNDArray zscal(IComplexFloat alpha, IComplexNDArray x) {
        Pointer dA = new Pointer();



        onePointerVi(dA, x, Sizeof.FLOAT);

        JCublas.cublasCscal(
                x.length(),
                cuComplex.cuCmplx(alpha.realComponent(),alpha.imaginaryComponent()),
                dA,
                2
        );


        gvi(dA,x, Sizeof.FLOAT);
        JCublas.cublasFree(dA);


        return x;
    }


    public static IComplexNDArray zscal(IComplexDouble alpha, IComplexNDArray x) {
        Pointer dA = new Pointer();

        

        onePointerVi(dA, x, Sizeof.FLOAT);

        JCublas.cublasZscal(
                x.length(),
                jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(),alpha.imaginaryComponent()),
                dA,
                2
                );
        gvi(dA,x, Sizeof.FLOAT);
        JCublas.cublasFree(dA);


        return x;
    }

    public static IComplexDouble dotu(IComplexNDArray x, IComplexNDArray y) {
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();

        

        twoPointersV(d_A, d_B, x, y, Sizeof.FLOAT);

        jcuda.cuDoubleComplex dott = jcuda.cuDoubleComplex.cuCmplx(0,0);
        dott = JCublas.cublasZdotu(x.length(), d_A, 1, d_B, 1);


        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);

        return NDArrays.createDouble(dott.x, dott.y);
    }

    public static IComplexNDArray geru(IComplexNDArray A,
                                             IComplexNDArray B,
                                             IComplexNDArray C, IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        

        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());

        ThreePointerMi(d_A, d_B, d_C, A, B, C, Sizeof.FLOAT);
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

        gmi(d_C, C, Sizeof.FLOAT);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);



        return C;
    }

    public static IComplexNDArray gerc(IComplexNDArray A, IComplexNDArray B, IComplexNDArray C,
                                             IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C

        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();
        

        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());

        ThreePointerMi(dA, dB, dC, A, B, C, Sizeof.FLOAT);

        JCublas.cublasZgerc(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                dA,        // dA or x
                A.rows(),   // incx
                dB,        // dB or y
                B.rows(),   // incy
                dC,        // dC or A
                C.rows()    // lda
        );

        gmi(dC, C, Sizeof.FLOAT);

        JCublas.cublasFree(dA);
        JCublas.cublasFree(dB);
        JCublas.cublasFree(dC);



        return C;
    }
}
