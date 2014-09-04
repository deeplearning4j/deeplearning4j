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
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexNDArray;

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
        Pointer ret =  Pointer.to(arr.data()).withByteOffset(arr.offset());
        JCublas.cublasAlloc(arr.length(),arr.length(),ret);
        JCublas.cublasSetVector(
                arr.length(),
                arr.length(),
                ret,
                arr.stride()[0]
                ,Pointer.to(ret),
                arr.stride()[0]);

        return ret;
    }


    public static void alloc(INDArray toAlloc) {
        Pointer p = pointerFor(toAlloc);
        JCublas.cublasSetVector(
                toAlloc.length(),
                toAlloc.length(),
                p,
                toAlloc.stride()[0]
                ,Pointer.to(p),
                toAlloc.stride()[0]);
    }



    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) {


        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        char trans = 'n';
        if (A.rows() == B.columns()) {
            trans = 'T';
        }
        JCublas.cublasDgemv(
                'n', //trans
                A.rows(),  // m
                A.columns(), // n
                alpha, //alpha
                cA.pointer(), // A
                A.rows(),  // lda
                cB.pointer(), // x
                B.rows(), // incx
                beta,  // beta
                cC.pointer(), // y
                cC.stride()[0]); // incy

        cC.getData();



        return C;
    }


    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexNDArray C,
                                       float Alpha, float Beta) {

        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha,0);
        cuDoubleComplex beta = cuDoubleComplex.cuCmplx(Beta,0);

        JCublas.cublasZgemm(
                'n', //trans
                'n',
                A.rows(),  // m
                B.columns(), // n
                B.rows(), //k,
                alpha,
                cA.pointer(), // A
                A.rows(),  // lda
                cB.pointer(), // x
                B.rows(), // incx
                beta,  // beta
                cC.pointer(), // y
                C.rows()); // incy


        cC.getData();


        return C;

    }
    public static INDArray gemm(INDArray A, INDArray B, INDArray C,
                                float alpha, float beta) {


        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;


        JCublas.cublasSgemm(
                'n', //trans
                'n',
                A.rows(),  // m
                B.columns(), // n
                B.rows(), //k,
                alpha,
                cA.pointer(), // A
                A.rows(),  // lda
                cB.pointer(), // x
                B.rows(), // ldb
                beta,  // beta
                cC.pointer(), // y
                C.rows()); // incy


        cC.getData();

        return C;

    }

    public static void dcopy(int length, float[] data, int offset, int i, float[] data1, int i1, int i2) {

    }

    public static float nrm2(IComplexNDArray A) {
        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;

        float s = JCublas.cublasSnrm2(A.length(), cA.pointer(), 2);



        return s;
    }

    public static void copy(IComplexNDArray x, IComplexNDArray y) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray yC = (JCublasComplexNDArray) y;
        JCublas.cublasZcopy(x.length(),
                xC.pointer(),
                xC.stride()[0],
                yC.pointer(),
                yC.stride()[0]);

        yC.getData();
    }

    public static int iamax(IComplexNDArray x) {

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;

        int max = JCublas.cublasIzamax(x.length(), xC.pointer(), x.stride()[0]);
        return max;
    }

    public static float asum(IComplexNDArray x) {

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;


        float sum = JCublas.cublasScasum(x.length(), xC.pointer(), 1);

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


        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        JCublas.cublasSswap(xC.length(),
                xC.pointer(),
                xC.stride()[0],
                yC.pointer(),
                yC.stride()[0]);

        yC.getData();


    }

    public static float asum(INDArray x) {

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;


        float sum = JCublas.cublasSasum(x.length(), xC.pointer(), x.stride()[0]);
        return sum;
    }

    public static float nrm2(INDArray x) {
        JCublasNDArray xC = (JCublasNDArray) x;


        float normal2 = JCublas.cublasSnrm2(x.length(), xC.pointer(), x.stride()[0]);

        return normal2;
    }

    public static int iamax(INDArray x) {

        JCublasNDArray xC = (JCublasNDArray) x;


        int max = JCublas.cublasIdamax(x.length(), xC.pointer(), x.stride()[0]);
        return max;

    }

    public static void axpy(float da, INDArray A, INDArray B) {
        JCublasNDArray xA = (JCublasNDArray) A;
        JCublasNDArray xB = (JCublasNDArray) B;
        JCublas.cublasDaxpy(xA.length(), da, xA.pointer(), A.stride()[0], xB.pointer(), B.stride()[0]);
        xB.getData();
    }


    public static void axpy(IComplexNumber da, IComplexNDArray A, IComplexNDArray B) {
        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;


        JCublas.cublasCaxpy(
                aC.length(),
                jcuda.cuComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                aC.pointer(),
                A.stride()[0],
                bC.pointer(),
                B.stride()[0]
        );

        ((JCublasComplexNDArray) B).getData();
    }

    public static INDArray scal(float alpha, INDArray x) {

        JCublasNDArray xC = (JCublasNDArray) x;
        JCublas.cublasDscal(xC.length(),alpha,xC.pointer(),xC.stride()[0]);
        xC.getData();
        return x;

    }

    public static void copy(INDArray x, INDArray y) {
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        JCublas.cublasDcopy(x.length(),
                xC.pointer(),
                xC.stride()[0],
                yC.pointer(),
                yC.stride()[0]);


        ((JCublasNDArray) y).getData();


    }

    public static float dot(INDArray x, INDArray y) {
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;
        return  JCublas.cublasSdot(x.length(), xC.pointer(), xC.stride()[0],yC.pointer(), yC.stride()[0]);
    }


    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        JCublasComplexNDArray aC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) y;


        jcuda.cuDoubleComplex dott = jcuda.cuDoubleComplex.cuCmplx(0,0);
        dott = JCublas.cublasZdotc(x.length(),aC.pointer(),aC.stride()[0],bC.pointer(),bC.stride()[0]);


        return  NDArrays.createDouble(dott.x,dott.y);
    }
    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) {
        // = alpha * A * tranpose(B) + C
        JCublasNDArray aC = (JCublasNDArray) A;
        JCublasNDArray bC = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;


        JCublas.cublasSger(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                aC.pointer(),        // d_A or x
                A.rows(),   // incx
                bC.pointer(),        // dB or y
                B.rows(),   // incy
                cC.pointer(),        // dC or A
                C.rows()    // lda
        );

        cC.getData();


        return C;
    }





    public static IComplexNDArray zscal(IComplexFloat alpha, IComplexNDArray x) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;

        JCublas.cublasCscal(
                x.length(),
                cuComplex.cuCmplx(alpha.realComponent(),alpha.imaginaryComponent()),
                xC.pointer(),
                x.stride()[0]
        );


        xC.getData();

        return x;
    }


    public static IComplexNDArray zscal(IComplexDouble alpha, IComplexNDArray x) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;

        JCublas.cublasZscal(
                x.length(),
                jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(),alpha.imaginaryComponent()),
                xC.pointer(),
                xC.stride()[0]
        );


        xC.getData();

        return x;
    }

    public static IComplexDouble dotu(IComplexNDArray x, IComplexNDArray y) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray yC = (JCublasComplexNDArray) y;

        jcuda.cuDoubleComplex dott = JCublas.cublasZdotu(x.length(), xC.pointer(), x.stride()[0], yC.pointer(), yC.stride()[0]);


        return NDArrays.createDouble(dott.x, dott.y);
    }

    public static IComplexNDArray geru(IComplexNDArray A,
                                       IComplexNDArray B,
                                       IComplexNDArray C, IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C

        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());

        JCublas.cublasZgeru(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                aC.pointer(),        // d_A or x
                A.rows(),   // incx
                bC.pointer(),        // d_B or y
                B.rows(),   // incy
                cC.pointer(),        // d_C or A
                C.rows()    // lda
        );


        cC.getData();



        return C;
    }

    public static IComplexNDArray gerc(IComplexNDArray A, IComplexNDArray B, IComplexNDArray C,
                                       IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C
        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());


        JCublas.cublasZgerc(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                aC.pointer(),        // dA or x
                A.rows(),   // incx
                bC.pointer(),        // dB or y
                B.rows(),   // incy
                cC.pointer(),        // dC or A
                C.rows()    // lda
        );

        cC.getData();

        return C;
    }
}
