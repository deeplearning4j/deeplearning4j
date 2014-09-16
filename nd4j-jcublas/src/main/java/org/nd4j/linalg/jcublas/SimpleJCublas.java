package org.nd4j.linalg.jcublas;

import jcuda.*;
import jcuda.jcublas.JCublas;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.complex.JCublasComplexNDArray;

/**
 * Created by mjk on 8/20/14.
 * @author mjk
 * @author Adam Gibson
 *
 */
public class SimpleJCublas {
    static {
        JCublas.setLogLevel(LogLevel.LOG_DEBUG);
        JCublas.setExceptionsEnabled(true);
        JCublas.cublasInit();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run() {
                JCublas.cublasShutdown();
            }
        });
    }

    public static void alloc(JCublasComplexNDArray...arrs) {
        for(JCublasComplexNDArray arr : arrs)
            arr.alloc();
    }

    public static void free(JCublasComplexNDArray...arrs) {
        for(JCublasComplexNDArray arr : arrs)
            arr.free();
    }
    public static void alloc(JCublasNDArray...arrs) {
        for(JCublasNDArray arr : arrs)
            arr.alloc();
    }
    public static void allocTest(JCublasNDArray...arrs) {
        for(JCublasNDArray arr : arrs)
            arr.allocTest();
    }
    public static void free(JCublasNDArray...arrs) {
        for(JCublasNDArray arr : arrs)
            arr.free();

    }


    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) {


        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        alloc(cA,cB,cC);

        JCublas.cublasSgemv('N',
                A.rows(),
                A.columns(),
                alpha,
                cA.pointer(),
                A.rows(),
                cB.pointer(),
                cB.stride()[0],
                beta,
                cC.pointer(),
                cC.stride()[0]);

        cC.getData();
        free(cA,cB,cC);


        return C;
    }


    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexNumber a,IComplexNDArray C
            , IComplexNumber b) {

        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;
        JCublasComplexNDArray cB = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;
        alloc(cA,cB,cC);

        cuComplex alpha = cuComplex.cuCmplx(a.realComponent().floatValue(),b.imaginaryComponent().floatValue());
        cuComplex beta = cuComplex.cuCmplx(b.realComponent().floatValue(),b.imaginaryComponent().floatValue());

        JCublas.cublasCgemm(
                'n', //trans
                'n',
                cC.rows(),  // m
                cC.columns(), // n
                cA.columns(), //k,
                alpha,
                cA.pointer().withByteOffset((cA.offset()) * Sizeof.FLOAT), // A
                A.rows(),  // lda
                cB.pointer().withByteOffset((cB.offset()) * Sizeof.FLOAT), // x
                B.rows(), // ldb
                beta,  // beta
                cC.pointer().withByteOffset((cC.offset()) * Sizeof.FLOAT), // y
                C.rows()); // ldc


        cC.getData();
        free(cA,cB,cC);

        return C;

    }



    public static INDArray gemm(INDArray A, INDArray B, INDArray C,
                                float alpha, float beta) {



        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        alloc(cA,cB,cC);

        JCublas.cublasSgemm(
                'n', //trans
                'n',
                C.rows(),  // m
                C.columns(), // n
                A.columns(), //k,
                alpha,
                cA.pointer(), // A
                A.rows(),  // lda
                cB.pointer(), // x
                B.rows(), // ldb
                beta,  // beta
                cC.pointer(), // y
                C.rows()); // incy
        cC.getData();

        free(cA,cB,cC);

        return C;

    }

    public static void dcopy(int length, float[] data, int offset, int i, float[] data1, int i1, int i2) {

    }

    public static float nrm2(IComplexNDArray A) {
        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;
        alloc(cA);
        float s = JCublas.cublasSnrm2(A.length(), cA.pointer(), 2);
        free(cA);


        return s;
    }

    public static void copy(IComplexNDArray x, IComplexNDArray y) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray yC = (JCublasComplexNDArray) y;
        alloc(xC,yC);
        JCublas.cublasScopy(x.length(),
                xC.pointer(),
                xC.stride()[0],
                yC.pointer(),
                yC.stride()[0]);

        yC.getData();

        free(xC,yC);
    }

    public static int iamax(IComplexNDArray x) {

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        alloc(xC);
        int max = JCublas.cublasIzamax(x.length(), xC.pointer(), 1);
        free(xC);
        return max;
    }

    public static float asum(IComplexNDArray x) {

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        alloc(xC);

        float sum = JCublas.cublasScasum(x.length(), xC.pointer(), 1);
        free(xC);
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
        alloc(xC,yC);
        JCublas.cublasSswap(xC.length(),
                xC.pointer(),
                1,
                yC.pointer(),
                1);

        yC.getData();
        free(xC,yC);

    }

    public static float asum(INDArray x) {

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        alloc(xC);

        float sum = JCublas.cublasSasum(x.length(), xC.pointer(),1);
        free(xC);
        return sum;
    }

    public static float nrm2(INDArray x) {
        JCublasNDArray xC = (JCublasNDArray) x;
        alloc(xC);

        float normal2 = JCublas.cublasSnrm2(x.length(), xC.pointer(), 1);
        free(xC);
        return normal2;
    }

    public static int iamax(INDArray x) {

        JCublasNDArray xC = (JCublasNDArray) x;
        alloc(xC);

        int max = JCublas.cublasIsamax(
                x.length(),
                xC.pointer(),
                1);
        free(xC);
        return max - 1;

    }

    public static void axpy(float da, INDArray A, INDArray B) {
        JCublasNDArray xA = (JCublasNDArray) A;
        JCublasNDArray xB = (JCublasNDArray) B;
        //allocate memory
        alloc(xA, xB);


        if(xA.ordering() == NDArrayFactory.C) {
            JCublas.cublasSaxpy(
                    xA.length(),
                    da,
                    xA.pointerWithOffset(),
                    1,
                    xB.pointerWithOffset(),
                    1);
            xB.getData();
        }
        else {
            JCublas.cublasSaxpy(
                    xA.length(),
                    da,
                    xA.pointer(),
                    1,
                    xB.pointer(),
                    1);
            xB.getData();
        }



        free(xA,xB);
    }


    public static void axpy(IComplexNumber da, IComplexNDArray A, IComplexNDArray B) {
        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;
        alloc(aC,bC);

        JCublas.cublasCaxpy(
                aC.length(),
                jcuda.cuComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                aC.pointer(),
                1,
                bC.pointer(),
                1
        );

        ((JCublasComplexNDArray) B).getData();
        free(aC,bC);
    }

    public static INDArray scal(float alpha, INDArray x) {

        JCublasNDArray xC = (JCublasNDArray) x;
        alloc(xC);
        JCublas.cublasSscal(
                xC.length(),
                alpha,
                xC.pointer(),
                1);
        xC.getData();
        free(xC);
        return x;

    }

    public static void copy(INDArray x, INDArray y) {
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;
        alloc(xC,yC);
        JCublas.cublasDcopy(x.length(),
                xC.pointer(),
                1,
                yC.pointer(),
                1);


        ((JCublasNDArray) y).getData();
        free(xC,yC);


    }

    public static float dot(INDArray x, INDArray y) {
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;
        alloc(xC,yC);
        float ret=  JCublas.cublasSdot(
                x.length(),
                xC.pointer(),
                xC.stride()[0]  * Sizeof.FLOAT
                ,yC.pointer(),
                yC.stride()[0]  * Sizeof.FLOAT);
        free(xC,yC);
        return ret;
    }


    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        JCublasComplexNDArray aC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) y;
        alloc(aC,bC);

        jcuda.cuDoubleComplex dott = JCublas.cublasZdotc(
                x.length(),
                aC.pointer(),
                aC.stride()[0]  * Sizeof.FLOAT,
                bC.pointer(),
                bC.stride()[0]  * Sizeof.FLOAT);

        IComplexDouble ret =   Nd4j.createDouble(dott.x, dott.y);
        free(aC,bC);
        return ret;
    }
    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) {
        // = alpha * A * tranpose(B) + C
        JCublasNDArray aC = (JCublasNDArray) A;
        JCublasNDArray bC = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;
        alloc(aC,bC,cC);

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
        free(aC,bC,cC);

        return C;
    }





    public static IComplexNDArray zscal(IComplexFloat alpha, IComplexNDArray x) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        alloc(xC);
        JCublas.cublasCscal(
                x.length(),
                cuComplex.cuCmplx(alpha.realComponent(),alpha.imaginaryComponent()),
                xC.pointer(),
                1
        );


        xC.getData();
        free(xC);
        return x;
    }


    public static IComplexNDArray zscal(IComplexDouble alpha, IComplexNDArray x) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        alloc(xC);
        JCublas.cublasZscal(
                x.length(),
                jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent()),
                xC.pointer(),
                1
        );


        xC.getData();
        free(xC);
        return x;
    }

    public static IComplexDouble dotu(IComplexNDArray x, IComplexNDArray y) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray yC = (JCublasComplexNDArray) y;
        alloc(xC,yC);
        jcuda.cuDoubleComplex dott = JCublas.cublasZdotu(x.length(), xC.pointer(), x.stride()[0]  * Sizeof.FLOAT, yC.pointer(), yC.stride()[0]  * Sizeof.FLOAT);


        IComplexDouble ret = Nd4j.createDouble(dott.x, dott.y);
        free(xC,yC);
        return ret;
    }

    public static IComplexNDArray geru(IComplexNDArray A,
                                       IComplexNDArray B,
                                       IComplexNDArray C, IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C

        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;
        alloc(aC,bC,cC);

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
        free(aC,bC,cC);


        return C;
    }

    public static IComplexNDArray gerc(IComplexNDArray A, IComplexNDArray B, IComplexNDArray C,
                                       IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C
        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;
        alloc(aC,bC,cC);

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
        free(aC,bC,cC);
        return C;
    }
}
