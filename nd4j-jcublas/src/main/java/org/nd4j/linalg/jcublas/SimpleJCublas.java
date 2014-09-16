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

    public static void free(Pointer...pointers) {
        for(Pointer arr : pointers)
            JCublas.cublasFree(arr);
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


    public static void getData(JCublasNDArray arr,Pointer from,Pointer to) {
        //p is typically the data vector which is strided access
        if(arr.length() == arr.data().length)
            JCublas.cublasGetVector(
                    arr.length(),
                    Sizeof.FLOAT,
                    from,
                    1,
                    to.withByteOffset(arr.offset() * Sizeof.FLOAT),
                    1);
        else
            JCublas.cublasGetVector(
                    arr.length(),
                    Sizeof.FLOAT,
                    from,
                    1,
                    to.withByteOffset(arr.offset() * Sizeof.FLOAT),
                    arr.majorStride());




    }



    public static Pointer alloc(JCublasNDArray ndarray) {
        Pointer ret = new Pointer();
        //allocate memory for the pointer
        JCublas.cublasAlloc(
                ndarray.length(),
                Sizeof.FLOAT
                , ret);

        /* Copy from data to pointer at majorStride() (you want to stride through the data properly) incrementing by 1 for the pointer on the GPU.
        * This allows us to copy only what we need. */

        if(ndarray.length() == ndarray.data().length)
            JCublas.cublasSetVector(
                    ndarray.length(),
                    Sizeof.FLOAT,
                    Pointer.to(ndarray.data()).withByteOffset(ndarray.offset() * Sizeof.FLOAT),
                    1,
                    ret,
                    1);
        else
            JCublas.cublasSetVector(
                    ndarray.length(),
                    Sizeof.FLOAT,
                    Pointer.to(ndarray.data()).withByteOffset(ndarray.offset() * Sizeof.FLOAT),
                    ndarray.majorStride(),
                    ret,
                    1);

        return ret;

    }

    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) {


        JCublas.cublasInit();

        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        Pointer cAPointer = alloc(cA);
        Pointer cBPointer = alloc(cB);
        Pointer cCPointer = alloc(cC);


        JCublas.cublasSgemv('N',
                A.rows(),
                A.columns(),
                alpha,
                cAPointer,
                A.rows(),
                cBPointer,
                cB.majorStride(),
                beta,
                cCPointer,
                cC.majorStride());

        getData(cC,cCPointer,Pointer.to(cC.data()));
        free(cAPointer,cBPointer,cCPointer);


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


        JCublas.cublasInit();

        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        Pointer cAPointer = alloc(cA);
        Pointer cBPointer = alloc(cB);
        Pointer cCPointer = alloc(cC);



        JCublas.cublasSgemm(
                'n', //trans
                'n',
                C.rows(),  // m
                C.columns(), // n
                A.columns(), //k,
                alpha,
                cAPointer, // A
                A.rows(),  // lda
                cBPointer, // x
                B.rows(), // ldb
                beta,  // beta
                cCPointer, // y
                C.rows()); // incy

        getData(cC,cCPointer,Pointer.to(cC.data()));

        free(cAPointer,cBPointer,cCPointer);

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
        JCublas.cublasInit();

        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;
        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);


        alloc(xC,yC);
        JCublas.cublasSswap(
                xC.length(),
                xCPointer,
                1,
                yCPointer,
                1);

        yC.getData();
        free(xC,yC);

    }

    public static float asum(INDArray x) {
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        Pointer xCPointer = alloc(xC);


        float sum = JCublas.cublasSasum(x.length(), xCPointer,1);
        JCublas.cublasFree(xCPointer);
        return sum;
    }

    public static float nrm2(INDArray x) {
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        Pointer xCPointer = alloc(xC);


        float normal2 = JCublas.cublasSnrm2(x.length(), xCPointer, 1);
        JCublas.cublasFree(xCPointer);
        return normal2;
    }

    public static int iamax(INDArray x) {
        JCublas.cublasInit();

        JCublasNDArray xC = (JCublasNDArray) x;
        Pointer xCPointer = alloc(xC);


        int max = JCublas.cublasIsamax(
                x.length(),
                xCPointer,
                1);
        free(xCPointer);
        return max - 1;

    }

    public static void axpy(float da, INDArray A, INDArray B) {
        JCublas.cublasInit();

        JCublasNDArray xA = (JCublasNDArray) A;
        JCublasNDArray xB = (JCublasNDArray) B;

        Pointer xAPointer = alloc(xA);
        Pointer xBPointer = alloc(xB);


        if(xA.ordering() == NDArrayFactory.C) {
            JCublas.cublasSaxpy(
                    xA.length(),
                    da,
                    xAPointer,
                    1,
                    xBPointer,
                    1);
            getData(xB,xBPointer,Pointer.to(xB.data()));
        }
        else {
            JCublas.cublasSaxpy(
                    xA.length(),
                    da,
                    xAPointer,
                    1,
                    xBPointer,
                    xB.majorStride());
            getData(xB, xBPointer, Pointer.to(xB.data()));

        }


        free(xAPointer,xBPointer);

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
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        alloc(xC);
        Pointer xCPointer = alloc(xC);
        JCublas.cublasSscal(
                xC.length(),
                alpha,
                xCPointer,
                1);
        getData(xC, xCPointer, Pointer.to(xC.data()));
        JCublas.cublasFree(xCPointer);
        return x;

    }

    public static void copy(INDArray x, INDArray y) {
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;
        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);
        JCublas.cublasDcopy(x.length(),
                xCPointer,
                1,
                yCPointer,
                1);


        getData(yC,yCPointer,Pointer.to(yC.data()));
        free(xCPointer,yCPointer);


    }

    public static float dot(INDArray x, INDArray y) {
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        float ret=  JCublas.cublasSdot(
                x.length(),
                xCPointer,
                xC.stride()[0]  * Sizeof.FLOAT
                ,yCPointer,
                yC.stride()[0]  * Sizeof.FLOAT);
        free(xCPointer,yCPointer);
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

        JCublas.cublasInit();
        // = alpha * A * transpose(B) + C
        JCublasNDArray aC = (JCublasNDArray) A;
        JCublasNDArray bC = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        Pointer aCPointer = alloc(aC);
        Pointer bCPointer = alloc(bC);
        Pointer cCPointer = alloc(cC);


        JCublas.cublasSger(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                aCPointer,        // d_A or x
                A.rows(),   // incx
                bCPointer,        // dB or y
                B.rows(),   // incy
                cCPointer,        // dC or A
                C.rows()    // lda
        );

        getData(cC,cCPointer,Pointer.to(cC.data()));
        free(aCPointer, bCPointer, cCPointer);

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
