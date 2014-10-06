package org.nd4j.linalg.jcublas;

import jcuda.*;
import jcuda.jcublas.JCublas;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.complex.JCublasComplexNDArray;
import org.springframework.core.io.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;

/**
 * Created by mjk on 8/20/14.
 * @author mjk
 * @author Adam Gibson
 *
 */
public class SimpleJCublas {
    static {
        //write the file to somewhere on java.library.path where there is permissions
        ClassPathResource resource = new ClassPathResource("/" + resourceName() );
        String home = findWritableLibDir();
        File cuBlastmp = new File(home);
        File shared = new File(cuBlastmp,resourceName());
        try {
            shared.createNewFile();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(shared));
            IOUtils.copy(resource.getInputStream(),bos);
            bos.flush();
            bos.close();

        } catch (IOException e) {
            throw new RuntimeException("Unable to initialize jcublas");
        }

        shared.deleteOnExit();
        cuBlastmp.deleteOnExit();


        JCublas.setLogLevel(LogLevel.LOG_DEBUG);

        JCublas.setExceptionsEnabled(true);
        JCublas.cublasInit();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run() {
                JCublas.cublasShutdown();
            }
        });
    }

    private static String resourceName() {
        LibUtils.OSType osType = LibUtils.calculateOS();
        switch(osType) {
            case APPLE:
                return "libJCublas-linux-x86_64.so";
            case LINUX:
                return "libJCublas-linux-x86_64.so";
            case SUN:
                return "libJCublas-linux-x86_64.so";
            case WINDOWS:
               return "libJCublas-windows-x86.dll";
            default:
                return null;
        }
    }

    //finds a writable directory on the java.library.path
    private static String findWritableLibDir() {
        String[] libPath = System.getProperty("java.library.path").split(File.pathSeparator);
        for(String s : libPath) {
            File dir = new File(s);
            if(dir.canWrite())
                return s;
        }

        throw new IllegalStateException("Unable to write to any library directories for jcublas");
    }

    public static void free(Pointer...pointers) {
        for(Pointer arr : pointers)
            JCublas.cublasFree(arr);
    }





    /**
     * Retrieve the data from the gpu
     * @param arr the array to get the data for
     * @param from the origin poitner
     * @param to the to pointer (DO NOT AUGMENT WITH OFFSET: THIS IS TAKEN CARE OF FOR YOU)
     */
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



    /**
     * Allocate and return a pointer
     * based on the length of the ndarray
     * @param ndarray the ndarray to allocate
     * @return the allocated pointer
     */
    public static Pointer alloc(JCublasComplexNDArray ndarray) {
        Pointer ret = new Pointer();
        //allocate memory for the pointer
        JCublas.cublasAlloc(
                ndarray.length() * 2,
                Sizeof.FLOAT
                , ret);

        /* Copy from data to pointer at majorStride() (you want to stride through the data properly) incrementing by 1 for the pointer on the GPU.
        * This allows us to copy only what we need. */

        if(ndarray.length() == ndarray.data().length)
            JCublas.cublasSetVector(
                    ndarray.length() * 2,
                    Sizeof.FLOAT,
                    Pointer.to(ndarray.data()).withByteOffset(ndarray.offset() * Sizeof.FLOAT),
                    1,
                    ret,
                    1);
        else
            JCublas.cublasSetVector(
                    ndarray.length() * 2,
                    Sizeof.FLOAT,
                    Pointer.to(ndarray.data()).withByteOffset(ndarray.offset() * Sizeof.FLOAT),
                    1,
                    ret,
                    1);

        return ret;

    }


    /**
     * Retrieve the data from the gpu
     * @param arr the array to get the data for
     * @param from the origin pointer
     * @param to the to pointer (DO NOT AUGMENT WITH OFFSET: THIS IS TAKEN CARE OF FOR YOU)
     */
    public static void getData(JCublasComplexNDArray arr,Pointer from,Pointer to) {
        //p is typically the data vector which is strided access
        if(arr.length() == arr.data().length)
            JCublas.cublasGetVector(
                    arr.length() * 2,
                    Sizeof.FLOAT,
                    from,
                    1,
                    to.withByteOffset(arr.offset() * Sizeof.FLOAT),
                    1);
        else
            JCublas.cublasGetVector(
                    arr.length() * 2,
                    Sizeof.FLOAT,
                    from,
                    1,
                    to.withByteOffset(arr.offset() * Sizeof.FLOAT),
                    1);




    }


    /**
     * Allocate and return a pointer
     * based on the length of the ndarray
     * @param ndarray the ndarray to allocate
     * @return the allocated pointer
     */
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

    /**
     * General matrix vector multiplication
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
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
                1,
                beta,
                cCPointer,
                1);

        getData(cC,cCPointer,Pointer.to(cC.data()));
        free(cAPointer,cBPointer,cCPointer);


        return C;
    }


    /**
     * General matrix multiply
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexNumber a,IComplexNDArray C
            , IComplexNumber b) {

        JCublas.cublasInit();

        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;
        JCublasComplexNDArray cB = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;

        Pointer cAPointer = alloc(cA);
        Pointer cBPointer = alloc(cB);
        Pointer cCPointer = alloc(cC);


        cuComplex alpha = cuComplex.cuCmplx(a.realComponent().floatValue(),b.imaginaryComponent().floatValue());
        cuComplex beta = cuComplex.cuCmplx(b.realComponent().floatValue(),b.imaginaryComponent().floatValue());

        JCublas.cublasCgemm(
                'n', //trans
                'n',
                cC.rows(),  // m
                cC.columns(), // n
                cA.columns(), //k,
                alpha,
                cAPointer, // A
                A.rows(),  // lda
                cBPointer, // x
                B.rows(), // ldb
                beta,  // beta
                cCPointer, // y
                C.rows()); // ldc


        getData(cC,cCPointer,Pointer.to(cC.data()));
        free(cAPointer,cBPointer,cCPointer);

        return C;

    }


    /**
     * General matrix multiply
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
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


    /**
     * Calculate the 2 norm of the ndarray
     * @param A
     * @return
     */
    public static float nrm2(IComplexNDArray A) {
        JCublas.cublasInit();

        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;
        Pointer cAPointer = alloc(cA);

        float s = JCublas.cublasSnrm2(A.length(), cAPointer, 2);
        free(cAPointer);


        return s;
    }

    /**
     * Copy x to y
     * @param x the origin
     * @param y the destination
     */
    public static void copy(IComplexNDArray x, IComplexNDArray y) {

        JCublas.cublasInit();

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray yC = (JCublasComplexNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        JCublas.cublasScopy(
                x.length(),
                xCPointer,
                1,
                yCPointer,
                1);


        getData(yC,yCPointer,Pointer.to(yC.data()));


        free(xCPointer,yCPointer);
    }


    /**
     * Return the index of the max in the given ndarray
     * @param x the ndarray to ge tthe max for
     * @return
     */
    public static int iamax(IComplexNDArray x) {

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        Pointer xCPointer = alloc(xC);

        int max = JCublas.cublasIzamax(x.length(), xCPointer, 1);
        free(xCPointer);
        return max;
    }

    /**
     *
     * @param x
     * @return
     */
    public static float asum(IComplexNDArray x) {

        JCublas.cublasInit();

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;

        Pointer xCPointer = alloc(xC);
        float sum = JCublas.cublasScasum(x.length(), xCPointer, 1);
        free(xCPointer);
        return sum;
    }


    /**
     * Swap the elements in each ndarray
     * @param x
     * @param y
     */
    public static void swap(INDArray x, INDArray y) {
        JCublas.cublasInit();

        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;
        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);



        JCublas.cublasSswap(
                xC.length(),
                xCPointer,
                1,
                yCPointer,
                1);

        getData(yC,yCPointer,Pointer.to(yC.data()));
        free(xCPointer,yCPointer);

    }

    /**
     *
     * @param x
     * @return
     */
    public static float asum(INDArray x) {
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        Pointer xCPointer = alloc(xC);


        float sum = JCublas.cublasSasum(x.length(), xCPointer,1);
        free(xCPointer);
        return sum;
    }

    /**
     * Returns the norm2 of the given ndarray
     * @param x
     * @return
     */
    public static float nrm2(INDArray x) {
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        Pointer xCPointer = alloc(xC);


        float normal2 = JCublas.cublasSnrm2(x.length(), xCPointer, 1);
        JCublas.cublasFree(xCPointer);
        return normal2;
    }

    /**
     * Returns the index of the max element
     * in the given ndarray
     * @param x
     * @return
     */
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

    /**
     * And and scale by the given scalar da
     * @param da
     * @param A
     * @param B
     */
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
                    1);
            getData(xB, xBPointer, Pointer.to(xB.data()));

        }


        free(xAPointer,xBPointer);

    }


    /**
     *
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexNumber da, IComplexNDArray A, IComplexNDArray B) {
        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;

        JCublas.cublasInit();

        Pointer aCPointer = alloc(aC);
        Pointer bCPointer = alloc(bC);



        JCublas.cublasCaxpy(
                aC.length(),
                jcuda.cuComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                aCPointer,
                1,
                bCPointer,
                1
        );

        getData(bC,bCPointer,Pointer.to(bC.data()));

        free(aCPointer,bCPointer);
    }

    /**
     * Multiply the given ndarray
     *  by alpha
     * @param alpha
     * @param x
     * @return
     */
    public static INDArray scal(float alpha, INDArray x) {
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;

        Pointer xCPointer = alloc(xC);
        JCublas.cublasSscal(
                xC.length(),
                alpha,
                xCPointer,
                1);
        getData(xC, xCPointer, Pointer.to(xC.data()));
        free(xCPointer);
        return x;

    }

    /**
     * Copy x to y
     * @param x
     * @param y
     */
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

    /**
     * Dot product between 2 ndarrays
     * @param x
     * @param y
     * @return
     */
    public static float dot(INDArray x, INDArray y) {
        JCublas.cublasInit();

        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        float ret =  JCublas.cublasSdot(
                x.length(),
                xCPointer,
                1
                ,yCPointer,
                1);

        free(xCPointer,yCPointer);
        return ret;
    }


    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {

        JCublas.cublasInit();

        JCublasComplexNDArray aC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) y;

        Pointer aCPointer = alloc(aC);
        Pointer bCPointer = alloc(bC);


        jcuda.cuDoubleComplex dott = JCublas.cublasZdotc(
                x.length(),
                aCPointer,
                1,
                bCPointer,
                1);

        IComplexDouble ret =   Nd4j.createDouble(dott.x, dott.y);
        free(aCPointer,bCPointer);
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


    /**
     *
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray zscal(IComplexFloat alpha, IComplexNDArray x) {

        JCublas.cublasInit();

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        Pointer xCPointer = alloc(xC);


        JCublas.cublasCscal(
                x.length(),
                cuComplex.cuCmplx(alpha.realComponent(),alpha.imaginaryComponent()),
                xCPointer,
                1
        );


        getData(xC,xCPointer,Pointer.to(xC.data()));
        free(xCPointer);
        return x;
    }


    /**
     * Complex multiplication of an ndarray
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray zscal(IComplexDouble alpha, IComplexNDArray x) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;

        JCublas.cublasInit();

        Pointer xCPointer = alloc(xC);

        JCublas.cublasZscal(
                x.length(),
                jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent()),
                xCPointer,
                1
        );


        getData(xC,xCPointer,Pointer.to(xC.data()));

        free(xCPointer);

        return x;
    }

    /**
     * Complex dot product
     * @param x
     * @param y
     * @return
     */
    public static IComplexDouble dotu(IComplexNDArray x, IComplexNDArray y) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray yC = (JCublasComplexNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        jcuda.cuDoubleComplex dott = JCublas.cublasZdotu(x.length(), xCPointer,1, yCPointer, 1);


        IComplexDouble ret = Nd4j.createDouble(dott.x, dott.y);
        free(xCPointer,yCPointer);
        return ret;
    }

    /***
     *
     * @param A
     * @param B
     * @param C
     * @param Alpha
     * @return
     */
    public static IComplexNDArray geru(IComplexNDArray A,
                                       IComplexNDArray B,
                                       IComplexNDArray C, IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C

        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;

        Pointer aCPointer = alloc(aC);
        Pointer bCPointer = alloc(bC);
        Pointer cCPointer = alloc(cC);

        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());

        JCublas.cublasZgeru(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                aCPointer,        // d_A or x
                A.rows(),   // incx
                bCPointer,        // d_B or y
                B.rows(),   // incy
                cCPointer,        // d_C or A
                C.rows()    // lda
        );


        getData(cC,cCPointer,Pointer.to(cC.data()));

        free(aCPointer,bCPointer,cCPointer);


        return C;
    }

    /**
     *
     * @param A
     * @param B
     * @param C
     * @param Alpha
     * @return
     */
    public static IComplexNDArray gerc(IComplexNDArray A, IComplexNDArray B, IComplexNDArray C,
                                       IComplexDouble Alpha) {
        // = alpha * A * tranpose(B) + C
        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;


        Pointer aCPointer = alloc(aC);
        Pointer bCPointer = alloc(bC);
        Pointer cCPointer = alloc(cC);


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());


        JCublas.cublasZgerc(
                A.rows(),   // m
                A.columns(),// n
                alpha,      // alpha
                aCPointer,        // dA or x
                A.rows(),   // incx
                bCPointer,        // dB or y
                B.rows(),   // incy
                cCPointer,        // dC or A
                C.rows()    // lda
        );


        getData(cC,cCPointer,Pointer.to(cC.data()));

        free(aCPointer,bCPointer,cCPointer);
        return C;
    }


    /**
     * Simpler version of saxpy
     * taking in to account the parameters of the ndarray
     * @param alpha the alpha to scale by
     * @param x the x
     * @param y the y
     */
    public static void saxpy(float alpha, INDArray x, INDArray y) {
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        JCublas.cublasSaxpy(x.length(),alpha,xCPointer,1,yCPointer,1);


        getData(yC,yCPointer,Pointer.to(yC.data()));
        free(xCPointer,yCPointer);

    }
}
