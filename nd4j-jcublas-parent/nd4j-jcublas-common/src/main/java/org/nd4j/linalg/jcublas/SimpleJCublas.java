package org.nd4j.linalg.jcublas;

import jcuda.*;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.complex.JCublasComplexNDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import javax.xml.crypto.Data;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Iterator;

/**
 * Created by mjk on 8/20/14.
 * @author mjk
 * @author Adam Gibson
 *
 */
public class SimpleJCublas {
    public final static String CUDA_HOME = "CUDA_HOME";
    public final static String JCUDA_HOME_PROP = "jcuda.home";
    private static boolean init = false;
    private static Logger log = LoggerFactory.getLogger(SimpleJCublas.class);
    static {
        init();
    }



    public static void assertCudaBuffer(INDArray...buffer) {
        for(INDArray b1 : buffer)
            if(!(b1.data() instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }


    public static void assertCudaBuffer(DataBuffer...buffer) {
        for(DataBuffer b1 : buffer)
            if(!(b1 instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }

    public static Pointer getPointer(INDArray array) {
        JCudaBuffer buffer = (JCudaBuffer) array.data();
        return buffer.pointer().withByteOffset(buffer.elementSize() * array.offset());
    }



    public static void loadJCublas(File cuBlastmp,ClassPathResource resource, LibUtils.OSType os) {
        File shared = new File(cuBlastmp,os == LibUtils.OSType.WINDOWS ? resourceName().replace("X","x").replaceAll("lib","") : resourceName().replace("X","x"));
        try {
            if(shared.exists())
                shared.delete();
            shared.createNewFile();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(shared));
            IOUtils.copy(resource.getInputStream(),bos);
            bos.flush();
            bos.close();
            shared.deleteOnExit();

        } catch (IOException e) {
            throw new RuntimeException("Unable to initialize jcublas",e);
        }
    }




    /**
     * Initialize jcublas only called once
     */
    public static void init() {
        if(init)
            return;


        String path = System.getProperty("java.library.path");
        log.info("Loading native artifacts from " + path);
        //write the file to somewhere on java.library.path where there is permissions
        String name = "/" + resourceName().substring(3).replace("X","x");
        ClassPathResource resource = new ClassPathResource(name);

        if(!resource.exists() && name.startsWith(File.separator + "lib" + File.separator))
            resource = new ClassPathResource(name.replaceAll("/lib/",""));
        else if(!resource.exists()) {
            resource = new ClassPathResource(name);
        }

        if(!resource.exists())
            throw new IllegalStateException("Unable to find resource with name " + resource.getFilename());

        log.info("Loading jcublas from " + resource.getFilename());


        String home = findWritableLibDir();
        File cuBlastmp = new File(home);
        LibUtils.OSType os = LibUtils.calculateOS();
        loadJCublas(cuBlastmp, resource, os);


        JCublas.setLogLevel(LogLevel.LOG_DEBUG);
        JCublas.setExceptionsEnabled(true);

        init = true;
    }



    private static String libDir() {
        int bits =  thirtyTwoOrSixtyFour();
        LibUtils.OSType o = LibUtils.calculateOS();
        if(o != LibUtils.OSType.WINDOWS) {
            String base = cudaBase() + File.separator  + libFolder();
            String ret =  base + (bits == 64 ? "64" : "");
            File test = new File(ret);
            boolean exists = test.exists();
            if(exists)
                return ret;
            if(!exists) {
                File maybeThirtyTwoBit = new File(base);
                if(bits == 64 && maybeThirtyTwoBit.exists()) {
                    log.warn("Loading 32 bit cuda...no 64 bit found");
                    return base;
                }
            }
            else {
                File testOther = new File(base);
                if(!exists && !testOther.exists())
                    throw new IllegalStateException("No lib directory found");
                else
                    return base;
            }

            return ret;

        }

        else {
            String base = cudaBase() + File.separator  + libFolder() + File.separator;
            if(bits == 32) {
                base += "Win32";
            }
            else
                base += "x64";
            return base;
        }
    }


    private static String libFolder() {
        return  "lib";
    }

    private static int thirtyTwoOrSixtyFour() {
        LibUtils.ARCHType ar = LibUtils.calculateArch();
        switch(ar) {
            case X86_64:
                return 64;
            case PPC_64:
                return 64;
            default:
                return 32;
        }

    }


    private static String cudaBase() {
        String cudaHome = System.getProperty(JCUDA_HOME_PROP,System.getenv(CUDA_HOME));

        if(cudaHome != null)
            return cudaHome;
        throw new IllegalStateException("Please specify a cuda home property in your environment (export CUDA_HOME=/path/to/your/dir) or via -Djcuda.home=/path/to/your/dir");

    }



    private static String jcudaRuntimeName() {
        LibUtils.OSType osType = LibUtils.calculateOS();
        LibUtils.ARCHType ar = LibUtils.calculateArch();

        switch(osType) {
            case APPLE:
                return String.format("JCudaRuntime-apple-%s.dylib",ar.toString());
            case LINUX:
                return String.format("JCudaRuntime-linux-%s.so",ar.toString());
            case SUN:
                return String.format("JCudaRuntime-linux-%s.so",ar.toString());
            case WINDOWS:
                return String.format("JCudaRuntime-windows-%s.dll",ar.toString());
            default:
                return null;
        }
    }

    private static String jcudaDriverName() {
        LibUtils.OSType osType = LibUtils.calculateOS();
        LibUtils.ARCHType ar = LibUtils.calculateArch();

        switch(osType) {
            case APPLE:
                return String.format("JCudaDriver-apple-%s.dylib",ar.toString());
            case LINUX:
                return String.format("JCudaDriver-linux-%s.so",ar.toString());
            case SUN:
                return String.format("JCudaDriver-linux-%s.so",ar.toString());
            case WINDOWS:
                return String.format("JCudaDriver-windows-%s.dll",ar.toString());
            default:
                return null;
        }
    }

    private static String resourceName() {
        LibUtils.OSType osType = LibUtils.calculateOS();
        LibUtils.ARCHType ar = LibUtils.calculateArch();

        switch(osType) {
            case APPLE:
                return String.format("libJCublas-apple-%s.dylib",ar.toString());
            case LINUX:
                return String.format("libJCublas-linux-%s.so",ar.toString());
            case SUN:
                return String.format("libJCublas-linux-%s.so",ar.toString());
            case WINDOWS:
                return String.format("libJCublas-windows-%s.dll",ar.toString());
            default:
                return null;
        }
    }

    //finds a writable directory on the java.library.path
    private static String findWritableLibDir() {
        String[] libPath = System.getProperty("java.library.path").split(File.pathSeparator);
        for(String s : libPath) {
            File dir = new File(s);
            if(canWrite(dir))
                return s;
        }

        throw new IllegalStateException("Unable to write to any library directories for jcublas");
    }

    private static boolean canWrite(File dir) {
        if(!dir.exists())
            return false;
        else if(dir.isFile())
            throw new IllegalArgumentException("Tests only directories");
        else {
            File newFile = new File(dir,"dummyfile");
            newFile.deleteOnExit();
            try {
                if(!newFile.createNewFile())
                    return false;
            }catch(IOException e) {
                return false;
            }

            return true;
        }
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
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, double alpha, double beta) {

        DataTypeValidation.assertDouble(A,B,C);
        assertCudaBuffer(A.data(),B.data(),C.data());



        Pointer cAPointer = getPointer(A);
        Pointer cBPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


        JCublas.cublasDgemv(
                'N',
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


        return C;
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

        DataTypeValidation.assertFloat(A,B,C);


        Pointer cAPointer = getPointer(A);
        Pointer cBPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


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


        return C;
    }






    /**
     * General matrix vector
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemv(IComplexNDArray A, IComplexNDArray B, IComplexDouble a,IComplexNDArray C
            , IComplexDouble b) {
        DataTypeValidation.assertSameDataType(A,B,C);




        Pointer cAPointer = getPointer(A);
        Pointer cBPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(a.realComponent().doubleValue(),b.imaginaryComponent().doubleValue());
        cuDoubleComplex beta = cuDoubleComplex.cuCmplx(b.realComponent().doubleValue(),b.imaginaryComponent().doubleValue());

        JCublas.cublasZgemv(
                'n', //trans
                A.rows(),  // m
                A.rows(), // n
                alpha,
                cAPointer, // A
                A.rows(),  // lda
                cBPointer, // x
                B.secondaryStride(), // ldb
                beta,  // beta
                cCPointer, // y
                C.secondaryStride()); // ldc



        return C;

    }

    /**
     * General matrix vector
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemv(IComplexNDArray A, IComplexNDArray B, IComplexFloat a,IComplexNDArray C
            , IComplexFloat b) {
        DataTypeValidation.assertFloat(A,B,C);
        assertCudaBuffer(A,B,C);

        Pointer cAPointer = getPointer(A);
        Pointer cBPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


        cuComplex alpha = cuComplex.cuCmplx(a.realComponent().floatValue(),b.imaginaryComponent().floatValue());
        cuComplex beta = cuComplex.cuCmplx(b.realComponent().floatValue(),b.imaginaryComponent().floatValue());

        JCublas.cublasCgemv(
                'n', //trans
                A.rows(),  // m
                A.columns(), // n
                alpha,
                cAPointer, // A
                A.rows(),  // lda
                cBPointer, // x
                B.secondaryStride(), // ldb
                beta,  // beta
                cCPointer, // y
                C.secondaryStride()); // ldc



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
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexDouble a,IComplexNDArray C
            , IComplexDouble b) {
        DataTypeValidation.assertSameDataType(A,B,C);




        Pointer cAPointer = getPointer(A);
        Pointer cBPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(a.realComponent().doubleValue(),b.imaginaryComponent().doubleValue());
        cuDoubleComplex beta = cuDoubleComplex.cuCmplx(b.realComponent().doubleValue(),b.imaginaryComponent().doubleValue());

        JCublas.cublasZgemm(
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
                C.rows()); // ldc


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
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexFloat a,IComplexNDArray C
            , IComplexFloat b) {
        DataTypeValidation.assertFloat(A,B,C);



        Pointer cAPointer = getPointer(A);
        Pointer cBPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


        cuComplex alpha = cuComplex.cuCmplx(a.realComponent().floatValue(),b.imaginaryComponent().floatValue());
        cuComplex beta = cuComplex.cuCmplx(b.realComponent().floatValue(),b.imaginaryComponent().floatValue());

        JCublas.cublasCgemm(
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
                C.rows()); // ldc



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
                                double alpha, double beta) {

        DataTypeValidation.assertDouble(A,B,C);


        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        Pointer cAPointer = getPointer(cA);
        Pointer cBPointer = getPointer(cB);
        Pointer cCPointer = getPointer(cC);



        JCublas.cublasDgemm(
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
        DataTypeValidation.assertFloat(A,B,C);



        Pointer cAPointer = getPointer(A);
        Pointer cBPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);



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


        return C;

    }


    /**
     * Calculate the 2 norm of the ndarray
     * @param A
     * @return
     */
    public static double nrm2(IComplexNDArray A) {


        Pointer cAPointer = getPointer(A);
        if(A.data().dataType() == DataBuffer.FLOAT) {
            float s = JCublas.cublasSnrm2(A.length(), cAPointer, 2);
            return s;
        }
        else {
            double s = JCublas.cublasDnrm2(A.length(), cAPointer, 2);
            return s;
        }

    }

    /**
     * Copy x to y
     * @param x the origin
     * @param y the destination
     */
    public static void copy(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x,y);


        Pointer xCPointer = getPointer(x);
        Pointer yCPointer = getPointer(y);

        if(x.data().dataType() == DataBuffer.FLOAT) {
            JCublas.cublasScopy(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);



        }

        else {
            JCublas.cublasDcopy(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);



        }


    }


    /**
     * Return the index of the max in the given ndarray
     * @param x the ndarray to ge tthe max for
     * @return
     */
    public static int iamax(IComplexNDArray x) {

        Pointer xCPointer = getPointer(x);
        if(x.data().dataType() == DataBuffer.FLOAT) {
            int max = JCublas.cublasIsamax(x.length(), xCPointer, 1);
            return max;
        }

        else {
            int max = JCublas.cublasIzamax(x.length(), xCPointer, 1);
            return max;
        }

    }

    /**
     *
     * @param x
     * @return
     */
    public static float asum(IComplexNDArray x) {
        Pointer xCPointer = getPointer(x);
        float sum = JCublas.cublasScasum(x.length(), xCPointer, 1);
        return sum;
    }


    /**
     * Swap the elements in each ndarray
     * @param x
     * @param y
     */
    public static void swap(INDArray x, INDArray y) {

        DataTypeValidation.assertSameDataType(x,y);

        Pointer xCPointer = getPointer(x);
        Pointer yCPointer = getPointer(y);


        if(x.data().dataType() == DataBuffer.FLOAT) {
            JCublas.cublasSswap(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);

        }

        else {
            JCublas.cublasDswap(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);

        }


    }

    /**
     *
     * @param x
     * @return
     */
    public static double asum(INDArray x) {

        Pointer xCPointer = getPointer(x);
        if(x.data().dataType() == DataBuffer.FLOAT) {
            float sum = JCublas.cublasSasum(x.length(), xCPointer,1);
            return sum;
        }
        else {
            double sum = JCublas.cublasDasum(x.length(),xCPointer,1);
            return sum;
        }

    }

    /**
     * Returns the norm2 of the given ndarray
     * @param x
     * @return
     */
    public static float nrm2(INDArray x) {

        Pointer xCPointer = getPointer(x);


        float normal2 = JCublas.cublasSnrm2(x.length(), xCPointer, 1);
        return normal2;
    }

    /**
     * Returns the index of the max element
     * in the given ndarray
     * @param x
     * @return
     */
    public static int iamax(INDArray x) {


        Pointer xCPointer = getPointer(x);


        int max = JCublas.cublasIsamax(
                x.length(),
                xCPointer,
                1);

        return max - 1;

    }







    /**
     * And and scale by the given scalar da
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(float da, INDArray A, INDArray B) {

        DataTypeValidation.assertFloat(A,B);

        Pointer xAPointer = getPointer(A);
        Pointer xBPointer = getPointer(B);


        if(A.ordering() == NDArrayFactory.C) {
            JCublas.cublasSaxpy(
                    A.length(),
                    da,
                    xAPointer,
                    1,
                    xBPointer,
                    1);

        }
        else {
            JCublas.cublasSaxpy(
                    A.length(),
                    da,
                    xAPointer,
                    1,
                    xBPointer,
                    1);


        }




    }

    /**
     *
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexFloat da, IComplexNDArray A, IComplexNDArray B) {
        DataTypeValidation.assertFloat(A,B);




        Pointer aCPointer = getPointer(A);
        Pointer bCPointer = getPointer(B);



        JCublas.cublasCaxpy(
                A.length(),
                jcuda.cuComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                aCPointer,
                1,
                bCPointer,
                1
        );


    }

    /**
     *
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexDouble da, IComplexNDArray A, IComplexNDArray B) {
        DataTypeValidation.assertDouble(A,B);




        Pointer aCPointer = getPointer(A);
        Pointer bCPointer = getPointer(B);



        JCublas.cublasZaxpy(
                A.length(),
                jcuda.cuDoubleComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                aCPointer,
                1,
                bCPointer,
                1
        );


    }


    /**
     * Multiply the given ndarray
     *  by alpha
     * @param alpha
     * @param x
     * @return
     */
    public static INDArray scal(double alpha, INDArray x) {
        DataTypeValidation.assertDouble(x);



        Pointer xCPointer = getPointer(x);
        JCublas.cublasDscal(
                x.length(),
                alpha,
                xCPointer,
                1);

        return x;

    }

    /**
     * Multiply the given ndarray
     *  by alpha
     * @param alpha
     * @param x
     * @return
     */
    public static INDArray scal(float alpha, INDArray x) {

        DataTypeValidation.assertFloat(x);


        Pointer xCPointer = getPointer(x);
        JCublas.cublasSscal(
                x.length(),
                alpha,
                xCPointer,
                1);

        return x;

    }

    /**
     * Copy x to y
     * @param x
     * @param y
     */
    public static void copy(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x,y);


        Pointer xCPointer = getPointer(x);
        Pointer yCPointer = getPointer(y);
        if(x.data().dataType() == DataBuffer.DOUBLE) {
            JCublas.cublasDcopy(
                    x.length(),
                    xCPointer,
                    x.secondaryStride(),
                    yCPointer,
                    y.secondaryStride());


        }
        else {
            JCublas.cublasScopy(
                    x.length(),
                    xCPointer,
                    x.secondaryStride(),
                    yCPointer,
                    y.secondaryStride());


        }




    }

    /**
     * Dot product between 2 ndarrays
     * @param x
     * @param y
     * @return
     */
    public static double dot(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x,y);



        Pointer xCPointer = getPointer(x);
        Pointer yCPointer = getPointer(y);

        if(x.data().dataType() == (DataBuffer.FLOAT)) {
            float ret =  JCublas.cublasSdot(
                    x.length(),
                    xCPointer,
                    1
                    ,yCPointer,
                    1);

            return ret;
        }


        else {
            double ret =  JCublas.cublasDdot(
                    x.length(),
                    xCPointer,
                    1
                    ,yCPointer,
                    1);

            return ret;
        }

    }


    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x,y);



        Pointer aCPointer = getPointer(x);
        Pointer bCPointer = getPointer(y);


        jcuda.cuDoubleComplex dott = JCublas.cublasZdotc(
                x.length(),
                aCPointer,
                1,
                bCPointer,
                1);

        IComplexDouble ret =   Nd4j.createDouble(dott.x, dott.y);
        return ret;
    }


    public static INDArray ger(INDArray A, INDArray B, INDArray C, double alpha) {
        DataTypeValidation.assertDouble(A,B,C);

        // = alpha * A * transpose(B) + C
        Pointer aCPointer = getPointer(A);
        Pointer bCPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


        JCublas.cublasDger(
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


        return C;
    }



    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) {
        DataTypeValidation.assertFloat(A,B,C);

        // = alpha * A * transpose(B) + C

        Pointer aCPointer = getPointer(A);
        Pointer bCPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


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


        return C;
    }



    /**
     * Complex multiplication of an ndarray
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        DataTypeValidation.assertFloat(x);


        Pointer xCPointer = getPointer(x);

        JCublas.cublasCscal(
                x.length(),
                jcuda.cuComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent()),
                xCPointer,
                1
        );



        return x;
    }

    /**
     * Complex multiplication of an ndarray
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        DataTypeValidation.assertDouble(x);


        Pointer xCPointer = getPointer(x);

        JCublas.cublasZscal(
                x.length(),
                jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent()),
                xCPointer,
                1
        );



        return x;
    }

    /**
     * Complex dot product
     * @param x
     * @param y
     * @return
     */
    public static IComplexDouble dotu(IComplexNDArray x, IComplexNDArray y) {

        DataTypeValidation.assertSameDataType(x,y);

        Pointer xCPointer = getPointer(x);
        Pointer yCPointer = getPointer(y);
        IComplexDouble ret = null;
        if(x.data().dataType() == DataBuffer.DOUBLE) {
            jcuda.cuDoubleComplex dott = JCublas.cublasZdotu(x.length(), xCPointer,1, yCPointer, 1);
            ret = Nd4j.createDouble(dott.x, dott.y);
        }

        else {
            jcuda.cuComplex dott = JCublas.cublasCdotu(x.length(), xCPointer, 1, yCPointer, 1);
            ret = Nd4j.createDouble(dott.x, dott.y);
        }

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

        DataTypeValidation.assertDouble(A,B,C);


        Pointer aCPointer = getPointer(A);
        Pointer bCPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);

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
                                       IComplexFloat Alpha) {
        DataTypeValidation.assertFloat(A,B,C);
        // = alpha * A * tranpose(B) + C

        Pointer aCPointer = getPointer(A);
        Pointer bCPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


        cuComplex alpha = cuComplex.cuCmplx(Alpha.realComponent(),Alpha.imaginaryComponent());


        JCublas.cublasCgerc(
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

        return C;
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
                                       IComplexNDArray C, IComplexFloat Alpha) {

        DataTypeValidation.assertFloat(A,B,C);
        // = alpha * A * tranpose(B) + C

        Pointer aCPointer = getPointer(A);
        Pointer bCPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);

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

        DataTypeValidation.assertDouble(A,B,C);
        // = alpha * A * tranpose(B) + C


        Pointer aCPointer = getPointer(A);
        Pointer bCPointer = getPointer(B);
        Pointer cCPointer = getPointer(C);


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


        return C;
    }

    /**
     * Simpler version of saxpy
     * taking in to account the parameters of the ndarray
     * @param alpha the alpha to scale by
     * @param x the x
     * @param y the y
     */
    public static void axpy(double alpha, INDArray x, INDArray y) {
        DataTypeValidation.assertDouble(x,y);


        Pointer xCPointer = getPointer(x);
        Pointer yCPointer = getPointer(y);

        JCublas.cublasDaxpy(x.length(), alpha, xCPointer, 1, yCPointer, 1);



    }
    /**
     * Simpler version of saxpy
     * taking in to account the parameters of the ndarray
     * @param alpha the alpha to scale by
     * @param x the x
     * @param y the y
     */
    public static void saxpy(float alpha, INDArray x, INDArray y) {
        DataTypeValidation.assertFloat(x,y);

        Pointer xCPointer = getPointer(x);
        Pointer yCPointer = getPointer(y);

        JCublas.cublasSaxpy(x.length(),alpha,xCPointer,1,yCPointer,1);



    }
}
