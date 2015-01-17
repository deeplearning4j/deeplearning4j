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
    private static Logger log = LoggerFactory.getLogger(SimpleJCublas.class);
    static {
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

        File libs = new File(libDir());
        log.info("Loading cuda from " + libs.getAbsolutePath());
        Iterator<File> iter = FileUtils.iterateFiles(libs,null,false);
        while(iter.hasNext()) {
            File next = iter.next();
            File dst = new File(cuBlastmp,next.getName());
            try {
                FileUtils.copyFile(next,dst);
            } catch (IOException e) {
                e.printStackTrace();
            }
            dst.deleteOnExit();
        }





        JCublas.setLogLevel(LogLevel.LOG_DEBUG);
        JCublas.setExceptionsEnabled(true);
        JCublas.cublasInit();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run() {
                JCublas.cublasShutdown();
            }
        });
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


    public static void free(Pointer...pointers) {
        for(Pointer arr : pointers)
            JCublas.cublasFree(arr);
    }


    private static int size(INDArray arr) {
        if(arr.data().dataType() == DataBuffer.FLOAT)
            return Sizeof.FLOAT;
        return Sizeof.DOUBLE;
    }


    /**
     * Retrieve the data from the gpu
     * @param arr the array to getFloat the data for
     * @param from the origin pointer
     * @param to the to pointer (DO NOT AUGMENT WITH OFFSET: THIS IS TAKEN CARE OF FOR YOU)
     */
    public static void getData(JCublasNDArray arr,Pointer from,Pointer to) {

        //p is typically the data vector which is strided access
        if(arr.length() == arr.data().length())
            JCublas.cublasGetVector(
                    arr.length(),
                    size(arr),
                    from,
                    1,
                    to.withByteOffset(arr.offset() * size(arr)),
                    1);
        else
            JCublas.cublasGetVector(
                    arr.length(),
                    size(arr),
                    from,
                    1,
                    to.withByteOffset(arr.offset() * size(arr)),
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
                size(ndarray)
                , ret);

        /* Copy from data to pointer at majorStride() (you want to stride through the data properly) incrementing by 1 for the pointer on the GPU.
        * This allows us to copy only what we need. */
        Pointer toData =null;
        if(ndarray.data().dataType() == DataBuffer.FLOAT)
            toData = Pointer.to(ndarray.data().asFloat()).withByteOffset(ndarray.offset() * size(ndarray));
        else
            toData =  Pointer.to(ndarray.data().asDouble()).withByteOffset(ndarray.offset() * size(ndarray));

        if(ndarray.length() == ndarray.data().length())
            JCublas.cublasSetVector(
                    ndarray.length() * 2,
                    size(ndarray),
                    toData,
                    1,
                    ret,
                    1);
        else
            JCublas.cublasSetVector(
                    ndarray.length() * 2,
                    size(ndarray),
                    toData,
                    1,
                    ret,
                    1);

        return ret;

    }


    /**
     * Retrieve the data from the gpu
     * @param arr the array to getFloat the data for
     * @param from the origin pointer
     * @param to the to pointer (DO NOT AUGMENT WITH OFFSET: THIS IS TAKEN CARE OF FOR YOU)
     */
    public static void getData(JCublasComplexNDArray arr,Pointer from,Pointer to) {
        //p is typically the data vector which is strided access
        if(arr.length() == arr.data().length())
            JCublas.cublasGetVector(
                    arr.length() * 2,
                    size(arr),
                    from,
                    1,
                    to.withByteOffset(arr.offset() * size(arr)),
                    1);
        else
            JCublas.cublasGetVector(
                    arr.length() * 2,
                    Sizeof.FLOAT,
                    from,
                    1,
                    to.withByteOffset(arr.offset() * size(arr)),
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

        Pointer toData =null;
        if(ndarray.data().dataType() == DataBuffer.FLOAT)
            toData = Pointer.to(ndarray.data().asFloat()).withByteOffset(ndarray.offset() * size(ndarray));
        else
            toData =  Pointer.to(ndarray.data().asDouble()).withByteOffset(ndarray.offset() * size(ndarray));


        JCublas.cublasAlloc(
                ndarray.length(),
                size(ndarray)
                , ret);

        /* Copy from data to pointer at majorStride() (you want to stride through the data properly) incrementing by 1 for the pointer on the GPU.
        * This allows us to copy only what we need. */

        if(ndarray.length() == ndarray.data().length())
            JCublas.cublasSetVector(
                    ndarray.length(),
                    size(ndarray),
                    toData,
                    1,
                    ret,
                    1);
        else
            JCublas.cublasSetVector(
                    ndarray.length(),
                    size(ndarray),
                    toData,
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
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, double alpha, double beta) {

        DataTypeValidation.assertDouble(A,B,C);
        JCublas.cublasInit();

        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        Pointer cAPointer = alloc(cA);
        Pointer cBPointer = alloc(cB);
        Pointer cCPointer = alloc(cC);


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

        getData(cC,cCPointer,Pointer.to(cC.data().asDouble()));
        free(cAPointer,cBPointer,cCPointer);


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

        getData(cC,cCPointer,Pointer.to(cC.data().asFloat()));
        free(cAPointer,cBPointer,cCPointer);


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

        JCublas.cublasInit();

        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;
        JCublasComplexNDArray cB = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;

        Pointer cAPointer = alloc(cA);
        Pointer cBPointer = alloc(cB);
        Pointer cCPointer = alloc(cC);


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


        getData(cC,cCPointer,Pointer.to(cC.data().asDouble()));
        free(cAPointer,cBPointer,cCPointer);

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
        JCublas.cublasInit();

        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;
        JCublasComplexNDArray cB = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;

        Pointer cAPointer = alloc(cA);
        Pointer cBPointer = alloc(cB);
        Pointer cCPointer = alloc(cC);


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
                cB.secondaryStride(), // ldb
                beta,  // beta
                cCPointer, // y
                cC.secondaryStride()); // ldc


        getData(cC,cCPointer,Pointer.to(cC.data().asFloat()));
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
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexDouble a,IComplexNDArray C
            , IComplexDouble b) {
        DataTypeValidation.assertSameDataType(A,B,C);

        JCublas.cublasInit();

        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;
        JCublasComplexNDArray cB = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;

        Pointer cAPointer = alloc(cA);
        Pointer cBPointer = alloc(cB);
        Pointer cCPointer = alloc(cC);


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(a.realComponent().doubleValue(),b.imaginaryComponent().doubleValue());
        cuDoubleComplex beta = cuDoubleComplex.cuCmplx(b.realComponent().doubleValue(),b.imaginaryComponent().doubleValue());

        JCublas.cublasZgemm(
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


        getData(cC,cCPointer,Pointer.to(cC.data().asDouble()));
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
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexFloat a,IComplexNDArray C
            , IComplexFloat b) {
        DataTypeValidation.assertFloat(A,B,C);
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


        getData(cC,cCPointer,Pointer.to(cC.data().asFloat()));
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
                                double alpha, double beta) {

        DataTypeValidation.assertDouble(A,B,C);
        JCublas.cublasInit();

        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        Pointer cAPointer = alloc(cA);
        Pointer cBPointer = alloc(cB);
        Pointer cCPointer = alloc(cC);



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

        getData(cC,cCPointer,Pointer.to(cC.data().asDouble()));

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
        DataTypeValidation.assertFloat(A,B,C);

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

        getData(cC,cCPointer,Pointer.to(cC.data().asFloat()));

        free(cAPointer,cBPointer,cCPointer);

        return C;

    }


    /**
     * Calculate the 2 norm of the ndarray
     * @param A
     * @return
     */
    public static double nrm2(IComplexNDArray A) {
        JCublas.cublasInit();

        JCublasComplexNDArray cA = (JCublasComplexNDArray) A;
        Pointer cAPointer = alloc(cA);
        if(A.data().dataType() == DataBuffer.FLOAT) {
            float s = JCublas.cublasSnrm2(A.length(), cAPointer, 2);
            free(cAPointer);
            return s;
        }
        else {
            double s = JCublas.cublasDnrm2(A.length(), cAPointer, 2);
            free(cAPointer);
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
        JCublas.cublasInit();

        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray yC = (JCublasComplexNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        if(xC.data().dataType() == DataBuffer.FLOAT) {
            JCublas.cublasScopy(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);


            getData(yC,yCPointer,Pointer.to(yC.data().asFloat()));

        }

        else {
            JCublas.cublasDcopy(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);


            getData(yC,yCPointer,Pointer.to(yC.data().asDouble()));

        }


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
        if(xC.data().dataType() == DataBuffer.FLOAT) {
            int max = JCublas.cublasIsamax(x.length(), xCPointer, 1);
            free(xCPointer);
            return max;
        }

        else {
            int max = JCublas.cublasIzamax(x.length(), xCPointer, 1);
            free(xCPointer);
            return max;
        }

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

        DataTypeValidation.assertSameDataType(x,y);
        JCublas.cublasInit();

        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;
        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);


        if(xC.data().dataType() == DataBuffer.FLOAT) {
            JCublas.cublasSswap(
                    xC.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);

            getData(yC,yCPointer,Pointer.to(yC.data().asFloat()));
        }

        else {
            JCublas.cublasDswap(
                    xC.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);

            getData(yC,yCPointer,Pointer.to(yC.data().asDouble()));
        }

        free(xCPointer,yCPointer);

    }

    /**
     *
     * @param x
     * @return
     */
    public static double asum(INDArray x) {
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        Pointer xCPointer = alloc(xC);
        if(x.data().dataType() == DataBuffer.FLOAT) {
            float sum = JCublas.cublasSasum(x.length(), xCPointer,1);
            free(xCPointer);
            return sum;
        }
        else {
            double sum = JCublas.cublasDasum(x.length(),xCPointer,1);
            free(xCPointer);
            return sum;
        }

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
        DataTypeValidation.assertFloat(A,B);
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
            getData(xB,xBPointer,Pointer.to(xB.data().asFloat()));
        }
        else {
            JCublas.cublasSaxpy(
                    xA.length(),
                    da,
                    xAPointer,
                    1,
                    xBPointer,
                    1);
            getData(xB, xBPointer, Pointer.to(xB.data().asFloat()));

        }


        free(xAPointer,xBPointer);

    }

    /**
     *
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexFloat da, IComplexNDArray A, IComplexNDArray B) {
        DataTypeValidation.assertFloat(A,B);

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

        getData(bC,bCPointer,Pointer.to(bC.data().asFloat()));

        free(aCPointer,bCPointer);
    }

    /**
     *
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexDouble da, IComplexNDArray A, IComplexNDArray B) {
        DataTypeValidation.assertDouble(A,B);

        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;

        JCublas.cublasInit();

        Pointer aCPointer = alloc(aC);
        Pointer bCPointer = alloc(bC);



        JCublas.cublasZaxpy(
                aC.length(),
                jcuda.cuDoubleComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                aCPointer,
                1,
                bCPointer,
                1
        );

        getData(bC,bCPointer,Pointer.to(bC.data().asDouble()));

        free(aCPointer,bCPointer);
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

        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;

        Pointer xCPointer = alloc(xC);
        JCublas.cublasDscal(
                xC.length(),
                alpha,
                xCPointer,
                1);
        getData(xC, xCPointer, Pointer.to(xC.data().asDouble()));
        free(xCPointer);
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
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;

        Pointer xCPointer = alloc(xC);
        JCublas.cublasSscal(
                xC.length(),
                alpha,
                xCPointer,
                1);
        getData(xC, xCPointer, Pointer.to(xC.data().asFloat()));
        free(xCPointer);
        return x;

    }

    /**
     * Copy x to y
     * @param x
     * @param y
     */
    public static void copy(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x,y);

        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);
        if(x.data().dataType() == DataBuffer.DOUBLE) {
            JCublas.cublasDcopy(x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);


            getData(yC, yCPointer, Pointer.to(yC.data().asDouble()));
        }
        else {
            JCublas.cublasScopy(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);


            getData(yC, yCPointer, Pointer.to(yC.data().asFloat()));
        }


        free(xCPointer,yCPointer);


    }

    /**
     * Dot product between 2 ndarrays
     * @param x
     * @param y
     * @return
     */
    public static double dot(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x,y);
        JCublas.cublasInit();

        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        if(x.data().dataType() == (DataBuffer.FLOAT)) {
            float ret =  JCublas.cublasSdot(
                    x.length(),
                    xCPointer,
                    1
                    ,yCPointer,
                    1);

            free(xCPointer,yCPointer);
            return ret;
        }


        else {
            double ret =  JCublas.cublasDdot(
                    x.length(),
                    xCPointer,
                    1
                    ,yCPointer,
                    1);

            free(xCPointer,yCPointer);
            return ret;
        }

    }


    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x,y);
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


    public static INDArray ger(INDArray A, INDArray B, INDArray C, double alpha) {
        DataTypeValidation.assertDouble(A,B,C);
        JCublas.cublasInit();
        // = alpha * A * transpose(B) + C
        JCublasNDArray aC = (JCublasNDArray) A;
        JCublasNDArray bC = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        Pointer aCPointer = alloc(aC);
        Pointer bCPointer = alloc(bC);
        Pointer cCPointer = alloc(cC);


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

        getData(cC,cCPointer,Pointer.to(cC.data().asDouble()));
        free(aCPointer, bCPointer, cCPointer);

        return C;
    }



    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) {
        DataTypeValidation.assertFloat(A,B,C);
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

        getData(cC,cCPointer,Pointer.to(cC.data().asFloat()));
        free(aCPointer, bCPointer, cCPointer);

        return C;
    }



    /**
     * Complex multiplication of an ndarray
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        DataTypeValidation.assertFloat(x);
        JCublas.cublasInit();

        Pointer xCPointer = alloc(xC);

        JCublas.cublasCscal(
                x.length(),
                jcuda.cuComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent()),
                xCPointer,
                1
        );


        getData(xC,xCPointer,Pointer.to(xC.data().asFloat()));

        free(xCPointer);

        return x;
    }

    /**
     * Complex multiplication of an ndarray
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        DataTypeValidation.assertDouble(x);
        JCublas.cublasInit();

        Pointer xCPointer = alloc(xC);

        JCublas.cublasZscal(
                x.length(),
                jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent()),
                xCPointer,
                1
        );


        getData(xC,xCPointer,Pointer.to(xC.data().asDouble()));

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

        DataTypeValidation.assertSameDataType(x,y);
        JCublasComplexNDArray xC = (JCublasComplexNDArray) x;
        JCublasComplexNDArray yC = (JCublasComplexNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);
        IComplexDouble ret = null;
        if(x.data().dataType() == DataBuffer.DOUBLE) {
            jcuda.cuDoubleComplex dott = JCublas.cublasZdotu(x.length(), xCPointer,1, yCPointer, 1);
            ret = Nd4j.createDouble(dott.x, dott.y);
        }

        else {
            jcuda.cuComplex dott = JCublas.cublasCdotu(x.length(), xCPointer, 1, yCPointer, 1);
            ret = Nd4j.createDouble(dott.x, dott.y);
        }

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

        DataTypeValidation.assertDouble(A,B,C);

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


        getData(cC,cCPointer,Pointer.to(cC.data().asDouble()));

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
                                       IComplexFloat Alpha) {
        DataTypeValidation.assertFloat(A,B,C);
        // = alpha * A * tranpose(B) + C
        JCublasComplexNDArray aC = (JCublasComplexNDArray) A;
        JCublasComplexNDArray bC = (JCublasComplexNDArray) B;
        JCublasComplexNDArray cC = (JCublasComplexNDArray) C;


        Pointer aCPointer = alloc(aC);
        Pointer bCPointer = alloc(bC);
        Pointer cCPointer = alloc(cC);


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


        getData(cC,cCPointer,Pointer.to(cC.data().asFloat()));

        free(aCPointer,bCPointer,cCPointer);
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


        getData(cC,cCPointer,Pointer.to(cC.data().asFloat()));

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

        DataTypeValidation.assertDouble(A,B,C);
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


        getData(cC,cCPointer,Pointer.to(cC.data().asDouble()));

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
    public static void axpy(double alpha, INDArray x, INDArray y) {
        DataTypeValidation.assertDouble(x,y);
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        JCublas.cublasDaxpy(x.length(), alpha, xCPointer, 1, yCPointer, 1);


        getData(yC,yCPointer,Pointer.to(yC.data().asDouble()));
        free(xCPointer,yCPointer);

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
        JCublas.cublasInit();
        JCublasNDArray xC = (JCublasNDArray) x;
        JCublasNDArray yC = (JCublasNDArray) y;

        Pointer xCPointer = alloc(xC);
        Pointer yCPointer = alloc(yC);

        JCublas.cublasSaxpy(x.length(),alpha,xCPointer,1,yCPointer,1);


        getData(yC,yCPointer,Pointer.to(yC.data().asFloat()));
        free(xCPointer,yCPointer);

    }
}
