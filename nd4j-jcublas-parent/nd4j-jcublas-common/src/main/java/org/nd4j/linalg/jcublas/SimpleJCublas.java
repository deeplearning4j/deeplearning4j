/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jcublas;

import java.nio.ByteBuffer;

import jcublas.cublasHandle;
import jcuda.CudaException;
import jcuda.LogLevel;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.utils.KernelLauncher;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.CopyOp;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.ops.executioner.JCudaExecutioner;

/**
 * Simple abstraction for jcublas operations
 *
 * @author mjk
 * @author Adam Gibson
 */
public class SimpleJCublas {

    private static boolean init = false;
    private static cublasHandle handle = new cublasHandle();


    static {
        init();
    }


    public static void assertCudaBuffer(INDArray... buffer) {
        for (INDArray b1 : buffer)
            if (!(b1.data() instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }


    public static void assertCudaBuffer(DataBuffer... buffer) {
        for (DataBuffer b1 : buffer)
            if (!(b1 instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }
    
    /**
     * The cublas handle
     *
     * @return the handle used for cublas
     */
    public static cublasHandle handle() {
        return handle;
    }
    
    static int checkResult(int result)
    {
        if (result != cudaError.cudaSuccess)
        {
            throw new CudaException(cudaError.stringFor(result));
        }
        return result;
    }


    /**
     * Initialize jcublas only called once
     */
    public static void init() {
        if (init)
            return;
//        JCublas2.initialize();
//        cublasHandle handle = new cublasHandle();
//        JCublas2.cublasCreate(handle);

        JCublas.setLogLevel(LogLevel.LOG_DEBUG);
        JCublas.setExceptionsEnabled(true);

        try {
            KernelFunctionLoader.getInstance().load();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        
        // Check if the device supports mapped host memory
        cudaDeviceProp deviceProperties = new cudaDeviceProp();
        checkResult(JCuda.cudaGetDeviceProperties(deviceProperties, 0));
        if (deviceProperties.canMapHostMemory == 0) {
            System.err.println("This device can not map host memory");
            System.err.println(deviceProperties.toFormattedString());
            return;
        }
        int[] version = new int[1];
        JCudaDriver.cuCtxGetApiVersion(ContextHolder.getInstance().getContext(), version);
        
        
        // Set the flag indicating that mapped memory will be used
        //checkResult(JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceMapHost));
        
        init = true;
    }


    public static void sync() {
    	JCuda.cudaDeviceSynchronize();
        KernelLauncher.setContext();
    }

    /**
     * General matrix vector multiplication
     *
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, double alpha, double beta) {

        DataTypeValidation.assertDouble(A, B, C);
        assertCudaBuffer(A.data(), B.data(), C.data());
        sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);

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
        
        cCPointer.copyToHost();
        releaseCublasPointers(cAPointer,cBPointer,cCPointer);
        
        sync();
        return C;
    }

    /**
     * G)eneral matrix vector multiplication
     *
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) {

        DataTypeValidation.assertFloat(A, B, C);
        sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);



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

        sync();
        
        cCPointer.copyToHost();
        releaseCublasPointers(cCPointer,cAPointer,cBPointer);

        return C;
    }


    /**
     * General matrix vector
     *
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemv(IComplexNDArray A, IComplexNDArray B, IComplexDouble a, IComplexNDArray C
            , IComplexDouble b) {
        DataTypeValidation.assertSameDataType(A, B, C);
        sync();


        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(a.realComponent().doubleValue(), b.imaginaryComponent().doubleValue());
        cuDoubleComplex beta = cuDoubleComplex.cuCmplx(b.realComponent().doubleValue(), b.imaginaryComponent().doubleValue());

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

        sync();

        return C;

    }

    /**
     * General matrix vector
     *
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemv(IComplexNDArray A, IComplexNDArray B, IComplexFloat a, IComplexNDArray C
            , IComplexFloat b) {
        DataTypeValidation.assertFloat(A, B, C);
        assertCudaBuffer(A, B, C);
        sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


        cuComplex alpha = cuComplex.cuCmplx(a.realComponent().floatValue(), b.imaginaryComponent().floatValue());
        cuComplex beta = cuComplex.cuCmplx(b.realComponent().floatValue(), b.imaginaryComponent().floatValue());

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

        sync();

        return C;

    }


    /**
     * General matrix multiply
     *
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexDouble a, IComplexNDArray C
            , IComplexDouble b) {
        DataTypeValidation.assertSameDataType(A, B, C);
        sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);



        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(a.realComponent().doubleValue(), b.imaginaryComponent().doubleValue());
        cuDoubleComplex beta = cuDoubleComplex.cuCmplx(b.realComponent().doubleValue(), b.imaginaryComponent().doubleValue());

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

        sync();

        return C;

    }

    /**
     * General matrix multiply
     *
     * @param A
     * @param B
     * @param a
     * @param C
     * @param b
     * @return
     */
    public static IComplexNDArray gemm(IComplexNDArray A, IComplexNDArray B, IComplexFloat a, IComplexNDArray C
            , IComplexFloat b) {
        DataTypeValidation.assertFloat(A, B, C);

        sync();

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


        cuComplex alpha = cuComplex.cuCmplx(a.realComponent().floatValue(), b.imaginaryComponent().floatValue());
        cuComplex beta = cuComplex.cuCmplx(b.realComponent().floatValue(), b.imaginaryComponent().floatValue());

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

        sync();

        return C;

    }

    /**
     * General matrix multiply
     *
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemm(INDArray A, INDArray B, INDArray C,
                                double alpha, double beta) {

        DataTypeValidation.assertDouble(A, B, C);

        sync();

        JCublasNDArray cA = (JCublasNDArray) A;
        JCublasNDArray cB = (JCublasNDArray) B;
        JCublasNDArray cC = (JCublasNDArray) C;

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


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

        sync();
        
        cCPointer.copyToHost();
        releaseCublasPointers(cAPointer, cBPointer, cCPointer);

        return C;

    }

    /**
     * General matrix multiply
     *
     * @param A
     * @param B
     * @param C
     * @param alpha
     * @param beta
     * @return
     */
    public static INDArray gemm(INDArray A, INDArray B, INDArray C,
                                float alpha, float beta) {
        DataTypeValidation.assertFloat(A, B, C);
        sync();


        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);
        
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
        sync();

        return C;

    }


    /**
     * Calculate the 2 norm of the ndarray.
     * Note that this is a standin for
     * no complex ndarray. It will treat this as a normal ndarray
     * with a stride of 2.
     *
     * @param A the ndarray to calculate the norm2 of
     * @return the ndarray to calculate the norm2 of
     */
    public static double nrm2(IComplexNDArray A) {

        sync();

        CublasPointer cAPointer = new CublasPointer(A);
        if (A.data().dataType() == DataBuffer.FLOAT) {
            float s = JCublas.cublasSnrm2(A.length(), cAPointer, 2);
            return s;
        } else {
            double s = JCublas.cublasDnrm2(A.length(), cAPointer, 2);
            return s;
        }

    }

    /**
     * Copy x to y
     *
     * @param x the origin
     * @param y the destination
     */
    public static void copy(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);

        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        CublasPointer yCPointer = new CublasPointer(y);


        JCudaBuffer buff = (JCudaBuffer) x.data();
        if (x.majorStride() == 2 && y.majorStride() == 2)
            JCuda.cudaMemcpy(
                    yCPointer
                    , xCPointer
                    , x.length() * buff.getElementSize() * 2
                    , cudaMemcpyKind.cudaMemcpyDeviceToDevice);
        else
            Nd4j.getExecutioner().exec(new CopyOp(x, y, y, x.length()));
        sync();


    }


    /**
     * Return the index of the max in the given ndarray
     *
     * @param x the ndarray to ge the max for
     * @return the max index of the given ndarray
     */
    public static int iamax(IComplexNDArray x) {
        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        if (x.data().dataType() == DataBuffer.FLOAT) {
            int max = JCublas.cublasIsamax(x.length(), xCPointer, 1);
            return max;
        } else {
            int max = JCublas.cublasIzamax(x.length(), xCPointer, 1);
            return max;
        }

    }

    /**
     * @param x
     * @return
     */
    public static float asum(IComplexNDArray x) {
    	CublasPointer xCPointer = new CublasPointer(x);
        float sum = JCublas.cublasScasum(x.length(), xCPointer, 1);
        return sum;
    }


    /**
     * Swap the elements in each ndarray
     *
     * @param x
     * @param y
     */
    public static void swap(INDArray x, INDArray y) {


        DataTypeValidation.assertSameDataType(x, y);

        CublasPointer xCPointer = new CublasPointer(x);
        CublasPointer yCPointer = new CublasPointer(y);
        sync();

        if (x.data().dataType() == DataBuffer.FLOAT) {
            JCublas.cublasSswap(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);

        } else {
            JCublas.cublasDswap(
                    x.length(),
                    xCPointer,
                    1,
                    yCPointer,
                    1);

        }
        sync();


    }

    /**
     * @param x
     * @return
     */
    public static double asum(INDArray x) {


    	CublasPointer xCPointer = new CublasPointer(x);
        if (x.data().dataType() == DataBuffer.FLOAT) {
            float sum = JCublas.cublasSasum(x.length(), xCPointer, 1);
            return sum;
        } else {
            double sum = JCublas.cublasDasum(x.length(), xCPointer, 1);
            return sum;
        }

    }

    /**
     * Returns the norm2 of the given ndarray
     *
     * @param x
     * @return
     */
    public static double nrm2(INDArray x) {


        if (x.data().dataType() == DataBuffer.FLOAT) {
        	CublasPointer xCPointer = new CublasPointer(x);


            float normal2 = JCublas.cublasSnrm2(x.length(), xCPointer, 1);
            return normal2;
        } else if (x.data().dataType() == DataBuffer.DOUBLE) {
        	CublasPointer xCPointer = new CublasPointer(x);
            double normal2 = JCublas.cublasDnrm2(x.length(), xCPointer, 1);
            return normal2;
        }
        throw new IllegalStateException("Illegal data type on array ");


    }

    /**
     * Returns the index of the max element
     * in the given ndarray
     *
     * @param x
     * @return
     */
    public static int iamax(INDArray x) {



    	CublasPointer xCPointer = new CublasPointer(x);

        if (x.data().dataType() == DataBuffer.FLOAT) {
            int max = JCublas.cublasIsamax(
                    x.length(),
                    xCPointer,
                    x.majorStride());

            return max - 1;
        } else if (x.data().dataType() == DataBuffer.DOUBLE) {
            int max = JCublas.cublasIdamax(
                    x.length(),
                    xCPointer,
                    x.majorStride());

            return max - 1;
        }

        throw new IllegalStateException("Illegal data type on array ");
    }


    /**
     * And and scale by the given scalar da
     *
     * @param da alpha
     * @param A  the element to add
     * @param B  the matrix to add to
     */
    public static void axpy(float da, INDArray A, INDArray B) {
        DataTypeValidation.assertFloat(A, B);
        CublasPointer xAPointer = new CublasPointer(A);
        CublasPointer xBPointer = new CublasPointer(B);
        sync();
        JCublas.cublasSaxpy(
                A.length(),
                da,
                xAPointer,
                A.majorStride(),
                xBPointer,
                B.majorStride());
        
        ((JCudaBuffer)A.data()).copyToHost();
        sync();


    }

    /**
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexFloat da, IComplexNDArray A, IComplexNDArray B) {
        DataTypeValidation.assertFloat(A, B);



        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(B);
        sync();

        JCublas.cublasCaxpy(
                A.length(),
                jcuda.cuComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                aCPointer,
                1,
                bCPointer,
                1
        );
        sync();


    }

    /**
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexDouble da, IComplexNDArray A, IComplexNDArray B) {
        DataTypeValidation.assertDouble(A, B);



        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(B);
        sync();

        JCublas.cublasZaxpy(
                A.length(),
                jcuda.cuDoubleComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue()),
                aCPointer,
                A.majorStride(),
                bCPointer,
                B.majorStride()
        );
        sync();


    }


    /**
     * Multiply the given ndarray
     * by alpha
     *
     * @param alpha
     * @param x
     * @return
     */
    public static INDArray scal(double alpha, INDArray x) {
        DataTypeValidation.assertDouble(x);

        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        JCublas.cublasDscal(
                x.length(),
                alpha,
                xCPointer,
                x.majorStride());
        sync();
        
        xCPointer.copyToHost();
        releaseCublasPointers(xCPointer);

        return x;

    }

    /**
     * Multiply the given ndarray
     * by alpha
     *
     * @param alpha
     * @param x
     * @return
     */
    public static INDArray scal(float alpha, INDArray x) {

        DataTypeValidation.assertFloat(x);
        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        JCublas.cublasSscal(
                x.length(),
                alpha,
                xCPointer,
                x.majorStride());
        sync();
        
        xCPointer.copyToHost();
        releaseCublasPointers(xCPointer);

        return x;

    }

    /**
     * Copy x to y
     *
     * @param x the src
     * @param y the destination
     */
    public static void copy(INDArray x, INDArray y) {
    	sync();
        DataTypeValidation.assertSameDataType(x, y);
        
        CublasPointer xCPointer = new CublasPointer(x);
        CublasPointer yCPointer = new CublasPointer(y);
        
        if(x.data().dataType() == DataBuffer.DOUBLE)
            JCublas.cublasDcopy(x.length(),xCPointer,x.majorStride(),yCPointer,y.majorStride());
        if(x.data().dataType() == DataBuffer.FLOAT)
            JCublas.cublasScopy(x.length(),xCPointer,x.majorStride(),yCPointer,y.majorStride());
        sync();
        
        yCPointer.copyToHost();
        releaseCublasPointers(yCPointer, xCPointer);
    }

    /**
     * Dot product between 2 ndarrays
     *
     * @param x the first ndarray
     * @param y the second ndarray
     * @return the dot product between the two ndarrays
     */
    public static double dot(INDArray x, INDArray y) {
        DataTypeValidation.assertSameDataType(x, y);

        sync();
        CublasPointer xCPointer = new CublasPointer(x);
        CublasPointer yCPointer = new CublasPointer(y);

        
        if (x.data().dataType() == (DataBuffer.FLOAT)) {
            float ret = JCublas.cublasSdot(
                    x.length(),
                    xCPointer,
                    x.majorStride()
                    , yCPointer,
                    y.majorStride());
            sync();
            
            releaseCublasPointers(xCPointer,yCPointer);

            return ret;
        } else {
            double ret = JCublas.cublasDdot(
                    x.length(),
                    xCPointer,
                    y.majorStride()
                    , yCPointer,
                    y.majorStride());
            sync();
            
            releaseCublasPointers(xCPointer,yCPointer);

            return ret;
        }
        
    	
        
        

    }


	private static void releaseCublasPointers(CublasPointer... pointers) {
		for(CublasPointer pointer : pointers)
			try { pointer.close(); } catch(Exception e) { throw new RuntimeException("Could not run cublas command", e); }
	}


    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);

        sync();

        CublasPointer aCPointer = new CublasPointer(x);
        CublasPointer bCPointer = new CublasPointer(y);


        jcuda.cuDoubleComplex dott = JCublas.cublasZdotc(
                x.length(),
                aCPointer,
                x.majorStride(),
                bCPointer,
                y.majorStride());

        IComplexDouble ret = Nd4j.createDouble(dott.x, dott.y);
        sync();

        releaseCublasPointers(aCPointer, bCPointer);
        return ret;
    }


    public static INDArray ger(INDArray A, INDArray B, INDArray C, double alpha) {
        DataTypeValidation.assertDouble(A, B, C);
        sync();

        // = alpha * A * transpose(B) + C
        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


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
        
        cCPointer.copyToHost();
        releaseCublasPointers(aCPointer,bCPointer,cCPointer);

        sync();

        return C;
    }


    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) {
        DataTypeValidation.assertFloat(A, B, C);

        sync();
        // = alpha * A * transpose(B) + C

        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


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
        sync();

        cCPointer.copyToHost();
        releaseCublasPointers(aCPointer,bCPointer,cCPointer);

        return C;
    }


    /**
     * Complex multiplication of an ndarray
     *
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) {
        DataTypeValidation.assertFloat(x);

        sync();

        CublasPointer xCPointer = new CublasPointer(x);

        JCublas.cublasCscal(
                x.length(),
                jcuda.cuComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent()),
                xCPointer,
                x.majorStride()
        );
        sync();
        
        xCPointer.copyToHost();
        releaseCublasPointers(xCPointer);


        return x;
    }

    /**
     * Complex multiplication of an ndarray
     *
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) {
        DataTypeValidation.assertDouble(x);
        sync();


        CublasPointer xCPointer = new CublasPointer(x);

        JCublas.cublasZscal(
                x.length(),
                jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent()),
                xCPointer,
                x.majorStride()
        );
        sync();

        xCPointer.copyToHost();
        releaseCublasPointers(xCPointer);

        return x;
    }

    /**
     * Complex dot product
     *
     * @param x
     * @param y
     * @return
     */
    public static IComplexDouble dotu(IComplexNDArray x, IComplexNDArray y) {

        DataTypeValidation.assertSameDataType(x, y);
        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        CublasPointer yCPointer = new CublasPointer(y);
        IComplexDouble ret = null;
        if (x.data().dataType() == DataBuffer.DOUBLE) {
            jcuda.cuDoubleComplex dott = JCublas.cublasZdotu(x.length(), xCPointer, x.majorStride(), yCPointer, y.majorStride());
            ret = Nd4j.createDouble(dott.x, dott.y);
        } else {
            jcuda.cuComplex dott = JCublas.cublasCdotu(x.length(), xCPointer, x.majorStride(), yCPointer, y.majorStride());
            ret = Nd4j.createDouble(dott.x, dott.y);
        }
        sync();
        
        releaseCublasPointers(xCPointer, yCPointer);

        return ret;
    }


    /**
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
        sync();
        DataTypeValidation.assertDouble(A, B, C);

        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);

        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(), Alpha.imaginaryComponent());

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
        sync();

        cCPointer.copyToHost();
        releaseCublasPointers(aCPointer,bCPointer,cCPointer);

        return C;
    }

    /**
     * @param A
     * @param B
     * @param C
     * @param Alpha
     * @return
     */
    public static IComplexNDArray gerc(IComplexNDArray A, IComplexNDArray B, IComplexNDArray C,
                                       IComplexFloat Alpha) {
        DataTypeValidation.assertFloat(A, B, C);
        // = alpha * A * tranpose(B) + C

        sync();
        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


        cuComplex alpha = cuComplex.cuCmplx(Alpha.realComponent(), Alpha.imaginaryComponent());


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
        sync();
        
        cCPointer.copyToHost();
        releaseCublasPointers(aCPointer,bCPointer,cCPointer);

        return C;
    }

    /**
     * @param A
     * @param B
     * @param C
     * @param Alpha
     * @return
     */
    public static IComplexNDArray geru(IComplexNDArray A,
                                       IComplexNDArray B,
                                       IComplexNDArray C, IComplexFloat Alpha) {

        DataTypeValidation.assertFloat(A, B, C);
        // = alpha * A * tranpose(B) + C
        sync();

        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);

        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(), Alpha.imaginaryComponent());

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

        sync();
        
        cCPointer.copyToHost();
        releaseCublasPointers(aCPointer,bCPointer,cCPointer);

        return C;
    }

    /**
     * @param A
     * @param B
     * @param C
     * @param Alpha
     * @return
     */
    public static IComplexNDArray gerc(IComplexNDArray A, IComplexNDArray B, IComplexNDArray C,
                                       IComplexDouble Alpha) {

        DataTypeValidation.assertDouble(A, B, C);
        // = alpha * A * tranpose(B) + C

        sync();

        CublasPointer aCPointer = new CublasPointer(A);
        CublasPointer bCPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(Alpha.realComponent(), Alpha.imaginaryComponent());


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

        sync();
        
        cCPointer.copyToHost();
        releaseCublasPointers(aCPointer,bCPointer,cCPointer);

        return C;
    }

    /**
     * Simpler version of saxpy
     * taking in to account the parameters of the ndarray
     *
     * @param alpha the alpha to scale by
     * @param x     the x
     * @param y     the y
     */
    public static void axpy(double alpha, INDArray x, INDArray y) {
        DataTypeValidation.assertDouble(x, y);

        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        CublasPointer yCPointer = new CublasPointer(y);

        JCublas.cublasDaxpy(x.length(), alpha, xCPointer, x.majorStride(), yCPointer, y.majorStride());

        sync();
        
        xCPointer.copyToHost();
        releaseCublasPointers(xCPointer, yCPointer);

    }

    /**
     * Simpler version of saxpy
     * taking in to account the parameters of the ndarray
     *
     * @param alpha the alpha to scale by
     * @param x     the x
     * @param y     the y
     */
    public static void saxpy(float alpha, INDArray x, INDArray y) {
        DataTypeValidation.assertFloat(x, y);
        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        CublasPointer yCPointer = new CublasPointer(y);

        JCublas.cublasSaxpy(x.length(), alpha, xCPointer, x.majorStride(), yCPointer, y.majorStride());
        sync();
        
        xCPointer.copyToHost();
        releaseCublasPointers(xCPointer, yCPointer);


    }
}
