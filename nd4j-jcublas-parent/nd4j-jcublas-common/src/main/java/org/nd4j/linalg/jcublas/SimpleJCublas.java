/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.jcublas;



import jcuda.*;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasOperation;
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
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.util.PointerUtil;

import javax.naming.Context;

/**
 * Simple abstraction for jcublas operations
 *
 * @author mjk
 * @author Adam Gibson
 */
public class SimpleJCublas {

    private static boolean init = false;


    static {
        init();
    }

    /**
     * Assert that the data buffer for each ndarray
     * is a cuda buffer
     * @param buffer the arrays to tests
     */
    public static void assertCudaBuffer(INDArray... buffer) {
        for (INDArray b1 : buffer)
            if (!(b1.data() instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }

    /**
     * Assert that the data buffer for each ndarray
     * is a cuda buffer
     * @param buffer the arrays to tests
     */
    public static void assertCudaBuffer(DataBuffer... buffer) {
        for (DataBuffer b1 : buffer)
            if (!(b1 instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }






    /**
     * Initialize JCublas2. Only called once
     */
    public static void init() {
        if (init)
            return;

        JCublas2.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);

        try {
            KernelFunctionLoader.getInstance().load();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Check if the device supports mapped host memory
        cudaDeviceProp deviceProperties = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(deviceProperties, 0);
        if (deviceProperties.canMapHostMemory == 0) {
            System.err.println("This device can not map host memory");
            System.err.println(deviceProperties.toFormattedString());
            return;
        }


        init = true;
    }


    /**
     * Sync the device
     */
    public static void sync() {
        JCudaDriver.cuCtxSynchronize();
        JCuda.cudaDeviceSynchronize();
        ContextHolder.syncStream();
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

        JCublas2.cublasDgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N,
                A.rows(),
                A.columns(),
                Pointer.to(new double[]{alpha}),
                cAPointer.getDevicePointer(),
                A.rows(),
                cBPointer.getDevicePointer(),
                1,
                Pointer.to(new double[]{beta}),
                cCPointer.getDevicePointer(),
                1);

        cCPointer.copyToHost();
        releaseCublasPointers(cAPointer,cBPointer,cCPointer);

        sync();
        return C;
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
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) {

        DataTypeValidation.assertFloat(A, B, C);

        CublasPointer cAPointer = new CublasPointer(A);
        CublasPointer cBPointer = new CublasPointer(B);
        CublasPointer cCPointer = new CublasPointer(C);

        sync();


        JCublas2.cublasSgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N,
                A.rows(),
                A.columns(),
                Pointer.to(new float[]{alpha}),
                cAPointer.getDevicePointer(),
                A.rows(),
                cBPointer.getDevicePointer(),
                1,
                Pointer.to(new float[]{beta}),
                cCPointer.getDevicePointer(),
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

        JCublas2.cublasZgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                A.rows(),  // m
                A.rows(), // n
                PointerUtil.getPointer(alpha),
                cAPointer.getDevicePointer(), // A
                A.rows(),  // lda
                cBPointer.getDevicePointer(), // x
                1, // ldb
                PointerUtil.getPointer(beta),  // beta
                cCPointer.getDevicePointer(), // ydoin
                1); // ldc

        sync();

        cCPointer.copyToHost();
        releaseCublasPointers(cAPointer,cBPointer,cCPointer);

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

        JCublas2.cublasCgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                A.rows(),  // m
                A.columns(), // n
                PointerUtil.getPointer(alpha),
                cAPointer.getDevicePointer(), // A
                A.rows(),  // lda
                cBPointer.getDevicePointer(), // x
                1, // ldb
                PointerUtil.getPointer(beta),  // beta
                cCPointer.getDevicePointer(), // y
                1); // ldc

        sync();

        cCPointer.copyToHost();
        releaseCublasPointers(cAPointer,cBPointer,cCPointer);

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

        JCublas2.cublasZgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                C.rows(),  // m
                C.columns(), // n
                A.columns(), //k,
                PointerUtil.getPointer(alpha),
                cAPointer.getDevicePointer(), // A
                A.rows(),  // lda
                cBPointer.getDevicePointer(), // x
                B.rows(), // ldb
                PointerUtil.getPointer(beta),  // beta
                cCPointer.getDevicePointer(), // y
                C.rows()); // ldc

        sync();

        cCPointer.copyToHost();
        releaseCublasPointers(cAPointer,cBPointer,cCPointer);

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

        JCublas2.cublasCgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                C.rows(),  // m
                C.columns(), // n
                A.columns(), //k,
                PointerUtil.getPointer(alpha),
                cAPointer.getDevicePointer(), // A
                A.rows(),  // lda
                cBPointer.getDevicePointer(), // x
                B.rows(), // ldb
                PointerUtil.getPointer(beta),  // beta
                cCPointer.getDevicePointer(), // y
                C.rows()); // ldc

        sync();

        cCPointer.copyToHost();
        releaseCublasPointers(cAPointer,cBPointer,cCPointer);

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


        JCublas2.cublasDgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                C.rows(),  // m
                C.columns(), // n
                A.columns(), //k,
                Pointer.to(new double[]{alpha}),
                cAPointer.getDevicePointer(), // A
                A.rows(),  // lda
                cBPointer.getDevicePointer(), // x
                B.rows(), // ldb
                Pointer.to(new double[]{beta}),
                cCPointer.getDevicePointer(), // y
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

        JCublas2.cublasSgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                C.rows(),  // m
                C.columns(), // n
                A.columns(), //k,
                Pointer.to(new float[]{alpha}),
                cAPointer.getDevicePointer(), // A
                A.rows(),  // lda
                cBPointer.getDevicePointer(), // x
                B.rows(), // ldb
                Pointer.to(new float[]{beta}),
                cCPointer.getDevicePointer(), // y
                C.rows()); // incy
        sync();

        cCPointer.copyToHost();
        releaseCublasPointers(cAPointer,cBPointer,cCPointer);

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

        if (A.data().dataType() == DataBuffer.Type.FLOAT) {
            float[] ret = new float[1];
            Pointer result = Pointer.to(ret);
            JCublas2.cublasSnrm2(
                    ContextHolder.getInstance().getHandle()
                    ,A.length()
                    ,cAPointer.getDevicePointer(),
                    2
                    , result);
            return ret[0];
        } else {
            double[] ret = new double[1];
            Pointer result = Pointer.to(ret);

            JCublas2.cublasDnrm2(
                    ContextHolder.getInstance().getHandle()
                    ,A.length(),
                    cAPointer.getDevicePointer()
                    ,2
                    , result);
            return ret[0];
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
                    yCPointer.getDevicePointer()
                    , xCPointer.getDevicePointer()
                    , x.length() * buff.getElementSize() * 2
                    , cudaMemcpyKind.cudaMemcpyDeviceToDevice);
        else
            Nd4j.getExecutioner().exec(new CopyOp(x, y, y, x.length()));

        sync();

        yCPointer.copyToHost();
        releaseCublasPointers(yCPointer, xCPointer);


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
        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            cuComplex complex = cuComplex.cuCmplx(0,0);
            Pointer resultPointer = PointerUtil.getPointer(complex);
            JCublas2.cublasIsamax(ContextHolder.getInstance().getHandle(),x.length(), xCPointer.getDevicePointer(), 1,resultPointer);
            return (int) complex.x - 1;
        } else {
            cuDoubleComplex complex = cuDoubleComplex.cuCmplx(0,0);
            Pointer resultPointer = PointerUtil.getPointer(complex);
            JCublas2.cublasIzamax(ContextHolder.getInstance().getHandle(),x.length(), xCPointer.getDevicePointer(), 1,resultPointer);
            return (int) complex.x;
        }

    }

    /**
     * @param x
     * @return
     */
    public static float asum(IComplexNDArray x) {
        CublasPointer xCPointer = new CublasPointer(x);
        float[] ret = new float[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasScasum(ContextHolder.getInstance().getHandle(),x.length(), xCPointer.getDevicePointer(), 1, result);
        return ret[0];
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

        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            JCublas2.cublasSswap(
                    ContextHolder.getInstance().getHandle(),
                    x.length(),
                    xCPointer.getDevicePointer(),
                    1,
                    yCPointer.getDevicePointer(),
                    1);

        } else {
            JCublas2.cublasDswap(
                    ContextHolder.getInstance().getHandle(),
                    x.length(),
                    xCPointer.getDevicePointer(),
                    1,
                    yCPointer.getDevicePointer(),
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
        Pointer result;
        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            float[] ret = new float[1];
            result = Pointer.to(ret);
            JCublas2.cublasSasum(ContextHolder.getInstance().getHandle(), x.length(), xCPointer.getDevicePointer(), 1, result);
            return ret[0];
        } else {
            double[] ret = new double[1];
            result = Pointer.to(ret);
            JCublas2.cublasDasum(ContextHolder.getInstance().getHandle(), x.length(), xCPointer.getDevicePointer(), 1, result);
            return ret[0];
        }

    }

    /**
     * Returns the norm2 of the given ndarray
     *
     * @param x
     * @return
     */
    public static double nrm2(INDArray x) {

        Pointer result;
        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            CublasPointer xCPointer = new CublasPointer(x);
            float[] ret = new float[1];
            result = Pointer.to(ret);
            JCublas2.cublasSnrm2(ContextHolder.getInstance().getHandle(),x.length(), xCPointer.getDevicePointer(), 1,result);
            return ret[0];
        } else if (x.data().dataType() == DataBuffer.Type.DOUBLE) {
            CublasPointer xCPointer = new CublasPointer(x);
            double[] ret = new double[1];
            result = Pointer.to(ret);
            double normal2 = JCublas2.cublasDnrm2(ContextHolder.getInstance().getHandle(),x.length(), xCPointer.getDevicePointer(), 1,result);
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
        Pointer result;
        sync();
        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            float[] ret = new float[1];
            result = Pointer.to(ret);
            JCublas2.cublasIsamax(
                    ContextHolder.getInstance().getHandle(),
                    x.length() * x.data().getElementSize(),
                    xCPointer.getDevicePointer(),
                    1,result);
            ContextHolder.syncStream();
            sync();
            return (int) (ret[0]- 1);
        }
        else if (x.data().dataType() == DataBuffer.Type.DOUBLE) {
            double[] ret = new double[1];
            result = Pointer.to(ret);
            sync();
            JCublas2.cublasIdamax(
                    ContextHolder.getInstance().getHandle(),
                    x.length(),
                    xCPointer.getDevicePointer(),
                    1, result);
            sync();
            return (int) (ret[0] - 1);
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
        JCublas2.cublasSaxpy(
                ContextHolder.getInstance().getHandle(),
                A.length(),
                Pointer.to(new float[]{da}),
                xAPointer.getDevicePointer(),
                1,
                xBPointer.getDevicePointer(),
                1);

        ((JCudaBuffer)A.data()).copyToHost();
        sync();

        xBPointer.copyToHost();
        releaseCublasPointers(xAPointer, xBPointer);


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

        JCublas2.cublasCaxpy(
                ContextHolder.getInstance().getHandle(),
                A.length(),
                PointerUtil.getPointer(jcuda.cuComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue())),
                aCPointer.getDevicePointer(),
                1,
                bCPointer.getDevicePointer(),
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

        JCublas2.cublasZaxpy(
                ContextHolder.getInstance().getHandle(),
                A.length(),
                PointerUtil.getPointer(jcuda.cuDoubleComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue())),
                aCPointer.getDevicePointer(),
                1,
                bCPointer.getDevicePointer(),
               1
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
        JCublas2.cublasDscal(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                Pointer.to(new double[]{alpha}),
                xCPointer.getDevicePointer(),
                1);
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
        JCublas2.cublasSscal(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                Pointer.to(new float[]{alpha}),
                xCPointer.getDevicePointer(),
               1);
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
        DataTypeValidation.assertSameDataType(x, y);
        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        CublasPointer yCPointer = new CublasPointer(y);

        if(x.data().dataType() == DataBuffer.Type.DOUBLE)
            JCublas2.cublasDcopy(
                    ContextHolder.getInstance().getHandle()
                    ,x.length(),xCPointer.getDevicePointer()
                    ,x.majorStride()
                    ,yCPointer.getDevicePointer()
                    ,y.majorStride());
        if(x.data().dataType() == DataBuffer.Type.FLOAT)
            JCublas2.cublasScopy(ContextHolder.getInstance().getHandle()
                    ,x.length()
                    ,xCPointer.getDevicePointer()
                    ,1
                    ,yCPointer.getDevicePointer()
                    ,1);
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

        Pointer result;
        if (x.data().dataType() == (DataBuffer.Type.FLOAT)) {
            float[] ret = new float[1];
            result = Pointer.to(ret);
            JCublas2.cublasSdot(
                    ContextHolder.getInstance().getHandle(),
                    x.length(),
                    xCPointer.getDevicePointer(),
                   1
                    , yCPointer.getDevicePointer(),
                    1,result);
            sync();

            releaseCublasPointers(xCPointer,yCPointer);

            return ret[0];
        } else {
            double[] ret = new double[1];
            result = Pointer.to(ret);
            JCublas2.cublasDdot(
                    ContextHolder.getInstance().getHandle(),
                    x.length(),
                    xCPointer.getDevicePointer(),
                    1
                    , yCPointer.getDevicePointer(),
                    1,result);
            sync();

            releaseCublasPointers(xCPointer,yCPointer);

            return ret[0];
        }





    }


    private static void releaseCublasPointers(CublasPointer... pointers) {
        for(CublasPointer pointer : pointers)
            try {
                if(pointer != null)
                    pointer.close();
            } catch(Exception e) {
                throw new RuntimeException("Could not run cublas command", e);
            }
    }


    /**
     * Dot product between to complex ndarrays
     * @param x
     * @param y
     * @return
     */
    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) {
        DataTypeValidation.assertSameDataType(x, y);

        sync();

        CublasPointer aCPointer = new CublasPointer(x);
        CublasPointer bCPointer = new CublasPointer(y);

        jcuda.cuDoubleComplex result = jcuda.cuDoubleComplex.cuCmplx(0,0);
        Pointer resultPointer = PointerUtil.getPointer(result);
        JCublas2.cublasZdotc(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                aCPointer.getDevicePointer(),
                1,
                bCPointer.getDevicePointer(),
                1,resultPointer);

        IComplexDouble ret = Nd4j.createDouble(result.x, result.y);
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


        JCublas2.cublasDger(
                ContextHolder.getInstance().getHandle(),
                A.rows(),   // m
                A.columns(),// n
                Pointer.to(new double[]{alpha}),      // alpha
                aCPointer.getDevicePointer(),        // d_A or x
                A.rows(),   // incx
                bCPointer.getDevicePointer(),        // dB or y
                B.rows(),   // incy
                cCPointer.getDevicePointer(),        // dC or A
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


        JCublas2.cublasSger(
                ContextHolder.getInstance().getHandle(),
                A.rows(),   // m
                A.columns(),// n
                Pointer.to(new float[]{alpha}),      // alpha
                aCPointer.getDevicePointer(),        // d_A or x
                A.rows(),   // incx
                bCPointer.getDevicePointer(),        // dB or y
                B.rows(),   // incy
                cCPointer.getDevicePointer(),        // dC or A
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

        JCublas2.cublasCscal(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                PointerUtil.getPointer(jcuda.cuComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent())),
                xCPointer.getDevicePointer(),
                1
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

        JCublas2.cublasZscal(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                PointerUtil.getPointer(jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent())),
                xCPointer.getDevicePointer(),
               1
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
        if (x.data().dataType() == DataBuffer.Type.DOUBLE) {
            cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(0, 0);
            Pointer p = PointerUtil.getPointer(alpha);
            JCublas2.cublasZdotu(
                    ContextHolder.getInstance().getHandle()
                    ,x.length()
                    , xCPointer.getDevicePointer(),
                    1
                    , yCPointer.getDevicePointer()
                    , 1,p);
            ret = Nd4j.createDouble(alpha.x, alpha.y);
        } else {
            cuComplex complex = cuComplex.cuCmplx(0, 0);
            Pointer p = PointerUtil.getPointer(complex);
            JCublas2.cublasCdotu(ContextHolder.getInstance().getHandle()
                    ,x.length()
                    , xCPointer.getDevicePointer()
                    , 1
                    , yCPointer.getDevicePointer()
                    , 1,p);
            ret = Nd4j.createDouble(complex.x, complex.y);
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

        JCublas2.cublasZgeru(
                ContextHolder.getInstance().getHandle(),
                A.rows(),   // m
                A.columns(),// n
                PointerUtil.getPointer(alpha),      // alpha
                aCPointer.getDevicePointer(),        // d_A or x
                A.rows(),   // incx
                bCPointer.getDevicePointer(),        // d_B or y
                B.rows(),   // incy
                cCPointer.getDevicePointer(),        // d_C or A
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


        JCublas2.cublasCgerc(
                ContextHolder.getInstance().getHandle(),
                A.rows(),   // m
                A.columns(),// n
                PointerUtil.getPointer(alpha),      // alpha
                aCPointer.getDevicePointer(),        // dA or x
                A.rows(),   // incx
                bCPointer.getDevicePointer(),        // dB or y
                B.rows(),   // incy
                cCPointer.getDevicePointer(),        // dC or A
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

        JCublas2.cublasZgeru(
                ContextHolder.getInstance().getHandle(),
                A.rows(),   // m
                A.columns(),// n
                PointerUtil.getPointer(alpha),      // alpha
                aCPointer.getDevicePointer(),        // d_A or x
                A.rows(),   // incx
                bCPointer.getDevicePointer(),        // d_B or y
                B.rows(),   // incy
                cCPointer.getDevicePointer(),        // d_C or A
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


        JCublas2.cublasZgerc(
                ContextHolder.getInstance().getHandle(),
                A.rows(),   // m
                A.columns(),// n
                PointerUtil.getPointer(alpha),      // alpha
                aCPointer.getDevicePointer(),        // dA or x
                A.rows(),   // incx
                bCPointer.getDevicePointer(),        // dB or y
                B.rows(),   // incy
                cCPointer.getDevicePointer(),        // dC or A
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

        JCublas2.cublasDaxpy(
                ContextHolder.getInstance().getHandle(),x.length()
                , Pointer.to(new double[]{alpha})
                , xCPointer.getDevicePointer()
                , 1
                , yCPointer.getDevicePointer()
                , 1);

        sync();

        yCPointer.copyToHost();
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

        JCublas2.cublasSaxpy(
                ContextHolder.getInstance().getHandle()
                ,x.length()
                , Pointer.to(new float[]{alpha})
                , xCPointer.getDevicePointer(),
                1,
                yCPointer.getDevicePointer()
                , 1);
        sync();

        xCPointer.copyToHost();
        releaseCublasPointers(xCPointer, yCPointer);


    }
}
