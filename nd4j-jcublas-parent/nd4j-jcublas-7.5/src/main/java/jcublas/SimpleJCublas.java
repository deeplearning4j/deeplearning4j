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

package jcublas;


import jcuda.Pointer;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import lombok.Cleanup;
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
import org.nd4j.linalg.util.LinearUtil;

/**
 * Simple abstraction for jcublas operations
 *
 * @author mjk
 * @author Adam Gibson
 */
public class SimpleJCublas {

    private static boolean init = false;


    static {
        try {
            init();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Assert that the data buffer for each ndarray
     * is a cuda buffer
     * @param buffer the arrays to tests
     */
    public static void assertCudaBuffer(INDArray... buffer) throws Exception {
        for (INDArray b1 : buffer)
            if (!(b1.data() instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }

    /**
     * Assert that the data buffer for each ndarray
     * is a cuda buffer
     * @param buffer the arrays to tests
     */
    public static void assertCudaBuffer(DataBuffer... buffer) throws Exception {
        for (DataBuffer b1 : buffer)
            if (!(b1 instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }






    /**
     * Initialize JCublas2. Only called once
     */
    public static synchronized void init() throws Exception {
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
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, double alpha, double beta) throws Exception {

        DataTypeValidation.assertDouble(A, B, C);
        assertCudaBuffer(A.data(), B.data(), C.data());
        sync();

        @Cleanup CublasPointer cAPointer = new CublasPointer(A.offset() > 0 ? A.ravel() : A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B.offset() > 0 ? B.ravel() : B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);

        JCublas2.cublasDgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N,
                A.rows(),
                A.columns(),
                Pointer.to(new double[]{alpha}),
                cAPointer.getDevicePointer(),
                A.rows(),
                cBPointer.getDevicePointer(),
                B.majorStride(),
                Pointer.to(new double[]{beta}),
                cCPointer.getDevicePointer(),
                C.majorStride());

        cCPointer.copyToHost();
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
    public static INDArray gemv(INDArray A, INDArray B, INDArray C, float alpha, float beta) throws Exception {

        DataTypeValidation.assertFloat(A, B, C);

        @Cleanup CublasPointer cAPointer = new CublasPointer(A.offset() > 0 ? A.ravel() : A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B.offset() > 0 ? B.ravel() : B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);

        sync();


        JCublas2.cublasSgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N,
                A.rows(),
                A.columns(),
                Pointer.to(new float[]{alpha}),
                cAPointer.getDevicePointer(),
                A.size(0),
                cBPointer.getDevicePointer(),
                B.majorStride(),
                Pointer.to(new float[]{beta}),
                cCPointer.getDevicePointer(),
                C.majorStride());

        sync();

        cCPointer.copyToHost();

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
            , IComplexDouble b) throws Exception {
        DataTypeValidation.assertSameDataType(A, B, C);
        sync();

        @Cleanup  CublasPointer cAPointer = new CublasPointer(A.ravel());
        @Cleanup CublasPointer cBPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


        cuDoubleComplex alpha = cuDoubleComplex.cuCmplx(a.realComponent().doubleValue(), b.imaginaryComponent().doubleValue());
        cuDoubleComplex beta = cuDoubleComplex.cuCmplx(b.realComponent().doubleValue(), b.imaginaryComponent().doubleValue());

        JCublas2.cublasZgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                A.rows(),  // m
                A.rows(), // n
                PointerUtil.getPointer(alpha),
                cAPointer.getDevicePointer(), // A
                A.size(0),  // lda
                cBPointer.getDevicePointer(), // x
                B.majorStride() / 2, // ldb
                PointerUtil.getPointer(beta),  // beta
                cCPointer.getDevicePointer(), // ydoin
                C.majorStride() / 2); // ldc

        sync();

        cCPointer.copyToHost();



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
            , IComplexFloat b) throws Exception {
        DataTypeValidation.assertFloat(A, B, C);
        assertCudaBuffer(A, B, C);
        sync();

        @Cleanup CublasPointer cAPointer = new CublasPointer(A.offset() > 0 ? A.ravel() : A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B.offset() > 0 ? B.ravel() : B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);



        cuComplex alpha = cuComplex.cuCmplx(a.realComponent().floatValue(), b.imaginaryComponent().floatValue());
        cuComplex beta = cuComplex.cuCmplx(b.realComponent().floatValue(), b.imaginaryComponent().floatValue());

        JCublas2.cublasCgemv(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                A.rows(),  // m
                A.columns(), // n
                PointerUtil.getPointer(alpha),
                cAPointer.getDevicePointer(), // A
                A.size(0),  // lda
                cBPointer.getDevicePointer(), // x
                B.majorStride() / 2, // ldb
                PointerUtil.getPointer(beta),  // beta
                cCPointer.getDevicePointer(), // y
                C.majorStride() / 2); // ldc

        sync();

        cCPointer.copyToHost();


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
            , IComplexDouble b)  throws Exception {
        DataTypeValidation.assertSameDataType(A, B, C);
        sync();

        @Cleanup CublasPointer cAPointer = new CublasPointer(A.offset() > 0 ? A.ravel() : A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B.offset() > 0 ? B.ravel() : B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);



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
                A.size(0),  // lda
                cBPointer.getDevicePointer(), // x
                B.size(0), // ldb
                PointerUtil.getPointer(beta),  // beta
                cCPointer.getDevicePointer(), // y
                C.size(0)); // ldc

        sync();

        cCPointer.copyToHost();

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
            , IComplexFloat b) throws Exception {
        DataTypeValidation.assertFloat(A, B, C);

        sync();


        cuComplex alpha = cuComplex.cuCmplx(a.realComponent().floatValue(), b.imaginaryComponent().floatValue());
        cuComplex beta = cuComplex.cuCmplx(b.realComponent().floatValue(), b.imaginaryComponent().floatValue());
        //custom striding for blas doesn't work
        if(A.offset() > 0) {

            @Cleanup CublasPointer cAPointer = new CublasPointer(A.ravel());
            @Cleanup CublasPointer cBPointer = new CublasPointer(B);
            @Cleanup CublasPointer cCPointer = new CublasPointer(C);


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
        }

        else {

            @Cleanup  CublasPointer cAPointer = new CublasPointer(A);
            @Cleanup CublasPointer cBPointer = new CublasPointer(B);
            @Cleanup CublasPointer cCPointer = new CublasPointer(C);


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

        }

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
                                double alpha, double beta) throws Exception {

        int m = A.rows();
        int n = B.columns();
        int k = A.columns();
        int lda = A.size(0);
        int ldb = B.size(0);
        int ldc = C.size(0);
        if(A.offset() > 0) {
            INDArray copy = Nd4j.create(A.shape());
            copy.assign(A);
            A = copy;
        }
        if(B.offset() > 0) {
            INDArray copy = Nd4j.create(B.shape());
            copy.assign(B);
            B = copy;
        }


        DataTypeValidation.assertDouble(A, B, C);

        sync();


        @Cleanup CublasPointer cAPointer = new CublasPointer(A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


        JCublas2.cublasDgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                m,  // m
                n, // n
                k, //k,
                Pointer.to(new double[]{alpha}),
                cAPointer.getDevicePointer(), // A
                lda,  // lda
                cBPointer.getDevicePointer(), // x
                ldb, // ldb
                Pointer.to(new double[]{beta}),
                cCPointer.getDevicePointer(), // y
                ldc); // incy

        sync();

        cCPointer.copyToHost();

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
                                float alpha, float beta) throws Exception {
        DataTypeValidation.assertFloat(A, B, C);
        sync();

        int m = A.rows();
        int n = B.columns();
        int k = A.columns();
        int lda = A.size(0);
        int ldb = B.size(0);
        int ldc = C.size(0);
        if(A.offset() > 0) {
            INDArray copy = Nd4j.create(A.shape());
            copy.assign(A);
            A = copy;
        }
        if(B.offset() > 0) {
            INDArray copy = Nd4j.create(B.shape());
            copy.assign(B);
            B = copy;
        }





        @Cleanup CublasPointer cAPointer = new CublasPointer(A);
        @Cleanup CublasPointer cBPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);

        JCublas2.cublasSgemm(
                ContextHolder.getInstance().getHandle(),
                cublasOperation.CUBLAS_OP_N, //trans
                cublasOperation.CUBLAS_OP_N,
                m,  // m
                n, // n
                k, //k,
                Pointer.to(new float[]{alpha}),
                cAPointer.getDevicePointer(), // A
                lda,  // lda
                cBPointer.getDevicePointer(), // x
                ldb, // ldb
                Pointer.to(new float[]{beta}),
                cCPointer.getDevicePointer(), // y
                ldc); // incy
        sync();

        cCPointer.copyToHost();

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
    public static double nrm2(IComplexNDArray A) throws Exception {

        sync();

        @Cleanup CublasPointer cAPointer = new CublasPointer(A);

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
    public static void copy(IComplexNDArray x, IComplexNDArray y) throws Exception {
        DataTypeValidation.assertSameDataType(x, y);
        Nd4j.getExecutioner().exec(new CopyOp(x, y, y, x.length()));
    }


    /**
     * Return the index of the max in the given ndarray
     *
     * @param x the ndarray to ge the max for
     * @return the max index of the given ndarray
     */
    public static int iamax(IComplexNDArray x) throws Exception {
        sync();

        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        if (x.data().dataType() == DataBuffer.Type.FLOAT)  {
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
    public static float asum(IComplexNDArray x) throws Exception {
        CublasPointer xCPointer = new CublasPointer(x);
        float[] ret = new float[1];
        Pointer result = Pointer.to(ret);
        JCublas2.cublasScasum(ContextHolder.getInstance().getHandle(), x.length(), xCPointer.getDevicePointer(), 1, result);
        return ret[0];
    }


    /**
     * Swap the elements in each ndarray
     *
     * @param x
     * @param y
     */
    public static void swap(INDArray x, INDArray y) throws Exception {


        DataTypeValidation.assertSameDataType(x, y);

        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        @Cleanup CublasPointer yCPointer = new CublasPointer(y);
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
    public static double asum(INDArray x) throws Exception {


        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
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
    public static double nrm2(INDArray x) throws Exception {

        Pointer result;
        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            @Cleanup CublasPointer xCPointer = new CublasPointer(x);
            float[] ret = new float[1];
            result = Pointer.to(ret);
            JCublas2.cublasSnrm2(ContextHolder.getInstance().getHandle(), x.length(), xCPointer.getDevicePointer(), 1, result);
            return ret[0];
        } else if (x.data().dataType() == DataBuffer.Type.DOUBLE) {
            @Cleanup CublasPointer xCPointer = new CublasPointer(x);
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
    public static int iamax(INDArray x) throws Exception {
        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        Pointer result;
        sync();
        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            int ret2 = JCublas.cublasIsamax(
                    x.length(),
                    xCPointer.getDevicePointer(),
                    x.majorStride());
            ContextHolder.syncStream();
            sync();
            return  (ret2 - 1);
        }
        else if (x.data().dataType() == DataBuffer.Type.DOUBLE) {
            sync();
            int ret2 = JCublas.cublasIdamax(
                    x.length(),
                    xCPointer.getDevicePointer(),
                    x.majorStride());
            sync();
            return ret2 - 1;
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
    public static void axpy(float da, INDArray A, INDArray B) throws Exception {
        DataTypeValidation.assertFloat(A, B);

        @Cleanup CublasPointer xAPointer = new CublasPointer(A);
        @Cleanup CublasPointer xBPointer = new CublasPointer(B);

        sync();
        int aStride = LinearUtil.linearStride(A);
        int bStride = LinearUtil.linearStride(B);
        JCublas2.cublasSaxpy(
                ContextHolder.getInstance().getHandle(),
                A.length(),
                Pointer.to(new float[]{da}),
                xAPointer.getDevicePointer(),
                aStride,
                xBPointer.getDevicePointer(),
                bStride);

        sync();

        xBPointer.copyToHost();


    }

    /**
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexFloat da, IComplexNDArray A, IComplexNDArray B) throws Exception {
        DataTypeValidation.assertFloat(A, B);



        @Cleanup CublasPointer aCPointer = new CublasPointer(A);
        @Cleanup CublasPointer bCPointer = new CublasPointer(B);
        sync();

        JCublas2.cublasCaxpy(
                ContextHolder.getInstance().getHandle(),
                A.length(),
                PointerUtil.getPointer(jcuda.cuComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue())),
                aCPointer.getDevicePointer(),
                A.majorStride() / 2,
                bCPointer.getDevicePointer(),
                B.majorStride() / 2
        );
        sync();


    }

    /**
     * @param da
     * @param A
     * @param B
     */
    public static void axpy(IComplexDouble da, IComplexNDArray A, IComplexNDArray B) throws Exception {
        DataTypeValidation.assertDouble(A, B);

        @Cleanup CublasPointer aCPointer = new CublasPointer(A);
        @Cleanup CublasPointer bCPointer = new CublasPointer(B);
        sync();

        JCublas2.cublasZaxpy(
                ContextHolder.getInstance().getHandle(),
                A.length(),
                PointerUtil.getPointer(jcuda.cuDoubleComplex.cuCmplx(da.realComponent().floatValue(), da.imaginaryComponent().floatValue())),
                aCPointer.getDevicePointer(),
                A.majorStride(),
                bCPointer.getDevicePointer(),
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
    public static INDArray scal(double alpha, INDArray x) throws Exception {
        DataTypeValidation.assertDouble(x);

        sync();

        CublasPointer xCPointer = new CublasPointer(x);
        JCublas2.cublasDscal(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                Pointer.to(new double[]{alpha}),
                xCPointer.getDevicePointer(),
                x.majorStride());
        sync();

        xCPointer.copyToHost();

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
    public static INDArray scal(float alpha, INDArray x) throws Exception {

        DataTypeValidation.assertFloat(x);
        sync();

        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        JCublas2.cublasSscal(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                Pointer.to(new float[]{alpha}),
                xCPointer.getDevicePointer(),
                x.majorStride());
        sync();

        xCPointer.copyToHost();

        return x;

    }

    /**
     * Copy x to y
     *
     * @param x the src
     * @param y the destination
     */
    public static void copy(INDArray x, INDArray y) throws Exception {
        DataTypeValidation.assertSameDataType(x, y);
        sync();

        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        @Cleanup CublasPointer yCPointer = new CublasPointer(y);

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
                    ,x.majorStride()
                    ,yCPointer.getDevicePointer()
                    ,y.majorStride());
        sync();

        yCPointer.copyToHost();
    }

    /**
     * Dot product between 2 ndarrays
     *
     * @param x the first ndarray
     * @param y the second ndarray
     * @return the dot product between the two ndarrays
     */
    public static double dot(INDArray x, INDArray y) throws Exception {
        DataTypeValidation.assertSameDataType(x, y);

        sync();
        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        @Cleanup CublasPointer yCPointer = new CublasPointer(y);

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
                    1, result);
            sync();


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
                    1, result);
            sync();


            return ret[0];
        }
    }



    /**
     * Dot product between to complex ndarrays
     * @param x
     * @param y
     * @return
     */
    public static IComplexDouble dot(IComplexNDArray x, IComplexNDArray y) throws Exception {
        DataTypeValidation.assertSameDataType(x, y);

        sync();

        @Cleanup CublasPointer aCPointer = new CublasPointer(x);
        @Cleanup CublasPointer bCPointer = new CublasPointer(y);

        jcuda.cuDoubleComplex result = jcuda.cuDoubleComplex.cuCmplx(0, 0);
        Pointer resultPointer = PointerUtil.getPointer(result);
        JCublas2.cublasZdotc(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                aCPointer.getDevicePointer(),
                1,
                bCPointer.getDevicePointer(),
                1, resultPointer);

        IComplexDouble ret = Nd4j.createDouble(result.x, result.y);
        sync();

        return ret;
    }


    public static INDArray ger(INDArray A, INDArray B, INDArray C, double alpha) throws Exception {
        DataTypeValidation.assertDouble(A, B, C);
        sync();

        // = alpha * A * transpose(B) + C
        @Cleanup CublasPointer aCPointer = new CublasPointer(A);
        @Cleanup CublasPointer bCPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


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

        sync();

        return C;
    }


    public static INDArray ger(INDArray A, INDArray B, INDArray C, float alpha) throws Exception {
        DataTypeValidation.assertFloat(A, B, C);

        sync();
        // = alpha * A * transpose(B) + C

        @Cleanup CublasPointer aCPointer = new CublasPointer(A);
        @Cleanup CublasPointer bCPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


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

        return C;
    }


    /**
     * Complex multiplication of an ndarray
     *
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray scal(IComplexFloat alpha, IComplexNDArray x) throws Exception {
        DataTypeValidation.assertFloat(x);

        sync();

        @Cleanup CublasPointer xCPointer = new CublasPointer(x);

        JCublas2.cublasCscal(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                PointerUtil.getPointer(jcuda.cuComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent())),
                xCPointer.getDevicePointer(),
                1
        );
        sync();

        xCPointer.copyToHost();

        return x;
    }

    /**
     * Complex multiplication of an ndarray
     *
     * @param alpha
     * @param x
     * @return
     */
    public static IComplexNDArray scal(IComplexDouble alpha, IComplexNDArray x) throws Exception {
        DataTypeValidation.assertDouble(x);
        sync();


        @Cleanup CublasPointer xCPointer = new CublasPointer(x);

        JCublas2.cublasZscal(
                ContextHolder.getInstance().getHandle(),
                x.length(),
                PointerUtil.getPointer(jcuda.cuDoubleComplex.cuCmplx(alpha.realComponent(), alpha.imaginaryComponent())),
                xCPointer.getDevicePointer(),
                1
        );
        sync();

        xCPointer.copyToHost();

        return x;
    }

    /**
     * Complex dot product
     *
     * @param x
     * @param y
     * @return
     */
    public static IComplexDouble dotu(IComplexNDArray x, IComplexNDArray y) throws Exception {

        DataTypeValidation.assertSameDataType(x, y);
        sync();

        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        @Cleanup CublasPointer yCPointer = new CublasPointer(y);
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
                                       IComplexNDArray C, IComplexDouble Alpha) throws Exception {
        // = alpha * A * tranpose(B) + C
        sync();
        DataTypeValidation.assertDouble(A, B, C);

        @Cleanup CublasPointer aCPointer = new CublasPointer(A);
        @Cleanup CublasPointer bCPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);

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
                                       IComplexFloat Alpha) throws Exception {
        DataTypeValidation.assertFloat(A, B, C);
        // = alpha * A * tranpose(B) + C

        sync();
        @Cleanup CublasPointer aCPointer = new CublasPointer(A);
        @Cleanup CublasPointer bCPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


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
                                       IComplexNDArray C, IComplexFloat Alpha) throws Exception {

        DataTypeValidation.assertFloat(A, B, C);
        // = alpha * A * tranpose(B) + C
        sync();

        @Cleanup CublasPointer aCPointer = new CublasPointer(A);
        @Cleanup CublasPointer bCPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);

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
                                       IComplexDouble Alpha) throws Exception {

        DataTypeValidation.assertDouble(A, B, C);
        // = alpha * A * tranpose(B) + C

        sync();

        @Cleanup CublasPointer aCPointer = new CublasPointer(A);
        @Cleanup CublasPointer bCPointer = new CublasPointer(B);
        @Cleanup CublasPointer cCPointer = new CublasPointer(C);


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
    public static void axpy(double alpha, INDArray x, INDArray y) throws Exception {
        DataTypeValidation.assertDouble(x, y);

        sync();

        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        @Cleanup CublasPointer yCPointer = new CublasPointer(y);

        JCublas2.cublasDaxpy(
                ContextHolder.getInstance().getHandle(), x.length()
                , Pointer.to(new double[]{alpha})
                , xCPointer.getDevicePointer()
                , 1
                , yCPointer.getDevicePointer()
                , 1);

        sync();

        yCPointer.copyToHost();

    }

    /**
     * Simpler version of saxpy
     * taking in to account the parameters of the ndarray
     *
     * @param alpha the alpha to scale by
     * @param x     the x
     * @param y     the y
     */
    public static void saxpy(float alpha, INDArray x, INDArray y) throws Exception {
        DataTypeValidation.assertFloat(x, y);
        sync();

        @Cleanup CublasPointer xCPointer = new CublasPointer(x);
        @Cleanup CublasPointer yCPointer = new CublasPointer(y);

        JCublas2.cublasSaxpy(
                ContextHolder.getInstance().getHandle()
                , x.length()
                , Pointer.to(new float[]{alpha})
                , xCPointer.getDevicePointer(),
                1,
                yCPointer.getDevicePointer()
                , 1);
        sync();

        xCPointer.copyToHost();


    }
}
