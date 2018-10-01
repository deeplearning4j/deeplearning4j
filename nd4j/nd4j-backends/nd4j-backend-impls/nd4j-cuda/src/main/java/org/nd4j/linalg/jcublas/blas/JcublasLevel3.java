/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.jcublas.blas;


import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.blas.impl.BaseLevel3;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.cublas.*;
import static org.bytedeco.javacpp.cuda.*;
import static org.nd4j.linalg.jcublas.blas.CudaBlas.*;

/**
 * Level 3 implementation of matrix matrix operations
 *
 * @author Adam Gibson
 */
public class JcublasLevel3 extends BaseLevel3 {
    private Allocator allocator = AtomicAllocator.getInstance();
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private static Logger logger = LoggerFactory.getLogger(JcublasLevel3.class);

    @Override
    protected void hgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda,
                    INDArray B, int ldb, float beta, INDArray C, int ldc) {
        //A = Shape.toOffsetZero(A);
        //B = Shape.toOffsetZero(B);

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer cAPointer = new CublasPointer(A, ctx);
        CublasPointer cBPointer = new CublasPointer(B, ctx);
        CublasPointer cCPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            int arch = CudaEnvironment.getInstance().getCurrentDeviceArchitecture();

            if ((CUDA_VERSION >= 8000 && (arch == 53 || arch == 60 || arch == 70)) || (CUDA_VERSION >= 8000 &&  CUDA_VERSION < 9020)) {
                // on these selected archs we run with cublasHgemm
                __half alphaHalf = new __half();
                __half betaHalf = new __half();
                new ShortPointer(alphaHalf).put((short) HalfIndexer.fromFloat(alpha));
                new ShortPointer(betaHalf).put((short) HalfIndexer.fromFloat(beta));

                cublasHgemm(new cublasContext(handle), convertTranspose(TransA), convertTranspose(TransB), M, N, K,
                                alphaHalf, new __half(cAPointer.getDevicePointer()), lda,
                                new __half(cBPointer.getDevicePointer()), ldb, betaHalf,
                                new __half(cCPointer.getDevicePointer()), ldc);
            } else {
                // CUDA_R_16F == 2 for CUDA 8
                // CUBLAS_DATA_HALF == 2 for CUDA 7.5
                cublasSgemmEx(new cublasContext(handle), convertTranspose(TransA), convertTranspose(TransB), M, N, K,
                                new FloatPointer(alpha), (ShortPointer) cAPointer.getDevicePointer(), 2, lda,
                                (ShortPointer) cBPointer.getDevicePointer(), 2, ldb, new FloatPointer(beta),
                                (ShortPointer) cCPointer.getDevicePointer(), 2, ldc);
            }
        }

        allocator.registerAction(ctx, C, A, B);
        OpExecutionerUtil.checkForAny(C);
    }


    @Override
    protected void sgemm(char Order, char TransA, char TransB, int M, int N, int K, float alpha, INDArray A, int lda,
                    INDArray B, int ldb, float beta, INDArray C, int ldc) {
        //A = Shape.toOffsetZero(A);
        //B = Shape.toOffsetZero(B);
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT gemm called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer cAPointer = new CublasPointer(A, ctx);
        CublasPointer cBPointer = new CublasPointer(B, ctx);
        CublasPointer cCPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasSgemm_v2(new cublasContext(handle), convertTranspose(TransA), convertTranspose(TransB), M, N, K,
                            new FloatPointer(alpha), (FloatPointer) cAPointer.getDevicePointer(), lda,
                            (FloatPointer) cBPointer.getDevicePointer(), ldb, new FloatPointer(beta),
                            (FloatPointer) cCPointer.getDevicePointer(), ldc);
        }

        allocator.registerAction(ctx, C, A, B);
        OpExecutionerUtil.checkForAny(C);
    }

    @Override
    protected void ssymm(char Order, char Side, char Uplo, int M, int N, float alpha, INDArray A, int lda, INDArray B,
                    int ldb, float beta, INDArray C, int ldc) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT symm called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer bPointer = new CublasPointer(B, ctx);
        CublasPointer cPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasSsymm_v2(new cublasContext(handle), convertSideMode(Side), convertUplo(Uplo), M, N,
                            new FloatPointer(alpha), (FloatPointer) aPointer.getDevicePointer(), lda,
                            (FloatPointer) bPointer.getDevicePointer(), ldb, new FloatPointer(beta),
                            (FloatPointer) cPointer.getDevicePointer(), ldc);
        }

        allocator.registerAction(ctx, C, A, B);
        OpExecutionerUtil.checkForAny(C);
    }

    @Override
    protected void ssyrk(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, float beta,
                    INDArray C, int ldc) {

        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT syrk called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer cPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasSsyrk_v2(new cublasContext(handle), convertUplo(Uplo), convertTranspose(Trans), N, K,
                            new FloatPointer(alpha), (FloatPointer) aPointer.getDevicePointer(), lda,
                            new FloatPointer(beta), (FloatPointer) cPointer.getDevicePointer(), ldc);
        }

        allocator.registerAction(ctx, C, A);
        OpExecutionerUtil.checkForAny(C);
    }

    @Override
    protected void ssyr2k(char Order, char Uplo, char Trans, int N, int K, float alpha, INDArray A, int lda, INDArray B,
                    int ldb, float beta, INDArray C, int ldc) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void strmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void strsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, float alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT trsm called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(B, A);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer bPointer = new CublasPointer(B, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasStrsm_v2(new cublasContext(handle), convertSideMode(Side), convertUplo(Uplo),
                            convertTranspose(TransA), convertDiag(Diag), M, N, new FloatPointer(alpha),
                            (FloatPointer) aPointer.getDevicePointer(), lda, (FloatPointer) bPointer.getDevicePointer(),
                            ldb);
        }

        allocator.registerAction(ctx, B, A);
        OpExecutionerUtil.checkForAny(B);
    }

    @Override
    protected void dgemm(char Order, char TransA, char TransB, int M, int N, int K, double alpha, INDArray A, int lda,
                    INDArray B, int ldb, double beta, INDArray C, int ldc) {
        //A = Shape.toOffsetZero(A);
        //B = Shape.toOffsetZero(B);
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE gemm called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        DataTypeValidation.assertDouble(A, B, C);

        CublasPointer cAPointer = new CublasPointer(A, ctx);
        CublasPointer cBPointer = new CublasPointer(B, ctx);
        CublasPointer cCPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDgemm_v2(new cublasContext(handle), convertTranspose(TransA), convertTranspose(TransB), M, N, K,
                            new DoublePointer(alpha), (DoublePointer) cAPointer.getDevicePointer(), lda,
                            (DoublePointer) cBPointer.getDevicePointer(), ldb, new DoublePointer(beta),
                            (DoublePointer) cCPointer.getDevicePointer(), ldc);
        }

        allocator.registerAction(ctx, C, A, B);
        OpExecutionerUtil.checkForAny(C);
    }

    @Override
    protected void dsymm(char Order, char Side, char Uplo, int M, int N, double alpha, INDArray A, int lda, INDArray B,
                    int ldb, double beta, INDArray C, int ldc) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE symm called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer bPointer = new CublasPointer(B, ctx);
        CublasPointer cPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDsymm_v2(new cublasContext(handle), convertSideMode(Side), convertUplo(Uplo), M, N,
                            new DoublePointer(alpha), (DoublePointer) aPointer.getDevicePointer(), lda,
                            (DoublePointer) bPointer.getDevicePointer(), ldb, new DoublePointer(beta),
                            (DoublePointer) cPointer.getDevicePointer(), ldc);
        }

        allocator.registerAction(ctx, C, A, B);
        OpExecutionerUtil.checkForAny(C);
    }

    @Override
    protected void dsyrk(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda,
                    double beta, INDArray C, int ldc) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE syrk called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer cPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDsyrk_v2(new cublasContext(handle), convertUplo(Uplo), Trans, N, K, new DoublePointer(alpha),
                            (DoublePointer) aPointer.getDevicePointer(), lda, new DoublePointer(beta),
                            (DoublePointer) cPointer.getDevicePointer(), ldc);
        }

        allocator.registerAction(ctx, C, A);
        OpExecutionerUtil.checkForAny(C);
    }

    @Override
    protected void dsyr2k(char Order, char Uplo, char Trans, int N, int K, double alpha, INDArray A, int lda,
                    INDArray B, int ldb, double beta, INDArray C, int ldc) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE syr2k called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(C, A, B);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer bPointer = new CublasPointer(B, ctx);
        CublasPointer cPointer = new CublasPointer(C, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDsyr2k_v2(new cublasContext(handle), convertUplo(Uplo), Trans, N, K, new DoublePointer(alpha),
                            (DoublePointer) aPointer.getDevicePointer(), lda,
                            (DoublePointer) bPointer.getDevicePointer(), ldb, new DoublePointer(beta),
                            (DoublePointer) cPointer.getDevicePointer(), ldc);
        }

        allocator.registerAction(ctx, C, A, B);
        OpExecutionerUtil.checkForAny(C);
    }

    @Override
    protected void dtrmm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE trmm called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(B, A);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer bPointer = new CublasPointer(B, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDtrmm_v2(new cublasContext(handle), convertSideMode(Side), convertUplo(Uplo),
                            convertTranspose(TransA), convertDiag(Diag), M, N, new DoublePointer(alpha),
                            (DoublePointer) aPointer.getDevicePointer(), lda,
                            (DoublePointer) bPointer.getDevicePointer(), ldb,
                            (DoublePointer) bPointer.getDevicePointer(), ldb);
        }

        allocator.registerAction(ctx, B, A);
        OpExecutionerUtil.checkForAny(B);
    }

    @Override
    protected void dtrsm(char Order, char Side, char Uplo, char TransA, char Diag, int M, int N, double alpha,
                    INDArray A, int lda, INDArray B, int ldb) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE trsm called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(B, A);

        CublasPointer aPointer = new CublasPointer(A, ctx);
        CublasPointer bPointer = new CublasPointer(B, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDtrsm_v2(new cublasContext(handle), convertSideMode(Side), convertUplo(Uplo),
                            convertTranspose(TransA), convertDiag(Diag), M, N, new DoublePointer(alpha),
                            (DoublePointer) aPointer.getDevicePointer(), lda,
                            (DoublePointer) bPointer.getDevicePointer(), ldb);
        }

        allocator.registerAction(ctx, B, A);
        OpExecutionerUtil.checkForAny(B);
    }
}
