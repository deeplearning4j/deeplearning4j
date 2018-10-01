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
import org.bytedeco.javacpp.IntPointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.linalg.api.blas.impl.BaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.ops.impl.accum.ASum;
import org.nd4j.linalg.api.ops.impl.accum.Dot;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.Axpy;
import org.nd4j.linalg.factory.DataTypeValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.ops.executioner.CudaExecutioner;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.cublas.*;
import static org.bytedeco.javacpp.cuda.CUstream_st;

/**
 * @author Adam Gibson
 */
public class JcublasLevel1 extends BaseLevel1 {
    private Allocator allocator = AtomicAllocator.getInstance();
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private static Logger logger = LoggerFactory.getLogger(JcublasLevel1.class);

    @Override
    protected float sdsdot(long N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double dsdot(long N, INDArray X, int incX, INDArray Y, int incY) {
        throw new UnsupportedOperationException();
    }


    @Override
    protected float hdot(long N, INDArray X, int incX, INDArray Y, int incY) {
        DataTypeValidation.assertSameDataType(X, Y);
        //        CudaContext ctx = allocator.getFlowController().prepareAction(null, X, Y);

        float ret = 1f;

        //        CublasPointer xCPointer = new CublasPointer(X, ctx);
        //        CublasPointer yCPointer = new CublasPointer(Y, ctx);

        Dot dot = new Dot(X, Y);
        Nd4j.getExecutioner().exec(dot);

        ret = dot.getFinalResult().floatValue();
        /*
        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            long result = cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new IllegalStateException("cublasSetStream failed");
        
            FloatPointer resultPointer = new FloatPointer(0.0f);
            cuBlasSdot_v2(new cublasContext(handle),
                    N,
                    xCPointer.getDevicePointer(),
                    incX,
                    yCPointer.getDevicePointer(),
                    incY, resultPointer);
            ret = resultPointer.get();
        }
        */

        //        allocator.registerAction(ctx, null, X, Y);

        return ret;
    }


    @Override
    protected float sdot(long N, INDArray X, int incX, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT dot called");

        DataTypeValidation.assertSameDataType(X, Y);

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(null, X, Y);

        float ret = 1f;

        CublasPointer xCPointer = new CublasPointer(X, ctx);
        CublasPointer yCPointer = new CublasPointer(Y, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            long result = cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new IllegalStateException("cublasSetStream failed");

            FloatPointer resultPointer = new FloatPointer(0.0f);
            result = cublasSdot_v2(new cublasContext(handle), (int) N, (FloatPointer) xCPointer.getDevicePointer(), incX,
                            (FloatPointer) yCPointer.getDevicePointer(), incY, resultPointer);
            ret = resultPointer.get();
        }

        allocator.registerAction(ctx, null, X, Y);

        return ret;
    }

    @Override
    protected float hdot(long N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected float sdot(long N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected double ddot(long N, INDArray X, int incX, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE dot called");

        Nd4j.getExecutioner().push();

        double ret;
        CudaContext ctx = allocator.getFlowController().prepareAction(null, X, Y);

        CublasPointer xCPointer = new CublasPointer(X, ctx);
        CublasPointer yCPointer = new CublasPointer(Y, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            DoublePointer resultPointer = new DoublePointer(0.0);
            cublasDdot_v2(new cublasContext(handle), (int) N, (DoublePointer) xCPointer.getDevicePointer(), incX,
                            (DoublePointer) yCPointer.getDevicePointer(), incY, resultPointer);
            ret = resultPointer.get();
        }

        allocator.registerAction(ctx, null, X, Y);

        return ret;
    }

    @Override
    protected double ddot(long N, DataBuffer X, int offsetX, int incX, DataBuffer Y, int offsetY, int incY) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected float snrm2(long N, INDArray X, int incX) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT nrm2 called");

        Nd4j.getExecutioner().push();


        CudaContext ctx = allocator.getFlowController().prepareAction(null, X);
        float ret;

        CublasPointer cAPointer = new CublasPointer(X, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            FloatPointer resultPointer = new FloatPointer(0.0f);
            cublasSnrm2_v2(new cublasContext(handle), (int) N, (FloatPointer) cAPointer.getDevicePointer(), incX,
                            resultPointer);
            ret = resultPointer.get();
        }

        allocator.registerAction(ctx, null, X);

        return ret;
    }

    @Override
    protected float hasum(long N, INDArray X, int incX) {

        ASum asum = new ASum(X);
        Nd4j.getExecutioner().exec(asum, Integer.MAX_VALUE);

        float ret = asum.getFinalResult().floatValue();

        return ret;
    }

    @Override
    protected float sasum(long N, INDArray X, int incX) {
        ASum asum = new ASum(X);
        Nd4j.getExecutioner().exec(asum, Integer.MAX_VALUE);

        float ret = asum.getFinalResult().floatValue();

        return ret;
/*
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT asum called");
        
        
        CudaContext ctx = allocator.getFlowController().prepareAction(null, X);
        float ret;
        
        CublasPointer xCPointer = new CublasPointer(X, ctx);
        
        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));
        
            FloatPointer resultPointer = new FloatPointer(0.0f);
            cublasSasum_v2(new cublasContext(handle),
                    N,
                    (FloatPointer) xCPointer.getDevicePointer(),
                    incX, resultPointer);
            ret = resultPointer.get();
        }
        
        allocator.registerAction(ctx, null, X);
        
        return ret;
*/
    }

    @Override
    protected float hasum(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected float sasum(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected double dnrm2(long N, INDArray X, int incX) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE nrm2 called");

        Nd4j.getExecutioner().push();

        double ret;

        CudaContext ctx = allocator.getFlowController().prepareAction(null, X);

        CublasPointer cAPointer = new CublasPointer(X, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            DoublePointer resultPointer = new DoublePointer(0.0f);
            cublasDnrm2_v2(new cublasContext(handle), (int) N, (DoublePointer) cAPointer.getDevicePointer(), incX,
                            resultPointer);
            ret = resultPointer.get();
        }

        allocator.registerAction(ctx, null, X);

        return ret;
    }

    @Override
    protected double dasum(long N, INDArray X, int incX) {
        ASum asum = new ASum(X);
        Nd4j.getExecutioner().exec(asum, Integer.MAX_VALUE);

        double ret = asum.getFinalResult().doubleValue();

        return ret;
        /*CudaContext ctx = allocator.getFlowController().prepareAction(null, X);
        double ret;
        
        CublasPointer xCPointer = new CublasPointer(X, ctx);
        
        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));
        
            DoublePointer resultPointer = new DoublePointer(0.0);
            cublasDasum_v2(new cublasContext(handle),
                    N,
                    xCPointer.getDevicePointer(),
                    incX, resultPointer);
            ret = resultPointer.get();
        }
        allocator.registerAction(ctx, null, X);
        
        return ret;
        */
    }

    @Override
    protected double dasum(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected int isamax(long N, INDArray X, int incX) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT iamax called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(null, X);
        int ret2;

        CublasPointer xCPointer = new CublasPointer(X, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            IntPointer resultPointer = new IntPointer(new int[] {0});
            cublasIsamax_v2(new cublasContext(handle), (int) N, (FloatPointer) xCPointer.getDevicePointer(), incX,
                            resultPointer);
            ret2 = resultPointer.get();
        }
        allocator.registerAction(ctx, null, X);

        return ret2 - 1;
    }

    @Override
    protected int isamax(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected int idamax(long N, INDArray X, int incX) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE imax called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(null, X);
        int ret2;

        CublasPointer xCPointer = new CublasPointer(X, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            IntPointer resultPointer = new IntPointer(new int[] {0});
            cublasIdamax_v2(new cublasContext(handle), (int) N, (DoublePointer) xCPointer.getDevicePointer(), incX,
                            resultPointer);
            ret2 = resultPointer.get();
        }

        allocator.registerAction(ctx, null, X);

        return ret2 - 1;
    }

    @Override
    protected int idamax(long N, DataBuffer X, int offsetX, int incX) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void sswap(long N, INDArray X, int incX, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT swap called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(Y, X);

        CublasPointer xCPointer = new CublasPointer(X, ctx);
        CublasPointer yCPointer = new CublasPointer(Y, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasSswap_v2(new cublasContext(handle), (int) N, (FloatPointer) xCPointer.getDevicePointer(), incX,
                            (FloatPointer) yCPointer.getDevicePointer(), incY);
        }

        allocator.registerAction(ctx, Y, X);
        OpExecutionerUtil.checkForAny(Y);
    }

    @Override
    protected void scopy(long N, INDArray X, int incX, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT copy called");

        Nd4j.getExecutioner().push();


        CudaContext ctx = allocator.getFlowController().prepareAction(Y, X);

        CublasPointer xCPointer = new CublasPointer(X, ctx);
        CublasPointer yCPointer = new CublasPointer(Y, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasScopy_v2(new cublasContext(handle), (int) N, (FloatPointer) xCPointer.getDevicePointer(), incX,
                            (FloatPointer) yCPointer.getDevicePointer(), incY);
        }

        allocator.registerAction(ctx, Y, X);
        OpExecutionerUtil.checkForAny(Y);
    }

    @Override
    protected void scopy(long N, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void saxpy(long N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT axpy called");

        //        CudaContext ctx = allocator.getFlowController().prepareAction(Y, X);
        Nd4j.getExecutioner().exec(new Axpy(X, Y, alpha, N));

        OpExecutionerUtil.checkForAny(Y);
        /*
        CublasPointer xAPointer = new CublasPointer(X, ctx);
        CublasPointer xBPointer = new CublasPointer(Y, ctx);
        
        cublasHandle_t handle = ctx.getHandle();
        
        
        
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));
        
            PointerPointer p = new cublasContext(handle);
            cublasSaxpy_v2(p,
                    N,
                    alpha,
                    xAPointer.getDevicePointer(),
                    incX,
                    xBPointer.getDevicePointer(),
                    incY);
        }
        */
        //        allocator.registerAction(ctx, Y, X);
    }

    @Override
    protected void haxpy(long N, float alpha, INDArray X, int incX, INDArray Y, int incY) {
        //        CudaContext ctx = allocator.getFlowController().prepareAction(Y, X);

        //        CublasPointer xAPointer = new CublasPointer(X, ctx);
        //        CublasPointer xBPointer = new CublasPointer(Y, ctx);

        //        cublasHandle_t handle = ctx.getHandle();

        ((CudaExecutioner) Nd4j.getExecutioner()).exec(new Axpy(X, Y, alpha, N));

        OpExecutionerUtil.checkForAny(Y);

        /*        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));
        
            PointerPointer p = new cublasContext(handle);
            cublasSaxpy_v2(p,
                    N,
                    alpha,
                    xAPointer.getDevicePointer(),
                    incX,
                    xBPointer.getDevicePointer(),
                    incY);
        }
        */
        //        allocator.registerAction(ctx, Y, X);
    }

    @Override
    protected void haxpy(long N, float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY,
                    int incrY) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void saxpy(long N, float alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY,
                    int incrY) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void dswap(long N, INDArray X, int incX, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE swap called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(Y, X);

        CublasPointer xCPointer = new CublasPointer(X, ctx);
        CublasPointer yCPointer = new CublasPointer(Y, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDswap_v2(new cublasContext(handle), (int) N, (DoublePointer) xCPointer.getDevicePointer(), incX,
                            (DoublePointer) yCPointer.getDevicePointer(), incY);
        }

        allocator.registerAction(ctx, Y, X);

        OpExecutionerUtil.checkForAny(Y);
    }

    @Override
    protected void dcopy(long N, INDArray X, int incX, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE copy called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(Y, X);

        CublasPointer xCPointer = new CublasPointer(X, ctx);
        CublasPointer yCPointer = new CublasPointer(Y, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDcopy_v2(new cublasContext(handle), (int) N, (DoublePointer) xCPointer.getDevicePointer(), incX,
                            (DoublePointer) yCPointer.getDevicePointer(), incY);
        }

        allocator.registerAction(ctx, Y, X);

        OpExecutionerUtil.checkForAny(Y);
    }

    @Override
    protected void dcopy(long N, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void daxpy(long N, double alpha, INDArray X, int incX, INDArray Y, int incY) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE axpy called");

        //CudaContext ctx = allocator.getFlowController().prepareAction(Y, X);


        //    logger.info("incX: {}, incY: {}, N: {}, X.length: {}, Y.length: {}", incX, incY, N, X.length(), Y.length());

        Nd4j.getExecutioner().exec(new Axpy(X, Y, alpha, N));

        OpExecutionerUtil.checkForAny(Y);

        /*
        CublasPointer xAPointer = new CublasPointer(X, ctx);
        CublasPointer xBPointer = new CublasPointer(Y, ctx);
        
        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));
        
            cublasDaxpy_v2(new cublasContext(handle),
                    N, alpha, xAPointer.getDevicePointer(),
                    incX, xBPointer.getDevicePointer(),
                    incY);
        }
        */
        //        allocator.registerAction(ctx, Y, X);
    }

    @Override
    protected void daxpy(long N, double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY,
                    int incrY) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    protected void srotg(float a, float b, float c, float s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotmg(float d1, float d2, float b1, float b2, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srot(long N, INDArray X, int incX, INDArray Y, int incY, float c, float s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void srotm(long N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drotg(double a, double b, double c, double s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drotmg(double d1, double d2, double b1, double b2, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drot(long N, INDArray X, int incX, INDArray Y, int incY, double c, double s) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void drotm(long N, INDArray X, int incX, INDArray Y, int incY, INDArray P) {
        throw new UnsupportedOperationException();

    }

    @Override
    protected void sscal(long N, float alpha, INDArray X, int incX) {
        if (Nd4j.dataType() != DataType.FLOAT)
            logger.warn("FLOAT scal called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(X);

        CublasPointer xCPointer = new CublasPointer(X, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasSscal_v2(new cublasContext(handle),(int) N, new FloatPointer(alpha),
                            (FloatPointer) xCPointer.getDevicePointer(), incX);
        }

        allocator.registerAction(ctx, X);

        OpExecutionerUtil.checkForAny(X);
    }

    @Override
    protected void dscal(long N, double alpha, INDArray X, int incX) {
        if (Nd4j.dataType() != DataType.DOUBLE)
            logger.warn("DOUBLE scal called");

        Nd4j.getExecutioner().push();

        CudaContext ctx = allocator.getFlowController().prepareAction(X);

        CublasPointer xCPointer = new CublasPointer(X, ctx);

        cublasHandle_t handle = ctx.getHandle();
        synchronized (handle) {
            cublasSetStream_v2(new cublasContext(handle), new CUstream_st(ctx.getOldStream()));

            cublasDscal_v2(new cublasContext(handle), (int) N, new DoublePointer(alpha),
                            (DoublePointer) xCPointer.getDevicePointer(), incX);
        }

        allocator.registerAction(ctx, X);

        OpExecutionerUtil.checkForAny(X);
    }

    @Override
    public boolean supportsDataBufferL1Ops() {
        return false;
    }
}
