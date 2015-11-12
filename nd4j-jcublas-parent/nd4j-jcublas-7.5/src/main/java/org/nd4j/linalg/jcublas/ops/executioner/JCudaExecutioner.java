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

package org.nd4j.linalg.jcublas.ops.executioner;


import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDimensions;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;


/**
 * JCuda executioner.
 * <p/>
 * Runs ops directly on the gpu
 *
 * @author Adam Gibson
 */
public class JCudaExecutioner extends DefaultOpExecutioner {
    private JCudaBuffer dummyFloatPointer, dummyDoublePointer;

    public JCudaExecutioner() {
        try {
            SimpleJCublas.init();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        dummyFloatPointer = KernelFunctions.alloc(new float[]{1});
        dummyDoublePointer =KernelFunctions.alloc(new double[]{1});
//        parallelExecutioner().setParallelEnabled(false);
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        if(op.isPassThrough()) {
            op.exec(dimension);
            return op.z();
        }


        if(dimension[0] == Integer.MAX_VALUE) {
            if(op.x() instanceof IComplexNDArray)
                return Nd4j.scalar(execAndReturn(op).getFinalResultComplex());
            return Nd4j.scalar(execAndReturn(op).getFinalResult().doubleValue());
        }

        if(op instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            IComplexNDArray linear = ret;
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).getFinalResultComplex();
                linear.putScalar(i, result);

            }

            if(ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));


            return ret;
        }

        else {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            //nothing to reduce
            if(ArrayUtil.prod(retShape) == op.x().length())
                return op.x();

            INDArray retArray = Nd4j.create(retShape);
            invoke(op,dimension,retArray,true);
            return retArray;
        }


    }

    @Override
    public INDArray execAndReturn(TransformOp op, int... dimension) {
        return super.execAndReturn(op, dimension);
    }



    @Override
    public INDArray execAndReturn(ScalarOp op, int... dimension) {
        return super.execAndReturn(op, dimension);
    }

    @Override
    public Op exec(Op op, int... dimension) {
        return super.exec(op, dimension);
    }


    @Override
    public Op exec(Op op) {
        //linear views and oblong offsets can't be handled by the gpu (due to the way the buffers are interpeted as vectors)
        if(op.x() instanceof IComplexNDArray
                || executionMode() == ExecutionMode.JAVA || op.isPassThrough())
            return super.exec(op);

        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t,true);
        } else if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            invoke(acc,null,Nd4j.scalar(0),true);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            invoke(sc,true);
        }
        return op;
    }

    private JCudaBuffer dummyDouble() {
        return dummyDoublePointer;
    }

    private JCudaBuffer dummyFloat() {
        return dummyFloatPointer;
    }

    @Override
    public INDArray execAndReturn(TransformOp op) {
        invoke(op,true);
        return op.z();
    }


    /**
     * Converts the given parameters
     * in to extra arguments to
     * pass to the kernel
     *
     * @param extraArgs the extra arguments
     * @param dataType  the data type
     * @return
     */
    private JCudaBuffer toArgs(Object[] extraArgs, String dataType) {
        if (dataType.equals("double")) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyDouble();
            return KernelFunctions.alloc(PointerUtil.toDoubles(extraArgs));
        } else if (dataType.equals("float")) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyFloat();
            return KernelFunctions.alloc(PointerUtil.toFloats(extraArgs));
        }
        throw new IllegalArgumentException("Illegal datatype");
    }


    private CudaContext invoke(BroadcastOp op,boolean sync) {
        CudaContext ctx;

        ContextHolder.getInstance().setContext();

        if(!KernelFunctionLoader.getInstance().exists(op.name()) || executionMode() == ExecutionMode.JAVA || op.isPassThrough())
            super.exec(op);


        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());
        metrics.setGridSize(op.x().data().length());
        metrics.setBlockSize(1024);
        metrics.setSharedMemoryNotOverMax(metrics.getBlockSize() * op.x().data().getElementSize());
        if(op.y() == null)
            throw new IllegalArgumentException("Op has no y to broadcast");


        int[] shape = op.broadcastShape();
        int[] smallerShape = op.x().shape();
        boolean compatible = true;
        int count = shape.length - 1;
        int thisCount = smallerShape.length - 1;
        for (int i = shape.length - 1; i > 0; i--) {
            if (count < 0 || thisCount < 0)
                break;
            if (shape[count] != smallerShape[thisCount] && shape[count] != 1 && smallerShape[thisCount] != 1) {
                compatible = false;
                break;
            }

            count--;
            thisCount--;
        }

        if (!compatible)
            throw new IllegalArgumentException("Incompatible broadcast from " + Arrays.toString(smallerShape) + " to " + Arrays.toString(shape));

        //total number of times to repeat each value over an element wise stride on the gpu
        int[] dimensions = BroadcastDimensions.getDimensions(op.y().shape());
        /**
         * 		T *x
         ,int *xShapeInfo
         ,T *y
         ,int *yShapeInfo
         ,T *result
         ,int *resultShapeInfo,
         int *dimension,
         int dimensionLength,
         int *gpuInformation
         */
        Object[] kernelParams = new Object[] {
                op.x(),
                KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x())),
                op.y(),
                KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.y())),
                op.z(),
                KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.z())),
                KernelFunctions.alloc(dimensions),
                dimensions.length,
                KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
        };


        /**
         *
         * Will need to get an element wise stride
         * along a broadcast dimension wrt the shape
         * This will allow us to setup a linear
         * operator along a subset of the original array
         * repeating along the desired dimensions.
         *
         * Will also need to figure out how to split
         * the bigger array wrt the original input
         * computing broadcast slices wrt the
         * specified y being broadcast.
         *
         * Will also need an element wise stride
         * for the bigger array
         * and a way to compute the offsets
         * (likely related to the major stride of the bigger array?)
         *
         * This will be very similar to how TAD is designed.
         *
         * There will likely be times when we need to compute a dup()
         * in order to force alignment of the data.
         */
        try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams)) {
            invokeFunction(op, sync,metrics,kParams.getContext(), kParams.getKernelParameters());
            ctx = kParams.getContext();
            if(sync)
                kParams.sync();
        } catch(Exception e) {
            throw new RuntimeException("Could not execute kernel: Kernel launch was: " + metrics, e);
        }



        return ctx;
    }

    private CudaContext invoke(Accumulation op,int[] dimension,INDArray result,boolean sync)  {
        CudaContext ctx;

        ContextHolder.getInstance().setContext();

        if(!KernelFunctionLoader.getInstance().exists(op.name()) || executionMode() == ExecutionMode.JAVA || op.isPassThrough())
            super.exec(op);


        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());
        if(dimension != null) {
            int length = op.x().tensorssAlongDimension(dimension);
            if(length > 1000)
                length = 1000;
            //of note here: THIS IS REVERSE OF WHAT IT SHOULD BE, THIS IS INTENDED.
            metrics.setGridSize(length);
            metrics.setBlockSize(op.x().tensorAlongDimension(0,dimension).length());
            int sharedMemBasedOnBlockSize = op.x().tensorAlongDimension(0,dimension).length() * 10 *  op.x().data().getElementSize();
            if(sharedMemBasedOnBlockSize < 1024)
                sharedMemBasedOnBlockSize = 1024;
            metrics.setSharedMemoryNotOverMax(sharedMemBasedOnBlockSize);
        }

        else {
            metrics.setGridSize(op.x().data().length());
            metrics.setBlockSize(1024);
            metrics.setSharedMemoryNotOverMax(metrics.getBlockSize() * op.x().data().getElementSize());
        }



        if (op.y() != null) {
            metrics.setSharedMemoryNotOverMax(metrics.getSharedMemory() * 2);
            int xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : op.x().tensorAlongDimension(0,dimension));
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            int yStride = BlasBufferUtil.getBlasStride(dimension == null ? op.y() : op.y().tensorAlongDimension(0,dimension));
            if(yStride < 0) {
                op.setY(op.y().dup());
            }
            else if(op.y().ordering() != op.x().ordering()) {
                op.setY(op.y().dup(op.x().ordering()));
            }



            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x(),
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x(),dimension)),
                    op.y(),
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.y(),dimension)),
                    toArgs(op.extraArgs(),
                            getType(op)),
                    result,
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(result)),
                    KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
                    KernelFunctions.alloc(dimension == null ? new int[] {1} : dimension),
                    dimension == null ? 1 : dimension.length
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultOp(op, result)) {
                invokeFunction(op, sync,metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();

            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }

            return ctx;


        } else {
            int xStride = BlasBufferUtil.getBlasStride(dimension == null ? op.x() : op.x().tensorAlongDimension(0,dimension));
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            int length = op.x().data().length();
            if(dimension == null && xStride == 1 && op.x().offset() == 0)
                length = op.n();

            Object[] kernelParams = new Object[] {
                    length,
                    op.x(),
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(op.x(),dimension)),
                    toArgs(op.extraArgs(), getType(op)),
                    result,
                    KernelFunctions.alloc(PointerUtil.toShapeInfoBuffer(result)),
                    KernelFunctions.alloc(metrics.getGpuDefinitionInfo()),
                    KernelFunctions.alloc(dimension == null ? new int[] {1} : dimension),
                    dimension == null ? 1 : dimension.length
            };



            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultOp(op, result)) {
                invokeFunction(op, sync,metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel: Kernel launch was: " + metrics, e);
            }



        }


        return ctx;
    }


    private CudaContext invoke(ScalarOp op,boolean sync) {

        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());
        metrics.setGridSize(op.n());
        metrics.setBlockSize(1024);
        metrics.setSharedMemory(metrics.getBlockSize() * op.x().data().getElementSize());

        CudaContext ctx = null;
        if(!KernelFunctionLoader.getInstance().exists(op.name())  || executionMode() == ExecutionMode.JAVA)
            super.exec(op);

        if (op.y() != null) {
            metrics.setSharedMemory(metrics.getSharedMemory() * 2);

            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            int yStride = BlasBufferUtil.getBlasStride(op.y());
            if(yStride < 0) {
                op.setY(op.y().dup());
            }

            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.x(),
                    op.y(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    BlasBufferUtil.getBlasStride(op.y()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z()
                    ,metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultArray(op.z())) {
                invokeFunction(op,sync,metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }



        } else {
            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }


            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    PointerUtil.getPointer(op),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultArray(op.z())) {
                invokeFunction(op,sync, metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();

            }

            catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }
        }


        return ctx;

    }




    private CudaContext invoke(TransformOp op,boolean sync) {
        if(!KernelFunctionLoader.getInstance().exists(op.name()) || op.x() instanceof IComplexNDArray || op.isPassThrough()) {
            super.exec(op);
            return null;
        }

        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());
        metrics.setGridSize(op.n());
        metrics.setBlockSize(1024);
        metrics.setSharedMemory(metrics.getBlockSize() * op.x().data().getElementSize());


        metrics.setGridMemoryNotOverMax(op.x().data().length());
        metrics.setBlockSize(1024);
        metrics.setSharedMemoryNotOverMax(metrics.getBlockSize() * op.x().data().getElementSize());

        CudaContext ctx;
        if (op.y() != null) {
            metrics.setSharedMemory(metrics.getSharedMemory() * 2);

            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            int yStride = BlasBufferUtil.getBlasStride(op.y());
            if(yStride < 0) {
                op.setY(op.y().dup());
            }
            else if(op.y().ordering() != op.x().ordering()) {
                op.setY(op.y().dup(op.x().ordering()));
            }

            /**
             * Construct pointer arguments in the following order:
             * n
             * offset,
             * pointer to buffer
             * increment,
             * extraArgs,
             * result
             */

            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.x(),
                    op.y(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    BlasBufferUtil.getBlasStride(op.y()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),
                    BlasBufferUtil.getBlasStride(op.z())
                    ,metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultArray(op.z())) {
                invokeFunction(op,sync, metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();

            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }


        } else {
            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultArray(op.z())) {
                invokeFunction(op,sync, metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }
        }


        return ctx;
    }


    private void invokeFunction(Op op,boolean sync,GpuMetrics metrics,CudaContext cudaContext, Object... kernelParams) {
        /**
         * Invoke a cuda kernel by name. This will be wrt the function name.
         * Functions that are accumulations or transforms have names that end with _strided.
         *
         */

        metrics.validate();
        String functionName = op instanceof TransformOp || op instanceof Accumulation ? op.name() + "_strided" : op.name();
        //force blocks and threads to be even
        KernelFunctions.invoke(
                metrics,
                sync
                ,functionName
                ,getType(op),cudaContext
                ,kernelParams);



    }

    private String getType(Op op) {
        return op.x().data().dataType() == DataBuffer.Type.DOUBLE ? "double" : "float";
    }

}


