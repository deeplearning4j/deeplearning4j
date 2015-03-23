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

package org.nd4j.linalg.jcublas.ops.executioner;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.util.PointerUtil;


/**
 * JCuda executioner.
 * <p/>
 * Runs ops directly on the gpu
 *
 * @author Adam Gibson
 */
public class JCudaExecutioner implements OpExecutioner {
    private JCudaBuffer dummyFloatPointer, dummyDoublePointer;

    public JCudaExecutioner() {
        dummyFloatPointer = KernelFunctions.alloc(new float[]{1});
        dummyDoublePointer =KernelFunctions.alloc(new double[]{1});
    }

    @Override
    public Op exec(Op op) {
        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t);
        } else if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            invoke(acc);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            invoke(sc);
        }
        return op;
    }


    private Pointer dummyDoublePointer() {
        return Pointer.to(dummyDoublePointer.pointer());
    }

    private Pointer dummyFloatPointer() {
        return Pointer.to(dummyFloatPointer.pointer());
    }

    @Override
    public INDArray execAndReturn(TransformOp op) {
        invoke(op);
        return op.z();
    }


    @Override
    public Accumulation execAndReturn(Accumulation op) {
        return (Accumulation) exec(op);
    }

    @Override
    public INDArray execAndReturn(ScalarOp op) {
        return exec(op).z();
    }


    @Override
    public Op exec(Op op, int dimension) {
        //only accumulate along a particular dimension
        if (op instanceof Accumulation) {
            Accumulation a = (Accumulation) op;
            return exec(a);
        }
        for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i, dimension);
            exec(op2);
            if (op instanceof TransformOp) {
                TransformOp t = (TransformOp) op;
                TransformOp t2 = (TransformOp) op2;
                t.z().vectorAlongDimension(i, dimension).assign(t2.z());
            }


        }
        return op;
    }


    @Override
    public INDArray execAndReturn(TransformOp op, int dimension) {
        for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i, dimension);
            exec(op2);
            if (op instanceof TransformOp) {
                TransformOp t = op;
                TransformOp t2 = (TransformOp) op2;
                t.z().vectorAlongDimension(i, dimension).assign(t2.z());
            }


        }
        return op.z();
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, int dimension) {
        return (Accumulation) exec(op, dimension);
    }

    @Override
    public INDArray execAndReturn(ScalarOp op, int dimension) {
        return exec(op, dimension).z();
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
    private Pointer toArgs(Object[] extraArgs, String dataType) {
        if (dataType.equals("double")) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyDoublePointer();
            return Pointer.to(KernelFunctions.alloc(PointerUtil.toDoubles(extraArgs)).pointer());
        } else if (dataType.equals("float")) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyFloatPointer();
            return Pointer.to(KernelFunctions.alloc(PointerUtil.toFloats(extraArgs)).pointer());
        }
        throw new IllegalArgumentException("Illegal datatype");
    }


    private void invoke(Accumulation op) {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        Pointer xPointer = xBuffer.pointer().withByteOffset(xBuffer.elementSize() * op.x().offset());
        JCudaBuffer result;
        int resultLength = 1000;
        if (op.x().data().dataType() == DataBuffer.DOUBLE) {
            double[] resultBuffer = new double[resultLength];
            for (int i = 0; i < resultBuffer.length; i++)
                resultBuffer[i] = op.zero().doubleValue();
            result = new CudaDoubleDataBuffer(resultBuffer);


        } else {
            float[] resultBuffer = new float[resultLength];
            for (int i = 0; i < resultBuffer.length; i++)
                resultBuffer[i] = op.zero().floatValue();
            result = new CudaFloatDataBuffer(resultBuffer);
        }

        if (op.y() != null) {
            JCudaBuffer yBuffer = (JCudaBuffer) op.y().data();
            Pointer yPointer = yBuffer.pointer().withByteOffset(op.y().offset() * yBuffer.elementSize());

            //int n,int xOffset,int yOffset, double *dx, double *dy,int incx,int incy,double *result
            Pointer kernelParams = KernelFunctions.constructKernelParameters(
                    Pointer.to(new int[]{op.n()}),
                    Pointer.to(new int[]{op.x().offset()}),
                    Pointer.to(new int[]{op.y().offset()}),
                    Pointer.to(xPointer),
                    Pointer.to(yPointer),
                    Pointer.to(new int[]{op.x().majorStride()}),
                    Pointer.to(new int[]{op.y().majorStride()}),
                    toArgs(op.extraArgs(), getType(op)),
                    Pointer.to(result.pointer())
            );

            invokeFunction(op, kernelParams);
            setResultForOp(op, result.pointer());


        } else {
            //int n, int xOffset,double *dx,int incx,double result
            Pointer kernelParams = KernelFunctions.constructKernelParameters(
                    Pointer.to(new int[]{op.n()}),
                    Pointer.to(new int[]{op.x().offset()}),
                    Pointer.to(xPointer),
                    Pointer.to(new int[]{op.x().majorStride()}),
                    toArgs(op.extraArgs(), getType(op)),
                    Pointer.to(result.pointer())
            );

            invokeFunction(op, kernelParams);
            setResultForOp(op, result.pointer());


        }

        result.destroy();
    }


    private void invokeFunction(Op op, Pointer kernelParams) {
        String functionName = op instanceof TransformOp || op instanceof Accumulation ? op.name() + "_strided" : op.name();
        CUfunction func = KernelFunctionLoader.getInstance().getFunction(functionName, op.x().data().dataType() == DataBuffer.DOUBLE ? "double" : "float");
        if (func == null)
            throw new IllegalArgumentException("Function " + functionName + " with data type " + (op.x().data().dataType() == DataBuffer.DOUBLE ? "double does not exist" : "float does not exist"));
        int blocks = PointerUtil.getNumBlocks(op.n(), KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(op.n(), KernelFunctions.THREADS);

        KernelFunctions.invoke(
                blocks,
                threads,
                func
                , kernelParams, getType(op));


    }


    private void setResultForOp(Accumulation acc, Pointer resultPointer) {
        JCudaBuffer buff = (JCudaBuffer) acc.x().data();

        if (buff.dataType() == DataBuffer.DOUBLE) {
            double[] data = new double[1];
            Pointer get = Pointer.to(data);
            JCuda.cudaMemcpy(get, resultPointer, Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            acc.setCurrentResult(data[0]);
        } else {
            float[] data = new float[1];
            Pointer get = Pointer.to(data);
            JCuda.cudaMemcpy(get, resultPointer, Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            acc.setCurrentResult(data[0]);
        }
    }


    private void invoke(ScalarOp op) {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        Pointer xPointer = xBuffer.pointer().withByteOffset(op.x().offset() * xBuffer.elementSize());

        JCudaBuffer zBuffer = (JCudaBuffer) op.z().data();
        Pointer zPointer = zBuffer.pointer().withByteOffset(zBuffer.elementSize() * op.z().offset());

        if (op.y() != null) {
            JCudaBuffer yBuffer = (JCudaBuffer) op.y().data();
            Pointer yPointer = yBuffer.pointer().withByteOffset(yBuffer.elementSize() * op.y().offset());
            Pointer kernelParams = KernelFunctions.constructKernelParameters(
                    Pointer.to(new int[]{op.n()}),
                    Pointer.to(new int[]{op.x().offset()}),
                    Pointer.to(new int[]{op.y().offset()}),
                    Pointer.to(xPointer),
                    Pointer.to(yPointer),
                    Pointer.to(new int[]{op.x().majorStride()}),
                    Pointer.to(new int[]{op.y().majorStride()}),
                    toArgs(op.extraArgs(), getType(op)),
                    Pointer.to(zPointer)
            );

            invokeFunction(op, kernelParams);


        } else {
            //int n,int idx,double *dy,int incy,double *result
            //int n, int idx,double dx,double *dy,int incy,double *result

            Pointer kernelParams = KernelFunctions.constructKernelParameters(
                    Pointer.to(new int[]{op.n()}),
                    Pointer.to(new int[]{op.x().offset()}),
                    PointerUtil.getPointer(op),
                    Pointer.to(xPointer),
                    Pointer.to(new int[]{op.x().majorStride()}),
                    toArgs(op.extraArgs(), getType(op)),
                    Pointer.to(zPointer)
            );

            invokeFunction(op, kernelParams);


        }

    }


    private String getType(Op op) {
        return op.x().data().dataType() == DataBuffer.DOUBLE ? "double" : "float";
    }


    private void invoke(TransformOp op) {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        Pointer xPointer = xBuffer.pointer().withByteOffset(xBuffer.elementSize() * op.x().offset());

        JCudaBuffer zBuffer = (JCudaBuffer) op.z().data();
        Pointer zPointer = zBuffer.pointer().withByteOffset(zBuffer.elementSize() * op.z().offset());

        if (op.y() != null) {
            JCudaBuffer yBuffer = (JCudaBuffer) op.y().data();
            Pointer yPointer = yBuffer.pointer().withByteOffset(op.y().offset() * yBuffer.elementSize());
            /**
             * Construct pointer arguments in the following order:
             * n
             * offset,
             * pointer to buffer
             * increment,
             * extraArgs,
             * result
             */
            Pointer[] params = new Pointer[9];
            params[0] = Pointer.to(new int[]{op.n()});
            params[1] = Pointer.to(new int[]{op.x().offset()});
            params[2] = Pointer.to(new int[]{op.y().offset()});
            params[3] = Pointer.to(xPointer);
            params[4] = Pointer.to(yPointer);
            params[5] = Pointer.to(new int[]{op.x().majorStride()});
            params[6] = Pointer.to(new int[]{op.y().majorStride()});
            params[7] = toArgs(op.extraArgs(), getType(op));
            params[8] = Pointer.to(zPointer);

            Pointer kernelParameters = KernelFunctions.constructKernelParameters(params);
            invokeFunction(op, kernelParameters);


        } else {
            //int n,int idx,double *dy,int incy,double *result
            Pointer kernelParams = KernelFunctions.constructKernelParameters(
                    Pointer.to(new int[]{op.n()}),
                    Pointer.to(new int[]{op.x().offset()}),
                    Pointer.to(xPointer),
                    Pointer.to(new int[]{op.x().majorStride()}),
                    toArgs(op.extraArgs(), getType(op)),
                    Pointer.to(zPointer)
            );

            invokeFunction(op, kernelParams);


        }

    }


}


