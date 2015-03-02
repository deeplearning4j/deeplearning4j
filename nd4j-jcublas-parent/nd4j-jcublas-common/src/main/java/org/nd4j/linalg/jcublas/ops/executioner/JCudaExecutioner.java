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
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.jcublas.JCublas;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;

import static jcuda.driver.JCudaDriver.cuMemAlloc;

/**
 * JCuda executioner.
 *
 * Runs ops directly on the gpu
 *
 * @author Adam Gibson
 */
public class JCudaExecutioner implements OpExecutioner {
    @Override
    public Op exec(Op op) {
        return exec(op,op.extraArgs());
    }

    @Override
    public Op exec(Op op, Object[] extraArgs) {
        if(op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t,extraArgs);
        }
        else if(op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            invoke(acc,extraArgs);
        }
        return op;
    }

    @Override
    public INDArray execAndReturn(TransformOp op) {
        return execAndReturn(op,op.extraArgs());
    }

    @Override
    public INDArray execAndReturn(TransformOp op, Object[] extraArgs) {
        invoke(op,extraArgs);
        return op.z();
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, Object[] extraArgs) {
        return (Accumulation) exec(op,extraArgs);
    }

    @Override
    public Accumulation execAndReturn(Accumulation op) {
        return execAndReturn(op,op.extraArgs());
    }

    @Override
    public Op exec(Op op, int dimension) {
        return exec(op,op.extraArgs(),dimension);
    }

    @Override
    public Op exec(Op op, Object[] extraArgs, int dimension) {
        //only accumulate along a particular dimension
        if(op instanceof Accumulation) {
            Accumulation a = (Accumulation) op;
            return exec(a,extraArgs);
        }
        for(int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i,dimension);
            exec(op2,extraArgs);
            if(op instanceof TransformOp) {
                TransformOp t = (TransformOp) op;
                TransformOp t2 = (TransformOp) op2;
                t.z().vectorAlongDimension(i,dimension).assign(t2.z());
            }


        }
        return op;
    }

    @Override
    public INDArray execAndReturn(TransformOp op, int dimension) {
        return execAndReturn(op,dimension,op.extraArgs());
    }

    @Override
    public INDArray execAndReturn(TransformOp op, int dimension, Object[] extraArgs) {
        for(int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i,dimension);
            exec(op2,extraArgs);
            if(op instanceof TransformOp) {
                TransformOp t =  op;
                TransformOp t2 = (TransformOp) op2;
                t.z().vectorAlongDimension(i,dimension).assign(t2.z());
            }


        }
        return op.z();
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, int dimension, Object[] extraArgs) {
        return (Accumulation) exec(op,extraArgs,dimension);
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, int dimension) {
        return execAndReturn(op,dimension,op.extraArgs());
    }

    private void invoke(Accumulation op,Object[] extraArgs) {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        Pointer xPointer = xBuffer.pointer();
        CUdeviceptr result = null;
        int maxThreads = 128;
        int maxBlocks = 64;
        int blocks = getNumBlocks(op.n(), maxBlocks, maxThreads);
        int threads = getNumThreads(op.n(),maxBlocks,maxThreads);

        if(op.x().data().dataType() == DataBuffer.DOUBLE) {
            double[] resultBuffer = new double[1024 * Sizeof.DOUBLE];
            resultBuffer[0] = op.zero().doubleValue();
            result = new CUdeviceptr();
            cuMemAlloc(result, 1024 * Sizeof.DOUBLE);
            JCublas.cublasSetVector(1024, Sizeof.DOUBLE,Pointer.to(resultBuffer),1,result,1);


        }
        else {
            float[] resultBuffer = new float[1024 * Sizeof.FLOAT];
            resultBuffer[0] = op.zero().floatValue();
            result = new CUdeviceptr();
            cuMemAlloc(result, 1024 * Sizeof.FLOAT);
            JCublas.cublasSetVector(1024, Sizeof.FLOAT,Pointer.to(resultBuffer),1,result,1);
        }

        if(op.y() != null) {
            JCudaBuffer yBuffer = (JCudaBuffer) op.y().data();
            Pointer yPointer = yBuffer.pointer();

            if(extraArgs == null || extraArgs.length < 1) {
                //int n,int xOffset,int yOffset, double *dx, double *dy,int incx,int incy,double *result
                Pointer kernelParams = KernelFunctions.constructKernelParameters(
                        Pointer.to(new int[]{op.n()}),
                        Pointer.to(new int[]{op.x().offset()}),
                        Pointer.to(new int[]{op.y().offset()}),
                        Pointer.to(xPointer),
                        Pointer.to(yPointer),
                        Pointer.to(new int[]{op.x().majorStride()}),
                        Pointer.to(new int[]{op.y().majorStride()}),
                        Pointer.to(result)
                );

                invokeFunction(op, kernelParams,threads,blocks);
                setResultForOp(op,result);


            }
            else {
                /**
                 * Construct pointer arguments in the following order:
                 * n
                 * offset,
                 * pointer to buffer
                 * increment,
                 * extraArgs,
                 * result
                 */
                Pointer[] results = new Pointer[5 + extraArgs.length];
                results[0] = Pointer.to(new int[]{op.n()});
                results[1] = Pointer.to(new int[]{op.x().offset()});
                results[2] = Pointer.to(xPointer);
                results[3] = Pointer.to(new int[]{op.x().majorStride()});

                addPointers(4,results,extraArgs);
                results[results.length - 1] = Pointer.to(result);

                Pointer kernelParameters = KernelFunctions.constructKernelParameters(results);
                invokeFunction(op, kernelParameters,threads,blocks);
                setResultForOp(op,result);

            }
        }
        else {
            //int n, int xOffset,double *dx,int incx,double result
            if(extraArgs == null || extraArgs.length < 1) {
                Pointer kernelParams = KernelFunctions.constructKernelParameters(
                        Pointer.to(new int[]{op.n()}),
                        Pointer.to(new int[]{op.x().offset()}),
                        Pointer.to(xPointer),
                        Pointer.to(new int[]{op.x().majorStride()}),
                        Pointer.to(result)
                );

                invokeFunction(op,kernelParams,threads,blocks);
                setResultForOp(op,result);

            }
            else {
                /**
                 * Construct pointer arguments in the following order:
                 * n
                 * offset,
                 * pointer to buffer
                 * increment,
                 * extraArgs,
                 * result
                 */
                Pointer[] results = new Pointer[5 + extraArgs.length];
                results[0] = Pointer.to(new int[]{op.n()});
                results[1] = Pointer.to(new int[]{op.x().offset()});
                results[2] = Pointer.to(xPointer);
                results[3] = Pointer.to(new int[]{op.x().majorStride()});

                addPointers(4,results,extraArgs);
                results[results.length - 1] = Pointer.to(result);

                Pointer kernelParameters = KernelFunctions.constructKernelParameters(results);
                invokeFunction(op,kernelParameters,threads,blocks);
                setResultForOp(op,result);

            }
        }

        if(result != null)
            JCublas.cublasFree(result);

    }


    /**
     * Returns the power of 2 that is equal to or greater than x
     *
     * @param x The input
     * @return The next power of 2
     */
    private static int nextPow2(int x)  {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }


    /**
     * Compute the number of blocks that should be used for the
     * given input size and limits
     *
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of blocks
     */
    private  int getNumBlocks(int n, int maxBlocks, int maxThreads)  {
        int blocks = 0;
        int threads = getNumThreads(n, maxBlocks, maxThreads);
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    /**
     * Compute the number of threads that should be used for the
     * given input size and limits
     *
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of threads
     */
    private  int getNumThreads(int n, int maxBlocks, int maxThreads) {
        int threads = 0;
        threads = (n < maxThreads * 2) ? nextPow2((n + 1)/ 2) : maxThreads;
        return threads;
    }


    private void invokeFunction(Op op,Pointer kernelParams,int...extraParams) {
        String functionName = op.name() + "_strided";
        CUfunction func =  KernelFunctionLoader.getInstance().getFunction(functionName, op.x().data().dataType() == DataBuffer.DOUBLE ? "double" : "float");
        if(func == null)
            throw new IllegalArgumentException("Function " + functionName + " with data type " + (op.x().data().dataType() == DataBuffer.DOUBLE ? "double does not exist" : "float does not exist"));
        if(KernelFunctions.isReduce(functionName)) {
             //specify threads and blocks
              KernelFunctions.invokeReduce(
                    extraParams[0],extraParams[1],
                    func, kernelParams);
        }
        else
            KernelFunctions.invoke(
                    op.n(),
                   func
                    , kernelParams);


    }




    private void setResultForOp(Accumulation acc,Pointer resultPointer) {
        JCudaBuffer buff = (JCudaBuffer) acc.x().data();

        if(buff.dataType() == DataBuffer.DOUBLE) {
            double[] data = new double[1];
            Pointer get = Pointer.to(data);
            JCublas.cublasGetVector(
                    1
                    ,buff.elementSize(),
                    resultPointer
                    ,1
                    ,get
                    ,1
            );
            acc.setCurrentResult(data[0]);
        }
        else {
            float[] data = new float[1];
            Pointer p = Pointer.to(data);
            JCublas.cublasGetVector(
                    1,
                    buff.elementSize(),
                    resultPointer,
                    1,
                    p,
                    1);
            acc.setCurrentResult(data[0]);
        }
    }


    private void invoke(TransformOp op,Object[] extraArgs) {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        Pointer xPointer = xBuffer.pointer();

        JCudaBuffer zBuffer = (JCudaBuffer) op.z().data();

        Pointer zPointer = zBuffer.pointer();

        if(op.y() != null) {
            JCudaBuffer yBuffer = (JCudaBuffer) op.y().data();
            Pointer yPointer = yBuffer.pointer();
            if(extraArgs == null || extraArgs.length < 1) {
                //int n,int xOffset,int yOffset, double *dx, double *dy,int incx,int incy,double *result
                Pointer kernelParams = KernelFunctions.constructKernelParameters(
                        Pointer.to(new int[]{op.n()}),
                        Pointer.to(new int[]{op.x().offset()}),
                        Pointer.to(new int[]{op.y().offset()}),
                        Pointer.to(xPointer),
                        Pointer.to(yPointer),
                        Pointer.to(new int[]{op.x().majorStride()}),
                        Pointer.to(new int[]{op.y().majorStride()}),
                        Pointer.to(zPointer)
                );

                invokeFunction(op,kernelParams);



            }
            else {
                /**
                 * Construct pointer arguments in the following order:
                 * n
                 * offset,
                 * pointer to buffer
                 * increment,
                 * extraArgs,
                 * result
                 */
                Pointer[] results = new Pointer[5 + extraArgs.length];
                results[0] = Pointer.to(new int[]{op.n()});
                results[1] = Pointer.to(new int[]{op.x().offset()});
                results[2] = Pointer.to(xPointer);
                results[3] = Pointer.to(new int[]{op.x().majorStride()});

                addPointers(4,results,extraArgs);
                results[results.length - 1] = Pointer.to(zPointer);

                Pointer kernelParameters = KernelFunctions.constructKernelParameters(results);
                invokeFunction(op,kernelParameters);


            }



        }

        else {
            if(extraArgs == null || extraArgs.length < 1) {
                //int n,int idx,double *dy,int incy,double *result
                Pointer kernelParams = KernelFunctions.constructKernelParameters(
                        Pointer.to(new int[]{op.n()}),
                        Pointer.to(new int[]{op.x().offset()}),
                        Pointer.to(xPointer),
                        Pointer.to(new int[]{op.x().majorStride()}),
                        Pointer.to(zPointer)
                );

                invokeFunction(op,kernelParams);

            }

            else {
                /**
                 * Construct pointer arguments in the following order:
                 * n
                 * offset,
                 * pointer to buffer
                 * increment,
                 * extraArgs,
                 * result
                 */
                Pointer[] results = new Pointer[5 + extraArgs.length];
                results[0] = Pointer.to(new int[]{op.n()});
                results[1] = Pointer.to(new int[]{op.x().offset()});
                results[2] = Pointer.to(xPointer);
                results[3] = Pointer.to(new int[]{op.x().majorStride()});

                addPointers(4,results,extraArgs);
                results[results.length - 1] = Pointer.to(zPointer);

                Pointer kernelParameters = KernelFunctions.constructKernelParameters(results);
                invokeFunction(op,kernelParameters);


            }
        }

    }


    private void addPointers(int start,Pointer[] results,Object[] extraArgs) {
        //start at the extra args slot and iterate over each argument
        for(int i = start,count = 0; count < extraArgs.length; i++,count++) {
            Object o = extraArgs[count];
            if(o instanceof Integer) {
                results[i] = Pointer.to(new int[]{Integer.valueOf(o.toString())});
            }
            else if(o instanceof Double) {
                results[i] = Pointer.to(new double[]{Double.valueOf(o.toString())});
            }
            else if(o instanceof Float) {
                results[i] = Pointer.to(new float[]{Float.valueOf(o.toString())});
            }
        }

    }




}


