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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;

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
        return exec(op,null);
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
        return execAndReturn(op,null);
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
        return execAndReturn(op,null);
    }

    @Override
    public Op exec(Op op, int dimension) {
        return exec(op,null,dimension);
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
        return execAndReturn(op,dimension,null);
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
        return execAndReturn(op,dimension,null);
    }

    private void invoke(Accumulation op,Object[] extraArgs) {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        Pointer xPointer = xBuffer.pointer();
        Pointer result = op.x().data().dataType() == DataBuffer.DOUBLE ? Pointer.to(new double[]{1}) : Pointer.to(new float[]{1});
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

                KernelFunctions.invoke(
                        op.n()
                        ,KernelFunctions.getFunction(op.name(),op.x().data().dataType() == DataBuffer.DOUBLE ? "double": "float")
                        ,kernelParams);
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
                KernelFunctions.invoke(
                        op.n()
                        ,KernelFunctions.getFunction(op.name(),op.x().data().dataType() == DataBuffer.DOUBLE ? "double": "float")
                        ,kernelParameters);

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
                        result
                );

                KernelFunctions.invoke(
                        op.n(),
                        KernelFunctions.getFunction(op.name(),op.x().data().dataType() == DataBuffer.DOUBLE ? "double": "float")
                        ,kernelParams);
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
                KernelFunctions.invoke(
                        op.n()
                        ,KernelFunctions.getFunction(op.name(),op.x().data().dataType() == DataBuffer.DOUBLE ? "double": "float")
                        ,kernelParameters);
            }
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

                KernelFunctions.invoke(
                        op.n()
                        ,KernelFunctions.getFunction(op.name(),op.x().data().dataType() == DataBuffer.DOUBLE ? "double": "float")
                        ,kernelParams);


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
                KernelFunctions.invoke(
                        op.n()
                        ,KernelFunctions.getFunction(op.name(),op.x().data().dataType() == DataBuffer.DOUBLE ? "double": "float")
                        ,kernelParameters);

            }



        }

        else {
            if(extraArgs == null || extraArgs.length < 1) {
                //int n,int idx,double *dy,int incy,double *result
                Pointer kernelParams = KernelFunctions.constructKernelParameters(
                        Pointer.to(new int[]{op.n()}),
                        Pointer.to(new int[]{op.x().offset()}),
                        Pointer.to(new int[]{op.y().offset()}),
                        Pointer.to(xPointer),
                        Pointer.to(new int[]{op.x().majorStride()}),
                        Pointer.to(zPointer)
                );

                KernelFunctions.invoke(
                        op.n()
                        ,KernelFunctions.getFunction(op.name(),op.x().data().dataType() == DataBuffer.DOUBLE ? "double": "float")
                        ,kernelParams);
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
                KernelFunctions.invoke(
                        op.n()
                        ,KernelFunctions.getFunction(op.name(),op.x().data().dataType() == DataBuffer.DOUBLE ? "double": "float")
                        ,kernelParameters);

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


