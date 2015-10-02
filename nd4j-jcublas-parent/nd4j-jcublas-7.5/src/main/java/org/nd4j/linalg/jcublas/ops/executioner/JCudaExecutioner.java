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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.LinearViewNDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.jcublas.util.PointerUtil;


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
    }

    @Override
    public Op exec(Op op) {
        checkOp(op);
        //linear views and oblong offsets can't be handled by the gpu (due to the way the buffers are interpeted as vectors)
        if(op.x() instanceof LinearViewNDArray
                || op.x() instanceof IComplexNDArray
                || op.x().offset() > 0 && op.x().shape().length >= 2
                || executionMode() == ExecutionMode.JAVA || op.isPassThrough())
            return super.exec(op);

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

    private JCudaBuffer dummyDouble() {
        return dummyDoublePointer;
    }

    private JCudaBuffer dummyFloat() {
        return dummyFloatPointer;
    }

    @Override
    public INDArray execAndReturn(TransformOp op) {
        invoke(op);
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


    private void invoke(Accumulation op)  {
        checkOp(op);
        if(!KernelFunctionLoader.getInstance().exists(op.name()) || executionMode() == ExecutionMode.JAVA || op.isPassThrough())
            super.exec(op);


        INDArray result = Nd4j.create(2);


        if (op.y() != null) {

            //int n,int xOffset,int yOffset, double *dx, double *dy,int incx,int incy,double *result
            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.x(),
                    op.y(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    BlasBufferUtil.getBlasStride(op.y()),
                    toArgs(op.extraArgs(), getType(op)),
                    result
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(kernelParams).setResultOp(op, result)) {
                invokeFunction(op, kParams.getKernelParameters());
                kParams.close();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }




        } else {
            //int n, int xOffset,double *dx,int incx,double result
            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    result
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(kernelParams).setResultOp(op, result)) {
                invokeFunction(op, kParams.getKernelParameters());
                kParams.close();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }



        }
    }


    private void invokeFunction(Op op, Object... kernelParams) {
        /**
         * Invoke a cuda kernel by name. This will be wrt the function name.
         * Functions that are accumulations or transforms have names that end with _strided.
         *
         */
        String functionName = op instanceof TransformOp || op instanceof Accumulation ? op.name() + "_strided" : op.name();
        int blocks = PointerUtil.getNumBlocks(op.n(), KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(op.n(), KernelFunctions.THREADS);
        KernelFunctions.invoke(
                blocks
                ,threads
                ,functionName
                ,getType(op)
                ,kernelParams);

    }






    private void invoke(ScalarOp op) {
        checkOp(op);
        if(!KernelFunctionLoader.getInstance().exists(op.name())  || executionMode() == ExecutionMode.JAVA)
            super.exec(op);

        if (op.y() != null) {

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
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(kernelParams).setResultArray(op.z())) {
                invokeFunction(op, kParams.getKernelParameters());
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }



        } else {
            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    PointerUtil.getPointer(op),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(kernelParams).setResultArray(op.z())) {
                invokeFunction(op, kParams.getKernelParameters());
                kParams.close();
            }

            catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }




        }

    }


    private String getType(Op op) {
        return op.x().data().dataType() == DataBuffer.Type.DOUBLE ? "double" : "float";
    }


    private void invoke(TransformOp op) {
        if(!KernelFunctionLoader.getInstance().exists(op.name()) || op.x() instanceof IComplexNDArray || op.isPassThrough()) {
            super.exec(op);
            return;
        }
        if (op.y() != null) {

            /**
             * Construct pointer arguments in the following order:
             * n
             * offset,
             * pointer to buffer
             * increment,
             * extraArgs,
             * result
             */

            Object[] kernelParams = new Object[]{
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
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(kernelParams).setResultArray(op.z())) {
                invokeFunction(op, kParams.getKernelParameters());
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }


        } else {
            //int n,int idx,double *dy,int incy,double *result
            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(kernelParams).setResultArray(op.z())) {
                invokeFunction(op, kParams.getKernelParameters());
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }
        }

    }
}


