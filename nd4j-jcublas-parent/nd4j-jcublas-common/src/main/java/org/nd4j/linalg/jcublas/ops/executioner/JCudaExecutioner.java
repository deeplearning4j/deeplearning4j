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



import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.complex.LinearViewComplexNDArray;
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
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.util.ArrayUtil;


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
        SimpleJCublas.init();
        dummyFloatPointer = KernelFunctions.alloc(new float[]{1});
        dummyDoublePointer =KernelFunctions.alloc(new double[]{1});
    }

    @Override
    public Op exec(Op op) {
        //linear views and oblong offsets can't be handled by the gpu (due to the way the buffers are interpeted as vectors)
        if(op.x() instanceof LinearViewNDArray || op.x() instanceof IComplexNDArray || op.x().offset() > 0 && op.x().shape().length >= 2)
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
    @Override
    public void iterateOverAllRows(Op op) {
        persist(op);
        if(op.x().isRowVector()) {
            //reset the op in case
            op.setX(op.x());
            if(op.y() != null)
                op.setY(op.y());
            op.setZ(op.z());
            exec(op);
        }
        //execute row wise
        else if(op.x().isMatrix() || op.x().isColumnVector()) {
            if(op.x() instanceof IComplexNDArray) {
                IComplexNDArray original = (IComplexNDArray) op.x();
                IComplexNDArray originalZ = (IComplexNDArray) op.z();
                IComplexNDArray y = (IComplexNDArray) op.y();

                for(int i = 0; i < original.rows(); i++) {
                    IComplexNDArray row = original.slice(i);
                    IComplexNDArray zRow = originalZ.slice(i);
                    IComplexNDArray rowRaveled = row.ravel();
                    IComplexNDArray zRowRaveled = zRow.ravel();
                    op.setX(rowRaveled);
                    op.setZ(zRowRaveled);
                    if(y != null)
                        op.setY(y.slice(i));
                    exec(op);
                    zRow.assign(op.z());

                }
            }
            else {
                INDArray original = op.x();
                INDArray originalZ = op.z();
                INDArray y = op.y();

                for(int i = 0; i < original.rows(); i++) {
                    INDArray row = original.getRow(i);
                    INDArray zRow = originalZ.getRow(i);
                    op.setX(row);
                    op.setZ(zRow);
                    if(y != null)
                        op.setY(y.getRow(i));
                    exec(op);
                    zRow.assign(op.z());
                }
            }

        }
        else {
            INDArray originalX = op.x();
            INDArray originalZ = op.z();
            for(int i = 0; i < originalX.slices(); i++) {
                INDArray slice = originalX.slice(i);
                INDArray zSlice = originalZ.slice(i);
                op.setX(slice);
                op.setZ(zSlice);
                iterateOverAllRows(op);
            }


        }

        //on the recursive case only free buffers where the buffer is the base case
        if(op.x().length() == op.x().data().length())
            unPersistAndFree(op);
    }

    @Override
    public void iterateOverAllColumns(Op op) {
        //persist for the duration of the usage of the buffer
        persist(op);
        if(op.x().isRowVector()) {
            exec(op);
        }
        //execute row wise
        else if(op.x().isMatrix() || op.x().isColumnVector()) {
            exec(op,1);
        }
        else {
            if(op.x() instanceof IComplexNDArray) {
                IComplexNDArray originalX = (IComplexNDArray) op.x();
                IComplexNDArray originalZ = (IComplexNDArray) op.z();
                IComplexNDArray y = (IComplexNDArray) op.y();
                for(int i = 0; i < op.x().slices(); i++) {
                    op.setX(originalX.getColumn(i));
                    op.setZ(originalZ.getColumn(i));
                    if(y != null)
                        op.setY(y.getColumn(i));
                    iterateOverAllColumns(op);
                }
            }
            else {
                INDArray originalX = op.x();
                INDArray originalZ = op.z();
                INDArray y = op.y();
                for(int i = 0; i < op.x().slices(); i++) {
                    op.setX(originalX.getColumn(i));
                    op.setZ(originalZ.getColumn(i));
                    if(y != null)
                        op.setY(y.getColumn(i));
                    iterateOverAllColumns(op);
                }
            }

        }

        //only free once the whole recursion has expired
        if(op.x().data().length() == op.x().length())
            unPersistAndFree(op);
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



    @Override
    public Op exec(Op op, int dimension) {
        persist(op);
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

        unPersistAndFree(op);

        return op;
    }

    @Override
    public INDArray exec(Accumulation op, int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            op.setX(op.x().linearView());
            if(op.x() instanceof IComplexNDArray)
                return Nd4j.scalar(execAndReturn(op).currentResultComplex());
            else
                return Nd4j.scalar(execAndReturn(op).currentResult());
        }
        else if(op.x().isScalar())
            return op.x();
        if(op.x() instanceof IComplexNDArray) {
            IComplexNDArray ret = Nd4j.createComplex(ArrayUtil.removeIndex(op.x().shape(), dimension));
            IComplexNDArray linear = ret.linearView();
            if(op.x().isRowVector()) {
                //same shape
                if(dimension == 0) {
                    //no reduction
                    return op.x();
                }
                else if(dimension == 1) {
                    return Nd4j.scalar(execAndReturn(op).currentResult());
                }
            }
            else if(op.x().isColumnVector()) {
                if(dimension == 0) {
                    return Nd4j.scalar(execAndReturn(op).currentResult());

                }
            }

            //cache the whole buffer on each piece until after the processing along
            //each dimension is done
            persist(op);
            for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).currentResultComplex();
                linear.putScalar(i, result);

            }

            unPersistAndFree(op);

            return ret;
        }
        else {
            if(op.x().isRowVector()) {
                //same shape
                if(dimension == 0) {
                    //no reduction
                    return op.x();
                }
                else if(dimension == 1) {
                    return Nd4j.scalar(execAndReturn(op).currentResult());
                }
            }
            else if(op.x().isColumnVector()) {
                if(dimension == 0) {
                    return Nd4j.scalar(execAndReturn(op).currentResult());

                }
            }

            INDArray ret = Nd4j.create(ArrayUtil.removeIndex(op.x().shape(), dimension));
            INDArray linear = ret.linearView();
            persist(op);
            for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                Number result = execAndReturn((Accumulation) op2).currentResult();
                linear.putScalar(i,result.doubleValue());

            }

            unPersistAndFree(op);


            return ret;

        }
    }


    @Override
    public INDArray execAndReturn(TransformOp op, int dimension) {
        //don't free device pointer until after operation is done
        persist(op);
        for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i, dimension);
            exec(op2);
            if (op instanceof TransformOp) {
                TransformOp t = op;
                TransformOp t2 = (TransformOp) op2;
                t.z().vectorAlongDimension(i, dimension).assign(t2.z());
            }


        }

        //don't cache buffers anymore
        unPersistAndFree(op);

        return op.z();
    }

    //save the pointers till they are all done being used
    private void persist(Op op) {
        persist(op.x());
        persist(op.y());
        persist(op.z());
    }

    //allow the pointers to be freed
    private void unPersistAndFree(Op op) {
        unPersistAndFree(op.x());
        unPersistAndFree(op.y());
        unPersistAndFree(op.z());
    }

    //persist() on a jcuda buffer forces the buffer uploaded to the gpu to be cached
    //this is useful for when you have arrays with offsets all viewing the same buffer
    //and intend on doing operations on slices of the same buffer
    private void persist(INDArray arr) {
        if(arr == null)
            return;
        arr.data().persist();
    }

    private void unPersistAndFree(INDArray buffer) {
        if(buffer == null)
            return;
        unPersistAndFree(buffer.data());
    }

    private void unPersistAndFree(DataBuffer buffer) {
        buffer.unPersist();
        //free the buffer after un persisting
        JCudaBuffer buf = (JCudaBuffer) buffer;
        buf.freeDevicePointer(0);
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
        if(!KernelFunctionLoader.getInstance().exists(op.name()))
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
                    op.x().majorStride(),
                    op.y().majorStride(),
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
            //NOTE THE STRIDE HERE. The stride should be set to 1.
            //The reason for this is because we only upload
            //the vector itself that is needed to the gpu
            //If you never need to use the whole array
            //with striding change this back to arr.majorStride()
            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    op.x(),
                    op.x().majorStride(),
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
        if(!KernelFunctionLoader.getInstance().exists(op.name()))
            super.exec(op);

        if (op.y() != null) {

            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.x(),
                    op.y(),
                    op.x().majorStride(),
                    op.y().majorStride(),
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
                    op.x().majorStride(),
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
        if(!KernelFunctionLoader.getInstance().exists(op.name()) || op.x() instanceof IComplexNDArray) {
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
                    op.x().majorStride(),
                    op.y().majorStride(),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(kernelParams).setResultArray(op.z())) {
                invokeFunction(op, kParams.getKernelParameters());
                kParams.close();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }


        } else {
            //int n,int idx,double *dy,int incy,double *result
            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    op.x(),
                    1,
                    toArgs(op.extraArgs(), getType(op)),
                    op.z()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(kernelParams).setResultArray(op.z())) {
                invokeFunction(op, kParams.getKernelParameters());
                kParams.close();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }
        }

    }
}


