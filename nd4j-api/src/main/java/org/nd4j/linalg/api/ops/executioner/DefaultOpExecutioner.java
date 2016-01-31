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

package org.nd4j.linalg.api.ops.executioner;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.parallel.ParallelExecutioner;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Basic op executioner. Knows how to iterate over
 * the buffers of each respective ndarray and apply transformations
 *
 * @author Adam Gibson
 */
public class DefaultOpExecutioner implements OpExecutioner {


    protected ExecutionMode executionMode = ExecutionMode.JAVA;
    protected TaskFactory taskFactory;

    public DefaultOpExecutioner() {
        taskFactory = Nd4j.getTaskFactory();
    }

    @Override
    public ParallelExecutioner parallelExecutioner() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Op exec(Op op) {
        if (op.isPassThrough()) {
            op.exec();
            return op;
        }

        if (op instanceof TransformOp) {
            doTransformOp((TransformOp) op);
        } else if (op instanceof Accumulation) {
            doAccumulationOp((Accumulation) op);
        } else if (op instanceof ScalarOp) {
            doScalarOp((ScalarOp) op);
        } else if (op instanceof IndexAccumulation) {
            doIndexAccumulationOp((IndexAccumulation) op);
        } else if (op instanceof BroadcastOp){
            doBroadcastOp((BroadcastOp) op);
        }
        return op;
    }

    @Override
    public INDArray execAndReturn(Op op) {
        if (op instanceof TransformOp) {
            return execAndReturn((TransformOp) op);
        } else if (op instanceof ScalarOp) {
            return execAndReturn((ScalarOp) op);
        } else if (op instanceof Accumulation) {
            return Nd4j.scalar(execAndReturn((Accumulation) op).getFinalResult());
        } else if (op instanceof IndexAccumulation) {
            return Nd4j.scalar(execAndReturn((IndexAccumulation) op).getFinalResult());
        }

        throw new IllegalArgumentException("Illegal type of op: " + op.getClass());
    }

    @Override
    public void iterateOverAllRows(Op op) {
        //column and row vectors should be treated the same
        if (op.x().isVector()) {
            //reset the op in case
            op.setX(op.x());
            if (op.y() != null)
                op.setY(op.y());
            op.setZ(op.z());
            exec(op);
        }
        //execute row wise
        else if (op.x().isMatrix()) {
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray original = (IComplexNDArray) op.x();
                IComplexNDArray originalZ = (IComplexNDArray) op.z();
                IComplexNDArray y = (IComplexNDArray) op.y();

                for (int i = 0; i < original.rows(); i++) {
                    IComplexNDArray row = original.slice(i);
                    IComplexNDArray zRow = originalZ.slice(i);
                    op.setX(row.dup());
                    op.setZ(zRow.dup());
                    if (y != null)
                        op.setY(y.slice(i));
                    exec(op);
                    originalZ.slice(i).assign(op.z());

                }
            } else {
                INDArray original = op.x();
                INDArray originalZ = op.z();
                INDArray y = op.y();

                for (int i = 0; i < original.rows(); i++) {
                    INDArray row = original.getRow(i);
                    INDArray zRow = originalZ.getRow(i);
                    op.setX(row.dup());
                    op.setZ(zRow.dup());
                    if (y != null)
                        op.setY(y.getRow(i).dup());
                    exec(op);
                    zRow.assign(op.z());
                }
            }
        } else {
            INDArray originalX = op.x();
            INDArray originalZ = op.z();
            for (int i = 0; i < originalX.slices(); i++) {
                INDArray slice = originalX.slice(i);
                INDArray zSlice = originalZ.slice(i);
                op.setX(slice);
                op.setZ(zSlice);
                iterateOverAllRows(op);
            }
        }
    }

    @Override
    public void iterateOverAllColumns(Op op) {
        if (op.x().isVector()) {
            exec(op);
        }
        //execute row wise
        else if (op.x().isMatrix() || op.x().isColumnVector()) {
            exec(op, 1);
        } else {
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray originalX = (IComplexNDArray) op.x();
                IComplexNDArray originalZ = (IComplexNDArray) op.z();
                IComplexNDArray y = (IComplexNDArray) op.y();
                for (int i = 0; i < op.x().slices(); i++) {
                    op.setX(originalX.getColumn(i));
                    op.setZ(originalZ.getColumn(i));
                    if (y != null)
                        op.setY(y.getColumn(i));
                    iterateOverAllColumns(op);
                }
            } else {
                INDArray originalX = op.x();
                INDArray originalZ = op.z();
                INDArray y = op.y();
                for (int i = 0; i < op.x().slices(); i++) {
                    op.setX(originalX.getColumn(i));
                    op.setZ(originalZ.getColumn(i));
                    if (y != null)
                        op.setY(y.getColumn(i));
                    iterateOverAllColumns(op);
                }
            }
        }
    }


    @Override
    public INDArray execAndReturn(TransformOp op) {
        Op result = exec(op);
        TransformOp t = (TransformOp) result;
        return t.z();
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
    public IndexAccumulation execAndReturn(IndexAccumulation op) {
        return (IndexAccumulation) exec(op);
    }

    @Override
    public INDArray execAndReturn(BroadcastOp op){
        return exec(op).z();
    }

    @Override
    public Op exec(Op op, int... dimension) {
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};

        if (op.isPassThrough()) {
            op.exec(dimension);
            return op;
        }

        if (op instanceof Accumulation || op instanceof IndexAccumulation) {
            //Overloaded exec(Accumulation,int...) and exec(IndexAccumulation,int...) should always be called instead of this
            throw new IllegalStateException("exec(Op,int...) should never be invoked for Accumulation/IndexAccumulation");
        } else if (op instanceof TransformOp) {
            execAndReturn((TransformOp) op,dimension);
            return op;
        } else if (op instanceof ScalarOp) {
            //Scalar op along dimension should be same as on the entire NDArray
            doScalarOp((ScalarOp) op);
            return op;
        } else {
            throw new UnsupportedOperationException("Unknown op type");
        }
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};

        if (op.isPassThrough()) {
            op.exec(dimension);
            return op.z();
        }


        if (dimension[0] == Integer.MAX_VALUE) {
            if (op.x() instanceof IComplexNDArray)
                return Nd4j.scalar(execAndReturn(op).getFinalResultComplex());
            return Nd4j.scalar(execAndReturn(op).getFinalResult().doubleValue());
        }

        if (op instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if (retShape.length == 1) {
                if (dimension[0] == 0)
                    retShape = new int[]{1, retShape[0]};
                else
                    retShape = new int[]{retShape[0], 1};
            } else if (retShape.length == 0) {
                retShape = new int[]{1, 1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).getFinalResultComplex();
                ret.putScalar(i, result);
            }

            if (ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));

            return ret;
        } else {
            Task<INDArray> task = taskFactory.getAccumulationTask(op, dimension);
            return task.invokeBlocking();
        }
    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};


        if (op.isPassThrough()) {
            op.exec(dimension);
            return op.z();
        }


        if (dimension[0] == Integer.MAX_VALUE) {
            return Nd4j.scalar(execAndReturn(op).getFinalResult());
        }

        if (op.x() instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if (retShape.length == 1) {
                if (dimension[0] == 0)
                    retShape = new int[]{1, retShape[0]};
                else
                    retShape = new int[]{retShape[0], 1};
            } else if (retShape.length == 0) {
                retShape = new int[]{1, 1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                int result = execAndReturn((IndexAccumulation) op2).getFinalResult();
                ret.putScalar(i, result);
            }

            if (ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));

            return ret;
        } else {
            Task<INDArray> task = taskFactory.getIndexAccumulationTask(op, dimension);
            return task.invokeBlocking();
        }
    }

    @Override
    public INDArray execAndReturn(TransformOp op, int... dimension) {
        if (dimension.length == op.x().rank()) {
            dimension = new int[]{Integer.MAX_VALUE};
        }

        if(op.isPassThrough()){
            op.exec(dimension);
            return op.z();
        }

        Task<Void> task = taskFactory.getTransformAction(op, dimension);
        task.invokeBlocking();
        return op.z();
    }

    @Override
    public INDArray execAndReturn(ScalarOp op, int... dimension) {
        return exec(op, dimension).z();
    }

    @Override
    public ExecutionMode executionMode() {
        return executionMode;
    }

    @Override
    public void setExecutionMode(ExecutionMode executionMode) {
        this.executionMode = executionMode;
    }

    protected void doTransformOp(TransformOp op) {
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

        if(x instanceof IComplexNDArray || y instanceof IComplexNDArray || z instanceof IComplexNDArray ){
            //Complex
            if(y != null){
                //Complex: x, y and/or z are complex
                if (z instanceof IComplexNDArray) {
                    IComplexNDArray cz = (IComplexNDArray) z;
                    if (x instanceof IComplexNDArray) {
                        IComplexNDArray cx = (IComplexNDArray) x;
                        if (y instanceof IComplexNDArray) {
                            IComplexNDArray cy = (IComplexNDArray) y;
                            //x,y,z all complex
                            for (int i = 0; i < op.n(); i++) {
                                cz.putScalar(i, op.op(cx.getComplex(i), cy.getComplex(i)));
                            }
                        } else {
                            //x,z complex, y real
                            for (int i = 0; i < op.n(); i++) {
                                cz.putScalar(i, op.op(cx.getComplex(i), y.getDouble(i)));
                            }
                        }
                    }
                } else {
                    //IComplexNDArray in, but real out
                    throw new UnsupportedOperationException("Invalid op: z is real but x.class=" + x.getClass().getName() + ", y.class=" + y.getClass().getName());
                }
            } else {
                //x and/or z are complex
                if (z instanceof IComplexNDArray) {
                    IComplexNDArray cz = (IComplexNDArray) z;
                    if (x instanceof IComplexNDArray) {
                        IComplexNDArray cx = (IComplexNDArray) x;
                        for (int i = 0; i < op.n(); i++) {
                            cz.putScalar(i, op.op(cx.getComplex(i)));
                        }
                    } else {
                        for (int i = 0; i < op.n(); i++) {
                            cz.putScalar(i, op.op(x.getDouble(i)));
                        }
                    }
                }
            }
        } else {
            Task task = taskFactory.getTransformAction(op);
            task.invokeBlocking();
        }
    }


    protected void doAccumulationOp(Accumulation op) {
        INDArray x = op.x();
        INDArray y = op.y();
        if (!(x instanceof IComplexNDArray) && !(y instanceof IComplexNDArray)) {
            Task<Double> task = taskFactory.getAccumulationTask(op);
            task.invokeBlocking();
        } else {
            //Complex
            if (y == null) {
                //Accumulation(x)
                //x must be complex
                IComplexNDArray cx = (IComplexNDArray) x;
                IComplexNumber accum = op.zeroComplex();
                for (int i = 0; i < op.n(); i++) {
                    accum = op.update(accum, cx.getComplex(i), i);
                }
                op.setFinalResultComplex(accum);
            } else {
                //Accumulation(x,y)
                if (!(x instanceof IComplexNDArray) || !(y instanceof IComplexNDArray)) {
                    throw new UnsupportedOperationException("Invalid input for accumulation op: x.class=" + x.getClass().getName() + ", y.class=" + y.getClass().getName());
                }
                IComplexNDArray cx = (IComplexNDArray) x;
                IComplexNDArray cy = (IComplexNDArray) y;
                IComplexNumber accum = op.zeroComplex();
                for (int i = 0; i < op.n(); i++) {
                    accum = op.update(accum, cx.getComplex(i), cy.getComplex(i));
                }
                op.setFinalResultComplex(accum);
            }
        }
    }

    protected void doScalarOp(ScalarOp op) {
        INDArray x = op.x();
        INDArray z = op.z();

        if (!(x instanceof IComplexNDArray) && !(z instanceof IComplexNDArray)) {
            Task task = taskFactory.getScalarAction(op);
            task.invokeBlocking();
        } else {
            //Complex
            if (z instanceof IComplexNDArray) {
                IComplexNDArray cz = (IComplexNDArray) z;
                if (x instanceof IComplexNDArray) {
                    IComplexNDArray cx = (IComplexNDArray) x;
                    for (int i = 0; i < op.n(); i++) cz.putScalar(i, op.op(cx.getComplex(i)));
                } else {
                    for (int i = 0; i < op.n(); i++) cz.putScalar(i, op.op(x.getDouble(i)));
                }
            } else {
                //Put complex into real -> not supported
                throw new UnsupportedOperationException("Scalar op with complex x but real z: not supported");
            }
        }
    }

    protected void doIndexAccumulationOp(IndexAccumulation op) {
        INDArray x = op.x();
        INDArray y = op.y();

        if (!(x instanceof IComplexNDArray) && !(y instanceof IComplexNDArray)) {
            Task<Pair<Double,Integer>> task = taskFactory.getIndexAccumulationTask(op);
            task.invokeBlocking();
        } else {
            //Complex
            if (y == null) {
                //IndexAccumulation(x)
                //x must be complex
                int accumIdx = -1;
                IComplexNDArray cx = (IComplexNDArray) x;
                IComplexNumber accum = op.zeroComplex();
                for (int i = 0; i < op.n(); i++) {
                    accumIdx = op.update(accum, accumIdx, cx.getComplex(i), i);
                    if (accumIdx == i) accum = op.op(cx.getComplex(i));
                }
                op.setFinalResult(accumIdx);
            } else {
                //IndexAccumulation(x,y)
                if (!(x instanceof IComplexNDArray) || !(y instanceof IComplexNDArray)) {
                    throw new UnsupportedOperationException("Invalid input for index accumulation op: x.class=" + x.getClass().getName() + ", y.class=" + y.getClass().getName());
                }
                int accumIdx = -1;
                IComplexNDArray cx = (IComplexNDArray) x;
                IComplexNDArray cy = (IComplexNDArray) y;
                IComplexNumber accum = op.zeroComplex();
                for (int i = 0; i < op.n(); i++) {
                    accumIdx = op.update(accum, accumIdx, cx.getComplex(i), cy.getComplex(i), i);
                    if (accumIdx == i) accum = op.op(cx.getComplex(i), cy.getComplex(i));
                }
                op.setFinalResult(accumIdx);
            }
        }
    }

    protected void doBroadcastOp(BroadcastOp op) {
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();
        if(!(x instanceof IComplexNDArray) && !(y instanceof IComplexNDArray) && !(z instanceof IComplexNDArray)) {
            taskFactory.getBroadcastOpAction(op).invokeBlocking();
        } else {
            //Complex vector op
            int nTensors = x.tensorssAlongDimension(op.getDimension());
            if(x instanceof IComplexNDArray){
                IComplexNDArray cx = (IComplexNDArray)x;
                IComplexNDArray cz = (IComplexNDArray)z;
                if(y instanceof IComplexNDArray){
                    IComplexNDArray cy = (IComplexNDArray)y;
                    for( int i = 0; i<nTensors; i++ ){
                        IComplexNDArray tx = (IComplexNDArray)cx.tensorAlongDimension(i,op.getDimension());
                        IComplexNDArray tz = (IComplexNDArray)cz.tensorAlongDimension(i,op.getDimension());
                        for( int j = 0; j < tx.length(); j++ ){
                            tz.put(j,Nd4j.scalar(op.op(tx.getComplex(j),cy.getComplex(j))));
                        }
                    }
                } else {
                    if(y == null) {
                        for (int i = 0; i < nTensors; i++) {
                            IComplexNDArray tx = (IComplexNDArray) cx.tensorAlongDimension(i,op.getDimension());
                            IComplexNDArray tz = (IComplexNDArray) cz.tensorAlongDimension(i,op.getDimension());
                            for( int j = 0; j < tz.length(); j++) {
                                tz.put(i,Nd4j.scalar(op.op(tx.getComplex(i))));
                            }
                        }
                    } else {
                        //Y is real
                        for( int i = 0; i < nTensors; i++) {
                            IComplexNDArray tx = (IComplexNDArray)cx.tensorAlongDimension(i,op.getDimension());
                            IComplexNDArray tz = (IComplexNDArray)cz.tensorAlongDimension(i,op.getDimension());
                            for( int j = 0; j<tx.length(); j++ ){
                                tz.put(j,Nd4j.scalar(op.op(tx.getComplex(j),y.getDouble(j))));
                            }
                        }
                    }
                }
            } else {
                throw new UnsupportedOperationException("Complex vector op with real x not supported/implemented");
            }
        }
    }
}
