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
import org.nd4j.linalg.api.ops.impl.accum.Variance;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Basic op executioner. Knows how to iterate over
 * the buffers of each respective ndarray and apply transformations
 *
 * @author Adam Gibson
 */
public  class DefaultOpExecutioner implements OpExecutioner {


    protected ExecutionMode executionMode = ExecutionMode.JAVA;

    public DefaultOpExecutioner() {
    }


    @Override
    public Op exec(Op op) {
        if (op.isPassThrough()) {
            op.exec();
            return op;
        }

        throw new IllegalStateException("Java computation no longer supported");
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
    public Accumulation execAndReturn(Variance op, boolean biasCorrected) {
        return null;
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
        } else if (op instanceof ScalarOp) {
            //Scalar op along dimension should be same as on the entire NDArray
            throw new IllegalStateException("Java computation no longer supported");
        }else if (op instanceof TransformOp) {
            throw new UnsupportedOperationException("Executing transform ops along a dimension should be done via exec special");
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

            // FIXME: this is wrong, it breaks shapeInfo immutability
            if (ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));

            return ret;
        }

        throw new UnsupportedOperationException("Java computation no longer supported");
    }

    @Override
    public INDArray exec(Variance accumulation, boolean biasCorrected, int... dimension) {
        accumulation.setBiasCorrected(biasCorrected);
        return exec(accumulation,dimension);
    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        throw new UnsupportedOperationException("Operation should use exec special");

    }

    public ExecutionMode executionMode() {
        return executionMode;
    }

    @Override
    public void setExecutionMode(ExecutionMode executionMode) {
        this.executionMode = executionMode;
    }




    @Override
    public INDArray exec(BroadcastOp broadcast, int... dimension) {
        if (dimension.length == broadcast.x().rank()) {
            dimension = new int[]{Integer.MAX_VALUE};
        }

        if(broadcast.isPassThrough()){
            broadcast.exec(dimension);
            return broadcast.z();
        }

        throw new IllegalStateException("Java computation no longer supported");

    }
}
