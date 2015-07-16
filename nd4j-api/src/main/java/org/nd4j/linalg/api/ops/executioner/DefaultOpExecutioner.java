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

import com.google.common.base.Preconditions;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.complex.LinearViewComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.LinearViewNDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.exception.BlasOpErrorMessage;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Basic op executioner. Knows how to iterate over
 * the buffers of each respective ndarray and apply transformations
 *
 * @author Adam Gibson
 */
public class DefaultOpExecutioner implements OpExecutioner {
    @Override
    public Op exec(Op op) {
        checkOp(op);

        if(op.isPassThrough()) {
            op.exec();
            return op;
        }
        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            //make assumption x and z are same type
            if (!op.x().getClass().equals(t.z().getClass()) && !(op.x() instanceof LinearViewNDArray) && !(t.z() instanceof LinearViewNDArray))
                throw new IllegalArgumentException("Illegal operation. Origin and output ndarray must be same types. op.x was " + op.x().getClass().getName() + " while t.z was " + t.z().getClass().getName());
            for (int c = 0; c < op.n(); c++) {
                apply(t, c);
            }
        }
        else if (op instanceof Accumulation) {
            Accumulation accumulation = (Accumulation) op;
            for (int c = 0; c < op.n(); c++)
                apply(accumulation, c);
        } else if (op instanceof ScalarOp) {
            ScalarOp scalarOp = (ScalarOp) op;
            if(op.isPassThrough())
                return scalarOp;
            INDArray zLinear = op.z().linearView();
            INDArray xLinear = op.x().linearView();

            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray ndArray = (IComplexNDArray) op.z();
                for(int c = 0; c < op.n(); c++)
                    ndArray.putScalar(c, op.op(((IComplexNDArray) op.x()).getComplex(c)));
            }
            else {
                for(int c = 0; c < op.n(); c++)
                    zLinear.putScalar(c, op.op(xLinear.getDouble(c)));

            }
        }


        return op;
    }

    @Override
    public INDArray execAndReturn(Op op) {
        if(op instanceof TransformOp) {
            return execAndReturn((TransformOp) op);
        }
        else if(op instanceof ScalarOp) {
            return execAndReturn((ScalarOp) op);
        }
        else if(op instanceof Accumulation) {
            return Nd4j.scalar(execAndReturn((Accumulation) op).currentResult());
        }

        throw new IllegalArgumentException("Illegal type of op " + op.getClass());
    }

    @Override
    public void iterateOverAllRows(Op op) {
        //column and row vectors should be treated the same
        if(op.x().isVector()) {
            //reset the op in case
            op.setX(op.x());
            if(op.y() != null)
                op.setY(op.y());
            op.setZ(op.z());
            exec(op);
        }
        //execute row wise
        else if(op.x().isMatrix()) {
            if(op.x() instanceof IComplexNDArray) {
                IComplexNDArray original = (IComplexNDArray) op.x();
                IComplexNDArray originalZ = (IComplexNDArray) op.z();
                IComplexNDArray y = (IComplexNDArray) op.y();

                for(int i = 0; i < original.rows(); i++) {
                    IComplexNDArray row = original.slice(i);
                    IComplexNDArray zRow = originalZ.slice(i);
                    op.setX(row.dup());
                    op.setZ(zRow.dup());
                    if(y != null)
                        op.setY(y.slice(i));
                    exec(op);
                    originalZ.slice(i).assign(op.z());

                }
            }
            else {
                INDArray original = op.x();
                INDArray originalZ = op.z();
                INDArray y = op.y();

                for(int i = 0; i < original.rows(); i++) {
                    INDArray row = original.getRow(i);
                    INDArray zRow = originalZ.getRow(i);
                    op.setX(row.dup());
                    op.setZ(zRow.dup());
                    if(y != null)
                        op.setY(y.getRow(i).dup());
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
    }

    @Override
    public void iterateOverAllColumns(Op op) {
        if(op.x().isVector()) {
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
    public Op exec(Op op, int...dimension) {
        if(dimension.length == 1)
            return exec(op,dimension[0]);
        else {
            //only accumulate along a particular dimension
            if (op instanceof Accumulation) {
                Accumulation a = (Accumulation) op;
                return exec(a);
            }
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                exec(op2);
                if (op instanceof TransformOp) {
                    TransformOp t = (TransformOp) op;
                    TransformOp t2 = (TransformOp) op2;
                    t.z().tensorAlongDimension(i, dimension).assign(t2.z());
                }


            }
            return op;
        }
    }

    protected Op exec(Op op,int dimension) {
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


    protected void checkOp(Op op) {
        if(op.x() instanceof LinearViewNDArray || op.y() instanceof LinearViewNDArray || op.z() instanceof LinearViewNDArray || op.x() instanceof LinearViewComplexNDArray || op.y() instanceof LinearViewComplexNDArray || op.z() instanceof LinearViewComplexNDArray)
            return;
        int xStride = op.x().offset() + (op.n() - op.x().elementStride()) * (op.x() instanceof IComplexNDArray ? BlasBufferUtil.getBlasStride(op.x()) / 2 : BlasBufferUtil.getBlasStride(op.x()));
        int zStride = op.z().offset() + (op.n() - op.z().elementStride()) * (op.z() instanceof IComplexNDArray ? BlasBufferUtil.getBlasStride(op.z()) / 2 : BlasBufferUtil.getBlasStride(op.z()));

        if(op.y() != null) {
            int yStride = op.y().offset() + (op.n() - op.y().elementStride()) * (op.y() instanceof IComplexNDArray ? BlasBufferUtil.getBlasStride(op.y()) / 2 : BlasBufferUtil.getBlasStride(op.y()));
            Preconditions.checkArgument(xStride < op.x().data().length(),new BlasOpErrorMessage(op).toString());
            Preconditions.checkArgument(yStride < op.y().data().length(),new BlasOpErrorMessage(op).toString());
            Preconditions.checkArgument(zStride < op.z().data().length(),new BlasOpErrorMessage(op).toString());

        }
        else {
            Preconditions.checkArgument(xStride < op.x().data().length(),new BlasOpErrorMessage(op).toString());
            Preconditions.checkArgument(zStride < op.z().data().length(),new BlasOpErrorMessage(op).toString());

        }
    }


    @Override
    public INDArray exec(Accumulation op, int...dimension) {
        if(dimension.length == 1)
            return execVector(op,dimension[0]);
        else {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(),dimension);
            //ensure vector is proper shape
            if(retShape.length == 1)
                retShape = new int[] {1,retShape[0]};
            if(op instanceof IComplexNDArray) {
                IComplexNDArray ret = Nd4j.complexZeros(retShape);
                IComplexNDArray linear = ret.linearView();
                for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                    Op op2 = op.opForDimension(i, dimension);
                    IComplexNumber result = execAndReturn((Accumulation) op2).currentResultComplex();
                    linear.putScalar(i, result);

                }

                return ret;
            }
            else {
                INDArray ret = Nd4j.zeros(retShape);
                INDArray linear = ret.linearView();
                for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                    Op op2 = op.opForDimension(i, dimension);
                    double result = execAndReturn((Accumulation) op2).currentResult().doubleValue();
                    linear.putScalar(i, result);

                }

                return ret;
            }

        }
    }


    protected INDArray execVector(Accumulation op,int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            op.setX(op.x().linearView());
            if(op.y() != null)
                op.setY(op.y().linearView());
            op.setZ(op.z().linearView());

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
                //row vector
                else if(dimension == 1) {
                    //make a row vector
                    return Nd4j.scalar(execAndReturn(op).currentResult());

                }
            }

            for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).currentResultComplex();
                linear.putScalar(i, result);

            }

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
                //row vector
                else if(dimension == 1) {
                    //make a row vector
                    return op.z().transpose();

                }
            }

            if(op.x().isMatrix() || op.x().isVector()) {
                int[] shape = ArrayUtil.removeIndex(op.x().shape(), dimension);
                if(shape.length < 2)
                    shape = new int[]{1,shape[0]};
                INDArray ret = Nd4j.create(shape);
                INDArray linear = ret.linearView();

                for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
                    Op op2 = op.opForDimension(i, dimension);
                    Number result = execAndReturn((Accumulation) op2).currentResult();
                    linear.putScalar(i,result.doubleValue());

                }

                return ret;
            }
            else {
                INDArray ret = Nd4j.create(ArrayUtil.removeIndex(op.x().shape(), dimension));
                INDArray linear = ret.linearView();
                for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
                    Op op2 = op.opForDimension(i, dimension);
                    Number result = execAndReturn((Accumulation) op2).currentResult();
                    linear.putScalar(i,result.doubleValue());

                }

                return ret;

            }



        }

    }

    @Override
    public INDArray execAndReturn(TransformOp op, int...dimension) {
        if(dimension.length == 1)
            return execAndReturnVector(op,dimension[0]);
        else {
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                exec(op2);
                op.z().tensorAlongDimension(i, dimension).assign(op2.z());
            }

            return op.z();

        }
    }

    protected INDArray execAndReturnVector(TransformOp op,int dimension) {
        for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i, dimension);
            exec(op2);
            op.z().vectorAlongDimension(i, dimension).assign(op2.z());
        }
        return op.z();
    }


    @Override
    public INDArray execAndReturn(ScalarOp op, int... dimension) {
        return exec(op, dimension).z();
    }




    //apply a pairwise op to x and store the result
    private void apply(TransformOp op, int c) {
        if(op.isPassThrough())
            return;
        if (op.y() != null) {
            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x().linearView();
                IComplexNDArray complexZ = (IComplexNDArray) op.z().linearView();

                IComplexNumber curr = complexX.getComplex(c);
                if (op.y() instanceof IComplexNDArray) {
                    IComplexNDArray complexY = (IComplexNDArray) op.y().linearView();
                    complexZ.putScalar(c, op.op(curr, complexY.getComplex(c)));
                } else
                    complexZ.putScalar(c, op.op(curr, op.y().getDouble(c)));
            }
            //x is real
            else {
                INDArray zLinear = op.z().linearView();
                INDArray xLinear = op.x().linearView();
                INDArray yLinear = op.y().linearView();
                zLinear.putScalar(c, op.op(xLinear.getDouble(c),yLinear.getDouble(c)));

            }

        }

        else {

            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x().linearView();
                IComplexNDArray complexZ = (IComplexNDArray) op.z().linearView();

                if (op.y() instanceof IComplexNDArray)
                    complexZ.putScalar(c, op.op(complexX.getComplex(c)));

                else
                    complexZ.putScalar(c, op.op(complexX.getComplex(c)));
            }
            //x is real
            else
                op.z().linearView().putScalar(c, op.op(op.x().linearView().getDouble(c)));
        }

    }

    private void apply(Accumulation op, int x) {
        if(op.isPassThrough())
            return;

        if (op.y() != null) {

            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x().linearView();
                IComplexNDArray complexY = (IComplexNDArray) op.y().linearView();
                IComplexNumber curr = complexX.getComplex(x);
                if (op.y() instanceof IComplexNDArray)
                    op.update(op.op(curr, complexY.getComplex(x)));

                else
                    op.update(op.op(curr, op.y().linearView().getDouble(x)));
            }
            //x is real
            else
                op.update(op.op(op.x().linearView().getDouble(x), op.y().linearView().getDouble(x)));
        }

        else {
            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x().linearView();
                op.update(op.op(complexX.getComplex(x)));
            }
            else
                op.update(op.op(op.x().linearView().getDouble(x)));
        }


    }


}
