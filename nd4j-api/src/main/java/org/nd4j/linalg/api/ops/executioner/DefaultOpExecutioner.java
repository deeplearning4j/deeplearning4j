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

package org.nd4j.linalg.api.ops.executioner;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
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
        if(op.isPassThrough()) {
            op.exec();
            return op;
        }
        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            //make assumption x and z are same type
            if (!op.x().getClass().equals(t.z().getClass()))
                throw new IllegalArgumentException("Illegal operation. Origin and output ndarray must be same types");
            for (int c = 0; c < op.n(); c++) {
                apply(t, c);

            }
        } else if (op instanceof Accumulation) {
            Accumulation accumulation = (Accumulation) op;
            for (int c = 0; c < op.n(); c++)
                apply(accumulation, c);
        } else if (op instanceof ScalarOp) {
            ScalarOp scalarOp = (ScalarOp) op;
            for (int c = 0; c < op.n(); c++) {
                apply(scalarOp, c);

            }
        }


        return op;
    }

    @Override
    public void iterateOverAllRows(Op op) {
        if(op.x().isRowVector()) {
            //reset the op in case
            op.setX(op.x());
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

                for(int i = 0; i < op.x().rows(); i++) {
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
    }

    @Override
    public void iterateOverAllColumns(Op op) {
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
    public INDArray exec(Accumulation op, int dimension) {
        if(dimension == Integer.MAX_VALUE) {
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
    public INDArray execAndReturn(ScalarOp op, int dimension) {
        return exec(op, dimension).z();
    }

    private void apply(ScalarOp op, int c) {
        if(op.isPassThrough())
            return;

        if (op.x() instanceof IComplexNDArray) {
            IComplexNDArray ndArray = (IComplexNDArray) op.z();
            ndArray.putScalar(c, op.op(((IComplexNDArray) op.x()).getComplex(c)));
        } else
            op.z().putScalar(c, op.op(op.x().getDouble(c)));
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
            else
                op.z().linearView().putScalar(c, op.op(op.x().linearView().getDouble(c), op.y().linearView().getDouble(c)));

        } else {

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
        } else {
            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x().linearView();
                op.update(op.op(complexX.getComplex(x)));
            } else
                op.update(op.op(op.x().linearView().getDouble(x)));
        }


    }


}
