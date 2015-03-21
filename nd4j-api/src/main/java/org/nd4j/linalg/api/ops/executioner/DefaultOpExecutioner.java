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

/**
 * Basic op executioner. Knows how to iterate over
 * the buffers of each respective ndarray and apply transformations
 *
 * @author Adam Gibson
 */
public class DefaultOpExecutioner implements OpExecutioner {
    @Override
    public Op exec(Op op) {
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

    private void apply(ScalarOp op, int c) {
        if (op.x() instanceof IComplexNDArray) {
            IComplexNDArray ndArray = (IComplexNDArray) op.z();
            ndArray.putScalar(c, op.op(((IComplexNDArray) op.x()).getComplex(c)));
        } else
            op.z().putScalar(c, op.op(op.x().getDouble(c)));
    }


    //apply a pairwise op to x and store the result
    private void apply(TransformOp op, int c) {
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
