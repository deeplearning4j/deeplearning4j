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

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;

/**
 * Basic op executioner. Knows how to iterate over
 * the buffers of each respective ndarray and apply transformations
 *
 * @author Adam Gibson
 */
public class DefaultOpExecutioner implements OpExecutioner {
    @Override
    public Op exec(Op op)  {
        return exec(op,null);
    }

    @Override
    public Op exec(Op op, Object[] extraArgs) {
        if(op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            //make assumption x and z are same type
            if(!op.x().getClass().equals(t.z().getClass()))
                throw new IllegalArgumentException("Illegal operation. Origin and output ndarray must be same types");
            if(op.x() != null && op.y() != null) {
                for(int c = 0,
                            x = op.x().offset()
                            ,y = op.y().offset(); c < op.n();
                    x+= op.x().majorStride(),y += op.y().majorStride(),c++) {
                    if(extraArgs != null)
                        apply(t,c,x,y,extraArgs);
                    else
                        apply(t,c,x,y);

                }

            }

            else {
                for(int c = 0,
                            x = op.x().offset(); c < op.n();
                    x+= op.x().majorStride(),c++) {
                    if(extraArgs != null)
                        apply(t,c,x,extraArgs);
                    else
                        apply(t,c,x);

                }

            }
        }

        else if(op instanceof Accumulation) {
            Accumulation accumulation = (Accumulation) op;
            if(op.x() != null && op.y() != null) {
                for(int c = 0,
                            x = op.x().offset()
                            ,y = op.y().offset(); c < op.n();
                    x+= op.x().majorStride(),y += op.y().majorStride(),c++) {
                    if(extraArgs != null)
                        apply(accumulation,x,y,extraArgs);
                    else
                        apply(accumulation,x,y);

                }

            }

            else {
                for(int c = 0, x = op.x().offset(); c < op.n();
                    x+= op.x().majorStride(),c++) {
                    if(extraArgs != null)
                        apply(accumulation,x,extraArgs);

                    else
                        apply(accumulation,x);

                }

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
    public INDArray execAndReturn(TransformOp op, Object[] extraArgs) {
        Op result = exec(op,extraArgs);
        TransformOp t = (TransformOp) result;
        return t.z();
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, Object[] extraArgs) {
        return (Accumulation) exec(op,extraArgs);
    }

    @Override
    public Accumulation execAndReturn(Accumulation op) {
        return (Accumulation) exec(op);
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
        return execAndReturn(op,extraArgs);
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, int dimension) {
        return execAndReturn(op,dimension,null);
    }


    //apply a singular op to x and store the result
    private void apply(TransformOp op,int c,int x,Object[] extraArgs) {
        DataBuffer xData = op.x().data();
        DataBuffer zData = op.z().data();

        //x is complex, y could be complex or real
        if(op.x() instanceof IComplexNDArray) {
            IComplexNumber curr = xData.getComplex(x);
            if(op.y() instanceof IComplexNDArray)
                zData.put(c,op.op(curr,extraArgs));

            else
                zData.put(c,op.op(curr,extraArgs));
        }
        //x is real
        else
            zData.put(c,op.op(xData.getDouble(x),extraArgs));
    }

    //apply a pairwise op to x and store the result
    private void apply(TransformOp op,int c,int x,int y,Object[] extraArgs) {
        DataBuffer xData = op.x().data();
        DataBuffer yData = op.y().data();
        DataBuffer zData = op.z().data();

        //x is complex, y could be complex or real
        if(op.x() instanceof IComplexNDArray) {
            IComplexNumber curr = xData.getComplex(x);
            if(op.y() instanceof IComplexNDArray)
                zData.put(c,op.op(curr, yData.getComplex(y),extraArgs));

            else
                zData.put(c,op.op(curr, op.y().getDouble(y),extraArgs));
        }
        //x is real
        else
            zData.put(c, op.op(xData.getDouble(x), yData.getDouble(y)));

    }

    private void apply(Accumulation op,int x,Object[] extraArgs) {
        DataBuffer xData = op.x().data();

        //x is complex, y could be complex or real
        if(op.x() instanceof IComplexNDArray) {
            op.update(op.op(xData.getComplex(x),extraArgs));
        }
        else
            op.update(op.op(xData.getDouble(x),extraArgs));

    }

    private void apply(Accumulation op,int x,int y,Object[] extraArgs) {
        DataBuffer xData = op.x().data();
        DataBuffer yData = op.y().data();


        //x is complex, y could be complex or real
        if(op.x() instanceof IComplexNDArray) {
            IComplexNumber curr = xData.getComplex(x);
            if(op.y() instanceof IComplexNDArray)
                op.update(op.op(curr, yData.getComplex(y)));

            else
                op.update(op.op(curr, op.y().getDouble(y),extraArgs));
        }
        //x is real
        else
            op.update(op.op(xData.getDouble(x), yData.getDouble(y),extraArgs));
    }




    //apply a singular op to x and store the result
    private void apply(TransformOp op,int c,int x) {
        DataBuffer xData = op.x().data();
        DataBuffer zData = op.z().data();

        //x is complex, y could be complex or real
        if(op.x() instanceof IComplexNDArray) {
            IComplexNumber curr = xData.getComplex(x);
            if(op.y() instanceof IComplexNDArray)
                zData.put(c,op.op(curr));

            else
                zData.put(c,op.op(curr));
        }
        //x is real
        else
            zData.put(c,op.op(xData.getDouble(x)));
    }

    //apply a pairwise op to x and store the result
    private void apply(TransformOp op,int c,int x,int y) {
        DataBuffer xData = op.x().data();
        DataBuffer yData = op.y().data();
        DataBuffer zData = op.z().data();

        //x is complex, y could be complex or real
        if(op.x() instanceof IComplexNDArray) {
            IComplexNumber curr = xData.getComplex(x);
            if(op.y() instanceof IComplexNDArray)
                zData.put(c,op.op(curr, yData.getComplex(y)));

            else
                zData.put(c,op.op(curr, op.y().getDouble(y)));
        }
        //x is real
        else
            zData.put(c, op.op(xData.getDouble(x), yData.getDouble(y)));

    }

    private void apply(Accumulation op,int x) {
        DataBuffer xData = op.x().data();

        //x is complex, y could be complex or real
        if(op.x() instanceof IComplexNDArray) {
            op.update(op.op(xData.getComplex(x)));
        }
        else
            op.update(op.op(xData.getDouble(x)));

    }

    private void apply(Accumulation op,int x,int y) {
        DataBuffer xData = op.x().data();
        DataBuffer yData = op.y().data();


        //x is complex, y could be complex or real
        if(op.x() instanceof IComplexNDArray) {
            IComplexNumber curr = xData.getComplex(x);
            if(op.y() instanceof IComplexNDArray)
                op.update(op.op(curr, yData.getComplex(y)));

            else
                op.update(op.op(curr, op.y().getDouble(y)));
        }
        //x is real
        else
            op.update(op.op(xData.getDouble(x), yData.getDouble(y)));
    }

}
