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


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.complex.LinearViewComplexNDArray;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.LinearViewNDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.DefaultParallelExecutionProvider;
import org.nd4j.linalg.api.parallel.DefaultParallelExecutioner;
import org.nd4j.linalg.api.parallel.ParallelExecutionProvider;
import org.nd4j.linalg.api.parallel.ParallelExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * Basic op executioner. Knows how to iterate over
 * the buffers of each respective ndarray and apply transformations
 *
 * @author Adam Gibson
 */
public class DefaultOpExecutioner implements OpExecutioner {

    protected ExecutionMode executionMode = ExecutionMode.JAVA;
    protected ParallelExecutionProvider parallelExecutionProvider;
    protected ParallelExecutioner executorService;

    public DefaultOpExecutioner() {
        String provider = System.getProperty(ParallelExecutionProvider.EXECUTOR_SERVICE_PROVIDER,DefaultParallelExecutionProvider.class.getName());
        try {
            Class<? extends ParallelExecutionProvider> executorServiceProvider = (Class<? extends ParallelExecutionProvider>) Class.forName(provider);
            this.parallelExecutionProvider = executorServiceProvider.newInstance();

        } catch (Exception e) {
            e.printStackTrace();
        }

        this.executorService = parallelExecutionProvider.getService();

    }

    @Override
    public ParallelExecutioner parallelExecutioner() {
        return executorService;
    }

    @Override
    public Op exec(Op op) {
        checkOp(op);

        if(op.isPassThrough()) {
            op.exec();
            return op;
        }
        if (op instanceof TransformOp) {
            final TransformOp t = (TransformOp) op;
            //make assumption x and z are same type
            if (!op.x().getClass().equals(t.z().getClass()) && !(op.x() instanceof LinearViewNDArray) && !(t.z() instanceof LinearViewNDArray))
                throw new IllegalArgumentException("Illegal operation. Origin and output ndarray must be same types. op.x was " + op.x().getClass().getName() + " while t.z was " + t.z().getClass().getName());
            if(op.y() != null && Shape.opIsWholeBufferWithMatchingStrides(op)) {
                for(int i = 0; i < op.n(); i++) {
                    op.z().data().put(i, op.op(op.x().data().getDouble(i), op.y().data().getDouble(i)));
                }

            }
            else if(op.y() != null && Shape.opIsWholeBufferWithMatchingStrides(op)) {
                int xStride = op.x().ordering() == 'f' ? op.x().stride(-1) : op.x().stride(0);
                int yStride = op.y().ordering() == 'f' ? op.y().stride(-1) : op.y().stride(0);
                int zStride = op.z().ordering() == 'f' ? op.z().stride(-1) : op.z().stride(0);
                for(int c = 0; c < op.n(); c ++)
                    op.z().data().put(c * zStride, op.op(op.x().data().getDouble(c * xStride),op.y().data().getDouble(c * yStride)));
            }
            else if(Shape.opIsWholeBufferWithMatchingStrides(op)){
                for(int i = 0; i < op.n(); i++) {
                    op.z().data().put(i, op.op(op.x().data().getDouble(i)));
                }
            }
              /*
            else if(op.y() != null && Shape.opIsWithMatchingStrides(op)) {
                int xStride = op.x().ordering() == 'f' ? op.x().stride(-1) : op.x().stride(0);
                int yStride = op.x().ordering() == 'f' ? op.y().stride(-1) : op.y().stride(0);
                int zStride = op.z().ordering() == 'f' ? op.z().stride(-1) : op.z().stride(0);
                for(int c = 0; c < op.n(); c ++)
                    op.z().data().put(op.z().offset() + c * zStride, op.op(op.x().data().getDouble(op.x().offset() + c * xStride),op.y().data().getDouble(op.y().offset() + c * yStride)));
            }

         else if(Shape.opIsWithMatchingStrides(op)) {
                int xStride = op.x().ordering() == 'f' ? op.x().stride(-1) : op.x().stride(0);
                int zStride = op.z().ordering() == 'f' ? op.z().stride(-1) : op.z().stride(0);
                for(int c = 0; c < op.n(); c ++)
                    op.z().data().put(op.z().offset() + c * zStride, op.op(op.x().data().getDouble(op.x().offset() + c * xStride)));
            }
*/

            else if(op.y() != null) {
                if(Arrays.equals(op.x().shape(),op.y().shape())) {
                    Shape.iterate(op.x(), new CoordinateFunction() {
                        @Override
                        public void process(int[]... coord) {
                            apply(t,coord[0],coord[0]);
                        }
                    });
                }
                else
                    Shape.iterate(op.x(), op.y(), new CoordinateFunction() {
                        @Override
                        public void process(int[]... coord) {
                            apply(t,coord[0],coord[1]);
                        }
                    });

            }

            else {
                NdIndexIterator iter = new NdIndexIterator(op.x().shape());
                for (int c = 0; c < op.n(); c++) {
                    apply(t, iter.next());
                }
            }

        }
        else if (op instanceof Accumulation) {
            Accumulation accumulation = (Accumulation) op;
            if(op.y() != null && Shape.opIsWholeBufferWithMatchingStrides(op)) {
                for(int i = 0; i < op.n(); i++) {
                    accumulation.update(op.op(op.x().data().getDouble(i), op.y().data().getDouble(i)));
                }

            }
       /*     else if(op.y() != null && Shape.opIsWithMatchingStrides(op)) {
                int xStride = op.x().ordering() == 'f' ? op.x().stride(-1) : op.x().stride(0);
                int yStride = op.y().ordering() == 'f' ? op.y().stride(-1) : op.y().stride(0);
                for(int i = 0; i < op.n(); i++) {
                    accumulation.update(op.op(op.x().getDouble(op.x().offset() + i * xStride), op.y().getDouble(op.y().offset() + i * yStride)));
                }
            }*/
            else if(Shape.opIsWholeBufferWithMatchingStrides(op)) {
                for(int i = 0; i < op.n(); i++) {
                    accumulation.update(op.op(op.x().data().getDouble(i)));
                }
            }


            else if(!(op.x() instanceof IComplexNDArray)) {
                if(op.y() != null) {
                    INDArray xLinear = op.x().reshape(1,op.x().length());
                    INDArray yLinear = op.y().reshape(1,op.y().length());
                    for(int i = 0; i < op.n(); i++) {
                        accumulation.update(op.op(xLinear.getDouble(0,i),yLinear.getDouble(0,i)));
                    }
                }
                else {
                    INDArray xLinear = op.x().reshape(1,op.x().length());
                    for(int i = 0; i < op.n(); i++) {
                        accumulation.update(op.op(xLinear.getDouble(0,i)));
                    }
                }

            }



            else {
                for (int c = 0; c < op.n(); c++) {
                    apply(accumulation, c);
                }
            }

        }

        else if (op instanceof ScalarOp) {
            ScalarOp scalarOp = (ScalarOp) op;
            if(op.isPassThrough())
                return scalarOp;
            INDArray zLinear = op.z();
            INDArray xLinear = op.x();
            if(Shape.opIsWholeBufferWithMatchingStrides(op)) {
                for(int c = 0; c < op.n(); c ++)
                    zLinear.data().put(c, op.op(xLinear.data().getDouble(c)));
            }

        /*    else if(Shape.opIsWithMatchingStrides(op)) {
                int xStride = op.x().ordering() == 'f' ? op.x().stride(-1) : op.x().stride(0);
                int zStride = op.z().ordering() == 'f' ? op.z().stride(-1) : op.z().stride(0);
                for(int c = 0; c < op.n(); c ++)
                    zLinear.data().put(zLinear.offset() + c * zStride, op.op(xLinear.data().getDouble(xLinear.offset() + c * xStride)));
            }*/

            else if (op.x() instanceof IComplexNDArray) {
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
        //do op along all dimensions
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        if(op.isPassThrough()) {
            op.exec(dimension);
            return op;
        }

        if(dimension.length == 1)
            return exec(op,dimension[0]);
        else {
            //only accumulate along a particular dimension
            if (op instanceof Accumulation) {
                throw new IllegalStateException("Should never be invoked");
            }


            parallelExecutioner().execBasedOnArraysAlongDimension(op.x(), op, this, dimension);

            return op;
        }
    }

    protected Op exec(Op op,int dimension) {
        if(op.isPassThrough()) {
            op.exec();
            return op;
        }

        //only accumulate along a particular dimension
        if (op instanceof Accumulation) {
            Accumulation a = (Accumulation) op;
            return exec(a);
        }
/*
        for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i, dimension);
            exec(op2);
            if (op instanceof TransformOp) {
                TransformOp t = (TransformOp) op;
                TransformOp t2 = (TransformOp) op2;
                t.z().tensorAlongDimension(i, dimension).assign(t2.z());
            }


        }*/
        parallelExecutioner().execBasedOnArraysAlongDimension(op.x(),op,this,dimension);

        return op;
    }


    protected void checkOp(Op op) {
        if(        op.x() instanceof LinearViewNDArray
                || op.y() != null && op.y() instanceof LinearViewNDArray
                || op.z() != null && op.z() instanceof LinearViewNDArray
                || op.x() != null && op.x() instanceof LinearViewComplexNDArray
                || op.y() != null && op.y() instanceof LinearViewComplexNDArray
                || op.z() != null && op.z() instanceof LinearViewComplexNDArray ||
                op.x() != null && op.x().isScalar() || op.y() != null && op.y().isScalar() || op.z() != null && op.z().isScalar())
            return;

    }


    @Override
    public INDArray exec(Accumulation op, int...dimension) {
        //do op along all dimensions
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        if(op.isPassThrough()) {
            op.exec(dimension);
            return op.z();
        }


        if(dimension[0] == Integer.MAX_VALUE) {
            if(op.x() instanceof IComplexNDArray)
                return Nd4j.scalar(execAndReturn(op).currentResultComplex());
            return Nd4j.scalar(execAndReturn(op).currentResult().doubleValue());
        }

        if(op instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(),dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            IComplexNDArray linear = ret;
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).currentResultComplex();
                linear.putScalar(i, result);

            }

            if(ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));


            return ret;
        }

        else
            return parallelExecutioner().execBasedOnArraysAlongDimension(op.x(), op, this,dimension);




    }


    @Override
    public INDArray execAndReturn(final TransformOp op, int...dimension) {
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};
        if(dimension.length == 1)
            return execAndReturnVector(op,dimension[0]);
        else {
            parallelExecutioner().execBasedOnArraysAlongDimension(op.x(), op, this, dimension);
            return op.z();

        }
    }

    protected INDArray execAndReturnVector(TransformOp op,int dimension) {
        if(op.isPassThrough()) {
            op.exec(dimension);
            return op.z();
        }

        parallelExecutioner().execBasedOnArraysAlongDimension(op.x(),op,this,dimension);

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
    //apply a pairwise op to x and store the result
    private void apply(TransformOp op, int[] c,int[] c2) {
        if(op.isPassThrough())
            return;
        if (op.y() != null) {
            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x();
                IComplexNDArray complexZ = (IComplexNDArray) op.z();

                IComplexNumber curr = complexX.getComplex(c);
                if (op.y() instanceof IComplexNDArray) {
                    IComplexNDArray complexY = (IComplexNDArray) op.y();
                    complexZ.putScalar(c, op.op(curr, complexY.getComplex(c)));
                } else
                    complexZ.putScalar(c, op.op(curr, op.y().getDouble(c)));
            }
            //x is real
            else {
                INDArray zLinear = op.z();
                INDArray xLinear = op.x();
                INDArray yLinear = op.y();
                zLinear.putScalar(c, op.op(xLinear.getDouble(c),yLinear.getDouble(c2)));

            }

        }

        else {

            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x();
                IComplexNDArray complexZ = (IComplexNDArray) op.z();

                if (op.y() instanceof IComplexNDArray)
                    complexZ.putScalar(c, op.op(complexX.getComplex(c)));

                else
                    complexZ.putScalar(c, op.op(complexX.getComplex(c)));
            }
            //x is real
            else
                op.z().putScalar(c, op.op(op.x().getDouble(c)));
        }

    }




    //apply a pairwise op to x and store the result
    private void apply(TransformOp op, int[] c) {
        if(op.isPassThrough())
            return;
        if (op.y() != null) {
            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x();
                IComplexNDArray complexZ = (IComplexNDArray) op.z();

                IComplexNumber curr = complexX.getComplex(c);
                if (op.y() instanceof IComplexNDArray) {
                    IComplexNDArray complexY = (IComplexNDArray) op.y();
                    complexZ.putScalar(c, op.op(curr, complexY.getComplex(c)));
                } else
                    complexZ.putScalar(c, op.op(curr, op.y().getDouble(c)));
            }
            //x is real
            else {
                INDArray zLinear = op.z();
                INDArray xLinear = op.x();
                INDArray yLinear = op.y();
                zLinear.putScalar(c, op.op(xLinear.getDouble(c),yLinear.getDouble(c)));

            }

        }

        else {

            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x();
                IComplexNDArray complexZ = (IComplexNDArray) op.z();

                if (op.y() instanceof IComplexNDArray)
                    complexZ.putScalar(c, op.op(complexX.getComplex(c)));

                else
                    complexZ.putScalar(c, op.op(complexX.getComplex(c)));
            }
            //x is real
            else
                op.z().putScalar(c, op.op(op.x().getDouble(c)));
        }

    }





    private void apply(Accumulation op, int x) {
        if(op.isPassThrough())
            return;

        if (op.y() != null) {

            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x();
                IComplexNDArray complexY = (IComplexNDArray) op.y();
                IComplexNumber curr = complexX.getComplex(x);
                if (op.y() instanceof IComplexNDArray)
                    op.update(op.op(curr, complexY.getComplex(x)));

                else
                    op.update(op.op(curr, op.y().getDouble(x)));
            }
            //x is real
            else
                op.update(op.op(op.x().getDouble(x), op.y().getDouble(x)));
        }

        else {
            //x is complex, y could be complex or real
            if (op.x() instanceof IComplexNDArray) {
                IComplexNDArray complexX = (IComplexNDArray) op.x();
                op.update(op.op(complexX.getComplex(x)));
            }
            else
                op.update(op.op(op.x().getDouble(x)));
        }


    }


}
