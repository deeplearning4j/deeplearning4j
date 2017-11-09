/*-
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

package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for accumulation, initiates the initial entry
 * with respect to the child class. Also contains baseline fields
 * for the over all field with accumulation.
 *
 * @author Adam Gibson
 */
public abstract class BaseAccumulation extends BaseOp implements Accumulation {
    protected Number finalResult;
    protected IComplexNumber finalResultComplex;
    protected boolean applyFinalTransform = true;
    protected boolean isComplex = false;

    public BaseAccumulation(SameDiff sameDiff,
                            DifferentialFunction i_v,
                            int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            this.args = new DifferentialFunction[] {i_v};
            this.dimensions = dimensions;
            this.shape = Shape.getReducedShape(i_v.getResultShape(),dimensions);
            f().validateDifferentialFunctionsameDiff(i_v);
            addAsNewVertexId();
            f().addFunctionEdges(this);
            this.opState.setAxes(dimensions);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

    }

    public BaseAccumulation(SameDiff sameDiff,
                            DifferentialFunction i_v,
                            DifferentialFunction i_v2,
                            int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            this.args = new DifferentialFunction[] {i_v,i_v2};
            this.dimensions = dimensions;
            this.shape = Shape.getReducedShape(i_v.getResultShape(),dimensions);
            f().validateDifferentialFunctionsameDiff(i_v);
            f().validateDifferentialFunctionsameDiff(i_v2);
            addAsNewVertexId();
            f().addFunctionEdges(this);
            this.opState.setAxes(dimensions);


        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

    }



    public BaseAccumulation() {}




    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     * @param n the number of elements
     */
    public BaseAccumulation(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        init();
        //      if (y != null)
        //            LinAlgExceptions.assertSameLength(x, y);
        //LinAlgExceptions.assertSameLength(x, z);

    }

    public BaseAccumulation(INDArray x, INDArray y, long n) {
        this(x, y, x, n);
    }

    public BaseAccumulation(INDArray x) {
        this(x, null, x, x.lengthLong());
    }

    public BaseAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.lengthLong());
        //if (y != null)
        //    LinAlgExceptions.assertSameLength(x, y);
    }

    public BaseAccumulation(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }




    private void init() {
        if (z == null || x == z)
            init(x, y, x, x.lengthLong());
        else
            init(x, y, z, x.lengthLong());
    }

    @Override
    public INDArray noOp() {
        if (z != null && x != z)
            return z().assign(x);
        else
            return x().dup(x().ordering());
    }

    @Override
    public boolean applyFinalTransform() {
        return applyFinalTransform;
    }

    @Override
    public void setApplyFinalTransform(boolean applyFinalTransform) {
        this.applyFinalTransform = applyFinalTransform;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        numProcessed++;
        return origin;
    }

    @Override
    public float op(float origin, float other) {
        numProcessed++;
        return origin;
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin;
    }

    @Override
    public double op(double origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public float op(float origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public double zeroDouble() {
        return 0.0;
    }

    @Override
    public float zeroFloat() {
        return 0.0f;
    }

    @Override
    public float zeroHalf() {
        return 0.0f;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0, 0.0);
    }

    @Override
    public long numProcessed() {
        return numProcessed;
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            this.extraArgs = new Object[] {zeroDouble()};
        } else if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            this.extraArgs = new Object[] {zeroFloat()};
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            this.extraArgs = new Object[] {zeroHalf()};
        }
    }

    @Override
    public double combineSubResults(double first, double second) {
        return update(first, second);
    }

    @Override
    public float combineSubResults(float first, float second) {
        return update(first, second);
    }

    @Override
    public IComplexNumber combineSubResults(IComplexNumber first, IComplexNumber second) {
        return update(first, second);
    }

    @Override
    public double getAndSetFinalResult(double accum) {
        this.finalResult = accum;
        if (z() != null && z.isScalar()) {
            z.assign(accum);
        }
        return accum;
    }

    @Override
    public float getAndSetFinalResult(float accum) {
        this.finalResult = accum;
        if (z() != null && z.isScalar()) {
            z.assign(accum);
        }
        return accum;
    }

    @Override
    public IComplexNumber getAndSetFinalResult(IComplexNumber accum) {
        this.finalResultComplex = accum;
        return accum;
    }

    @Override
    public double calculateFinalResult(double accum, long n) {
        return accum;
    }

    @Override
    public float calculateFinalResult(float accum, long n) {
        return accum;
    }

    @Override
    public Number currentResult() {
        return finalResult;
    }

    @Override
    public void setFinalResult(Number number) {
        this.finalResult = number;
        if (z() != null && z.isScalar()) {
            z.assign(number);
        }
    }

    @Override
    public Type opType() {
        if(args() != null && args().length > 1)
            return Type.REDUCE3;
        return Type.REDUCE;
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        ret.add(Shape.getReducedShape(arg().getResultShape(),dimensions));
        return ret;
    }


    @Override
    public void setFinalResultComplex(IComplexNumber number) {
        this.finalResultComplex = number;
    }


    @Override
    public Number getFinalResult() {
        return finalResult;
    }

    @Override
    public IComplexNumber getFinalResultComplex() {
        return finalResultComplex;
    }

    /**
     * This method is only used for Distance functions
     * @return
     */
    public boolean isComplexAccumulation() {
        return isComplex;
    }
}
