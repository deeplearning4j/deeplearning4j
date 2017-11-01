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

package org.nd4j.linalg.api.ops.impl.accum;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Calculate the max over a vector
 *
 * @author Adam Gibson
 */
public class Max extends BaseAccumulation {
    public Max(SameDiff sameDiff, DifferentialFunction i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Max(SameDiff sameDiff, DifferentialFunction i_v, DifferentialFunction i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Max() {}

    public Max(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

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
    public Max(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Max(INDArray x) {
        super(x);
    }

    public Max(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public double op(double origin, double other) {
        return origin;
    }

    @Override
    public float op(float origin, float other) {
        return origin;
    }

    @Override
    public double update(double accum, double x) {
        return FastMath.max(accum, x);
    }

    @Override
    public double update(double accum, double x, double y) {
        return FastMath.max(accum, x);
    }

    @Override
    public float update(float accum, float x) {
        return FastMath.max(accum, x);
    }

    @Override
    public float update(float accum, float x, float y) {
        return FastMath.max(accum, x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x) {
        return (accum.absoluteValue().doubleValue() > x ? accum : Nd4j.createComplexNumber(x, 0));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x, double y) {
        return (accum.absoluteValue().doubleValue() > x ? accum : Nd4j.createComplexNumber(x, 0));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x) {
        return (accum.absoluteValue().doubleValue() > x.absoluteValue().doubleValue() ? accum : x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        return (accum.absoluteValue().doubleValue() > x.absoluteValue().doubleValue() ? accum : x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return (accum.absoluteValue().doubleValue() > x.absoluteValue().doubleValue() ? accum : x);
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String name() {
        return "max";
    }

    @Override
    public double zeroDouble() {
        return -Double.MAX_VALUE;
    }

    @Override
    public float zeroHalf() {
        return -65503.0f;
    }

    @Override
    public float zeroFloat() {
        return -Float.MAX_VALUE;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Max(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Max(x.vectorAlongDimension(index, dimension));
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Max(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Max(x.tensorAlongDimension(index, dimension));
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        f().validateDifferentialFunctionsameDiff(i_v1);
        List<DifferentialFunction> ret = new ArrayList<>(1);
        ret.add(sameDiff.f().doGradChoose(this,i_v1.get(0),dimensions));
        return ret;
    }
}
