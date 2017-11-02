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

import java.util.List;

/**
 * Logical And reduction op
 *
 * @author raver119@gmail.com
 */
public class All extends BaseAccumulation {
    public All(SameDiff sameDiff, DifferentialFunction i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public All(SameDiff sameDiff, DifferentialFunction i_v, DifferentialFunction i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public All() {}

    public All(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public All(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public All(INDArray x) {
        super(x);
    }

    public All(INDArray x, INDArray y) {
        super(x, y);
    }

    public All(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    @Override
    public double update(double accum, double x) {
        return accum + x;
    }

    @Override
    public double update(double accum, double x, double y) {
        return accum + x;
    }

    @Override
    public float update(float accum, float x) {
        return accum + x;
    }

    @Override
    public float update(float accum, float x, float y) {
        return accum + x;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x, double y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double op(double origin) {
        return FastMath.pow(origin, 2) * FastMath.log(FastMath.pow(origin, 2));
    }

    @Override
    public float op(float origin) {
        return (float) FastMath.pow(origin, 2) * (float) FastMath.log(FastMath.pow(origin, 2));
    }

    @Override
    public double calculateFinalResult(double accum, long n) {
        return -accum;
    }

    @Override
    public float calculateFinalResult(float accum, long n) {
        return -accum;
    }

    @Override
    public int opNum() {
        return 20;
    }

    @Override
    public String name() {
        return "all";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new All(xAlongDimension, y.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new All(xAlongDimension);

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new All(xAlongDimension, y.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new All(xAlongDimension);
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
