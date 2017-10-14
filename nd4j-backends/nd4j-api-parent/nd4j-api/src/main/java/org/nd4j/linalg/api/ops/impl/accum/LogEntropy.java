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
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;

import java.util.List;

/**
 * Log Entropy Op - returns the entropy (information gain, or uncertainty of a random variable).
 *
 * @author raver119@gmail.com
 */
public class  LogEntropy extends BaseAccumulation {

    public LogEntropy() {}

    public LogEntropy(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public LogEntropy(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public LogEntropy(INDArray x) {
        super(x);
    }

    public LogEntropy(INDArray x, INDArray y) {
        super(x, y);
    }

    public LogEntropy(INDArray x, INDArray y, INDArray z) {
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
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x, double y) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return accum.add(x);
    }

    @Override
    public double op(double origin) {
        return FastMath.log(FastMath.pow(origin, 2));
    }

    @Override
    public float op(float origin) {
        return (float) FastMath.log(FastMath.pow(origin, 2));
    }

    @Override
    public int opNum() {
        return 17;
    }

    @Override
    public String name() {
        return "logentropy";
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
            return new LogEntropy(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new LogEntropy(xAlongDimension);

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new LogEntropy(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new LogEntropy(xAlongDimension);
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
