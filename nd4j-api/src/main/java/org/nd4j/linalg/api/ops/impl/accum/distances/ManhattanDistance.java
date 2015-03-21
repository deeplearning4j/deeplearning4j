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

package org.nd4j.linalg.api.ops.impl.accum.distances;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Euclidean distance
 *
 * @author Adam Gibson
 */
public class ManhattanDistance extends BaseAccumulation {
    public ManhattanDistance(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public ManhattanDistance(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public ManhattanDistance(INDArray x) {
        super(x);
    }

    public ManhattanDistance(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public void update(Number result) {
        currentResult = currentResult.doubleValue() + FastMath.pow(result.doubleValue(), 2.0);
    }

    @Override
    public void update(IComplexNumber result) {
        currentComplexResult.addi(ComplexUtil.pow(result, 2));
    }

    @Override
    public Number zero() {
        return 0.0;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0, 0.0);
    }

    @Override
    public String name() {
        return "manhattan";
    }


    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.sub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.sub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.sub(other);
    }

    @Override
    public float op(float origin, float other) {
        return origin - other;
    }

    @Override
    public double op(double origin, double other) {
        return origin - other;
    }

    @Override
    public double op(double origin) {
        return origin;
    }

    @Override
    public float op(float origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        if (y() != null)
            return new ManhattanDistance(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), x.length());
        else
            return new ManhattanDistance(x.vectorAlongDimension(index, dimension));

    }
}
