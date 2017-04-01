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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;

/**
 * Operation for fast INDArrays equality checks
 *
 * @author raver119@gmail.com
 */
public class EqualsWithEps extends BaseAccumulation {
    private double eps;

    public EqualsWithEps() {}

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, long n, double eps) {
        super(x, y, z, n);
        this.extraArgs = new Object[] {eps};
    }

    public EqualsWithEps(INDArray x, INDArray y, long n, double eps) {
        super(x, y, n);
        this.extraArgs = new Object[] {eps};
    }

    public EqualsWithEps(INDArray x, INDArray y, double eps) {
        super(x, y);
        this.extraArgs = new Object[] {eps};
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String name() {
        return "equals_with_eps";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new EqualsWithEps(xAlongDimension, y.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            throw new UnsupportedOperationException("This Op is suited only as comparison for two arrays");
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new EqualsWithEps(xAlongDimension, y.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            throw new UnsupportedOperationException("This Op is suited only as comparison for two arrays");
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float op(float origin, float other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double op(double origin, double other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double op(double origin) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float op(float origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double update(double accum, double x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double update(double accum, double x, double y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float update(float accum, float x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float update(float accum, float x, float y) {
        throw new UnsupportedOperationException();
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
    public double combineSubResults(double first, double second) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float combineSubResults(float first, float second) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber combineSubResults(IComplexNumber first, IComplexNumber second) {
        throw new UnsupportedOperationException();
    }
}
