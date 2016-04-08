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

package org.nd4j.linalg.api.ops.impl.scalar.comparison;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Return a binary (0 or 1) when greater than a number
 *
 * @author Adam Gibson
 */
public class ScalarEquals extends BaseScalarOp {
    public ScalarEquals() {
    }

    public ScalarEquals(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarEquals(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarEquals(INDArray x, INDArray y, INDArray z, long n, IComplexNumber num) {
        super(x, y, z, n, num);
    }

    public ScalarEquals(INDArray x, IComplexNumber num) {
        super(x, num);
    }

    public ScalarEquals(INDArray x) {
        this(x,0);
    }

    @Override
    public int opNum() {
        return 9;
    }

    @Override
    public String name() {
        return "equals_scalar";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.absoluteValue().doubleValue() == num.doubleValue() ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.absoluteValue().doubleValue() == num.doubleValue() ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.absoluteValue().doubleValue() == num.doubleValue() ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0);
    }

    @Override
    public float op(float origin, float other) {
        return origin == num.floatValue() ? 1 : 0;
    }

    @Override
    public double op(double origin, double other) {
        return origin == num.doubleValue() ? 1 : 0;
    }

    @Override
    public double op(double origin) {
        return origin == num.doubleValue() ? 1 : 0;

    }

    @Override
    public float op(float origin) {
        return origin == num.floatValue() ? 1 : 0;

    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin.absoluteValue().doubleValue() == num.doubleValue() ? Nd4j.createComplexNumber(1, 0) : Nd4j.createComplexNumber(0, 0);

    }

    @Override
    public Op opForDimension(int index, int dimension) {
        if (num != null)
            return new ScalarEquals(x.vectorAlongDimension(index, dimension), num);
        else
            return new ScalarEquals(x.vectorAlongDimension(index, dimension), complexNumber);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        if (num != null)
            return new ScalarEquals(x.tensorAlongDimension(index, dimension), num);
        else
            return new ScalarEquals(x.tensorAlongDimension(index, dimension), complexNumber);
    }

    @Override
    public int[] getDimension() {
        return new int[0];
    }

    @Override
    public void setDimension(int... dimension) {

    }
}
