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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * Scalar value set operation.
 * Anything less than the scalar value will
 * set the current element to be that value.
 *
 * @author Adam Gibson
 */
public class ScalarSetValue extends BaseScalarOp {
    public ScalarSetValue() {
    }

    public ScalarSetValue(INDArray x, INDArray y, INDArray z, int n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarSetValue(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarSetValue(INDArray x, INDArray y, INDArray z, int n, IComplexNumber num) {
        super(x, y, z, n, num);
    }

    public ScalarSetValue(INDArray x, IComplexNumber num) {
        super(x, num);
    }

    @Override
    public int opNum() {
        return 13;
    }

    @Override
    public String name() {
        return "setvalorless_scalar";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return setValueIfLess(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return setValueIfLess(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return setValueIfLess(origin);
    }

    @Override
    public float op(float origin, float other) {
        return origin < num.floatValue() ? num.floatValue() : origin;

    }

    @Override
    public double op(double origin, double other) {
        return origin < num.doubleValue() ? num.floatValue() : origin;

    }

    @Override
    public double op(double origin) {
        return origin < num.doubleValue() ? num.floatValue() : origin;
    }

    @Override
    public float op(float origin) {
        return origin < num.floatValue() ? num.floatValue() : origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return setValueIfLess(origin);
    }

    private IComplexNumber setValueIfLess(IComplexNumber num) {
        if (num.realComponent().doubleValue() < this.num.doubleValue())
            return num.set(this.num.doubleValue(), 0.0);
        return num;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        if (num != null)
            return new ScalarSetValue(x.vectorAlongDimension(index, dimension), num);
        else
            return new ScalarSetValue(x.vectorAlongDimension(index, dimension), complexNumber);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        if (num != null)
            return new ScalarSetValue(x.tensorAlongDimension(index, dimension), num);
        else
            return new ScalarSetValue(x.tensorAlongDimension(index, dimension), complexNumber);
    }
}
