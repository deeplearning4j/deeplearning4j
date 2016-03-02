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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * Scalar max operation.
 * Returns the max of an element
 * in the ndarray of the specified number.
 *
 * @author Adam Gibson
 */
public class ScalarMin extends BaseScalarOp {
    public ScalarMin() {
    }

    public ScalarMin(INDArray x, INDArray y, INDArray z, int n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarMin(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarMin(INDArray x, INDArray y, INDArray z, int n, IComplexNumber num) {
        super(x, y, z, n, num);
    }

    public ScalarMin(INDArray x, IComplexNumber num) {
        super(x, num);
    }

    @Override
    public int opNum() {
        return 12;
    }

    @Override
    public String name() {
        return "max_scalar";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        if (origin.absoluteValue().doubleValue() > complexNumber.absoluteValue().doubleValue())
            return origin;
        return complexNumber;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        if (origin.absoluteValue().doubleValue() > complexNumber.absoluteValue().doubleValue())
            return origin;
        return complexNumber;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        if (origin.absoluteValue().doubleValue() > complexNumber.absoluteValue().doubleValue())
            return origin;
        return complexNumber;
    }

    @Override
    public float op(float origin, float other) {
        return FastMath.max(origin, num.floatValue());
    }

    @Override
    public double op(double origin, double other) {
        return FastMath.max(origin, num.doubleValue());
    }

    @Override
    public double op(double origin) {
        return FastMath.max(origin, num.doubleValue());

    }

    @Override
    public float op(float origin) {
        return FastMath.max(origin, num.floatValue());

    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        if (origin.absoluteValue().doubleValue() > complexNumber.absoluteValue().doubleValue())
            return origin;
        return complexNumber;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        if (num != null)
            return new ScalarMin(x.vectorAlongDimension(index, dimension), num);
        else
            return new ScalarMin(x.vectorAlongDimension(index, dimension), complexNumber);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        if (num != null)
            return new ScalarMin(x.tensorAlongDimension(index, dimension), num);
        else
            return new ScalarMin(x.tensorAlongDimension(index, dimension), complexNumber);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        if (num != null)
            this.extraArgs = new Object[]{num};
        else
            this.extraArgs = new Object[]{complexNumber};

    }
}
