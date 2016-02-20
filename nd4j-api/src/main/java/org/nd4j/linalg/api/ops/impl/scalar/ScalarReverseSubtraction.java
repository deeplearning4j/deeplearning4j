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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * Scalar reverse subtraction
 *
 * @author Adam Gibson
 */
public class ScalarReverseSubtraction extends BaseScalarOp {

    public ScalarReverseSubtraction() {
    }

    public ScalarReverseSubtraction(INDArray x, INDArray y, INDArray z, int n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarReverseSubtraction(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarReverseSubtraction(INDArray x, INDArray y, INDArray z, int n, IComplexNumber num) {
        super(x, y, z, n, num);
    }

    public ScalarReverseSubtraction(INDArray x, IComplexNumber num) {
        super(x, num);
    }

    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String name() {
        return "rsub_scalar";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        if (complexNumber != null)
            return complexNumber.rsub(origin);
        return complexNumber.rsub(num);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        if (complexNumber != null)
            return origin.rsub(complexNumber);
        return complexNumber.rsub(num);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        if (complexNumber != null)
            return origin.div(complexNumber);
        return complexNumber.rsub(num);
    }

    @Override
    public float op(float origin, float other) {
        return (num.floatValue() - origin);
    }

    @Override
    public double op(double origin, double other) {
        return num.doubleValue() - origin;
    }

    @Override
    public double op(double origin) {
        return num.doubleValue() - origin;
    }

    @Override
    public float op(float origin) {
        return (num.floatValue() - origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        if (complexNumber != null)
            return origin.rsub(complexNumber);
        return complexNumber.rsub(num);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        if (num != null)
            return new ScalarReverseSubtraction(x.vectorAlongDimension(index, dimension), num);
        else
            return new ScalarReverseSubtraction(x.vectorAlongDimension(index, dimension), complexNumber);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        if (num != null)
            return new ScalarReverseSubtraction(x.tensorAlongDimension(index, dimension), num);
        else
            return new ScalarReverseSubtraction(x.tensorAlongDimension(index, dimension), complexNumber);
    }
}
