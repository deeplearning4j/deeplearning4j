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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.Op;

/**
 *  Scalar reverse subtraction
 *  @author Adam Gibson
 */
public class ScalarReverseSubtraction extends BaseScalarOp {
    public ScalarReverseSubtraction(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public ScalarReverseSubtraction(INDArray x) {
        super(x);
    }

    @Override
    public String name() {
        return "scalar_rsub";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        if(complexNumber != null)
            return complexNumber.rsub(origin);
        return complexNumber.rsub(num);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        if(complexNumber != null)
            return origin .rsub(complexNumber);
        return complexNumber.rsub(num);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        if(complexNumber != null)
            return origin.div(complexNumber);
        return complexNumber.rsub(num);
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        return (float) (num - origin);
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        return num - origin;
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        return num - origin;
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        return (float) (num - origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        if(complexNumber != null)
            return origin.rsub(complexNumber);
        return complexNumber.rsub(num);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return new ScalarReverseSubtraction(x.vectorAlongDimension(index,dimension));
    }
}
