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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Pow function
 *
 * @author Adam Gibson
 */
public class Pow extends BaseTransformOp {
    public Pow(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Pow(INDArray x, INDArray z) {
        super(x, z);
    }

    public Pow(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public Pow(INDArray x) {
        super(x);
    }

    @Override
    public String name() {
        return "pow";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        double pow = pow(extraArgs);
        return ComplexUtil.pow(origin,pow);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        double pow = pow(extraArgs);
        return ComplexUtil.pow(origin,pow);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        double pow = pow(extraArgs);
        return ComplexUtil.pow(origin,pow);
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        double pow = pow(extraArgs);
        return (float) FastMath.pow(origin, pow);
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        double pow = pow(extraArgs);
        return  FastMath.pow(origin, pow);
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        double pow = pow(extraArgs);
        return  FastMath.pow(origin, pow);
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        double pow = pow(extraArgs);
        return (float) FastMath.pow(origin, pow);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        double pow = pow(extraArgs);
        return ComplexUtil.pow(origin,pow);
    }

    private double pow(Object[] extraArgs) {
        if(extraArgs.length < 1)
            throw new IllegalArgumentException("Extra arguments must contain a power");
        return Double.valueOf(extraArgs[0].toString());
    }
}
