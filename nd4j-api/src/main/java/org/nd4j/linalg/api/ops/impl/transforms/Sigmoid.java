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
import org.nd4j.linalg.factory.Nd4j;

/**
 * Sigmoid function
 * @author Adam Gibson
 */
public class Sigmoid extends BaseTransformOp {


    public Sigmoid(INDArray x, INDArray z) {
        super(x, z);
    }

    public Sigmoid(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public Sigmoid(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Sigmoid(INDArray ndArray) {
        super(ndArray);
    }

    @Override
    public String name() {
        return "sigmoid";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        return sigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        return sigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        return sigmoid(origin);
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        return (float) sigmoid(origin);
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        return sigmoid(origin);
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        return sigmoid(origin);
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        return (float) sigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        return sigmoid(origin);
    }


    private double sigmoid(double input) {
        double inputf = input;
        double val = 1 / (1 + FastMath.exp(-inputf));
        if (Nd4j.ENFORCE_NUMERICAL_STABILITY) {
            if (Double.isNaN(val) || Double.isInfinite(val))
                val = Nd4j.EPS_THRESHOLD;
        }
        return val;
    }


    private IComplexNumber sigmoid(IComplexNumber number) {
        double arg = number.complexArgument().doubleValue();
        double sigArg = 1 / 1 + (FastMath.exp(-arg)) - 1 + .5f;
        double ret = Math.exp(sigArg);
        return Nd4j.createDouble(ret, 0);
    }

}
