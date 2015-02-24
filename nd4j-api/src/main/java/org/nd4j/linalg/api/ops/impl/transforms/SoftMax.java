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
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Soft max function
 * row_maxes is a row vector (max for each row)
 * row_maxes = rowmaxes(input)
 * diff = exp(input - max) / diff.rowSums()
 * Outputs a probability distribution.
 * Note that this is a parameterized model and requires
 * the sum and max for the vector being calculated
 *
 * @author Adam Gibson
 */

public class SoftMax extends BaseTransformOp {
    private double sum = Double.NaN;
    private double max = Double.NaN;

    public SoftMax(INDArray x, INDArray z) {
        super(x, z);
    }

    public SoftMax(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public SoftMax(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public SoftMax(INDArray x) {
        super(x);
    }

    @Override
    public String name() {
        return "softmax";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        initFromArgs(extraArgs);
        return ComplexUtil.exp(origin.sub(max)).divi(sum);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        initFromArgs(extraArgs);
        return ComplexUtil.exp(origin.sub(max)).divi(sum);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        initFromArgs(extraArgs);
        return ComplexUtil.exp(origin.sub(max)).divi(sum);
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        initFromArgs(extraArgs);
        return (float) ((FastMath.exp(origin - max)) / sum);
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        initFromArgs(extraArgs);
        return ((FastMath.exp(origin - max)) / sum);
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        initFromArgs(extraArgs);
        return ((FastMath.exp(origin - max)) / sum);
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        initFromArgs(extraArgs);
        return (float) ((FastMath.exp(origin - max)) / sum);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        initFromArgs(extraArgs);
        return ComplexUtil.exp(origin.sub(max)).divi(sum);
    }

    @Override
    public TransformOp derivative() {
        return new SoftMaxDerivative(x,y,z,n);
    }

    private void initFromArgs(Object[] extraArgs) {
        if(extraArgs.length < 2)
            throw new IllegalArgumentException("Please specify the max and the sum for this vector");
        if(Double.isNaN(max)) {
            max = Double.valueOf(extraArgs[0].toString());
        }
        if(Double.isNaN(sum)) {
            sum = Double.valueOf(extraArgs[1].toString());
        }
    }

}
