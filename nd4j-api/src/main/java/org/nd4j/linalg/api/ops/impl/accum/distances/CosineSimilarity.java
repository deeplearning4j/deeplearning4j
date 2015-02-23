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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Cosine similarity
 * Note that you need to initialize a scaling constant equal to the norm2 of the
 * vector
 * @author Adam Gibson
 */
public class CosineSimilarity extends BaseAccumulation {
    private double constantNormalizedByNorm2 = Double.MIN_VALUE;

    public CosineSimilarity(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public CosineSimilarity(INDArray x) {
        super(x);
    }

    @Override
    public void update(Number result) {
        currentResult = currentResult.doubleValue() + result.doubleValue();
    }

    @Override
    public void update(IComplexNumber result) {
        currentComplexResult.addi(result);
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
        return "cosinesimilarity";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        initNormalizationConstant(extraArgs);
        return origin.mul(other * constantNormalizedByNorm2);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        initNormalizationConstant(extraArgs);
        return origin.mul(other * constantNormalizedByNorm2);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        initNormalizationConstant(extraArgs);
        return origin.mul(other.mul(constantNormalizedByNorm2));
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        initNormalizationConstant(extraArgs);
        return (float) (origin * other * constantNormalizedByNorm2);
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        initNormalizationConstant(extraArgs);
        return origin * other * constantNormalizedByNorm2;
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        initNormalizationConstant(extraArgs);
        return origin * constantNormalizedByNorm2;
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        initNormalizationConstant(extraArgs);
        return (float) (origin * constantNormalizedByNorm2);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        initNormalizationConstant(extraArgs);
        return origin.mul(constantNormalizedByNorm2);
    }

    private void initNormalizationConstant(Object[] extraArgs) {
        if(constantNormalizedByNorm2 == Double.MIN_VALUE)
            this.constantNormalizedByNorm2 = Double.valueOf(extraArgs[0].toString());
    }

}
