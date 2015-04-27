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

package org.nd4j.linalg.api.ops.tests;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 2/22/15.
 */
public class TestAccumulation extends BaseAccumulation {
    public TestAccumulation(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public TestAccumulation(INDArray x) {
        super(x);
    }

    @Override
    public void update(Number result) {
        this.currentResult = result.doubleValue() + this.currentResult.doubleValue();
    }

    @Override
    public void update(IComplexNumber result) {
        this.currentComplexResult.addi(result);
    }

    @Override
    public Number zero() {
        return 0;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0, 0);
    }

    @Override
    public String name() {
        return "test";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.add(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.add(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.add(other);
    }

    @Override
    public float op(float origin, float other) {
        return origin + other;
    }

    @Override
    public double op(double origin, double other) {
        return origin + other;
    }

    @Override
    public double op(double origin) {
        return origin;
    }

    @Override
    public float op(float origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }
}
