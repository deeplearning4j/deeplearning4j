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

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for accumulation, initiates the initial entry
 * with respect to the child class. Also contains baseline fields
 * for the over all field with accumulation.
 *
 * @author Adam Gibson
 */
public abstract class BaseAccumulation extends BaseOp implements Accumulation {
    protected Number currentResult;
    protected IComplexNumber currentComplexResult;
    protected int numProcessed = 0;
    protected List<Number> otherAccum;
    protected List<IComplexNumber> otherAccumComplex;

    public BaseAccumulation(INDArray x, INDArray y, int n) {
        super(x, y, n);
        init();
    }

    public BaseAccumulation(INDArray x) {
        super(x);
        init();
    }

    public BaseAccumulation(INDArray x, INDArray y) {
        this(x,y,x.length());
    }

    private void init() {
        currentResult = zero();
        currentComplexResult = zeroComplex();
        otherAccum = new ArrayList<>();
        otherAccumComplex = new ArrayList<>();
    }

    @Override
    public IComplexNumber currentResultComplex() {
        return currentComplexResult;
    }

    @Override
    public Number currentResult() {
        return currentResult;
    }

    @Override
    public int numProcessed() {
        return numProcessed;
    }


    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        return origin;
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        return origin;
    }

    @Override
    public List<IComplexNumber> otherAccumComplex() {
        return otherAccumComplex;
    }

    @Override
    public List<Number> otherAccum() {
        return otherAccum;
    }


}
