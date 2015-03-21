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
import org.nd4j.linalg.factory.Nd4j;

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
    protected List<Number> otherAccum;
    protected List<IComplexNumber> otherAccumComplex;
    protected Number initial;
    protected IComplexNumber initialComplex;

    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     * @param n the number of elements
     */
    public BaseAccumulation(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
        init();
    }

    public BaseAccumulation(INDArray x, INDArray y, int n) {
        this(x, y, x, n);
    }

    public BaseAccumulation(INDArray x) {
        this(x, null, x, x.length());
    }

    public BaseAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.length());
    }

    private void init() {
        currentResult = zero();
        currentComplexResult = zeroComplex();
        otherAccum = new ArrayList<>();
        otherAccumComplex = new ArrayList<>();
        init(x, y, x, x.length());
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        numProcessed++;
        return origin;
    }

    @Override
    public float op(float origin, float other) {
        numProcessed++;
        return origin;
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin;
    }

    @Override
    public double op(double origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public float op(float origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        numProcessed++;
        return origin;
    }


    @Override
    public Number zero() {
        return initial;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return initialComplex.dup();
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
    public List<IComplexNumber> otherAccumComplex() {
        return otherAccumComplex;
    }

    @Override
    public List<Number> otherAccum() {
        return otherAccum;
    }

    @Override
    public void setCurrentResult(Number number) {
        this.currentResult = number;
    }

    @Override
    public void setCurrentResultComplex(IComplexNumber complexNumber) {
        this.currentComplexResult = complexNumber;
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        if (initial == null)
            initial = 0.0;
        if (initialComplex == null)
            initialComplex = Nd4j.createComplexNumber(0.0, 0.0);
        this.extraArgs = new Object[]{zero()};
    }
}
