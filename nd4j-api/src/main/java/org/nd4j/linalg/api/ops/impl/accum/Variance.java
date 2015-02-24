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

package org.nd4j.linalg.api.ops.impl.accum;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Variance
 * @author Adam Gibson
 */
public class Variance extends BaseAccumulation {
    private double mean;

    public Variance(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Variance(INDArray x) {
        super(x);
    }

    public Variance(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public void update(Number result) {
        if(otherAccum().isEmpty()) {
            otherAccum.add(0.0);

        }
        double dev = result.doubleValue() - mean;
        currentResult = currentResult().doubleValue() + FastMath.pow(dev, 2);
        otherAccum().set(0,otherAccum().get(0).doubleValue() + dev);
        numProcessed++;

        if(numProcessed() == n())
            currentResult = (currentResult.doubleValue() - (FastMath.pow(otherAccum.get(0).doubleValue(),2.0) / n())) / (n() - 1.0);


    }

    @Override
    public void update(IComplexNumber result) {
        if(otherAccumComplex().isEmpty()) {
            otherAccumComplex().add(Nd4j.createComplexNumber(0.0,0.0));
        }


        IComplexNumber dev = result.sub(mean);
        currentComplexResult.addi(ComplexUtil.pow(dev, 2));
        otherAccumComplex().get(0).addi(dev);
        numProcessed++;

        if(numProcessed() == n())
            currentComplexResult = (currentComplexResult.sub(ComplexUtil.pow(otherAccumComplex.get(0),2.0).div(Nd4j.createComplexNumber(n(),0))).div(Nd4j.createComplexNumber(n() - 1.0,0.0)));


    }

    @Override
    public Number zero() {
        return 0.0;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createDouble(0, 0);
    }

    @Override
    public String name() {
        return "var";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        return super.op(origin, other, extraArgs);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        return super.op(origin, other, extraArgs);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        return super.op(origin, other, extraArgs);
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        return super.op(origin, other, extraArgs);
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        return super.op(origin, other, extraArgs);
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        return super.op(origin, extraArgs);
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        return super.op(origin, extraArgs);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        return super.op(origin, extraArgs);
    }

    @Override
    public Op opForDimension(int index,int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index,dimension);


        if(y() != null)
            return new Variance(xAlongDimension,y.vectorAlongDimension(index,dimension),xAlongDimension.length());
        else
            return new Variance(x.vectorAlongDimension(index,dimension));

    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        this.mean = Nd4j.getExecutioner().execAndReturn(new Mean(x)).currentResult().doubleValue();
    }
}
