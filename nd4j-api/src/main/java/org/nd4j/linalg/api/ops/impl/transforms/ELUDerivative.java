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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * Derivative of ELU: Exponential Linear Unit (alpha=1.0)<br>
 * Introduced in paper:<br>
 * Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)<br>
 * Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015)<br>
 * <a href="http://arxiv.org/abs/1511.07289">http://arxiv.org/abs/1511.07289</a>
 *
 * @author Alex Black
 */
public class ELUDerivative extends BaseTransformOp {

    public ELUDerivative(){

    }

    public ELUDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public ELUDerivative(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public ELUDerivative(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public ELUDerivative(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.length());
    }

    public ELUDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 22;
    }

    @Override
    public String name() {
        return "eluderivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.realComponent().doubleValue() >= 0.0 ? Nd4j.createComplexNumber(1,0) :
                Nd4j.createComplexNumber(FastMath.exp(origin.realComponent().doubleValue()),0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.realComponent().doubleValue() >= 0.0 ? Nd4j.createComplexNumber(1,0) :
                Nd4j.createComplexNumber(FastMath.exp(origin.realComponent().doubleValue()),0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.realComponent().doubleValue() >= 0.0 ? Nd4j.createComplexNumber(1,0) :
                Nd4j.createComplexNumber(FastMath.exp(origin.realComponent().doubleValue()),0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin.realComponent().doubleValue() >= 0.0 ? Nd4j.createComplexNumber(1,0) :
                Nd4j.createComplexNumber(FastMath.exp(origin.realComponent().doubleValue()),0);
    }

    @Override
    public float op(float origin, float other) {
        return origin >= 0.0 ? 1.0f : (float)(FastMath.exp(origin));
    }

    @Override
    public double op(double origin, double other) {
        return origin >= 0.0 ? 1.0 : FastMath.exp(origin);
    }

    @Override
    public double op(double origin) {
        return origin >= 0.0 ? 1.0 : FastMath.exp(origin);
    }

    @Override
    public float op(float origin) {
        return origin >= 0.0 ? 1.0f : (float)(FastMath.exp(origin));
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new ELUDerivative(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new ELUDerivative(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new ELUDerivative(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new ELUDerivative(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());
    }
}
