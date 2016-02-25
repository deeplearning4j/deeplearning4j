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

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.factory.Nd4j;

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

    public SoftMax() {
    }

    public SoftMax(INDArray x, INDArray z) {
        super(x, z);
    }

    public SoftMax(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public SoftMax(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public SoftMax(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.length());
    }

    public SoftMax(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String name() {
        return "softmax";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        IComplexNDArray arr = (IComplexNDArray) y;
        IComplexNumber ret = arr.getComplex(numProcessed);
        numProcessed++;
        return ret;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        IComplexNDArray arr = (IComplexNDArray) y;
        IComplexNumber ret = arr.getComplex(numProcessed);
        numProcessed++;
        return ret;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        IComplexNDArray arr = (IComplexNDArray) y;
        IComplexNumber ret = arr.getComplex(numProcessed);
        numProcessed++;
        return ret;
    }

    @Override
    public float op(float origin, float other) {
        float ret = other;
        numProcessed++;
        return ret;
    }

    @Override
    public double op(double origin, double other) {
        double ret = other;
        numProcessed++;
        return ret;
    }

    @Override
    public double op(double origin) {
        double ret = y.getDouble(numProcessed);
        numProcessed++;
        return ret;
    }

    @Override
    public float op(float origin) {
        float ret = (y.getFloat(numProcessed));
        numProcessed++;
        return ret;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        IComplexNDArray arr = (IComplexNDArray) y;
        IComplexNumber ret = arr.getComplex(numProcessed);
        numProcessed++;
        return ret;
    }

    @Override
    public TransformOp derivative() {
        return new SoftMaxDerivative(x, y, z, n);
    }


    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new SoftMax(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SoftMax(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new SoftMax(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SoftMax(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public void exec() {
        exec(1);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        passThrough = true;
    }


    @Override
    public void exec(int... dimensions) {
        if(dimensions[0] != 1)
            throw new IllegalArgumentException("Only supports row wise calculations");
        if(x.isMatrix()) {
            INDArray maxAlongDimension = x.max(dimensions);
            if(!maxAlongDimension.isVector() && !maxAlongDimension.isScalar())
                throw new IllegalStateException("Max along dimension for input must either be a row vector or scalar");

            INDArray xMinusMax = x.subColumnVector(maxAlongDimension);

            INDArray exp;
            if(z != null) {
                exp = Nd4j.getExecutioner().execAndReturn(new Exp(xMinusMax, z));
            } else {
                exp = Nd4j.getExecutioner().execAndReturn(new Exp(xMinusMax));
            }
            INDArray sum = exp.sum(dimensions);
            exp.diviColumnVector(sum);

            if(z == null) z = exp;
        }
        else if(x.isVector()) {
           double max = x.maxNumber().doubleValue();
            INDArray exp;
            if(z != null){
                exp = Nd4j.getExecutioner().execAndReturn(new Exp(x.sub(max), z));
            } else {
                exp = Nd4j.getExecutioner().execAndReturn(new Exp(x.sub(max)));
            }
            exp.divi(exp.sumNumber().doubleValue());
            this.z = exp;
        }
    }
}
