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

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

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
    private Number sum;
    private Number max;
    private IComplexNumber maxComplex, sumComplex;


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
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        if (x instanceof IComplexNDArray) {
            this.maxComplex = Nd4j.getExecutioner().execAndReturn(new Max(x)).currentResultComplex();
            IComplexNDArray complexX = (IComplexNDArray) x;
            this.y = Transforms.exp(complexX.sub(maxComplex));
            this.sumComplex = Nd4j.getExecutioner().execAndReturn(new Sum(y)).currentResultComplex();
            this.y.divi(sumComplex);
            this.extraArgs = new Object[]{maxComplex, sumComplex};
        } else {
            this.max = Nd4j.getExecutioner().execAndReturn(new Max(x)).currentResult();
            INDArray xMinusMax = x.sub(max);
            this.y = Transforms.exp(xMinusMax);
            this.sum = Nd4j.getExecutioner().execAndReturn(new Sum(this.y)).currentResult();
            this.y.divi(sum);
            this.extraArgs = new Object[]{max, sum};
        }


    }
}
