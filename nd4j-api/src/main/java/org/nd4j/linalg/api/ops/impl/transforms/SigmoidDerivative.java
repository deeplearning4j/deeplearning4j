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
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Sigmoid derivative
 *
 * @author Adam Gibson
 */
public class SigmoidDerivative extends BaseTransformOp {

    public SigmoidDerivative() {
    }

    public SigmoidDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public SigmoidDerivative(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public SigmoidDerivative(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public SigmoidDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 29;
    }

    @Override
    public String name() {
        return "sigmoidderivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
    	return sigmoidDeriv(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
    	return sigmoidDeriv(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
    	return sigmoidDeriv(origin);
    }

    @Override
    public float op(float origin, float other) {
        return (float)sigmoidDeriv(origin);
    }

    @Override
    public double op(double origin, double other) {
    	return sigmoidDeriv(origin);
    }

    @Override
    public double op(double origin) {
    	return sigmoidDeriv(origin);
    }

    @Override
    public float op(float origin) {
    	return (float)sigmoidDeriv(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
    	return sigmoidDeriv(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new SigmoidDerivative(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SigmoidDerivative(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new SigmoidDerivative(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SigmoidDerivative(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }
    
    private static double sigmoidDeriv(double input) {
        double sigmoid = 1 / (1 + FastMath.exp(-input));
    	double out = sigmoid * (1.0 - sigmoid);
        if (Nd4j.ENFORCE_NUMERICAL_STABILITY) {
            if (Double.isNaN(out) || Double.isInfinite(out))
                out = Nd4j.EPS_THRESHOLD;
        }
        return out;
    }
    
    private static IComplexNumber sigmoidDeriv(IComplexNumber number) {
        double arg = number.complexArgument().doubleValue();
        double sigArg = 1 / 1 + (FastMath.exp(-arg)) - 1 + .5f;
        double ret = Math.exp(sigArg);
        IComplexDouble sigmoid = Nd4j.createDouble(ret, 0);
        IComplexNumber oneMinus = Nd4j.createComplexNumber(1, 1).subi(sigmoid);
        return sigmoid.mul(oneMinus);
    }
}
