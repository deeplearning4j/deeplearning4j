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
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Tanh derivative
 */
public class TanhDerivative extends BaseTransformOp {

    public TanhDerivative() {
    }

    public TanhDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public TanhDerivative(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public TanhDerivative(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public TanhDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 23;
    }

    @Override
    public String name() {
        return "tanhderivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
    	IComplexNumber tanh = ComplexUtil.tanh(origin);
    	return Nd4j.createComplexNumber(1, 1).sub(tanh.mul(tanh));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
    	IComplexNumber tanh = ComplexUtil.tanh(origin);
    	return Nd4j.createComplexNumber(1, 1).sub(tanh.mul(tanh));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
    	IComplexNumber tanh = ComplexUtil.tanh(origin);
    	return Nd4j.createComplexNumber(1, 1).sub(tanh.mul(tanh));
    }

    @Override
    public float op(float origin, float other) {
    	double tanh = FastMath.tanh(origin);
        return (float)(1.0 - tanh*tanh);
    }

    @Override
    public double op(double origin, double other) {
    	double tanh = FastMath.tanh(origin);
        return 1.0 - tanh*tanh;
    }

    @Override
    public double op(double origin) {
    	double tanh = FastMath.tanh(origin);
        return 1.0 - tanh*tanh;
    }

    @Override
    public float op(float origin) {
    	double tanh = FastMath.tanh(origin);
        return (float)(1.0 - tanh*tanh);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
    	IComplexNumber tanh = ComplexUtil.tanh(origin);
    	return Nd4j.createComplexNumber(1, 1).sub(tanh.mul(tanh));
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new TanhDerivative(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new TanhDerivative(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new TanhDerivative(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new TanhDerivative(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }
}
