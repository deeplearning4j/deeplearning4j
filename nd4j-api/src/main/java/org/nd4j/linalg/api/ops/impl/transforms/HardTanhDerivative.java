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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Hard tanh elementwise derivative function
 *
 * @author Adam Gibson
 */
public class HardTanhDerivative extends BaseTransformOp {
    public HardTanhDerivative() {
    }

    public HardTanhDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public HardTanhDerivative(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public HardTanhDerivative(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public HardTanhDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 25;
    }

    @Override
    public String name() {
        return "hardtanhderivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
    	double r = origin.realComponent().doubleValue();
    	return (r >= -1 && r <= 1) ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(0, 0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
    	double r = origin.realComponent().doubleValue();
    	return (r >= -1 && r <= 1) ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(0, 0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
    	double r = origin.realComponent().doubleValue();
    	return (r >= -1 && r <= 1) ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(0, 0);
    }

    @Override
    public float op(float origin, float other) {
        return hardTanhDeriv(origin);
    }

    @Override
    public double op(double origin, double other) {
        return hardTanhDeriv(origin);
    }

    @Override
    public double op(double origin) {
        return hardTanhDeriv(origin);
    }

    @Override
    public float op(float origin) {
        return hardTanhDeriv(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
    	double r = origin.realComponent().doubleValue();
    	return (r >= -1 && r <= 1) ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(0, 0);
    }


    private static float hardTanhDeriv(float num) {
    	return ((num>=-1.0f && num<=1.0f) ? 1.0f : 0.0f);
    }

    private static double hardTanhDeriv(double num) {
    	return ((num >= - 1.0 && num <= 1.0) ? 1.0 : 0.0);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new HardTanhDerivative(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new HardTanhDerivative(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new HardTanhDerivative(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new HardTanhDerivative(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

}
