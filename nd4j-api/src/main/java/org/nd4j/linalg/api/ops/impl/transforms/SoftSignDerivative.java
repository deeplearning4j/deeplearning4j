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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.util.ComplexUtil;

/**SoftSign derivative.
 */
public class SoftSignDerivative extends BaseTransformOp {

    public SoftSignDerivative() {
    }

    public SoftSignDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public SoftSignDerivative(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public SoftSignDerivative(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public SoftSignDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 30;
    }

    @Override
    public String name() {
        return "softsignderivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
    	IComplexNumber oneMinusAbs = ComplexUtil.abs(origin).rsubi(1.0);
    	return oneMinusAbs.muli(oneMinusAbs).rdivi(1.0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
    	IComplexNumber oneMinusAbs = ComplexUtil.abs(origin).rsubi(1.0);
    	return oneMinusAbs.muli(oneMinusAbs).rdivi(1.0);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
    	IComplexNumber oneMinusAbs = ComplexUtil.abs(origin).rsubi(1.0);
    	return oneMinusAbs.muli(oneMinusAbs).rdivi(1.0);
    }
    
    @Override
    public IComplexNumber op(IComplexNumber origin) {
    	IComplexNumber oneMinusAbs = ComplexUtil.abs(origin).rsubi(1.0);
    	return oneMinusAbs.muli(oneMinusAbs).rdivi(1.0);
    }

    @Override
    public float op(float origin, float other) {
    	return (float)ssderiv(origin);
    }

    @Override
    public double op(double origin, double other) {
    	return ssderiv(origin);
    }

    @Override
    public double op(double origin) {
    	return ssderiv(origin);
    }

    @Override
    public float op(float origin) {
    	return (float)ssderiv(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new SoftSignDerivative(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SoftSignDerivative(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new SoftSignDerivative(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SoftSignDerivative(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
    }
    
    private static double ssderiv(double x){
    	double y = 1 + FastMath.abs(x);
    	return 1.0 / (y * y);
    }
}
