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

/**Leaky ReLU derivative. Default alpha = 0.01. Cutoff = 0
 */
public class LeakyReLUDerivative extends BaseTransformOp {
	private double alpha = 0.01;
    private boolean alphaSpecified = false;
    public LeakyReLUDerivative() { }

    public LeakyReLUDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public LeakyReLUDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public LeakyReLUDerivative(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public LeakyReLUDerivative(INDArray x) {
        super(x);
    }
    
    public LeakyReLUDerivative(INDArray x, INDArray z, double alpha) {
        super(x, z);
        this.alpha = alpha;
        alphaSpecified = true;
    }

    public LeakyReLUDerivative(INDArray x, INDArray z, long n, double alpha) {
        super(x, z, n);
        this.alpha = alpha;
        alphaSpecified = true;
    }

    public LeakyReLUDerivative(INDArray x, INDArray y, INDArray z, long n, double alpha) {
        super(x, y, z, n);
        this.alpha = alpha;
        alphaSpecified = true;
    }

    public LeakyReLUDerivative(INDArray x, double alpha) {
        super(x);
        this.alpha = alpha;
    }

    @Override
    public int opNum() {
        return 32;
    }

    @Override
    public String name() {
        return "leakyreluderivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return (origin.realComponent().doubleValue() >= 0.0 ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(alpha, 0));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
    	return (origin.realComponent().doubleValue() >= 0.0 ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(alpha, 0)); 
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
    	return (origin.realComponent().doubleValue() >= 0.0 ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(alpha, 0));
    }

    @Override
    public float op(float origin, float other) {
        return (origin >= 0f ? 1.0f : (float) alpha);
    }

    @Override
    public double op(double origin, double other) {
    	return (origin >= 0 ? 1.0 : alpha);
    }

    @Override
    public double op(double origin) {
    	return (origin >= 0 ? 1.0 : alpha);
    }

    @Override
    public float op(float origin) {
    	return (origin >= 0f ? 1.0f : (float)alpha);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
    	return (origin.realComponent().doubleValue() >= 0 ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(alpha, 0));
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new LeakyReLUDerivative(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(),alpha);
        else
            return new LeakyReLUDerivative(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(),alpha);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new LeakyReLUDerivative(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(),alpha);
        else
            return new LeakyReLUDerivative(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(),alpha);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if(!alphaSpecified)
            alpha = 0.01;
        this.extraArgs = new Object[] {alpha};
    }
}
