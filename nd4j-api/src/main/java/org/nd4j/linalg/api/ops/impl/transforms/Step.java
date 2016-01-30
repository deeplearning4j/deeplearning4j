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
 * Unit step function.
 * f(x) = 1 if x > cutoff; 0 otherwise
 * cutoff = 0.0 usually.
 */
public class Step extends BaseTransformOp {
	private final double cutoff;
    public Step() {
    	cutoff = 0.0;
    }

    public Step(INDArray x, INDArray z) {
        super(x, z);
        cutoff = 0.0;
    }

    public Step(INDArray x, INDArray z, int n) {
        super(x, z, n);
        cutoff = 0.0;
    }

    public Step(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
        cutoff = 0.0;
    }

    public Step(INDArray x) {
        super(x);
        cutoff = 0.0;
    }
    
    public Step(INDArray x, INDArray z, double cutoff) {
        super(x, z);
        this.cutoff = cutoff;
    }

    public Step(INDArray x, INDArray z, int n, double cutoff) {
        super(x, z, n);
        this.cutoff = cutoff;
    }

    public Step(INDArray x, INDArray y, INDArray z, int n, double cutoff) {
        super(x, y, z, n);
        this.cutoff = cutoff;
    }

    public Step(INDArray x, double cutoff) {
        super(x);
        this.cutoff = cutoff;
    }

    @Override
    public int opNum() {
        return 34;
    }

    @Override
    public String name() {
        return "step";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return (origin.realComponent().doubleValue() > cutoff ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(0, 0));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
    	return (origin.realComponent().doubleValue() > cutoff ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(0, 0)); 
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
    	return (origin.realComponent().doubleValue() > cutoff ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(0, 0));
    }

    @Override
    public float op(float origin, float other) {
        return (origin > cutoff ? 1.0f : 0.0f);
    }

    @Override
    public double op(double origin, double other) {
    	return (origin > cutoff ? 1.0 : 0.0);
    }

    @Override
    public double op(double origin) {
    	return (origin > cutoff ? 1.0 : 0.0);
    }

    @Override
    public float op(float origin) {
    	return (origin > cutoff ? 1.0f : 0.0f);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
    	return (origin.realComponent().doubleValue() > cutoff ? Nd4j.createDouble(1, 0) : Nd4j.createDouble(0, 0));
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Step(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(),cutoff);
        else
            return new Step(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(),cutoff);

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Step(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(),cutoff);
        else
            return new Step(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(),cutoff);

    }

}
