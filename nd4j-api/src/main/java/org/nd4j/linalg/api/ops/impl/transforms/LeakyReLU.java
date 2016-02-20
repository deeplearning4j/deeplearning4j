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
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.factory.Nd4j;

/**Leaky Rectified linear unit. Default alpha=0.01, cutoff=0<br>
 * Out(x) = alpha*x if x<0<br>
 * Out(x) = x if x >= 0<br>
 * Leaky ReLU may avoid zero gradient "dying ReLU" problem by having non-zero
 * gradient below 0.<br>
 * See for example http://arxiv.org/abs/1505.00853 for a comparison of
 * ReLU variants.
 * @author Alex Black
 */
public class LeakyReLU extends BaseTransformOp {
    private double alpha = 0.01;

    public LeakyReLU() {
        alpha = 0.01;
    }
    
    public LeakyReLU(INDArray x, double alpha) {
        super(x);
        this.alpha = alpha;
    }
    
    public LeakyReLU(INDArray x, INDArray z, double alpha) {
        super(x, z);
        this.alpha = alpha;
    }

    public LeakyReLU(INDArray x, INDArray z, int n, double alpha) {
        super(x, z, n);
        this.alpha = alpha;
    }

    public LeakyReLU(INDArray x, INDArray y, INDArray z, int n, double alpha) {
        super(x, y, z, n);
        this.alpha = alpha;
    }

    public LeakyReLU(INDArray x, INDArray z) {
        super(x, z);
    }

    public LeakyReLU(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public LeakyReLU(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public LeakyReLU(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 31;
    }

    @Override
    public String name() {
        return "leakyrelu";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
    	double rv = origin.realComponent().doubleValue(); 
        return rv < 0 ? Nd4j.createComplexNumber(alpha*rv,0) : origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
    	double rv = origin.realComponent().doubleValue(); 
        return rv < 0 ? Nd4j.createComplexNumber(alpha*rv,0) : origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
    	double rv = origin.realComponent().doubleValue(); 
        return rv < 0 ? Nd4j.createComplexNumber(alpha*rv,0) : origin;
    }

    @Override
    public float op(float origin, float other) {
        return origin < 0 ? (float)alpha*origin : origin;
    }

    @Override
    public double op(double origin, double other) {
        return origin < 0 ?  alpha*origin : origin;
    }

    @Override
    public double op(double origin) {
    	return origin < 0 ?  alpha * origin : origin;
    }

    @Override
    public float op(float origin) {
    	return origin < 0 ? (float)alpha * origin : origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
    	double rv = origin.realComponent().doubleValue(); 
        return rv < 0 ? Nd4j.createComplexNumber(alpha*rv,0) : origin;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new LeakyReLU(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), alpha);
        else
            return new LeakyReLU(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length(), alpha);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new LeakyReLU(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(), alpha);
        else
            return new LeakyReLU(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length(), alpha);

    }

    @Override
    public TransformOp derivative() {
        return new LeakyReLUDerivative(x,y,z,n,alpha);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {alpha};
    }
}
