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
import org.nd4j.linalg.util.ComplexUtil;

/**
 *
 * Softsign element-wise activation function. f(x) = x/(1+abs(x))<br>
 * Similar in shape to tanh but may outperform it due to
 * 'gentler' nonlinearity (smoother asymptotes).<br>
 * See for example: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 * @author Alex Black
 */
public class SoftSign extends BaseTransformOp {

    public SoftSign() {
    }

    public SoftSign(INDArray x, INDArray z) {
        super(x, z);
    }

    public SoftSign(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public SoftSign(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public SoftSign(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 20;
    }

    @Override
    public String name() {
        return "softsign";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.div(ComplexUtil.abs(origin).addi(1.0));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
    	return origin.div(ComplexUtil.abs(origin).addi(1.0));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
    	return origin.div(ComplexUtil.abs(origin).addi(1.0));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
    	return origin.div(ComplexUtil.abs(origin).addi(1.0));
    }
    
    @Override
    public float op(float origin, float other) {
        return (float) softsign(origin);
    }

    @Override
    public double op(double origin, double other) {
        return softsign(origin);
    }

    @Override
    public double op(double origin) {
        return softsign(origin);
    }

    @Override
    public float op(float origin) {
        return (float) softsign(origin);
    }


    @Override
    public TransformOp derivative() {
        return new SoftSignDerivative(x, y, z, n);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new SoftSign(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SoftSign(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new SoftSign(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SoftSign(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());
    }
    
    private static double softsign(double x){
    	return x / (1.0 + Math.abs(x));
    }
}
