/*-
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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SELUDerivative;

import java.util.Collections;
import java.util.List;

/**
 * SELU activation function
 *
 * https://arxiv.org/pdf/1706.02515.pdf
 *
 * @author raver119@gmail.com
 */
public class SELU extends BaseTransformOp {

    private static final double SELU_ALPHA = 1.6732632423543772848170429916717;
    private static final double SELU_LAMBDA = 1.0507009873554804934193349852946;

    public SELU(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public SELU(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public SELU(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public SELU() {}

    public SELU(INDArray x, INDArray z) {
        super(x, z);
    }

    public SELU(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public SELU(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 67;
    }

    @Override
    public String name() {
        return "selu";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float op(float origin, float other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double op(double origin, double other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double op(double d1) {
        return d1 > 0.0f ? SELU_LAMBDA * d1 : SELU_LAMBDA * (SELU_ALPHA * FastMath.exp(d1) - SELU_ALPHA);
    }

    @Override
    public float op(float d1) {
        return d1 > 0.0f ? (float) SELU_LAMBDA * d1
                        : (float) ((float) SELU_LAMBDA * ((float) SELU_ALPHA * FastMath.exp(d1) - (float) SELU_ALPHA));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        throw new UnsupportedOperationException();
    }

    @Override
    public TransformOp derivative() {
        return new SELUDerivative(x, z, n);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        return new SELU(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        return new SELU(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().div(arg(),f().selu(arg()));
        return Collections.singletonList(ret);
    }

}
