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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * Rational Tanh Derivative, as described at as described at https://github.com/deeplearning4j/libnd4j/issues/351
 *
 * @author raver119@gmail.com
 * @author AlexDBlack
 */
public class RationalTanhDerivative extends BaseTransformOp {

    public RationalTanhDerivative() {}

    public RationalTanhDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public RationalTanhDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public RationalTanhDerivative(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public RationalTanhDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 54;
    }

    @Override
    public String name() {
        return "rational_tanh_derivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return null;
    }

    @Override
    public float op(float origin, float other) {
        return 0;
    }

    @Override
    public double op(double origin, double other) {
        return 0;
    }

    @Override
    public double op(double origin) {
        return 0;
    }

    @Override
    public float op(float origin) {
        return 0;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return null;
    }


    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new RationalTanhDerivative(x.vectorAlongDimension(index, dimension),
                            y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new RationalTanhDerivative(x.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new RationalTanhDerivative(x.tensorAlongDimension(index, dimension),
                            y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new RationalTanhDerivative(x.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }
}
