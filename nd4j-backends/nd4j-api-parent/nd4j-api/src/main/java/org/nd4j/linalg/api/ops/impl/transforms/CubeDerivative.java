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
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * Cube derivative, e.g. 3x^2
 */
public class CubeDerivative extends BaseTransformOp {

    public CubeDerivative() {}

    public CubeDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public CubeDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public CubeDerivative(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public CubeDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 50;
    }

    @Override
    public String name() {
        return "cubederivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        throw new UnsupportedOperationException("Cube Derivative not supported on Complex Numbers");
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        throw new UnsupportedOperationException("Cube Derivative not supported on Complex Numbers");

    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        throw new UnsupportedOperationException("Cube Derivative not supported on Complex Numbers");
    }

    @Override
    public float op(float origin, float other) {
        return (float) (3 * FastMath.pow(origin, 2));
    }

    @Override
    public double op(double origin, double other) {
        return 3 * FastMath.pow(origin, 2);
    }

    @Override
    public double op(double origin) {
        return 3 * FastMath.pow(origin, 2);
    }

    @Override
    public float op(float origin) {
        return (float) (3 * FastMath.pow(origin, 2));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        throw new UnsupportedOperationException("Cube Derivative not supported on Complex Numbers");
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new CubeDerivative(x.vectorAlongDimension(index, dimension),
                            y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new CubeDerivative(x.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new CubeDerivative(x.tensorAlongDimension(index, dimension),
                            y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new CubeDerivative(x.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }
}
