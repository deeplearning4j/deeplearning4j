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
 * Tan Derivative elementwise function
 *
 * @author raver119@gmail.com
 */
public class TanDerivative extends BaseTransformOp {

    public TanDerivative() {}

    public TanDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public TanDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public TanDerivative(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public TanDerivative(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public TanDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 66;
    }

    @Override
    public String name() {
        return "tanderivative";
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
        return 1.0f / (float) FastMath.pow(FastMath.cos(origin), 2);
    }

    @Override
    public double op(double origin, double other) {
        return 1.0f / (float) FastMath.pow(FastMath.cos(origin), 2);
    }

    @Override
    public double op(double origin) {
        return 1.0f / (float) FastMath.pow(FastMath.cos(origin), 2);
    }

    @Override
    public float op(float origin) {
        return 1.0f / (float) FastMath.pow(FastMath.cos(origin), 2);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new TanDerivative(xAlongDimension, y.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new TanDerivative(xAlongDimension, z.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new TanDerivative(xAlongDimension, y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new TanDerivative(xAlongDimension, z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());

    }
}
