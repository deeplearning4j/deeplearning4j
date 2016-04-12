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
 * Signum function
 *
 * @author Adam Gibson
 */
public class Sign extends BaseTransformOp {
    public Sign() {
    }

    public Sign(INDArray x, INDArray z) {
        super(x, z);
    }

    public Sign(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Sign(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Sign(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 11;
    }

    @Override
    public String name() {
        return "sign";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return sign(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return sign(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return sign(origin);
    }

    @Override
    public float op(float origin, float other) {
        return (float) sign(origin);
    }

    @Override
    public double op(double origin, double other) {
        return sign(origin);
    }

    @Override
    public double op(double origin) {
        return sign(origin);
    }

    @Override
    public float op(float origin) {
        return (float) sign(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return sign(origin);
    }

    private double sign(double n) {
        if (n < 0)
            return (double) -1;
        else if (n > 0)
            return (double) 1;
        return (double) 0;
    }

    private IComplexNumber sign(IComplexNumber n) {
        if (n.realComponent().doubleValue() > 0)
            return Nd4j.createDouble(1, 0);
        else if (n.realComponent().doubleValue() < 0)
            return Nd4j.createDouble(-1, 0);

        return Nd4j.createDouble(0, 0);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Sign(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Sign(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Sign(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Sign(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

}
