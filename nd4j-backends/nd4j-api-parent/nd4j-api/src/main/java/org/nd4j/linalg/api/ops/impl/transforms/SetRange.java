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
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Set range to a particular set of values
 *
 * @author Adam Gibson
 */
public class SetRange extends BaseTransformOp {

    private double min, max;

    public SetRange() {
    }

    public SetRange(INDArray x) {
        this(x,0,1);
    }

    public SetRange(INDArray x, INDArray z, double min, double max) {
        super(x, z);
        this.min = min;
        this.max = max;
        init(x, y, z, n);
    }

    public SetRange(INDArray x, INDArray z, long n, double min, double max) {
        super(x, z, n);
        this.min = min;
        this.max = max;
        init(x, y, z, n);
    }

    public SetRange(INDArray x, INDArray y, INDArray z, long n, double min, double max) {
        super(x, y, z, n);
        this.min = min;
        this.max = max;
        init(x, y, z, n);
    }

    public SetRange(INDArray x, double min, double max) {
        super(x);
        this.min = min;
        this.max = max;
        init(x, y, z, n);
    }

    @Override
    public int opNum() {
        return 9;
    }

    @Override
    public String name() {
        return "setrange";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return op(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return op(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return op(origin);
    }

    @Override
    public float op(float origin, float other) {
        return op(origin);
    }

    @Override
    public double op(double origin, double other) {
        return op(origin);
    }

    @Override
    public double op(double origin) {
        if (origin >= min && origin <= max)
            return origin;
        if (min == 0 && max == 1) {
            double val = 1 / (1 + FastMath.exp(-origin));
            return (FastMath.floor(val * (max - min)) + min);
        }

        double ret = (FastMath.floor(origin * (max - min)) + min);
        return ret;
    }

    @Override
    public float op(float origin) {
        return (float) op((double) origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return Nd4j.createComplexNumber(op(origin.realComponent().doubleValue()), op(origin.imaginaryComponent().doubleValue()));
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[]{min, max};
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new SetRange(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), min, max);
        else
            return new SetRange(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), min, max);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new SetRange(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(), min, max);
        else
            return new SetRange(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(), min, max);

    }
}
