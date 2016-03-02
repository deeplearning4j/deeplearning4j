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
 * Stabilization function, forces values to be within a range
 *
 * @author Adam Gibson
 */
public class Stabilize extends BaseTransformOp {
    double realMin = 1.1755e-38f;
    double cutOff = FastMath.log(realMin);
    double k;

    public Stabilize() {
    }

    public Stabilize(INDArray x, INDArray z, double k) {
        super(x, z);
        this.k = k;
    }

    public Stabilize(INDArray x, INDArray z, int n, double k) {
        super(x, z, n);
        this.k = k;
    }

    public Stabilize(INDArray x, INDArray y, INDArray z, int n, double k) {
        super(x, y, z, n);
        this.k = k;
    }

    public Stabilize(INDArray x, double k) {
        super(x);
        this.k = k;
    }

    @Override
    public int opNum() {
        return 28;
    }

    @Override
    public String name() {
        return "stabilize";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return stabilize(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return stabilize(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return stabilize(origin);
    }

    @Override
    public float op(float origin, float other) {
        return stabilize(origin);
    }

    private float stabilize(float curr) {
        if (curr * k > -cutOff)
            return (float) (-cutOff / k);
        else if (curr * k < cutOff)
            return (float) (cutOff / k);
        return curr;
    }

    private double stabilize(double curr) {
        if (curr * k > -cutOff)
            return (float) (-cutOff / k);
        else if (curr * k < cutOff)
            return (float) (cutOff / k);
        return curr;
    }

    @Override
    public double op(double origin, double other) {
        return stabilize(origin);
    }

    @Override
    public double op(double origin) {
        return stabilize(origin);
    }

    @Override
    public float op(float origin) {
        return stabilize(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return stabilize(origin);
    }


    private IComplexNumber stabilize(IComplexNumber c) {
        double realMin = 1.1755e-38f;
        double cutOff = FastMath.log(realMin);
        double curr = c.realComponent().doubleValue();
        if (curr * k > -cutOff)
            return Nd4j.createDouble(-cutOff / k, c.imaginaryComponent().doubleValue());
        else if (curr * k < cutOff)
            return Nd4j.createDouble(cutOff / k, c.imaginaryComponent().doubleValue());
        return c;

    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[]{k};
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new Stabilize(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), k);
        else
            return new Stabilize(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length(), k);

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new Stabilize(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(), k);
        else
            return new Stabilize(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length(), k);

    }

}
