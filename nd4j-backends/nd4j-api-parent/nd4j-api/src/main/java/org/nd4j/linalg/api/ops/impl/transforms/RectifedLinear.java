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

/**
 *
 * Rectified linear units
 *
 * @author Adam Gibson
 */
public class RectifedLinear extends BaseTransformOp {
    private double cutoff = 0.0;

    public RectifedLinear() {
    }

    public RectifedLinear(INDArray x, INDArray z, double cutoff) {
        super(x, z);
        this.cutoff = cutoff;
    }

    public RectifedLinear(INDArray x, INDArray z, long n, double cutoff) {
        super(x, z, n);
        this.cutoff = cutoff;
    }

    public RectifedLinear(INDArray x, INDArray y, INDArray z, long n, double cutoff) {
        super(x, y, z, n);
        this.cutoff = cutoff;
    }

    public RectifedLinear(INDArray x, double cutoff) {
        super(x);
        this.cutoff = cutoff;
    }

    public RectifedLinear(INDArray x, INDArray z) {
        super(x, z);
    }

    public RectifedLinear(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public RectifedLinear(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public RectifedLinear(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public RectifedLinear(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 33;
    }

    @Override
    public String name() {
        return "relu";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.realComponent().doubleValue() < cutoff ? Nd4j.createComplexNumber(cutoff,0) : origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.realComponent().doubleValue() < cutoff ? Nd4j.createComplexNumber(cutoff,0) : origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.realComponent().doubleValue() < cutoff ? Nd4j.createComplexNumber(cutoff,0) : origin;
    }

    @Override
    public float op(float origin, float other) {
        return origin < cutoff ? (float) cutoff : origin;
    }

    @Override
    public double op(double origin, double other) {
        return origin < cutoff ?  cutoff : origin;
    }

    @Override
    public double op(double origin) {
        return origin < cutoff ? cutoff : origin;

    }

    @Override
    public float op(float origin) {
        return origin < cutoff ? (float) cutoff : origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin.realComponent().doubleValue() < cutoff ? Nd4j.createComplexNumber(cutoff,0) : origin;

    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new RectifedLinear(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), cutoff);
        else
            return new RectifedLinear(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length(), cutoff);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new RectifedLinear(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(), cutoff);
        else
            return new RectifedLinear(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length(), cutoff);

    }

    @Override
    public TransformOp derivative() {
        return new Step(x,y,z,n,cutoff);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {cutoff};
    }
}
