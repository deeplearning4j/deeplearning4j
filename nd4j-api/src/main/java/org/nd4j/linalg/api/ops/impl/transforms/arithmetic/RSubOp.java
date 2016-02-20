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

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * Subtraction operation
 *
 * @author Adam Gibson
 */
public class RSubOp extends BaseTransformOp {
    public RSubOp() {
    }

    public RSubOp(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public RSubOp(INDArray x) {
        super(x);
    }

    public RSubOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public RSubOp(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public RSubOp(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.length());
    }

    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String name() {
        return "rsub";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.rsub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.rsub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.rsub(other);
    }

    @Override
    public float op(float origin, float other) {
        return other - origin;
    }

    @Override
    public double op(double origin, double other) {
        return other - origin;
    }

    @Override
    public double op(double origin) {
        return origin;
    }

    @Override
    public float op(float origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }


    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);


        if (y() != null)
            return new RSubOp(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new RSubOp(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);


        if (y() != null)
            return new RSubOp(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new RSubOp(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        if (y == null)
            throw new IllegalArgumentException("No components to subtract");
    }
}
