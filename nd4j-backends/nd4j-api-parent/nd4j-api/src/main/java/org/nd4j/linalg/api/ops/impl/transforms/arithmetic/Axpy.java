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

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

/**
 *  Level 1 blas op Axpy as libnd4j native op
 *
 * @author raver119@gmail.com
 */
public class Axpy extends BaseTransformOp {

    private double p;

    public Axpy() {

    }

    public Axpy(INDArray x, INDArray z, double p) {
        //      super(x, z, z, z.lengthLong());
        this.p = p;
        init(x, z, z, x.length());
    }

    public Axpy(INDArray x, INDArray z, double p, long n) {
        //        super(x, z, n);
        this.p = p;
        init(x, z, z, n);
    }

    public Axpy(INDArray x, INDArray y, INDArray z, double p, long n) {
        //        super(x,y,z,n);
        this.p = p;
        init(x, y, z, x.length());
    }

    @Override
    public int opNum() {
        return 17;
    }

    @Override
    public String name() {
        return "axpy";
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
            return new Axpy(xAlongDimension, z.vectorAlongDimension(index, dimension), p, xAlongDimension.length());
        else
            throw new IllegalStateException("op.Y can't be null");
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Axpy(xAlongDimension, z.tensorAlongDimension(index, dimension), p, xAlongDimension.length());
        else
            throw new IllegalStateException("op.Y can't be null");

    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);

        if (x.lengthLong() < n || y.lengthLong() < n || z.lengthLong() < n)
            throw new IllegalStateException("Mis matched lengths: X: [" + x.lengthLong() + "], Y: [" + y.lengthLong()
                            + "], Z: [" + z.lengthLong() + "], N: [" + n + "]");

        this.extraArgs = new Object[] {p, (double) n};
    }
}
