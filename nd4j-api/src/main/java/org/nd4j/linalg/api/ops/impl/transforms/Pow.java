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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Pow function
 *
 * @author Adam Gibson
 */
public class Pow extends BaseTransformOp {
    private double pow;

    public Pow() {
    }

    public Pow(INDArray x, INDArray z, double pow) {
        super(x, z);
        this.pow = pow;
        init(x, null, z, x.length());
    }

    public Pow(INDArray x, INDArray z, int n, double pow) {
        super(x, z, n);
        this.pow = pow;
        init(x, null, z, n);

    }

    public Pow(INDArray x, INDArray y, INDArray z, int n, double pow) {
        super(x, y, z, n);
        this.pow = pow;
        init(x, y, z, n);

    }

    public Pow(INDArray x, double pow) {
        super(x);
        this.pow = pow;
        init(x, null, x, x.length());
    }

    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String name() {
        return "pow";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return ComplexUtil.pow(origin, pow);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return ComplexUtil.pow(origin, pow);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return ComplexUtil.pow(origin, pow);
    }

    @Override
    public float op(float origin, float other) {
        return (float) FastMath.pow(origin, pow);
    }

    @Override
    public double op(double origin, double other) {
        return FastMath.pow(origin, pow);
    }

    @Override
    public double op(double origin) {
        return FastMath.pow(origin, pow);
    }

    @Override
    public float op(float origin) {
        return (float) FastMath.pow(origin, pow);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return ComplexUtil.pow(origin, pow);
    }


    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Pow(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), pow);
        else
            return new Pow(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length(), pow);

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Pow(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(), pow);
        else
            return new Pow(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length(), pow);

    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        this.extraArgs = new Object[]{pow};
    }
}
