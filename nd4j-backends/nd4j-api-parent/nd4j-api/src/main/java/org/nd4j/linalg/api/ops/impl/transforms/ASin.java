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
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Arcsin elementwise function
 *
 * @author Adam Gibson
 */
public class ASin extends BaseTransformOp {

    public ASin() {
    }

    public ASin(INDArray x, INDArray z) {
        super(x, z);
    }

    public ASin(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public ASin(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public ASin(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 17;
    }

    @Override
    public String name() {
        return "asin";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return ComplexUtil.asin(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return ComplexUtil.asin(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return ComplexUtil.asin(origin);
    }

    @Override
    public float op(float origin, float other) {
        return (float) FastMath.asin(origin);
    }

    @Override
    public double op(double origin, double other) {
        return FastMath.asin(origin);
    }

    @Override
    public double op(double origin) {
        return FastMath.asin(origin);
    }

    @Override
    public float op(float origin) {
        return (float) FastMath.asin(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return ComplexUtil.asin(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new ASin(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new ASin(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new ASin(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new ASin(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }
}
