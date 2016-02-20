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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Max out activation:
 * http://arxiv.org/pdf/1302.4389.pdf
 *
 * @author Adam Gibson
 */
public class MaxOut extends BaseTransformOp {

    private IComplexNumber maxComplex = Nd4j.createComplexNumber(Double.NaN, Double.NaN);
    private Number max = Double.NaN;

    public MaxOut() {
    }

    public MaxOut(INDArray x, INDArray z) {
        super(x, z);
    }

    public MaxOut(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public MaxOut(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public MaxOut(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        throw new UnsupportedOperationException();
    }


    @Override
    public String name() {
        return "maxout";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        setMax(extraArgs);
        return maxComplex;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        setMax(extraArgs);
        return maxComplex;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        setMax(extraArgs);
        return maxComplex;
    }

    @Override
    public float op(float origin, float other) {
        setMax(extraArgs);
        return max.floatValue();
    }

    @Override
    public double op(double origin, double other) {
        setMax(extraArgs);
        return max.doubleValue();
    }

    @Override
    public double op(double origin) {
        setMax(extraArgs);
        return max.doubleValue();
    }

    @Override
    public float op(float origin) {
        setMax(extraArgs);
        return max.floatValue();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        setMax(extraArgs);
        return maxComplex;
    }

    private void setMax(Object[] extraArgs) {
        if (extraArgs.length < 1)
            throw new IllegalArgumentException("Please specify a max value");
        if (Double.isNaN(max.doubleValue()) && extraArgs[0] instanceof Number) {
            max = Double.valueOf(extraArgs[0].toString());
        } else if (Double.isNaN(maxComplex.realComponent().doubleValue()) && extraArgs[0] instanceof IComplexNumber) {
            maxComplex = (IComplexNumber) extraArgs[0];
        }
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new MaxOut(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new MaxOut(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new MaxOut(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new MaxOut(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

}
