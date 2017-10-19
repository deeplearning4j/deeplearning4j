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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * Swish function
 *
 * @author raver119@gmail.com
 */
public class Swish extends BaseTransformOp {
    public Swish(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Swish(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public Swish(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Swish() {}

    public Swish(INDArray x, INDArray z) {
        super(x, z);
    }

    public Swish(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Swish(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Swish(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public Swish(INDArray ndArray) {
        super(ndArray);
    }

    @Override
    public int opNum() {
        return 74;
    }

    @Override
    public String name() {
        return "swish";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return sigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return sigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return sigmoid(origin);
    }

    @Override
    public float op(float origin, float other) {
        return origin * (float) sigmoid(origin);
    }

    @Override
    public double op(double origin, double other) {
        return origin * sigmoid(origin);
    }

    @Override
    public double op(double origin) {
        return origin * sigmoid(origin);
    }

    @Override
    public float op(float origin) {
        return origin * (float) sigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        throw new UnsupportedOperationException();
    }


    private double sigmoid(double input) {
        double inputf = input;
        double val = 1 / (1 + FastMath.exp(-inputf));
        if (Nd4j.ENFORCE_NUMERICAL_STABILITY && (Double.isNaN(val) || Double.isInfinite(val))) {
            val = Nd4j.EPS_THRESHOLD;
        }
        return val;
    }

    @Override
    public TransformOp derivative() {
        return new SwishDerivative(x, y, z, n);
    }

    private IComplexNumber sigmoid(IComplexNumber number) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Swish(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Swish(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Swish(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Swish(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());

    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().swishDerivative(arg(), i_v.get(0));

        return Collections.singletonList(ret);
    }


}
