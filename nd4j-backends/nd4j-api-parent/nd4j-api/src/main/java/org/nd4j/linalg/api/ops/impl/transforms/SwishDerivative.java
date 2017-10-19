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
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Swish derivative
 *
 * @author Adam Gibson
 */
public class SwishDerivative extends BaseTransformOp {
    public SwishDerivative(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public SwishDerivative(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public SwishDerivative(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public SwishDerivative(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public SwishDerivative(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public SwishDerivative() {}

    public SwishDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public SwishDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public SwishDerivative(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public SwishDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 75;
    }

    @Override
    public String name() {
        return "_swishderivative";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return sigmoidDeriv(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return sigmoidDeriv(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return sigmoidDeriv(origin);
    }

    @Override
    public float op(float origin, float other) {
        return (float) swishDeriv(origin);
    }

    @Override
    public double op(double origin, double other) {
        return swishDeriv(origin);
    }

    @Override
    public double op(double origin) {
        return swishDeriv(origin);
    }

    @Override
    public float op(float origin) {
        return (float) swishDeriv(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return sigmoidDeriv(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new SwishDerivative(x.vectorAlongDimension(index, dimension),
                            y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new SwishDerivative(x.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new SwishDerivative(x.tensorAlongDimension(index, dimension),
                            y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new SwishDerivative(x.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

    private static double swishDeriv(double input) {
        double ex = FastMath.pow(Math.E, input);
        return (ex * (input + ex + 1)) / FastMath.pow((ex + 1) , 2);
    }

    private static IComplexNumber sigmoidDeriv(IComplexNumber number) {
        throw new UnsupportedOperationException();
    }

       @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
       throw new UnsupportedOperationException();
    }
}
