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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;


import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.List;

/**
 * HardSigmoid derivative
 *
 * @author raver119@gmail.com
 */
public class HardSigmoidDerivative extends BaseTransformOp {
    public HardSigmoidDerivative(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public HardSigmoidDerivative(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public HardSigmoidDerivative(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public HardSigmoidDerivative() {}

    public HardSigmoidDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public HardSigmoidDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public HardSigmoidDerivative(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public HardSigmoidDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 52;
    }

    @Override
    public String name() {
        return "hard_sigmoidderivative";
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
        return (float) hardSigmoidDeriv(origin);
    }

    @Override
    public double op(double origin, double other) {
        return hardSigmoidDeriv(origin);
    }

    @Override
    public double op(double origin) {
        return hardSigmoidDeriv(origin);
    }

    @Override
    public float op(float origin) {
        return (float) hardSigmoidDeriv(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return sigmoidDeriv(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new HardSigmoidDerivative(x.vectorAlongDimension(index, dimension),
                            y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new HardSigmoidDerivative(x.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new HardSigmoidDerivative(x.tensorAlongDimension(index, dimension),
                            y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());
        else
            return new HardSigmoidDerivative(x.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

    private static double hardSigmoidDeriv(double input) {
        return input < -2.5 || input > 2.5 ? 0.0 : 0.2;
    }

    private static IComplexNumber sigmoidDeriv(IComplexNumber number) {
        return null;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
