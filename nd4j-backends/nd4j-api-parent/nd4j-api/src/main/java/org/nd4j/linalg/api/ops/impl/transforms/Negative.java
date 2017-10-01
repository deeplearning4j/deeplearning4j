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

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.Arrays;
import java.util.List;

/**
 * Negative function
 *
 * @author Adam Gibson
 */
public class Negative extends BaseTransformOp {
    public Negative(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Negative(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public Negative(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Negative() {}

    public Negative(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Negative(INDArray x, INDArray z) {
        super(x, z);
    }

    public Negative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Negative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 6;
    }

    @Override
    public String name() {
        return "neg";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin.neg();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin.neg();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin.neg();
    }

    @Override
    public float op(float origin, float other) {
        return -origin;
    }

    @Override
    public double op(double origin, double other) {
        return -origin;
    }

    @Override
    public double op(double origin) {
        return -origin;
    }

    @Override
    public float op(float origin) {
        return -origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin.neg();
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Negative(xAlongDimension, y.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Negative(xAlongDimension, z.vectorAlongDimension(index, dimension), x.lengthLong());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Negative(xAlongDimension, y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Negative(xAlongDimension, z.tensorAlongDimension(index, dimension), x.lengthLong());

    }

    @Override
    public ArrayField doGetValue() {
        return arg().getValue(true).negate();
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        return Arrays.asList(f().neg(arg().diff(i_v).get(0)));
    }

    @Override
    public String toString() {
        return "-" + arg().toString();
    }

    @Override
    public String doGetFormula(List<Variable> variables) {
        return "-" + arg().doGetFormula(variables);
    }


}
