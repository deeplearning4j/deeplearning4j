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

import java.util.Collections;
import java.util.List;

/**
 * Arcsin elementwise function
 *
 * @author Adam Gibson
 */
public class ASinh extends BaseTransformOp {
    public ASinh(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public ASinh(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public ASinh(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public ASinh() {}

    public ASinh(INDArray x, INDArray z) {
        super(x, z);
    }

    public ASinh(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public ASinh(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public ASinh(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 17;
    }

    @Override
    public String name() {
        return "asinh";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float op(float origin, float other) {
        return (float) FastMath.asinh(origin);
    }

    @Override
    public double op(double origin, double other) {
        return FastMath.asinh(origin);
    }

    @Override
    public double op(double origin) {
        return FastMath.asinh(origin);
    }

    @Override
    public float op(float origin) {
        return (float) FastMath.asinh(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new ASinh(xAlongDimension, y.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new ASinh(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new ASinh(xAlongDimension, y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new ASinh(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().div(f().one(getResultShape()),f().sqrt(f().add(f().pow(arg(),2),
                f().one(getResultShape()))));

        return Collections.singletonList(ret);
    }
}
