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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Soft max function
 * row_maxes is a row vector (max for each row)
 * row_maxes = rowmaxes(input)
 * diff = exp(input - max) / diff.rowSums()
 * Outputs a probability distribution.
 * Note that this is a parameterized model and requires
 * the sum and max for the vector being calculated
 *
 * @author Adam Gibson
 */

public class OldSoftMax extends BaseTransformOp {
    public OldSoftMax(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public OldSoftMax(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public OldSoftMax(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public OldSoftMax(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public OldSoftMax(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public OldSoftMax() {
    }

    public OldSoftMax(INDArray x, INDArray z) {
        this(x, null, z);

    }

    public OldSoftMax(INDArray x, INDArray z, long n) {
        this(x, null, z, n);
    }

    public OldSoftMax(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public OldSoftMax(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, x.lengthLong());
    }

    public OldSoftMax(INDArray x) {
        super(x);

    }

    @Override
    public int opNum() {
        return 38;
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public String opName() {
        return "old_softmax";
    }


    @Override
    public String onnxName() {
        return "Softmax";
    }

    @Override
    public String tensorflowName() {
        return "Softmax";
    }


    @Override
    public void exec() {
        exec(1);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        passThrough = true;
    }


    @Override
    public void exec(int... dimensions) {
        if (dimensions[0] != 1)
            throw new IllegalArgumentException("Only supports row wise calculations");
        if (x.isMatrix()) {
            INDArray maxAlongDimension = x.max(dimensions);
            if (!maxAlongDimension.isVector() && !maxAlongDimension.isScalar())
                throw new IllegalStateException("Max along dimension for input must either be a row vector or scalar");

            INDArray xMinusMax = x.subColumnVector(maxAlongDimension);

            INDArray exp;
            if (z != null) {
                exp = Nd4j.getExecutioner().execAndReturn(new Exp(xMinusMax, z));
            } else {
                exp = Nd4j.getExecutioner().execAndReturn(new Exp(xMinusMax));
            }

            INDArray sum = exp.sum(dimensions);
            exp.diviColumnVector(sum);

            if (z == null)
                z = exp;
        } else if (x.isVector()) {
            double max = x.maxNumber().doubleValue();
            INDArray exp;
            if (z != null) {
                exp = Nd4j.getExecutioner().execAndReturn(new Exp(x.sub(max), z));
            } else {
                exp = Nd4j.getExecutioner().execAndReturn(new Exp(x.sub(max)));
            }
            exp.divi(exp.sumNumber().doubleValue());
            this.z = exp;
        }
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = f().softmaxDerivative(arg(), i_v.get(0));
        return Arrays.asList(ret);
    }
}
