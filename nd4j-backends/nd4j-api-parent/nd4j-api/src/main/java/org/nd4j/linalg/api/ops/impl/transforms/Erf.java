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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Gaussian error function (erf) function, which is defined as
 * <p>
 * erf(x) = 1 / sqrt(pi) * integral_(-x, x) exp(-t^2) dt
 *
 * @author raver119@gmail.com
 */
public class Erf extends BaseTransformOp {
    public Erf(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Erf(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public Erf(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Erf() {
    }

    public Erf(INDArray x, INDArray z) {
        super(x, z);
    }

    public Erf(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Erf(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Erf(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 78;
    }

    @Override
    public String opName() {
        return "erf";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Erf";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Derivative of erf(z) is 2 / sqrt(pi) * e^(-z^2)
        SDVariable gradient = i_v.get(0);
        SDVariable constant = sameDiff.onesLike(gradient).mul(2).div(Math.sqrt(Math.PI));
        SDVariable ret = constant.mul(sameDiff.exp(gradient.mul(gradient).mul(-1)));
        return Arrays.asList(ret);
    }

}
