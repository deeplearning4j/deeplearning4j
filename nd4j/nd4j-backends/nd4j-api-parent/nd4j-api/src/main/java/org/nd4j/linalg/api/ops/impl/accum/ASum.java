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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;

import java.util.Collections;
import java.util.List;

/**
 * Absolute sum the components
 *
 * @author raver119@gmail.com
 */
public class ASum extends BaseAccumulation {
    public ASum(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public ASum(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public ASum() {}

    public ASum(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public ASum(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public ASum(INDArray x) {
        super(x);
    }

    public ASum(INDArray x, INDArray y) {
        super(x, y);
    }


    @Override
    public int opNum() {
        return 11;
    }

    @Override
    public String opName() {
        return "asum";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable sgn = sameDiff.sign(arg());
        SDVariable meanBp = f().sumBp(sameDiff.abs(arg()), f1.get(0), false, dimensions);
        return Collections.singletonList(sgn.mul(meanBp));
    }
}
