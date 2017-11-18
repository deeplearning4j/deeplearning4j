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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;

import java.util.Collections;
import java.util.List;

/**
 * Sum the components
 *
 * @author Adam Gibson
 */
@Slf4j
public class Sum extends BaseAccumulation {
    public Sum(SameDiff sameDiff, DifferentialFunction i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Sum(SameDiff sameDiff, DifferentialFunction i_v, DifferentialFunction i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }



    public Sum() {}

    public Sum(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Sum(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Sum(INDArray x) {
        super(x);
    }

    public Sum(INDArray x, INDArray y) {
        super(x, y);
    }

    public Sum(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "sum";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        f().validateDifferentialFunctionsameDiff(i_v1);
        f().validateFunctionReference(i_v1);
        DifferentialFunction repeat =  f().doRepeat(
                i_v1.get(0),
                arg(),dimensions);
        return Collections.singletonList(repeat);
    }


    @Override
    public String onnxName() {
       return "ReduceSum";
    }

    @Override
    public String tensorflowName() {
        return "reduce_sum";
    }
}
