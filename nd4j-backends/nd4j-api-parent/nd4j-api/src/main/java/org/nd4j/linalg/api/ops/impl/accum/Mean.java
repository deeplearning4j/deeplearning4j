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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;

/**
 * Calculate the mean of the vector
 *
 * @author Adam Gibson
 */
public class Mean extends Sum {
    public Mean(SameDiff sameDiff, DifferentialFunction i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Mean(SameDiff sameDiff, DifferentialFunction i_v, DifferentialFunction i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Mean() {}

    public Mean(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Mean(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Mean(INDArray x) {
        super(x);
    }

    public Mean(INDArray x, INDArray y) {
        super(x, y);
    }

    public Mean(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "mean";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        f().validateDifferentialFunctionsameDiff(i_v1);
        DifferentialFunction ret = f().div(f().doRepeat(this,i_v1.get(0),dimensions),
                f().mul(f().one(i_v1.get(0).getResultShape()),
                        f().getInputLength(i_v1.get(0))));

        return Collections.singletonList(ret);
    }

    @Override
    public String onnxName() {
        return "ReduceMean";
    }

    @Override
    public String tensorflowName() {
        return "reduce_mean";
    }
}
