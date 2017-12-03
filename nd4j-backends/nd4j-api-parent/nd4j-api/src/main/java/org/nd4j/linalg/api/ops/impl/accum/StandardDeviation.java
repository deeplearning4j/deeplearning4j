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
 * Standard deviation (sqrt of variance)
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends Variance {
    public StandardDeviation(SameDiff sameDiff, DifferentialFunction i_v, int[] dimensions, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, DifferentialFunction i_v, DifferentialFunction i_v2, int[] dimensions, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions, biasCorrected);
    }

    public StandardDeviation(INDArray x, boolean biasCorrected) {
        super(x, biasCorrected);
    }

    public StandardDeviation(INDArray x, INDArray y, INDArray z, long n, boolean biasCorrected) {
        super(x, y, z, n, biasCorrected);
    }

    public StandardDeviation() {}

    public StandardDeviation(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public StandardDeviation(INDArray x) {
        super(x);
    }

    public StandardDeviation(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "std";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        f().validateDifferentialFunctionsameDiff(i_v1);
        int inputs = f().getInputLength(i_v1.get(0));
        DifferentialFunction g =  f().doRepeat(this,i_v1.get(0),dimensions);
        DifferentialFunction ret = f().div(f().sub(f().mul(g,arg()),f().mean(arg(),dimensions)),f().mul(f()
                .one(g.getResultShape()),inputs));

        return Collections.singletonList(ret);
    }

}
