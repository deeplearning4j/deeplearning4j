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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Standard deviation (sqrt of variance)
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends Variance {
    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, int[] dimensions, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions, biasCorrected);
    }

    public StandardDeviation(INDArray x, boolean biasCorrected) {
        super(x, biasCorrected);
    }

    public StandardDeviation(INDArray x, INDArray y, INDArray z, long n, boolean biasCorrected) {
        super(x, y, z, n, biasCorrected);
    }

    public StandardDeviation() {
    }

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
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //Here: calculating dL/dIn given dL/dOut (i.e., i_v1) and input/output
        //If out = stdev(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        //dOut/dIn_i = (in_i-mean)/(stdev * (n-1))
        //TODO KEEP DIMS SUPPORT
        return Collections.singletonList(f().stdBp(arg(), i_v1.get(0), biasCorrected, false, dimensions));
    }

}
