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

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Negative function
 *
 * @author Adam Gibson
 */
public class Negative extends BaseTransformOp {
    public Negative(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Negative(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public Negative(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
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
    public String opName() {
        return "neg";
    }


    @Override
    public String onnxName() {
        return "Neg";
    }

    @Override
    public String tensorflowName() {
        return "Neg";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Arrays.asList(f().neg(i_v.get(0)));
    }



}
