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

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Subtraction operation
 *
 * @author Adam Gibson
 */
public class SubOp extends DynamicCustomOp {

    public SubOp() {}

    public SubOp( SameDiff sameDiff, DifferentialFunction[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public SubOp( INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }

    @Override
    public List<int[]> calculateOutputShape() {
        return Arrays.asList(arg().getResultShape());
    }

    @Override
    public String opName() {
        return "sub";
    }


    @Override
    public String onnxName() {
        return "Sub";
    }

    @Override
    public String tensorflowName() {
        return "Sub";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction gradWrtX = i_v.get(0);
        DifferentialFunction gradWrtY = f().neg(i_v.get(0));
        List<DifferentialFunction> ret = new ArrayList<>();
        ret.add(gradWrtX);
        ret.add(gradWrtY);
        return ret;
    }

}
