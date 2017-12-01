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
 * Addition operation
 *
 * @author Adam Gibson
 */
public class AddOp extends DynamicCustomOp {

    public AddOp() {}

    public AddOp( SameDiff sameDiff, DifferentialFunction[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public AddOp( INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }

    @Override
    public List<int[]> calculateOutputShape() {
        return Arrays.asList(arg().getResultShape());
    }

    @Override
    public String opName() {
        return "add";
    }


    @Override
    public String onnxName() {
        return "Add";
    }

    @Override
    public String tensorflowName() {
        return "Add";
    }




    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction g = sameDiff.setupFunction(i_v.get(0));
        List<DifferentialFunction> ret = new ArrayList<>();
        for(int i = 0; i < 2; i++)
            ret.add(g);

        return ret;
    }


}
