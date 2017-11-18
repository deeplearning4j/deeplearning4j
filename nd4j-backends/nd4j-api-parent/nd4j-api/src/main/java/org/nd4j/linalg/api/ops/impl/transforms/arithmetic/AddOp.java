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
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.ArrayList;
import java.util.List;

/**
 * Addition operation
 *
 * @author Adam Gibson
 */
public class AddOp extends BaseTransformOp {
    public AddOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public AddOp(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public AddOp() {}

    public AddOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public AddOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public AddOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public AddOp(INDArray x) {
        super(x, x);
    }

    public AddOp(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    @Override
    public int opNum() {
        return 0;
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
        return "add";
    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (y == null)
            throw new IllegalArgumentException("No components to add");
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
