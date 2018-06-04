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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardSigmoidDerivative;

import java.util.Collections;
import java.util.List;

/**
 * HardSigmoid function
 *
 * @author raver119@gmail.com
 */
public class HardSigmoid extends BaseTransformOp {
    public HardSigmoid() {}

    public HardSigmoid(INDArray x, INDArray z) {
        super(x, z);
    }

    public HardSigmoid(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public HardSigmoid(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public HardSigmoid(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public HardSigmoid(INDArray ndArray) {
        super(ndArray);
    }

    public HardSigmoid(SameDiff sameDiff, SDVariable in, boolean inPlace){
        super(sameDiff, in, inPlace);
    }

    @Override
    public int opNum() {
        return 51;
    }

    @Override
    public String opName() {
        return "hard_sigmoid";
    }

    @Override
    public String onnxName() {
        return "HardSigmoid";
    }

    @Override
    public String tensorflowName() {
        return "HardSigmoid";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable in = arg();
        SDVariable dOutdIn = new HardSigmoidDerivative(sameDiff, in, false).outputVariables()[0];
        return Collections.singletonList(dOutdIn.mul(f1.get(0)));
    }


}
