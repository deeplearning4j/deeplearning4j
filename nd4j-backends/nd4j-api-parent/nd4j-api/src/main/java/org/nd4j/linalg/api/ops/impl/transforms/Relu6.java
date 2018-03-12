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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Rectified linear unit 6, i.e. min(max(input, 0), 6).
 *
 * @author Max Pumperla
 */
public class Relu6 extends BaseTransformOp {


    public Relu6(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Relu6() {}

    public Relu6(INDArray x, INDArray z) {
        super(x, z);
    }

    public Relu6(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Relu6(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Relu6(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public Relu6(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 96;
    }

    @Override
    public String opName() {
        return "relu6";
    }

    @Override
    public String onnxName() { throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "relu6";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable step = new Step(sameDiff, arg(), false, 6.0).outputVariables()[0];
        SDVariable ret = step.mul(i_v.get(0));
        return Arrays.asList(ret);
    }
}
