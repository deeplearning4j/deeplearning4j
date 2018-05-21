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

package org.nd4j.linalg.api.ops.impl.transforms.comparison;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Not equal to function:
 * Bit mask over whether 2 elements are not equal or not
 *
 * @author Adam Gibson
 */
public class OldNotEqualTo extends BaseTransformOp {
    public OldNotEqualTo(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public OldNotEqualTo(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public OldNotEqualTo() {
    }

    public OldNotEqualTo(INDArray x) {
        super(x);
    }

    public OldNotEqualTo(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public OldNotEqualTo(INDArray x, INDArray z) {
        super(x, z);
    }

    public OldNotEqualTo(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    @Override
    public int opNum() {
        return 15;
    }

    @Override
    public String opName() {
        return "old_neq";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No op name found");
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {

        return Arrays.asList(f().neg(i_v.get(0)));
    }


}
