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

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;

import java.util.List;

/**
 * Count the number of non-zero elements
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class CountNonZero extends BaseAccumulation {

    public CountNonZero(SameDiff sameDiff, SDVariable input) {
        super(sameDiff, input);
    }


    public CountNonZero(INDArray x) {
        super(x);
    }


    @Override
    public int opNum() {
        return 22;
    }

    @Override
    public String opName() {
        return "countNonZero";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public String tensorflowName() {
        return "count_nonzero";
    }

    @Override
    public Type getOpType() {
        return Type.AGGREGATION;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }

}
