/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.impl.accum.longer;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceLongOp;
import org.nd4j.linalg.api.ops.BaseReduceOp;

import java.util.Collections;
import java.util.List;

/**
 * Count the number of zero elements
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class CountZero extends BaseReduceLongOp {

    public CountZero(SameDiff sameDiff, SDVariable input, int... dimensions) {
        super(sameDiff, input, dimensions);
    }


    public CountZero(INDArray x) {
        super(x);
    }


    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "countZero";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public String tensorflowName() {
        return "count_zero";
    }

    @Override
    public Type getOpType() {
        return Type.AGGREGATION;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f().zerosLike(arg()));
    }

}
