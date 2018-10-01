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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceOp;

import java.util.Collections;
import java.util.List;

/**
 * Boolean AND accumulation
 *
 * @author raver119@gmail.com
 */
public class All extends BaseReduceOp {
    public All(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public All() {}

    public All(INDArray x) {
        super(x);
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }


    @Override
    public int opNum() {
        return 21;
    }

    @Override
    public String opName() {
        return "all";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f().zerosLike(arg()));
    }

    @Override
    public String onnxName() {
        return "All";
    }

    @Override
    public String tensorflowName() {
        return "All";
    }
}
