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

package org.nd4j.linalg.api.ops.impl.reduce.bool;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceBoolOp;

import java.util.Collections;
import java.util.List;

/**
 * IsInf function
 *
 * @author raver119@gmail.com
  */
public class IsInf extends BaseReduceBoolOp {
    public IsInf(SameDiff sameDiff, SDVariable i_v, int[] dims) {
        super(sameDiff, i_v, dims);
    }

    public IsInf(SameDiff sameDiff, SDVariable i_v, int[] dims, boolean keepDims) {
        super(sameDiff, i_v, dims, keepDims);
    }


    public IsInf() {}

    public IsInf(INDArray x, INDArray z) {
        super(x, z, false, null);
    }

    public IsInf(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String opName() {
        return "hasinf";
    }

    @Override
    public String onnxName() {
        return "HasInf";
    }

    @Override
    public String tensorflowName() {
        return "HasInf";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(f().zerosLike(arg()));
    }

}
