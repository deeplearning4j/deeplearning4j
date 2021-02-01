/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.transforms.strict;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhBp;

import java.util.List;

/**
 * Hard tanh elementwise function
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class HardTanh extends BaseTransformStrictOp {
    public HardTanh(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public HardTanh(SameDiff sameDiff, SDVariable i_v) {
        this(sameDiff, i_v, false);
    }

    public HardTanh(INDArray x, INDArray z) {
        super(x, z);
    }

    public HardTanh(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 33;
    }

    @Override
    public String opName() {
        return "hardtanh";
    }


    @Override
    public String onnxName() {
        return "HardTanh";
    }

    @Override
    public String tensorflowName() {
        return "HardTanh";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return new HardTanhBp(sameDiff, arg(), i_v.get(0)).outputs();
    }
}
