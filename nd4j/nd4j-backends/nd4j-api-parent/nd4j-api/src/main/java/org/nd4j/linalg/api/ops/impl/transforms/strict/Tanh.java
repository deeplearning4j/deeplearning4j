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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative;

import java.util.List;

/**
 * Tanh elementwise function
 *
 * @author Adam Gibson
 */
public class Tanh extends BaseTransformStrictOp {
    public Tanh(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Tanh(SameDiff sameDiff, SDVariable i_v) {
        this(sameDiff, i_v,false);
    }

    public Tanh() {
    }

    public Tanh(INDArray x, INDArray z) {
        super(x, z);
    }

    public Tanh(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 29;
    }

    @Override
    public String opName() {
        return "tanh";
    }

    @Override
    public String onnxName() {
        return "Tanh";
    }

    @Override
    public String tensorflowName() {
        return "Tanh";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return new TanhDerivative(sameDiff, arg(), i_v.get(0)).outputs();
    }
}
