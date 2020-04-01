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

package org.nd4j.linalg.api.ops.impl.transforms.strict;

import java.util.Collections;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformFloatOp;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;

import java.util.Arrays;
import java.util.List;

/**
 * SELU activation function
 * <p>
 * https://arxiv.org/pdf/1706.02515.pdf
 *
 * @author raver119@gmail.com
 */
public class SELU extends BaseTransformStrictOp {

    private static final double SELU_ALPHA = 1.6732632423543772848170429916717;
    private static final double SELU_LAMBDA = 1.0507009873554804934193349852946;

    public SELU(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public SELU(SameDiff sameDiff, SDVariable i_v) {
        this(sameDiff, i_v, false);
    }

    public SELU() {
    }

    public SELU(INDArray x, INDArray z) {
        super(x, z);
    }

    public SELU(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 42;
    }

    @Override
    public String opName() {
        return "selu";
    }

    @Override
    public String onnxName() {
        return "Selu";
    }

    @Override
    public String tensorflowName() {
        return "Selu";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(f().seluBp(arg(), i_v.get(0)));
    }

}
