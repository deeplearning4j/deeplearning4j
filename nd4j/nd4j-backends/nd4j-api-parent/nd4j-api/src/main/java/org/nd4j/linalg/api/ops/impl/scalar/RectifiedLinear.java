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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.ThresholdReluBp;

import java.util.List;

/**
 * Rectified linear units
 *
 * @author Adam Gibson
 */
public class RectifiedLinear extends BaseScalarOp {
    public RectifiedLinear(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double cutoff) {
        super(sameDiff, i_v, cutoff, inPlace);
    }

    public RectifiedLinear(SameDiff sameDiff, SDVariable i_v, double cutoff) {
        this(sameDiff, i_v, false, cutoff);
    }

    public RectifiedLinear() {
        super();
    }

    public RectifiedLinear(INDArray x, INDArray z, double cutoff) {
        super(x, null, z, cutoff);
    }

    public RectifiedLinear(INDArray x, double cutoff) {
        super(x, cutoff);
    }

    public RectifiedLinear(INDArray x, INDArray z) {
        this(x, z, 0.0f);
    }

    public RectifiedLinear(INDArray x) {
        this(x, 0.0f);
    }

    @Override
    public int opNum() {
        return 39;
    }

    @Override
    public String opName() {
        return "relu";
    }

    @Override
    public String onnxName() {
        return "Relu";
    }

    @Override
    public String tensorflowName() {
        return "Relu";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return new ThresholdReluBp(sameDiff, arg(), i_v.get(0), scalarValue.getDouble(0)).outputs();
    }
}
