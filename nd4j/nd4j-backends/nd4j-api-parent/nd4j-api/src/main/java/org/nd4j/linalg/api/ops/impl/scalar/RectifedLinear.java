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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.Arrays;
import java.util.List;

/**
 * Rectified linear units
 *
 * @author Adam Gibson
 */
public class RectifedLinear extends BaseScalarOp {
    public RectifedLinear(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace, double cutoff) {
        super(sameDiff, i_v1, cutoff, inPlace);
    }

    public RectifedLinear(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs, double cutoff) {
        super(sameDiff, i_v1, cutoff, extraArgs);
    }

    public RectifedLinear(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double cutoff) {
        super(sameDiff, i_v, cutoff, inPlace);
    }

    public RectifedLinear() {

    }

    public RectifedLinear(INDArray x, INDArray z, double cutoff) {
        super(x, null, z, x.length(), cutoff);

        init(x, null, z, n); //Need to re-init to properly set cutoff in extra args array
    }

    public RectifedLinear(INDArray x, INDArray z, long n, double cutoff) {
        super(x, null, z, n, cutoff);
        init(x, null, z, n);
    }

    public RectifedLinear(INDArray x, double cutoff) {
        super(x, cutoff);

        init(x, null, x, x.length());
    }

    public RectifedLinear(INDArray x, INDArray z) {
        this(x, z, 0.0f);
    }

    public RectifedLinear(INDArray x, INDArray z, long n) {
        this(x, z, n, 0.0f);
    }

    public RectifedLinear(INDArray x) {
        this(x, 0.0f);
    }

    @Override
    public int opNum() {
        return 41;
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
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable step = new Step(sameDiff, arg(), false, scalarValue.getDouble(0)).outputVariables()[0];
        SDVariable ret = step.mul(i_v.get(0));
        return Arrays.asList(ret);
    }
}
