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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;

/**
 * Inverted DropOut implementation as Op
 *
 * PLEASE NOTE: This is legacy DropOutInverted implementation, please consider using op with the same opName from randomOps
 * @author raver119@gmail.com
 */
public class LegacyDropOutInverted extends BaseTransformOp {

    private double p;

    public LegacyDropOutInverted(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double p) {
        super(sameDiff, i_v, inPlace);
        this.p = p;
    }

    public LegacyDropOutInverted(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs, double p) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.p = p;
    }

    public LegacyDropOutInverted(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double p) {
        super(sameDiff, i_v, extraArgs);
        this.p = p;
    }

    public LegacyDropOutInverted() {

    }

    public LegacyDropOutInverted(INDArray x, double p) {
        super(x);
        this.p = p;
        init(x, null, x, x.length());
    }

    public LegacyDropOutInverted(INDArray x, INDArray z, double p) {
        super(x, z);
        this.p = p;
        init(x, null, z, x.length());
    }

    public LegacyDropOutInverted(INDArray x, INDArray z, double p, long n) {
        super(x, z, n);
        this.p = p;
        init(x, null, z, n);
    }

    @Override
    public int opNum() {
        return 44;
    }

    @Override
    public String opName() {
        return "legacy_dropout_inverted";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {p, (double) n};
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
