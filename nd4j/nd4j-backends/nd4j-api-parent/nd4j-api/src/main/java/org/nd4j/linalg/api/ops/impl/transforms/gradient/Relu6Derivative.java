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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

/**
 * Derivative of Rectified linear unit 6, i.e. min(max(input, cutoff), 6), where cutoff can be chosen.
 *
 * @author Alex Black
 */
public class Relu6Derivative extends DynamicCustomOp {

    private double cutoff = 0.0;

    public Relu6Derivative(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, double cutoff) {
        super("relu6_bp", sameDiff, new SDVariable[]{i_v1, i_v2});
        this.cutoff = cutoff;
        this.extraArgs = new Object[]{cutoff};
    }

    public Relu6Derivative() {
        this.extraArgs = new Object[]{cutoff};
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "relu6_bp";
    }

    @Override
    public String onnxName() { throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not supported");
    }
}
