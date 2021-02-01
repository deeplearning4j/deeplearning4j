/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.transforms.gradient;


import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.List;

@NoArgsConstructor
public class LeakyReLUDerivative extends BaseScalarOp {
    private double alpha = 0.01;

    public LeakyReLUDerivative(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double alpha) {
        super(sameDiff, i_v, alpha, inPlace);
        this.alpha = alpha;
        this.extraArgs = new Object[] {alpha};
    }

    public LeakyReLUDerivative(SameDiff sameDiff, SDVariable i_v, double alpha) {
        this(sameDiff, i_v, false, alpha);
    }

    public LeakyReLUDerivative(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double alpha) {
        super(sameDiff, i_v, alpha, extraArgs);
        this.alpha = alpha;
        this.extraArgs = new Object[] {alpha};
    }

    public LeakyReLUDerivative(INDArray x, INDArray z) {
        this(x, z, 0.01);
    }

    public LeakyReLUDerivative(INDArray x) {
        this(x,x,0.01);
    }

    public LeakyReLUDerivative(INDArray x, INDArray z, double alpha) {
        super(x, null, z, alpha);
        this.alpha = alpha;
        this.extraArgs = new Object[] {alpha};
    }

    public LeakyReLUDerivative(INDArray x, double alpha) {
        super(x, alpha);
        this.alpha = alpha;
        this.extraArgs = new Object[] {alpha};
    }

    @Override
    public int opNum() {
        return 36;
    }

    @Override
    public String opName() {
        return "leakyreluderivative";
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
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not supported");
    }
}
