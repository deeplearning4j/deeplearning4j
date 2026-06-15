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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Backward pass for Fused RMS Norm + SwiGLU Operation
 *
 * Computes gradients for:
 * - input [batch, seq_len, hidden_dim]
 * - gamma [hidden_dim]
 * - wGate [hidden_dim, intermediate_dim]
 * - wUp [hidden_dim, intermediate_dim]
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FusedRmsNormSwiGLUBp extends DynamicCustomOp {

    private double epsilon = 1e-5;

    public FusedRmsNormSwiGLUBp(SameDiff sameDiff, SDVariable input, SDVariable gamma,
                                SDVariable wGate, SDVariable wUp, SDVariable gradOut, double epsilon) {
        super(null, sameDiff, new SDVariable[]{input, gamma, wGate, wUp, gradOut}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public FusedRmsNormSwiGLUBp(INDArray input, INDArray gamma, INDArray wGate, INDArray wUp, INDArray gradOut,
                                INDArray gradInput, INDArray gradGamma, INDArray gradWGate, INDArray gradWUp,
                                double epsilon) {
        super(new INDArray[]{input, gamma, wGate, wUp, gradOut},
              new INDArray[]{gradInput, gradGamma, gradWGate, gradWUp});
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) {
            this.epsilon = tArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "fused_rms_norm_swiglu_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        // Outputs: gradInput, gradGamma, gradWGate, gradWUp
        return Arrays.asList(dt, dt, dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 4;
    }
}
