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

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Fused RMS Norm + SwiGLU Operation
 *
 * This mega-kernel fuses the feed-forward network computation used in LLaMA-style models:
 * output = silu(rms_norm(input) @ W_gate) * (rms_norm(input) @ W_up)
 *
 * By fusing these operations into a single kernel, we:
 * 1. Avoid multiple memory passes over the input data
 * 2. Compute RMS normalization only once
 * 3. Minimize intermediate memory allocations
 *
 * Input shapes:
 * - input: [batch, seq_len, hidden_dim]
 * - gamma: [hidden_dim] (RMS norm scale)
 * - wGate: [hidden_dim, intermediate_dim] (gate projection)
 * - wUp: [hidden_dim, intermediate_dim] (up projection)
 *
 * Output shape: [batch, seq_len, intermediate_dim]
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FusedRmsNormSwiGLU extends DynamicCustomOp {

    @Getter private double epsilon = 1e-5;

    /**
     * Create a fused RMS norm + SwiGLU operation.
     *
     * @param sameDiff The SameDiff instance
     * @param input Input tensor [batch, seq_len, hidden_dim]
     * @param gamma RMS norm scale [hidden_dim]
     * @param wGate Gate projection weights [hidden_dim, intermediate_dim]
     * @param wUp Up projection weights [hidden_dim, intermediate_dim]
     * @param epsilon RMS norm epsilon
     */
    public FusedRmsNormSwiGLU(SameDiff sameDiff, SDVariable input, SDVariable gamma,
                              SDVariable wGate, SDVariable wUp, double epsilon) {
        super(null, sameDiff, new SDVariable[]{input, gamma, wGate, wUp}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public FusedRmsNormSwiGLU(SameDiff sameDiff, SDVariable input, SDVariable gamma,
                              SDVariable wGate, SDVariable wUp) {
        this(sameDiff, input, gamma, wGate, wUp, 1e-5);
    }

    public FusedRmsNormSwiGLU(INDArray input, INDArray gamma, INDArray wGate, INDArray wUp,
                              INDArray output, double epsilon) {
        super(new INDArray[]{input, gamma, wGate, wUp}, output != null ? new INDArray[]{output} : null);
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
        return "fused_rms_norm_swiglu";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Arrays.asList(new FusedRmsNormSwiGLUBp(sameDiff, arg(0), arg(1), arg(2), arg(3),
                gradients.get(0), epsilon).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
