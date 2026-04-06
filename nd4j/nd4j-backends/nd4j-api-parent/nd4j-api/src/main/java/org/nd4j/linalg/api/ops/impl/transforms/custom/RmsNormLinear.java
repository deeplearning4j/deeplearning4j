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
 * Fused RMSNorm + Linear (MatMul) operation.
 *
 * Computes: matmul(rms_norm(x, gamma, eps), W)
 *
 * This fused op avoids materializing the intermediate normalized tensor,
 * reducing memory bandwidth and improving performance in transformer models
 * where RMSNorm is immediately followed by a linear projection.
 *
 * Inputs:
 *   0: x       - input tensor [batch, ..., features]
 *   1: gamma   - RMSNorm scale weights [features]
 *   2: W       - weight matrix for linear projection [features, outFeatures]
 *
 * Float args:
 *   0: epsilon (default 1e-6)
 *
 * Output:
 *   0: result tensor [batch, ..., outFeatures]
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class RmsNormLinear extends DynamicCustomOp {

    @Getter private double epsilon = 1e-6;

    /**
     * Create RmsNormLinear with SameDiff variables.
     */
    public RmsNormLinear(SameDiff sd, SDVariable input, SDVariable gamma, SDVariable weights, double epsilon) {
        super(null, sd, new SDVariable[]{input, gamma, weights}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    /**
     * Create RmsNormLinear with SameDiff variables, default epsilon.
     */
    public RmsNormLinear(SameDiff sd, SDVariable input, SDVariable gamma, SDVariable weights) {
        this(sd, input, gamma, weights, 1e-6);
    }

    /**
     * Create RmsNormLinear for INDArray execution.
     */
    public RmsNormLinear(INDArray input, INDArray gamma, INDArray weights, INDArray output, double epsilon) {
        super(new INDArray[]{input, gamma, weights},
                output != null ? new INDArray[]{output} : null);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    /**
     * Create RmsNormLinear for INDArray execution with default epsilon.
     */
    public RmsNormLinear(INDArray input, INDArray gamma, INDArray weights) {
        this(input, gamma, weights, null, 1e-6);
    }

    /**
     * Create RmsNormLinear for INDArray execution with explicit epsilon.
     */
    public RmsNormLinear(INDArray input, INDArray gamma, INDArray weights, double epsilon) {
        this(input, gamma, weights, null, epsilon);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!tArguments.isEmpty()) {
            this.epsilon = tArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "rms_norm_linear";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        // Use the fused backward op which recomputes intermediates internally
        return Arrays.asList(new RmsNormLinearBp(sameDiff, arg(0), arg(1), arg(2),
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
