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
 * RMS Norm - Root Mean Square Layer Normalization
 *
 * Implements RMSNorm as used in LLaMA, Mistral, and other modern LLMs:
 *   RMSNorm(x) = x * rsqrt(mean(x^2, axis=-1) + eps) * gamma
 *
 * This is a lighter-weight alternative to LayerNorm that skips the mean-centering
 * step, only normalizing by the root mean square.
 *
 * Inputs:
 *   0: input tensor [batch, ..., features]
 *   1: gamma weights [features] (optional - scale parameter)
 *
 * Float args:
 *   0: epsilon (default 1e-5)
 *
 * Output:
 *   0: normalized tensor [batch, ..., features]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class RmsNorm extends DynamicCustomOp {

    @Getter private double epsilon = 1e-5;

    /**
     * Create RmsNorm with input and gamma (scale weights).
     */
    public RmsNorm(SameDiff sameDiff, SDVariable input, SDVariable gamma, double epsilon) {
        super(null, sameDiff, new SDVariable[]{input, gamma}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    /**
     * Create RmsNorm with input and gamma, default epsilon.
     */
    public RmsNorm(SameDiff sameDiff, SDVariable input, SDVariable gamma) {
        this(sameDiff, input, gamma, 1e-5);
    }

    /**
     * Create RmsNorm with input only (no gamma).
     */
    public RmsNorm(SameDiff sameDiff, SDVariable input, double epsilon) {
        super(null, sameDiff, new SDVariable[]{input}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    /**
     * Create RmsNorm for INDArray execution.
     */
    public RmsNorm(INDArray input, INDArray gamma, INDArray output, double epsilon) {
        super(gamma != null ? new INDArray[]{input, gamma} : new INDArray[]{input},
                output != null ? new INDArray[]{output} : null);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public RmsNorm(INDArray input, INDArray gamma) {
        this(input, gamma, null, 1e-5);
    }

    /**
     * Create RmsNorm for INDArray execution with explicit epsilon.
     */
    public RmsNorm(INDArray input, INDArray gamma, double epsilon) {
        this(input, gamma, null, epsilon);
    }

    /**
     * Create RmsNorm for INDArray execution without gamma.
     */
    public RmsNorm(INDArray input, double epsilon) {
        this(input, null, null, epsilon);
    }

    /**
     * Create RmsNorm for INDArray execution without gamma, default epsilon.
     */
    public RmsNorm(INDArray input) {
        this(input, null, null, 1e-5);
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
        return "rms_norm";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Arrays.asList(new RmsNormBp(sameDiff, arg(0), gradients.get(0), epsilon).outputVariables());
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
