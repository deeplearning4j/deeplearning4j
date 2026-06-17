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

import java.util.Collections;
import java.util.List;

/**
 * Fused Skip (Residual Add) + RMS Normalization
 *
 * Combines residual add with RMS normalization in a single kernel:
 *   hidden = input + skip [+ bias]
 *   output = hidden * rsqrt(mean(hidden^2) + eps) * gamma
 *
 * Inputs:
 *   0: input tensor [batch, ..., features]
 *   1: skip tensor [batch, ..., features] (residual)
 *   2: gamma weights [features]
 *   3: bias [features] (optional)
 *
 * Float args:
 *   0: epsilon (default 1e-5)
 *
 * Outputs:
 *   0: normalized tensor [batch, ..., features]
 *   1: pre-norm hidden states (input + skip [+ bias]) - optional
 */
@NoArgsConstructor
public class SkipRmsNorm extends DynamicCustomOp {

    @Getter private double epsilon = 1e-5;
    @Getter private int numOutputs = 1;

    public SkipRmsNorm(SameDiff sameDiff, SDVariable input, SDVariable skip,
                       SDVariable gamma, double epsilon) {
        super(null, sameDiff, new SDVariable[]{input, skip, gamma}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public SkipRmsNorm(SameDiff sameDiff, SDVariable input, SDVariable skip,
                       SDVariable gamma, SDVariable bias, double epsilon) {
        super(null, sameDiff, bias != null
                ? new SDVariable[]{input, skip, gamma, bias}
                : new SDVariable[]{input, skip, gamma}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public SkipRmsNorm(SameDiff sameDiff, SDVariable input, SDVariable skip,
                       SDVariable gamma, SDVariable bias, double epsilon, int numOutputs) {
        super(null, sameDiff, bias != null
                ? new SDVariable[]{input, skip, gamma, bias}
                : new SDVariable[]{input, skip, gamma}, false);
        this.epsilon = epsilon;
        this.numOutputs = numOutputs;
        addTArgument(epsilon);
    }

    public SkipRmsNorm(INDArray input, INDArray skip, INDArray gamma, INDArray output, double epsilon) {
        super(new INDArray[]{input, skip, gamma},
                output != null ? new INDArray[]{output} : null);
        this.epsilon = epsilon;
        addTArgument(epsilon);
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
        return "skip_rms_norm";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        if (numOutputs > 1) {
            return List.of(inputDataTypes.get(0), inputDataTypes.get(0));
        }
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return numOutputs;
    }
}
