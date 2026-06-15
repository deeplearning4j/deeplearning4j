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
import java.util.Collections;
import java.util.List;

/**
 * Fused Layer Normalization
 *
 * Computes layer normalization in a single fused kernel using Welford's
 * algorithm for numerical stability:
 * output = (input - mean) / sqrt(variance + epsilon) * gain + bias
 *
 * This mega-kernel minimizes memory bandwidth by computing mean and variance
 * in a single pass and fusing the normalization with scale and shift.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FusedLayerNorm extends DynamicCustomOp {

    private double epsilon = 1e-5;

    public FusedLayerNorm(SameDiff sameDiff, SDVariable input, SDVariable gain, SDVariable bias, double epsilon) {
        super(null, sameDiff, bias != null ? new SDVariable[]{input, gain, bias} : new SDVariable[]{input, gain}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public FusedLayerNorm(SameDiff sameDiff, SDVariable input, SDVariable gain, double epsilon) {
        this(sameDiff, input, gain, null, epsilon);
    }

    public FusedLayerNorm(INDArray input, INDArray gain, INDArray bias, double epsilon) {
        this(input, gain, bias, null, epsilon);
    }

    public FusedLayerNorm(INDArray input, INDArray gain, INDArray bias, INDArray output, double epsilon) {
        super(bias != null ? new INDArray[]{input, gain, bias} : new INDArray[]{input, gain},
              output != null ? new INDArray[]{output} : null);
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
        return "fused_layer_norm";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        SDVariable bias = args().length > 2 ? arg(2) : null;
        return Arrays.asList(new FusedLayerNormBp(sameDiff, arg(0), arg(1), gradients.get(0), bias, epsilon).outputVariables());
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
