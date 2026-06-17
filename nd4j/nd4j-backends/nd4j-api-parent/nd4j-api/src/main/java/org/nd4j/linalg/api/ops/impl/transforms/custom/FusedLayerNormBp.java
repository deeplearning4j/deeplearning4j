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
 * Backward pass for Fused Layer Normalization
 *
 * Computes gradients for input, gain (gamma), and optionally bias (beta).
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FusedLayerNormBp extends DynamicCustomOp {

    private double epsilon = 1e-5;

    public FusedLayerNormBp(SameDiff sameDiff, SDVariable input, SDVariable gain, SDVariable gradOut, SDVariable bias, double epsilon) {
        super(null, sameDiff, bias != null ? new SDVariable[]{input, gain, gradOut, bias} : new SDVariable[]{input, gain, gradOut}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public FusedLayerNormBp(SameDiff sameDiff, SDVariable input, SDVariable gain, SDVariable gradOut, double epsilon) {
        this(sameDiff, input, gain, gradOut, null, epsilon);
    }

    public FusedLayerNormBp(INDArray input, INDArray gain, INDArray gradOut, INDArray bias,
                            INDArray gradInput, INDArray gradGain, INDArray gradBias, double epsilon) {
        super(bias != null ? new INDArray[]{input, gain, gradOut, bias} : new INDArray[]{input, gain, gradOut},
              gradBias != null ? new INDArray[]{gradInput, gradGain, gradBias} : new INDArray[]{gradInput, gradGain});
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
        return "fused_layer_norm_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        // Output: gradInput, gradGain (same shapes as input, gain)
        return Arrays.asList(dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
