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
import java.util.List;

/**
 * Causal depthwise 1D convolution with state for autoregressive decoding.
 * <p>
 * Used in Gated Delta Networks (GDN) and Mamba architectures.
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: x [B, L, D] - input sequence</li>
 *   <li>1: weight [D, K] - depthwise conv weights (K = kernel size)</li>
 *   <li>2: bias [D] (optional)</li>
 *   <li>3: state_in [B, D, K-1] (optional - conv state for autoregressive decode)</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: output [B, L, D] - convolved output</li>
 *   <li>1: state_out [B, D, K-1] - updated conv state</li>
 * </ul>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class CausalConv1d extends DynamicCustomOp {

    @Getter private int activation = 0;  // 0=none, 1=silu

    public CausalConv1d(SameDiff sameDiff, SDVariable x, SDVariable weight) {
        super(null, sameDiff, new SDVariable[]{x, weight}, false);
    }

    public CausalConv1d(SameDiff sameDiff, SDVariable x, SDVariable weight, SDVariable bias) {
        super(null, sameDiff, new SDVariable[]{x, weight, bias}, false);
    }

    public CausalConv1d(SameDiff sameDiff, SDVariable x, SDVariable weight, SDVariable bias,
                         SDVariable stateIn, int activation) {
        super(null, sameDiff, bias != null ?
                (stateIn != null ? new SDVariable[]{x, weight, bias, stateIn} : new SDVariable[]{x, weight, bias}) :
                new SDVariable[]{x, weight}, false);
        this.activation = activation;
        addIArgument(activation);
    }

    public CausalConv1d(INDArray x, INDArray weight) {
        this(x, weight, null, null, 0);
    }

    public CausalConv1d(INDArray x, INDArray weight, INDArray bias, INDArray stateIn, int activation) {
        super(null, buildInputs(x, weight, bias, stateIn), null);
        this.activation = activation;
        addIArgument(activation);
    }

    private static INDArray[] buildInputs(INDArray x, INDArray weight, INDArray bias, INDArray stateIn) {
        if (bias != null && stateIn != null) return new INDArray[]{x, weight, bias, stateIn};
        if (bias != null) return new INDArray[]{x, weight, bias};
        return new INDArray[]{x, weight};
    }

    @Override
    public String opName() {
        return "causal_conv1d";
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.activation = iArguments.get(0).intValue();
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
