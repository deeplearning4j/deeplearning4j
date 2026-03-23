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
 * Full Gated Delta Network layer block.
 * <p>
 * Fuses: linear projection -> causal_conv1d + SiLU -> gated_delta_rule
 *        -> RMSNorm + Swish gate -> output projection
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: x [B, L, D] - input</li>
 *   <li>1: Wqkv [D, qkv_dim] - QKV projection weights</li>
 *   <li>2: Wbeta [D, H] - beta projection weights</li>
 *   <li>3: Wgate [D, H] - gate projection weights</li>
 *   <li>4: Wout [H*D_v, D] - output projection weights</li>
 *   <li>5: conv_weight [D, K] - causal conv1d weights</li>
 *   <li>6: conv_bias [D] - causal conv1d bias</li>
 *   <li>7: state_in [B, H, D_k, D_v] (optional) - recurrent state</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: output [B, L, D]</li>
 *   <li>1: recurrent_state_out [B, H, D_k, D_v]</li>
 *   <li>2: conv_state_out [B, D, K-1]</li>
 * </ul>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class GatedDeltaNetBlock extends DynamicCustomOp {

    @Getter private int numHeads;
    @Getter private int headDimK;
    @Getter private int headDimV;
    @Getter private double rmsNormEpsilon = 1e-5;

    public GatedDeltaNetBlock(SameDiff sameDiff, SDVariable x, SDVariable wqkv, SDVariable wbeta,
                               SDVariable wgate, SDVariable wout, SDVariable convWeight,
                               SDVariable convBias, int numHeads, int headDimK, int headDimV) {
        super(null, sameDiff, new SDVariable[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias}, false);
        this.numHeads = numHeads;
        this.headDimK = headDimK;
        this.headDimV = headDimV;
        addIArgument(numHeads, headDimK, headDimV);
        addTArgument(rmsNormEpsilon);
    }

    public GatedDeltaNetBlock(SameDiff sameDiff, SDVariable x, SDVariable wqkv, SDVariable wbeta,
                               SDVariable wgate, SDVariable wout, SDVariable convWeight,
                               SDVariable convBias, SDVariable stateIn,
                               int numHeads, int headDimK, int headDimV, double rmsNormEpsilon) {
        super(null, sameDiff, new SDVariable[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias, stateIn}, false);
        this.numHeads = numHeads;
        this.headDimK = headDimK;
        this.headDimV = headDimV;
        this.rmsNormEpsilon = rmsNormEpsilon;
        addIArgument(numHeads, headDimK, headDimV);
        addTArgument(rmsNormEpsilon);
    }

    public GatedDeltaNetBlock(INDArray x, INDArray wqkv, INDArray wbeta, INDArray wgate,
                               INDArray wout, INDArray convWeight, INDArray convBias,
                               int numHeads, int headDimK, int headDimV) {
        super(null, new INDArray[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias}, null);
        this.numHeads = numHeads;
        this.headDimK = headDimK;
        this.headDimV = headDimV;
        addIArgument(numHeads, headDimK, headDimV);
        addTArgument(rmsNormEpsilon);
    }

    public GatedDeltaNetBlock(INDArray x, INDArray wqkv, INDArray wbeta, INDArray wgate,
                               INDArray wout, INDArray convWeight, INDArray convBias,
                               INDArray stateIn, int numHeads, int headDimK, int headDimV,
                               double rmsNormEpsilon) {
        super(null, stateIn != null
                ? new INDArray[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias, stateIn}
                : new INDArray[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias}, null);
        this.numHeads = numHeads;
        this.headDimK = headDimK;
        this.headDimV = headDimV;
        this.rmsNormEpsilon = rmsNormEpsilon;
        addIArgument(numHeads, headDimK, headDimV);
        addTArgument(rmsNormEpsilon);
    }

    @Override
    public String opName() {
        return "gated_delta_net_block";
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() >= 3) {
            this.numHeads = iArguments.get(0).intValue();
            this.headDimK = iArguments.get(1).intValue();
            this.headDimV = iArguments.get(2).intValue();
        }
        if (!tArguments.isEmpty()) {
            this.rmsNormEpsilon = tArguments.get(0);
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(dt, dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 3;
    }
}
