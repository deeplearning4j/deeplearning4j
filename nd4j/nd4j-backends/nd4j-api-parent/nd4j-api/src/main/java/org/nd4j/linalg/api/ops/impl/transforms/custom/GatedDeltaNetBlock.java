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
 * Full Gated Delta Network layer.
 *
 * Fuses: linear projection -> causal_conv1d + SiLU -> gated_delta_rule
 *        -> RMSNorm + Swish gate -> output projection
 *
 * Inputs:
 *   0: x          [B, L, D]
 *   1: Wqkv       [D, qkv_dim]   (projects to Q, K, V)
 *   2: Wbeta      [D, H]         (projects to beta gates)
 *   3: Wgate      [D, H]         (projects to decay gates)
 *   4: Wout       [H*D_v, D]     (output projection)
 *   5: convWeight [D, K]         (causal conv1d kernel)
 *   6: convBias   [D]            (causal conv1d bias)
 *   7: stateIn    [B, H, D_k, D_v] (optional recurrent state)
 *
 * Int args:
 *   0: numHeads
 *   1: headDimK
 *   2: headDimV
 *
 * Float args:
 *   0: rmsNormEpsilon (default 1e-5)
 *
 * Outputs:
 *   0: output          [B, L, D]
 *   1: recurrentState  [B, H, D_k, D_v]
 *   2: convState       [B, D, K-1]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class GatedDeltaNetBlock extends DynamicCustomOp {

    public GatedDeltaNetBlock(INDArray x, INDArray wqkv, INDArray wbeta, INDArray wgate,
                              INDArray wout, INDArray convWeight, INDArray convBias,
                              int numHeads, int headDimK, int headDimV, double rmsEps) {
        super(new INDArray[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias}, null);
        addIArgument(numHeads, headDimK, headDimV);
        addTArgument(rmsEps);
    }

    public GatedDeltaNetBlock(INDArray x, INDArray wqkv, INDArray wbeta, INDArray wgate,
                              INDArray wout, INDArray convWeight, INDArray convBias,
                              INDArray stateIn,
                              int numHeads, int headDimK, int headDimV, double rmsEps) {
        super(new INDArray[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias, stateIn}, null);
        addIArgument(numHeads, headDimK, headDimV);
        addTArgument(rmsEps);
    }

    public GatedDeltaNetBlock(SameDiff sd, SDVariable x, SDVariable wqkv, SDVariable wbeta,
                              SDVariable wgate, SDVariable wout, SDVariable convWeight,
                              SDVariable convBias,
                              int numHeads, int headDimK, int headDimV, double rmsEps) {
        super(null, sd, new SDVariable[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias});
        addIArgument(numHeads, headDimK, headDimV);
        addTArgument(rmsEps);
    }

    public GatedDeltaNetBlock(SameDiff sd, SDVariable x, SDVariable wqkv, SDVariable wbeta,
                              SDVariable wgate, SDVariable wout, SDVariable convWeight,
                              SDVariable convBias, SDVariable stateIn,
                              int numHeads, int headDimK, int headDimV, double rmsEps) {
        super(null, sd, stateIn != null
                ? new SDVariable[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias, stateIn}
                : new SDVariable[]{x, wqkv, wbeta, wgate, wout, convWeight, convBias});
        addIArgument(numHeads, headDimK, headDimV);
        addTArgument(rmsEps);
    }

    @Override
    public String opName() {
        return "gated_delta_net_block";
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
