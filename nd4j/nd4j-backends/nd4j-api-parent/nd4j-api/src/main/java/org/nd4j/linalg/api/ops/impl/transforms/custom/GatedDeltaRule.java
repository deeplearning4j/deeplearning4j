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
 * Gated Delta Rule recurrent state update.
 *
 * Implements the gated delta rule from arXiv:2412.06464 (ICLR 2025):
 *   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)
 *   output_t = S_t^T * q_t
 *
 * Inputs:
 *   0: Q      [B, L, H, D_k]
 *   1: K      [B, L, H, D_k]
 *   2: V      [B, L, H, D_v]
 *   3: beta   [B, L, H]
 *   4: gate   [B, L, H]
 *   5: stateIn [B, H, D_k, D_v] (optional)
 *
 * Outputs:
 *   0: output    [B, L, H, D_v]
 *   1: state_out [B, H, D_k, D_v]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class GatedDeltaRule extends DynamicCustomOp {

    public GatedDeltaRule(INDArray q, INDArray k, INDArray v, INDArray beta, INDArray gate) {
        super(new INDArray[]{q, k, v, beta, gate}, null);
    }

    public GatedDeltaRule(INDArray q, INDArray k, INDArray v, INDArray beta, INDArray gate, INDArray stateIn) {
        super(new INDArray[]{q, k, v, beta, gate, stateIn}, null);
    }

    public GatedDeltaRule(SameDiff sd, SDVariable q, SDVariable k, SDVariable v,
                          SDVariable beta, SDVariable gate) {
        super(null, sd, new SDVariable[]{q, k, v, beta, gate});
    }

    public GatedDeltaRule(SameDiff sd, SDVariable q, SDVariable k, SDVariable v,
                          SDVariable beta, SDVariable gate, SDVariable stateIn) {
        super(null, sd, stateIn != null
                ? new SDVariable[]{q, k, v, beta, gate, stateIn}
                : new SDVariable[]{q, k, v, beta, gate});
    }

    @Override
    public String opName() {
        return "gated_delta_rule";
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
