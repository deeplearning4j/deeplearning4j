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
 * Gated Delta Rule operation (arXiv:2412.06464, ICLR 2025, NVIDIA Research).
 * <p>
 * Implements the gated delta rule for linear attention with recurrent state:
 * <pre>
 *   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)
 *   output_t = S_t^T * q_t
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: Q [B, L, H, D_k] - query</li>
 *   <li>1: K [B, L, H, D_k] - key (L2-normalized)</li>
 *   <li>2: V [B, L, H, D_v] - value</li>
 *   <li>3: beta [B, L, H] - per-step learning rate</li>
 *   <li>4: gate [B, L, H] - decay gate (pre-exp)</li>
 *   <li>5: state_in [B, H, D_k, D_v] (optional) - previous recurrent state</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: output [B, L, H, D_v] - attention output</li>
 *   <li>1: state_out [B, H, D_k, D_v] - final recurrent state</li>
 * </ul>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class GatedDeltaRule extends DynamicCustomOp {

    public GatedDeltaRule(SameDiff sameDiff, SDVariable q, SDVariable k, SDVariable v,
                          SDVariable beta, SDVariable gate) {
        super(null, sameDiff, new SDVariable[]{q, k, v, beta, gate}, false);
    }

    public GatedDeltaRule(SameDiff sameDiff, SDVariable q, SDVariable k, SDVariable v,
                          SDVariable beta, SDVariable gate, SDVariable stateIn) {
        super(null, sameDiff, new SDVariable[]{q, k, v, beta, gate, stateIn}, false);
    }

    public GatedDeltaRule(INDArray q, INDArray k, INDArray v, INDArray beta, INDArray gate) {
        this(q, k, v, beta, gate, null, null, null);
    }

    public GatedDeltaRule(INDArray q, INDArray k, INDArray v, INDArray beta, INDArray gate,
                          INDArray stateIn) {
        this(q, k, v, beta, gate, stateIn, null, null);
    }

    public GatedDeltaRule(INDArray q, INDArray k, INDArray v, INDArray beta, INDArray gate,
                          INDArray stateIn, INDArray output, INDArray stateOut) {
        super(null, stateIn != null ?
                new INDArray[]{q, k, v, beta, gate, stateIn} :
                new INDArray[]{q, k, v, beta, gate},
                output != null && stateOut != null ?
                        new INDArray[]{output, stateOut} : null);
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
