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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Gated Delta Rule recurrent state update with per-timestep state checkpoints.
 *
 * Same recurrence as {@link GatedDeltaRule} (arXiv:2412.06464) but requires the
 * actualLen scalar and additionally produces:
 *
 *   output 2: prefix [W, B, H, D_k, D_v] time-leading C-order, where slot t holds
 *   the recurrent state AFTER consuming input rows 0..t. Slot t is a snapshot of
 *   the unrounded working state at that boundary; later timesteps never re-read it.
 *
 * This is the capture primitive for accepted-prefix state selection in bundled MTP
 * speculative decoding: a partially accepted verification window commits
 * prefix[consumed-1] instead of re-executing the target for the consumed prefix.
 *
 * Inputs:
 *   0: Q      [B, L, H, D_k]
 *   1: K      [B, L, H, D_k]
 *   2: V      [B, L, H, D_v]
 *   3: beta   [B, L, H]
 *   4: gate   [B, L, H]
 *   5: stateIn   [B, H, D_k, D_v] (optional)
 *   6: actualLen scalar INT64 (REQUIRED)
 *
 * Outputs:
 *   0: output    [B, L, H, D_v]
 *   1: state_out [B, H, D_k, D_v]
 *   2: prefix    [W, B, H, D_k, D_v]  (W = L at shape time)
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class GatedDeltaRuleWithPrefix extends DynamicCustomOp {

    public GatedDeltaRuleWithPrefix(INDArray q, INDArray k, INDArray v, INDArray beta, INDArray gate,
                                    INDArray stateIn, INDArray actualLen) {
        super(buildInputs(q, k, v, beta, gate, stateIn, actualLen), null);
    }

    public GatedDeltaRuleWithPrefix(SameDiff sd, SDVariable q, SDVariable k, SDVariable v,
                                    SDVariable beta, SDVariable gate, SDVariable stateIn,
                                    SDVariable actualLen) {
        super(null, sd, buildSdInputs(q, k, v, beta, gate, stateIn, actualLen));
    }

    private static INDArray[] buildInputs(INDArray q, INDArray k, INDArray v, INDArray beta, INDArray gate,
                                          INDArray stateIn, INDArray actualLen) {
        List<INDArray> inputs = new ArrayList<>();
        inputs.add(q);
        inputs.add(k);
        inputs.add(v);
        inputs.add(beta);
        inputs.add(gate);
        if (stateIn != null) inputs.add(stateIn);
        inputs.add(actualLen);
        return inputs.toArray(new INDArray[0]);
    }

    private static SDVariable[] buildSdInputs(SDVariable q, SDVariable k, SDVariable v,
                                              SDVariable beta, SDVariable gate,
                                              SDVariable stateIn, SDVariable actualLen) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(q);
        inputs.add(k);
        inputs.add(v);
        inputs.add(beta);
        inputs.add(gate);
        if (stateIn != null) inputs.add(stateIn);
        inputs.add(actualLen);
        return inputs.toArray(new SDVariable[0]);
    }

    @Override
    public String opName() {
        return "gated_delta_rule_with_prefix";
    }

    /**
     * All three outputs share Q's dtype per the native contract (state and prefix
     * are built from the q dtype in the shape function; the op validates stateIn
     * matches Q).
     */
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
