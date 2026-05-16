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

import java.util.Collections;
import java.util.List;

/**
 * Lightning Attention — O(N) linear attention with per-head exponential decay
 * via intra/inter-chunk decomposition.
 *
 * Based on cuLA (inclusionAI/cuLA) algorithmic patterns.
 * Reference: https://arxiv.org/abs/2405.17381 (Lightning Attention-2)
 *
 * The algorithm decomposes each chunk into:
 *   - Inter-chunk: O_inter = Q_i @ S_{i-1}
 *   - Intra-chunk: A = tril(Q_i @ K_i^T) * decay_mask; O_intra = A @ V_i
 *   - State update: S_i = decay^C * S_{i-1} + K_i^T @ V_i
 *
 * Inputs:
 *   0: query      [batch, seqLen, numHeads, headDim]
 *   1: key        [batch, seqLen, numHeads, headDim]
 *   2: value      [batch, seqLen, numHeads, headDim]
 *   3: decayRates [numHeads] float32 — per-head scalar decay rate
 *   4: state      [batch, numHeads, headDim, headDim] float32 — recurrent state (in-place)
 *
 * Int args:
 *   0: isCausal (0 or 1, default 1)
 *
 * Output:
 *   0: attention output [batch, seqLen, numHeads, headDim]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class LightningAttention extends DynamicCustomOp {

    public LightningAttention(INDArray query, INDArray key, INDArray value,
                              INDArray decayRates, INDArray state) {
        super(new INDArray[]{query, key, value, decayRates, state}, null);
        addIArgument(1L); // isCausal default true
    }

    public LightningAttention(INDArray query, INDArray key, INDArray value,
                              INDArray decayRates, INDArray state, boolean isCausal) {
        super(new INDArray[]{query, key, value, decayRates, state}, null);
        addIArgument(isCausal ? 1L : 0L);
    }

    public LightningAttention(INDArray query, INDArray key, INDArray value,
                              INDArray decayRates, INDArray state, int isCausal) {
        super(new INDArray[]{query, key, value, decayRates, state}, null);
        addIArgument((long) isCausal);
    }

    public LightningAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                              SDVariable decayRates, SDVariable state) {
        super(null, sd, new SDVariable[]{query, key, value, decayRates, state});
        addIArgument(1L); // isCausal default true
    }

    public LightningAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                              SDVariable decayRates, SDVariable state, boolean isCausal) {
        super(null, sd, new SDVariable[]{query, key, value, decayRates, state});
        addIArgument(isCausal ? 1L : 0L);
    }

    public LightningAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                              SDVariable decayRates, SDVariable state, int isCausal) {
        super(null, sd, new SDVariable[]{query, key, value, decayRates, state});
        addIArgument((long) isCausal);
    }

    @Override
    public String opName() {
        return "lightning_attention";
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
