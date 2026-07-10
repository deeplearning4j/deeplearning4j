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
 * MoE Weighted Sum — fused top-K expert output accumulation.
 *
 * Reduces expert outputs into the final hidden state:
 *
 *   out[t, d] = Σ_k  weights[t, k] · expertOutputs[t, k, d]   (dense mode)
 *   out[t, d] = Σ_{r: rowIndex[r,0]==t} weights[t, rowIndex[r,1]] · eo[r, d]  (flat mode)
 *
 * fp32 accumulation; output type matches expertOutputs type.
 *
 * <h3>Dense mode (mode=0, default)</h3>
 * <pre>
 *   Inputs:
 *     0: expertOutputs [T, topK, D]  — stacked expert outputs per token
 *     1: weights       [T, topK]     — router scores (already normalised)
 *   Output:
 *     0: [T, D]
 * </pre>
 *
 * <h3>Flat/segment_gemm mode (mode=1)</h3>
 * <pre>
 *   Inputs:
 *     0: expertOutputs [totalRows, D] — sorted flat output from segment_gemm
 *     1: weights       [T, topK]      — router scores
 *     2: rowIndex      [totalRows, 2] INT32/INT64 = {token_idx, k_idx} per row
 *   Output:
 *     0: [T, D]
 * </pre>
 */
@NoArgsConstructor
public class MoeWeightedSum extends DynamicCustomOp {

    /** Dense mode: expertOutputs [T, topK, D] + weights [T, topK] → [T, D] */
    public MoeWeightedSum(INDArray expertOutputs, INDArray weights) {
        super(new INDArray[]{expertOutputs, weights}, null);
        addIArgument(0);  // mode = dense
    }

    /** Dense mode (SameDiff) */
    public MoeWeightedSum(SameDiff sd, SDVariable expertOutputs, SDVariable weights) {
        super(null, sd, new SDVariable[]{expertOutputs, weights}, false);
        addIArgument(0);  // mode = dense
    }

    /**
     * Flat mode (segment_gemm layout): expertOutputs [totalRows, D] + rowIndex [totalRows, 2]
     * → [T, D], where T is inferred from weights.shape[0].
     */
    public MoeWeightedSum(INDArray expertOutputs, INDArray weights, INDArray rowIndex) {
        super(new INDArray[]{expertOutputs, weights, rowIndex}, null);
        addIArgument(1);  // mode = flat
    }

    /** Flat mode (SameDiff) */
    public MoeWeightedSum(SameDiff sd, SDVariable expertOutputs, SDVariable weights, SDVariable rowIndex) {
        super(null, sd, new SDVariable[]{expertOutputs, weights, rowIndex}, false);
        addIArgument(1);  // mode = flat
    }

    @Override
    public String opName() {
        return "moe_weighted_sum";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // Output type matches expertOutputs type (input 0)
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
