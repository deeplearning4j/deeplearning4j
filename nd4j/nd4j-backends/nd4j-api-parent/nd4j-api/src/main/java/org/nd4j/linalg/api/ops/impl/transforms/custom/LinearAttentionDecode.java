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
 * Linear Attention Decode — single-token decode step for linear attention.
 *
 * Implements the recurrent form of linear attention:
 *   state_new = exp(-decay[h]) * state_old + k (x) v   (rank-1 outer-product update)
 *   output    = q @ state_new                            (matrix-vector product)
 *
 * State is always stored and computed as float32 regardless of the input dtype,
 * matching the convention established by gated_delta_rule / mamba2_ssm.
 *
 * Inputs:
 *   0: query      [batch, 1, numHeads, headDimK]
 *   1: key        [batch, 1, numHeads, headDimK]
 *   2: value      [batch, 1, numHeads, headDimV]
 *   3: decayRates [numHeads] float32 — per-head log-decay magnitude
 *   4: state      [batch, numHeads, headDimV, headDimK] float32 — in-place update
 *
 * Output:
 *   0: output [batch, 1, numHeads, headDimV]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class LinearAttentionDecode extends DynamicCustomOp {

    public LinearAttentionDecode(INDArray query, INDArray key, INDArray value,
                                 INDArray decayRates, INDArray state) {
        super(new INDArray[]{query, key, value, decayRates, state}, null);
    }

    public LinearAttentionDecode(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                 SDVariable decayRates, SDVariable state) {
        super(null, sd, new SDVariable[]{query, key, value, decayRates, state});
    }

    @Override
    public String opName() {
        return "linear_attention_decode";
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
