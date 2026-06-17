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
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Grouped Query Attention Backward Pass.
 *
 * Computes gradients for query, key, and value tensors.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class GroupedQueryAttentionBp extends DynamicCustomOp {
    private int numHeads = 8;
    private int numKvHeads = 8;
    private boolean isCausal = true;
    private double scale = 0.0;

    /**
     * Create a grouped query attention backward pass operation.
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param gradOutput Gradient of loss w.r.t. output [batch, seq_len, num_heads, head_dim]
     * @param numHeads Number of query heads
     * @param numKvHeads Number of key-value heads
     * @param isCausal Whether causal mask was applied
     * @param scale Scale factor used in forward pass
     */
    public GroupedQueryAttentionBp(INDArray query, INDArray key, INDArray value, INDArray gradOutput,
                                    int numHeads, int numKvHeads, boolean isCausal, double scale) {
        super(wrapFilterNull(query, key, value, gradOutput), null);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.isCausal = isCausal;
        this.scale = scale;
        addIArgument(numHeads, numKvHeads, isCausal ? 1 : 0);
        addTArgument(scale);
    }

    /**
     * Create a grouped query attention backward pass operation for SameDiff.
     */
    public GroupedQueryAttentionBp(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                    SDVariable gradOutput, int numHeads, int numKvHeads,
                                    boolean isCausal, double scale) {
        super(null, sd, new SDVariable[]{query, key, value, gradOutput}, false);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.isCausal = isCausal;
        this.scale = scale;
        addIArgument(numHeads, numKvHeads, isCausal ? 1 : 0);
        addTArgument(scale);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.numHeads = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.numKvHeads = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.isCausal = iArguments.get(2) != 0;
        }
        if (!tArguments.isEmpty()) {
            this.scale = tArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "grouped_query_attention_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        Preconditions.checkState(first.isFPType(),
            "Input datatype must be a floating point type, got %s", first);
        // Returns gradients for query, key, value
        return Arrays.asList(first, first, first);
    }

    @Override
    public int getNumOutputs() {
        return 3;  // gradQuery, gradKey, gradValue
    }
}
