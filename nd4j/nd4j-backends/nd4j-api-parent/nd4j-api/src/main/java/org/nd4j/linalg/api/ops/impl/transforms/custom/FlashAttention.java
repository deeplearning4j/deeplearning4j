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
import java.util.Collections;
import java.util.List;

/**
 * Flash Attention - Memory-efficient attention computation.
 *
 * Implements the Flash Attention algorithm which:
 * 1. Processes attention in tiles to reduce memory from O(N^2) to O(N)
 * 2. Uses online softmax for numerical stability
 * 3. Never materializes the full attention matrix
 *
 * Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * https://arxiv.org/abs/2205.14135
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FlashAttention extends DynamicCustomOp {
    private boolean isCausal = true;
    private double scale = 0.0;  // 0 = auto (1/sqrt(head_dim))
    private int numHeads = 8;
    private int numKvHeads = 0;  // 0 = same as numHeads

    /**
     * Create a flash attention operation.
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param scale Scale factor for attention scores (0 = auto: 1/sqrt(head_dim))
     * @param isCausal Whether to apply causal (lower triangular) mask
     * @param numHeads Number of query attention heads
     * @param numKvHeads Number of KV heads (0 = same as numHeads, for GQA use smaller value)
     */
    public FlashAttention(INDArray query, INDArray key, INDArray value,
                          double scale, boolean isCausal, int numHeads, int numKvHeads) {
        super(wrapFilterNull(query, key, value), null);
        this.scale = scale;
        this.isCausal = isCausal;
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        addTArgument(scale);
        addBArgument(isCausal);
        addIArgument(numHeads, numKvHeads);
    }

    /**
     * Create a flash attention operation (simplified constructor).
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param isCausal Whether to apply causal (lower triangular) mask
     * @param scale Scale factor for attention scores (0 = auto: 1/sqrt(head_dim))
     */
    public FlashAttention(INDArray query, INDArray key, INDArray value,
                          boolean isCausal, double scale) {
        super(wrapFilterNull(query, key, value), null);
        this.isCausal = isCausal;
        this.scale = scale;
        this.numHeads = (int) query.size(2);
        this.numKvHeads = (int) key.size(2);
        addTArgument(scale);
        addBArgument(isCausal);
        addIArgument(numHeads, numKvHeads);
    }

    /**
     * Create a flash attention operation with mask.
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param mask Attention mask [batch, 1, seq_len, kv_len] (optional)
     * @param isCausal Whether to apply causal (lower triangular) mask
     * @param scale Scale factor for attention scores (0 = auto: 1/sqrt(head_dim))
     */
    public FlashAttention(INDArray query, INDArray key, INDArray value, INDArray mask,
                          boolean isCausal, double scale) {
        super(wrapFilterNull(query, key, value, mask), null);
        this.isCausal = isCausal;
        this.scale = scale;
        this.numHeads = (int) query.size(2);
        this.numKvHeads = (int) key.size(2);
        addTArgument(scale);
        addBArgument(isCausal);
        addIArgument(numHeads, numKvHeads);
    }

    /**
     * Create a flash attention operation for SameDiff.
     */
    public FlashAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                          double scale, boolean isCausal, int numHeads, int numKvHeads) {
        super(null, sd, new SDVariable[]{query, key, value}, false);
        this.scale = scale;
        this.isCausal = isCausal;
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        addTArgument(scale);
        addBArgument(isCausal);
        addIArgument(numHeads, numKvHeads);
    }

    /**
     * Create a flash attention operation for SameDiff (simplified).
     */
    public FlashAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                          boolean isCausal, double scale) {
        super(null, sd, new SDVariable[]{query, key, value}, false);
        this.isCausal = isCausal;
        this.scale = scale;
        addTArgument(scale);
        addBArgument(isCausal);
        addIArgument(numHeads, numKvHeads);
    }

    /**
     * Create a flash attention operation for SameDiff with mask.
     */
    public FlashAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                          SDVariable mask, boolean isCausal, double scale) {
        super(null, sd, mask != null ?
              new SDVariable[]{query, key, value, mask} :
              new SDVariable[]{query, key, value}, false);
        this.isCausal = isCausal;
        this.scale = scale;
        addTArgument(scale);
        addBArgument(isCausal);
        addIArgument(numHeads, numKvHeads);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        // T_ARG: scale
        if (!tArguments.isEmpty()) {
            this.scale = tArguments.get(0);
        }
        // B_ARG: isCausal
        if (!bArguments.isEmpty()) {
            this.isCausal = bArguments.get(0);
        }
        // I_ARG: numHeads, numKvHeads
        if (!iArguments.isEmpty()) {
            this.numHeads = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.numKvHeads = iArguments.get(1).intValue();
        }
    }

    @Override
    public String opName() {
        return "flash_attention";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        return Arrays.asList(new FlashAttentionBp(sameDiff,
                arg(0),              // query
                arg(1),              // key
                arg(2),              // value
                gradient.get(0),     // gradOutput
                isCausal,
                scale).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        Preconditions.checkState(first.isFPType(),
            "Input datatype must be a floating point type, got %s", first);
        return Collections.singletonList(first);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
