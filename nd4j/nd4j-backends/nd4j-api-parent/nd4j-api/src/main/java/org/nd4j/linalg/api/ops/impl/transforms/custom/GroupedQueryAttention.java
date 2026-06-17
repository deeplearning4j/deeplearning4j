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
 * Grouped Query Attention (GQA).
 *
 * Implements grouped query attention as used in LLaMA 2, Mistral, and other modern LLMs.
 * Multiple query heads share the same key-value heads, reducing memory usage while
 * maintaining model quality.
 *
 * - MHA (Multi-Head Attention): numHeads == numKvHeads
 * - GQA (Grouped Query Attention): numHeads > numKvHeads (e.g., 32 query heads, 8 KV heads)
 * - MQA (Multi-Query Attention): numKvHeads == 1
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class GroupedQueryAttention extends DynamicCustomOp {
    private int numHeads = 8;
    private int numKvHeads = 8;
    private boolean isCausal = true;
    private double scale = 0.0;  // 0 = auto (1/sqrt(head_dim))

    /**
     * Create a grouped query attention operation.
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param scale Scale factor (0 = auto: 1/sqrt(head_dim))
     * @param isCausal Whether to apply causal (lower triangular) mask
     * @param numHeads Number of query heads
     * @param numKvHeads Number of key-value heads (must divide numHeads evenly)
     */
    public GroupedQueryAttention(INDArray query, INDArray key, INDArray value,
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
     * Create a grouped query attention operation (legacy parameter order).
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param numHeads Number of query heads
     * @param numKvHeads Number of key-value heads (must divide numHeads evenly)
     * @param isCausal Whether to apply causal (lower triangular) mask
     * @param scale Scale factor (0 = auto: 1/sqrt(head_dim))
     */
    public GroupedQueryAttention(INDArray query, INDArray key, INDArray value,
                                  int numHeads, int numKvHeads, boolean isCausal, double scale) {
        this(query, key, value, scale, isCausal, numHeads, numKvHeads);
    }

    /**
     * Create a grouped query attention operation with mask.
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param mask Attention mask [batch, 1, seq_len, kv_len] (optional)
     * @param numHeads Number of query heads
     * @param numKvHeads Number of key-value heads
     * @param isCausal Whether to apply causal mask
     * @param scale Scale factor
     */
    public GroupedQueryAttention(INDArray query, INDArray key, INDArray value, INDArray mask,
                                  int numHeads, int numKvHeads, boolean isCausal, double scale) {
        super(wrapFilterNull(query, key, value, mask), null);
        this.scale = scale;
        this.isCausal = isCausal;
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        addTArgument(scale);
        addBArgument(isCausal);
        addIArgument(numHeads, numKvHeads);
    }

    /**
     * Create a grouped query attention operation for SameDiff.
     */
    public GroupedQueryAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
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
     * Create a grouped query attention operation for SameDiff (legacy parameter order).
     */
    public GroupedQueryAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                  int numHeads, int numKvHeads, boolean isCausal, double scale) {
        this(sd, query, key, value, scale, isCausal, numHeads, numKvHeads);
    }

    /**
     * Create a grouped query attention operation for SameDiff with mask.
     */
    public GroupedQueryAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                  SDVariable mask, int numHeads, int numKvHeads,
                                  boolean isCausal, double scale) {
        super(null, sd, mask != null ?
              new SDVariable[]{query, key, value, mask} :
              new SDVariable[]{query, key, value}, false);
        this.scale = scale;
        this.isCausal = isCausal;
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
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
        return "grouped_query_attention";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        return Arrays.asList(new GroupedQueryAttentionBp(sameDiff,
                arg(0),              // query
                arg(1),              // key
                arg(2),              // value
                gradient.get(0),     // gradOutput
                numHeads,
                numKvHeads,
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
