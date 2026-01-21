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

import java.util.Collections;
import java.util.List;

/**
 * Sliding Window Attention.
 *
 * Implements sliding window attention as used in Mistral and other modern LLMs.
 * Each token only attends to a fixed window of previous tokens, enabling
 * efficient processing of very long sequences.
 *
 * This is useful for:
 * - Long context LLMs (e.g., Mistral's 32K context with 4K window)
 * - Memory-efficient inference
 * - Controlling attention patterns
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class SlidingWindowAttention extends DynamicCustomOp {
    private int windowSize = 4096;
    private int numHeads = 8;
    private int numKvHeads = 8;
    private double scale = 0.0;

    /**
     * Create a sliding window attention operation.
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, seq_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, seq_len, num_kv_heads, head_dim]
     * @param windowSize Number of positions to attend to (tokens can only see this many previous positions)
     * @param numHeads Number of query heads
     * @param numKvHeads Number of key-value heads (for GQA support)
     * @param scale Scale factor (0 = auto: 1/sqrt(head_dim))
     */
    public SlidingWindowAttention(INDArray query, INDArray key, INDArray value,
                                   int windowSize, int numHeads, int numKvHeads, double scale) {
        super(wrapFilterNull(query, key, value), null);
        this.windowSize = windowSize;
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.scale = scale;
        addIArgument(windowSize, numHeads, numKvHeads);
        addTArgument(scale);
    }

    /**
     * Create a sliding window attention operation for SameDiff.
     */
    public SlidingWindowAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                   int windowSize, int numHeads, int numKvHeads, double scale) {
        super(null, sd, new SDVariable[]{query, key, value}, false);
        this.windowSize = windowSize;
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.scale = scale;
        addIArgument(windowSize, numHeads, numKvHeads);
        addTArgument(scale);
    }

    /**
     * Create a sliding window attention with defaults.
     *
     * @param query Query tensor
     * @param key Key tensor
     * @param value Value tensor
     * @param windowSize Sliding window size
     */
    public SlidingWindowAttention(INDArray query, INDArray key, INDArray value, int windowSize) {
        super(wrapFilterNull(query, key, value), null);
        this.windowSize = windowSize;
        this.numHeads = (int) query.size(2);
        this.numKvHeads = (int) key.size(2);
        addIArgument(windowSize, numHeads, numKvHeads);
        addTArgument(0.0);  // auto scale
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.windowSize = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.numHeads = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.numKvHeads = iArguments.get(2).intValue();
        }
        if (!tArguments.isEmpty()) {
            this.scale = tArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "sliding_window_attention";
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
