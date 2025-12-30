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

package org.nd4j.ggml.architecture;

import lombok.Builder;
import lombok.Data;

/**
 * Configuration for model architecture.
 * Contains hyperparameters needed to construct the model graph.
 */
@Data
@Builder
public class ArchitectureConfig {

    /**
     * Number of transformer layers
     */
    private int numLayers;

    /**
     * Hidden size / embedding dimension
     */
    private int hiddenSize;

    /**
     * Intermediate size for feed-forward layers
     */
    private int intermediateSize;

    /**
     * Number of attention heads
     */
    private int numAttentionHeads;

    /**
     * Number of key-value heads (for grouped-query attention)
     */
    private int numKVHeads;

    /**
     * Vocabulary size
     */
    private int vocabSize;

    /**
     * Maximum context/sequence length
     */
    private int contextLength;

    /**
     * Layer normalization epsilon
     */
    @Builder.Default
    private float layerNormEpsilon = 1e-5f;

    /**
     * RoPE frequency base
     */
    @Builder.Default
    private float ropeFreqBase = 10000.0f;

    /**
     * RoPE dimension count
     */
    private int ropeDimensionCount;

    /**
     * Whether to use RMS normalization (LLaMA-style) vs layer norm (BERT-style)
     */
    @Builder.Default
    private boolean useRmsNorm = true;

    /**
     * Whether to use SwiGLU activation (LLaMA-style) vs GELU (BERT-style)
     */
    @Builder.Default
    private boolean useSwiGLU = true;

    /**
     * Whether to use rotary position embeddings
     */
    @Builder.Default
    private boolean useRotaryEmbeddings = true;

    /**
     * Whether this is a decoder-only model
     */
    @Builder.Default
    private boolean decoderOnly = true;

    /**
     * Get the head dimension (hidden_size / num_heads)
     */
    public int getHeadDimension() {
        if (numAttentionHeads == 0) return 0;
        return hiddenSize / numAttentionHeads;
    }

    /**
     * Check if grouped-query attention is used
     */
    public boolean hasGroupedQueryAttention() {
        return numKVHeads > 0 && numKVHeads < numAttentionHeads;
    }
}
