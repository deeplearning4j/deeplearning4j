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

import java.util.ArrayList;
import java.util.List;

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
     * Explicit head dimension. 0 means derive from hiddenSize / numAttentionHeads.
     * Some models (e.g. Qwen3.5) have head_dim != hidden_size / num_attention_heads.
     */
    private int headDim;

    /**
     * Per-layer type strings from GGUF metadata.
     * Example: ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]
     * Empty list means all layers use the default attention type.
     */
    @Builder.Default
    private List<String> layerTypes = new ArrayList<>();

    /**
     * Full attention interval: every Nth layer is full attention, rest are linear/GDN.
     * 0 means not specified (all layers same type). E.g., 4 means layers 3,7,11,15,... are full attention.
     */
    private int fullAttentionInterval;

    /**
     * RoPE type: 0 = standard/LLaMA (split-half pairing), 1 = NeoX/interleaved (adjacent pairing).
     * Qwen models use interleaved (type 1). LLaMA/Mistral use standard (type 0).
     */
    private int ropeType;

    // ========================================================================
    // MoE configuration
    // ========================================================================

    /**
     * Number of routed experts for MoE models. 0 = dense (no MoE).
     */
    private int numExperts;

    /**
     * Number of experts selected per token for MoE routing.
     */
    @Builder.Default
    private int numExpertsPerToken = 2;

    /**
     * Shared expert intermediate size. 0 = no shared experts.
     */
    private int sharedIntermediateSize;

    // ========================================================================
    // muP (maximal update parametrization) scaling multipliers
    // ========================================================================

    /**
     * Embedding multiplier (muP). Applied after token embedding gather.
     * Default 1.0 means no scaling.
     */
    @Builder.Default
    private double embeddingMultiplier = 1.0;

    /**
     * Attention multiplier (muP). Replaces 1/sqrt(d_k) scaling.
     * Default 0.0 means use standard 1/sqrt(d_k).
     */
    @Builder.Default
    private double attentionMultiplier = 0.0;

    /**
     * Residual multiplier (muP). Applied to residual connections.
     * Default 1.0 means no scaling.
     */
    @Builder.Default
    private double residualMultiplier = 1.0;

    /**
     * Logits scaling (muP). Applied to final logits before softmax.
     * Default 1.0 means no scaling.
     */
    @Builder.Default
    private double logitsScaling = 1.0;

    /**
     * Get the head dimension. Prefers explicit headDim when set,
     * otherwise falls back to hiddenSize / numAttentionHeads.
     */
    public int getHeadDimension() {
        if (headDim > 0) return headDim;
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
