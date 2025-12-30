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

package org.nd4j.ggml.format;

import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * High-level metadata container for GGML/GGUF models.
 * Provides a unified interface for accessing model information regardless of the underlying format.
 */
@Data
@Builder
public class GGMLMetadata {

    /**
     * Source file format
     */
    private GGMLFormat format;

    /**
     * Model architecture (e.g., "llama", "bert", "gpt2")
     */
    private String architecture;

    /**
     * Model name
     */
    private String modelName;

    /**
     * Number of transformer layers/blocks
     */
    private int numLayers;

    /**
     * Hidden/embedding dimension
     */
    private int hiddenSize;

    /**
     * Intermediate/feed-forward dimension
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
     * Maximum context/sequence length
     */
    private int contextLength;

    /**
     * Vocabulary size
     */
    private int vocabSize;

    /**
     * Layer norm epsilon
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
     * List of tensor information
     */
    @Builder.Default
    private List<GGMLTensorInfo> tensors = new ArrayList<>();

    /**
     * Raw metadata key-value pairs
     */
    @Builder.Default
    private Map<String, Object> rawMetadata = new HashMap<>();

    /**
     * Tokenizer information
     */
    private TokenizerInfo tokenizerInfo;

    /**
     * Create metadata from a GGUF header
     */
    public static GGMLMetadata fromGGUF(GGUFHeader header, List<GGMLTensorInfo> tensors) {
        return GGMLMetadata.builder()
                .format(GGMLFormat.GGUF)
                .architecture(header.getArchitecture())
                .modelName(header.getModelName())
                .numLayers(header.getBlockCount())
                .hiddenSize(header.getEmbeddingLength())
                .intermediateSize(header.getFeedForwardLength())
                .numAttentionHeads(header.getAttentionHeadCount())
                .numKVHeads(header.getAttentionHeadCountKV())
                .contextLength(header.getContextLength())
                .vocabSize(header.getVocabSize())
                .layerNormEpsilon(header.getLayerNormRmsEpsilon())
                .ropeFreqBase(header.getRopeFreqBase())
                .tensors(tensors != null ? tensors : new ArrayList<>())
                .rawMetadata(header.getMetadata())
                .tokenizerInfo(TokenizerInfo.fromGGUFHeader(header))
                .build();
    }

    /**
     * Create metadata from a legacy GGML header
     */
    public static GGMLMetadata fromGGML(GGMLHeader header, List<GGMLTensorInfo> tensors) {
        return GGMLMetadata.builder()
                .format(GGMLFormat.GGML)
                .architecture(inferArchitectureFromTensors(tensors))
                .numLayers(header.getNumLayers())
                .hiddenSize(header.getEmbeddingSize())
                .numAttentionHeads(header.getNumHeads())
                .numKVHeads(header.getNumKVHeads() > 0 ? header.getNumKVHeads() : header.getNumHeads())
                .contextLength(header.getContextLength())
                .vocabSize(header.getVocabSize())
                .tensors(tensors != null ? tensors : new ArrayList<>())
                .build();
    }

    /**
     * Try to infer the architecture from tensor names
     */
    private static String inferArchitectureFromTensors(List<GGMLTensorInfo> tensors) {
        if (tensors == null || tensors.isEmpty()) {
            return "unknown";
        }

        for (GGMLTensorInfo tensor : tensors) {
            String name = tensor.getName().toLowerCase();

            // LLaMA-style naming
            if (name.contains("blk.") && name.contains("attn_q")) {
                return "llama";
            }

            // BERT-style naming
            if (name.contains("encoder.layer") || name.contains("bert")) {
                return "bert";
            }

            // GPT-2 style naming
            if (name.contains("h.") && name.contains("attn.c_attn")) {
                return "gpt2";
            }

            // Falcon-style naming
            if (name.contains("transformer.h") && name.contains("self_attention")) {
                return "falcon";
            }
        }

        return "unknown";
    }

    /**
     * Get the head dimension (hidden_size / num_heads)
     */
    public int getHeadDimension() {
        if (numAttentionHeads == 0) return 0;
        return hiddenSize / numAttentionHeads;
    }

    /**
     * Check if this model uses grouped-query attention
     */
    public boolean hasGroupedQueryAttention() {
        return numKVHeads > 0 && numKVHeads < numAttentionHeads;
    }

    /**
     * Get a tensor by name
     */
    public GGMLTensorInfo getTensor(String name) {
        for (GGMLTensorInfo tensor : tensors) {
            if (tensor.getName().equals(name)) {
                return tensor;
            }
        }
        return null;
    }

    /**
     * Get all tensors matching a pattern
     */
    public List<GGMLTensorInfo> getTensorsMatching(String pattern) {
        List<GGMLTensorInfo> result = new ArrayList<>();
        for (GGMLTensorInfo tensor : tensors) {
            if (tensor.getName().matches(pattern)) {
                result.add(tensor);
            }
        }
        return result;
    }

    /**
     * Calculate the total model size in bytes
     */
    public long getTotalModelSize() {
        long total = 0;
        for (GGMLTensorInfo tensor : tensors) {
            total += tensor.getDataSize();
        }
        return total;
    }

    /**
     * Calculate the total number of parameters
     */
    public long getTotalParameters() {
        long total = 0;
        for (GGMLTensorInfo tensor : tensors) {
            total += tensor.getNumElements();
        }
        return total;
    }

    /**
     * Tokenizer information
     */
    @Data
    @Builder
    public static class TokenizerInfo {
        private String model;
        private List<String> tokens;
        private float[] scores;
        private int[] tokenTypes;
        private int bosTokenId;
        private int eosTokenId;
        private int padTokenId;
        private int unkTokenId;

        public static TokenizerInfo fromGGUFHeader(GGUFHeader header) {
            return TokenizerInfo.builder()
                    .model(header.getTokenizerModel())
                    .tokens(header.getTokens())
                    .bosTokenId(header.getBosTokenId())
                    .eosTokenId(header.getEosTokenId())
                    .build();
        }
    }
}
