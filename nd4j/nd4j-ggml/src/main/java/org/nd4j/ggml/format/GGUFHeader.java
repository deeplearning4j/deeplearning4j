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

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * GGUF file header containing metadata and model information.
 */
@Data
@Builder
public class GGUFHeader {

    /**
     * Magic number (should be 0x46554747 for 'GGUF')
     */
    private int magic;

    /**
     * GGUF format version (1, 2, or 3)
     */
    private int version;

    /**
     * Number of tensors in the file
     */
    private long tensorCount;

    /**
     * Number of key-value metadata pairs
     */
    private long metadataKVCount;

    /**
     * Parsed key-value metadata
     */
    @Builder.Default
    private Map<String, Object> metadata = new HashMap<>();

    // Standard metadata key constants
    public static final String KEY_GENERAL_ARCHITECTURE = "general.architecture";
    public static final String KEY_GENERAL_NAME = "general.name";
    public static final String KEY_GENERAL_AUTHOR = "general.author";
    public static final String KEY_GENERAL_DESCRIPTION = "general.description";
    public static final String KEY_GENERAL_FILE_TYPE = "general.file_type";
    public static final String KEY_GENERAL_QUANTIZATION_VERSION = "general.quantization_version";
    public static final String KEY_GENERAL_ALIGNMENT = "general.alignment";

    // Architecture-specific keys (prefix with architecture name)
    public static final String KEY_CONTEXT_LENGTH = ".context_length";
    public static final String KEY_EMBEDDING_LENGTH = ".embedding_length";
    public static final String KEY_BLOCK_COUNT = ".block_count";
    public static final String KEY_FEED_FORWARD_LENGTH = ".feed_forward_length";
    public static final String KEY_ATTENTION_HEAD_COUNT = ".attention.head_count";
    public static final String KEY_ATTENTION_HEAD_COUNT_KV = ".attention.head_count_kv";
    public static final String KEY_ATTENTION_LAYER_NORM_RMS_EPS = ".attention.layer_norm_rms_epsilon";
    public static final String KEY_ROPE_FREQ_BASE = ".rope.freq_base";
    public static final String KEY_ROPE_DIMENSION_COUNT = ".rope.dimension_count";
    public static final String KEY_VOCAB_SIZE = ".vocab_size";

    // Tokenizer keys
    public static final String KEY_TOKENIZER_MODEL = "tokenizer.ggml.model";
    public static final String KEY_TOKENIZER_TOKENS = "tokenizer.ggml.tokens";
    public static final String KEY_TOKENIZER_SCORES = "tokenizer.ggml.scores";
    public static final String KEY_TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type";
    public static final String KEY_TOKENIZER_BOS_ID = "tokenizer.ggml.bos_token_id";
    public static final String KEY_TOKENIZER_EOS_ID = "tokenizer.ggml.eos_token_id";
    public static final String KEY_TOKENIZER_PAD_ID = "tokenizer.ggml.padding_token_id";

    /**
     * Get the model architecture name (e.g., "llama", "bert", "gpt2")
     */
    public String getArchitecture() {
        return getMetadataString(KEY_GENERAL_ARCHITECTURE);
    }

    /**
     * Get the model name
     */
    public String getModelName() {
        return getMetadataString(KEY_GENERAL_NAME);
    }

    /**
     * Get the context length for the model
     */
    public int getContextLength() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_CONTEXT_LENGTH, 0);
    }

    /**
     * Get the embedding/hidden size
     */
    public int getEmbeddingLength() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_EMBEDDING_LENGTH, 0);
    }

    /**
     * Get the number of transformer blocks/layers
     */
    public int getBlockCount() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_BLOCK_COUNT, 0);
    }

    /**
     * Get the feed-forward network dimension
     */
    public int getFeedForwardLength() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_FEED_FORWARD_LENGTH, 0);
    }

    /**
     * Get the number of attention heads
     */
    public int getAttentionHeadCount() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_ATTENTION_HEAD_COUNT, 0);
    }

    /**
     * Get the number of key-value attention heads (for grouped-query attention)
     */
    public int getAttentionHeadCountKV() {
        String arch = getArchitecture();
        if (arch == null) return getAttentionHeadCount();
        int kvHeads = getMetadataInt(arch + KEY_ATTENTION_HEAD_COUNT_KV, 0);
        return kvHeads > 0 ? kvHeads : getAttentionHeadCount();
    }

    /**
     * Get the RMS layer norm epsilon
     */
    public float getLayerNormRmsEpsilon() {
        String arch = getArchitecture();
        if (arch == null) return 1e-5f;
        return getMetadataFloat(arch + KEY_ATTENTION_LAYER_NORM_RMS_EPS, 1e-5f);
    }

    /**
     * Get the RoPE frequency base
     */
    public float getRopeFreqBase() {
        String arch = getArchitecture();
        if (arch == null) return 10000.0f;
        return getMetadataFloat(arch + KEY_ROPE_FREQ_BASE, 10000.0f);
    }

    /**
     * Get the vocabulary size
     */
    public int getVocabSize() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_VOCAB_SIZE, 0);
    }

    /**
     * Get the tokenizer model type (e.g., "llama", "gpt2", "bert")
     */
    public String getTokenizerModel() {
        return getMetadataString(KEY_TOKENIZER_MODEL);
    }

    /**
     * Get the token vocabulary as a list of strings
     */
    @SuppressWarnings("unchecked")
    public List<String> getTokens() {
        Object tokens = metadata.get(KEY_TOKENIZER_TOKENS);
        if (tokens instanceof List) {
            return (List<String>) tokens;
        }
        return Collections.emptyList();
    }

    /**
     * Get the BOS (beginning of sequence) token ID
     */
    public int getBosTokenId() {
        return getMetadataInt(KEY_TOKENIZER_BOS_ID, 1);
    }

    /**
     * Get the EOS (end of sequence) token ID
     */
    public int getEosTokenId() {
        return getMetadataInt(KEY_TOKENIZER_EOS_ID, 2);
    }

    /**
     * Get the data alignment in bytes
     */
    public int getAlignment() {
        return getMetadataInt(KEY_GENERAL_ALIGNMENT, 32);
    }

    // Helper methods for type-safe metadata access

    public String getMetadataString(String key) {
        Object value = metadata.get(key);
        return value != null ? value.toString() : null;
    }

    public int getMetadataInt(String key, int defaultValue) {
        Object value = metadata.get(key);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return defaultValue;
    }

    public long getMetadataLong(String key, long defaultValue) {
        Object value = metadata.get(key);
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        return defaultValue;
    }

    public float getMetadataFloat(String key, float defaultValue) {
        Object value = metadata.get(key);
        if (value instanceof Number) {
            return ((Number) value).floatValue();
        }
        return defaultValue;
    }

    public double getMetadataDouble(String key, double defaultValue) {
        Object value = metadata.get(key);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return defaultValue;
    }

    public boolean getMetadataBoolean(String key, boolean defaultValue) {
        Object value = metadata.get(key);
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        return defaultValue;
    }
}
