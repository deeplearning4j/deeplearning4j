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
    public static final String KEY_ATTENTION_KEY_LENGTH = ".attention.key_length";
    public static final String KEY_ATTENTION_VALUE_LENGTH = ".attention.value_length";
    public static final String KEY_EXPERT_COUNT = ".expert_count";
    public static final String KEY_EXPERT_USED_COUNT = ".expert_used_count";
    public static final String KEY_LAYER_TYPES = ".layer_types";
    public static final String KEY_FULL_ATTENTION_INTERVAL = ".full_attention_interval";
    public static final String KEY_SSM_CONV_KERNEL = ".ssm.conv_kernel";
    public static final String KEY_SSM_STATE_SIZE = ".ssm.state_size";
    public static final String KEY_SSM_GROUP_COUNT = ".ssm.group_count";
    public static final String KEY_SSM_TIME_STEP_RANK = ".ssm.time_step_rank";
    public static final String KEY_SSM_INNER_SIZE = ".ssm.inner_size";

    // Tokenizer keys
    public static final String KEY_TOKENIZER_MODEL = "tokenizer.ggml.model";
    public static final String KEY_TOKENIZER_TOKENS = "tokenizer.ggml.tokens";
    public static final String KEY_TOKENIZER_SCORES = "tokenizer.ggml.scores";
    public static final String KEY_TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type";
    public static final String KEY_TOKENIZER_BOS_ID = "tokenizer.ggml.bos_token_id";
    public static final String KEY_TOKENIZER_EOS_ID = "tokenizer.ggml.eos_token_id";
    public static final String KEY_TOKENIZER_PAD_ID = "tokenizer.ggml.padding_token_id";
    public static final String KEY_TOKENIZER_CHAT_TEMPLATE = "tokenizer.chat_template";

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
     * Get the number of key-value attention heads (for grouped-query attention).
     * If the GGUF metadata stores a per-layer array (as in LFM-2), this returns
     * the first non-zero entry, or falls back to the full head count.
     */
    public int getAttentionHeadCountKV() {
        String arch = getArchitecture();
        if (arch == null) return getAttentionHeadCount();
        Object raw = metadata.get(arch + KEY_ATTENTION_HEAD_COUNT_KV);
        if (raw instanceof Number) {
            int kvHeads = ((Number) raw).intValue();
            return kvHeads > 0 ? kvHeads : getAttentionHeadCount();
        }
        // GGUFReader.readArray() returns primitive int[] for INT32/UINT32 arrays
        if (raw instanceof int[]) {
            for (int val : (int[]) raw) {
                if (val > 0) return val;
            }
        }
        if (raw instanceof long[]) {
            for (long val : (long[]) raw) {
                if (val > 0) return (int) val;
            }
        }
        if (raw instanceof List) {
            List<?> perLayer = (List<?>) raw;
            for (Object entry : perLayer) {
                if (entry instanceof Number) {
                    int val = ((Number) entry).intValue();
                    if (val > 0) return val;
                }
            }
        }
        return getAttentionHeadCount();
    }

    /**
     * Get the per-layer KV head count array.
     * Some architectures (e.g., LFM-2) store head_count_kv as a per-layer array
     * where 0 means the layer has no attention. Returns null if the metadata
     * is a scalar or absent.
     */
    @SuppressWarnings("unchecked")
    public List<Integer> getAttentionHeadCountKVPerLayer() {
        String arch = getArchitecture();
        if (arch == null) return null;
        Object raw = metadata.get(arch + KEY_ATTENTION_HEAD_COUNT_KV);
        // GGUFReader.readArray() returns primitive int[] for INT32/UINT32 arrays
        if (raw instanceof int[]) {
            int[] arr = (int[]) raw;
            List<Integer> result = new ArrayList<>(arr.length);
            for (int v : arr) result.add(v);
            return result;
        }
        if (raw instanceof long[]) {
            long[] arr = (long[]) raw;
            List<Integer> result = new ArrayList<>(arr.length);
            for (long v : arr) result.add((int) v);
            return result;
        }
        if (raw instanceof List) {
            List<?> perLayer = (List<?>) raw;
            List<Integer> result = new ArrayList<>(perLayer.size());
            for (Object entry : perLayer) {
                if (entry instanceof Number) {
                    result.add(((Number) entry).intValue());
                } else {
                    result.add(0);
                }
            }
            return result;
        }
        return null;
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
        if (arch != null) {
            int declared = getMetadataInt(arch + KEY_VOCAB_SIZE, 0);
            if (declared > 0) return declared;
        }
        // Many GGUFs (e.g. qwen35) omit <arch>.vocab_size — per the GGUF spec the
        // vocabulary is then defined by the embedded tokenizer's token list. Note
        // this counts the EMBEDDING rows, which may exceed the usable tokenizer
        // vocabulary (padding rows); tokenizer.json remains the decode authority.
        Object tokens = getMetadata() != null ? getMetadata().get("tokenizer.ggml.tokens") : null;
        if (tokens instanceof java.util.List) {
            return ((java.util.List<?>) tokens).size();
        }
        if (tokens instanceof Object[]) {
            return ((Object[]) tokens).length;
        }
        return 0;
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
        return getMetadataInt(KEY_TOKENIZER_BOS_ID, -1);
    }

    /**
     * Get the EOS (end of sequence) token ID
     */
    public int getEosTokenId() {
        return getMetadataInt(KEY_TOKENIZER_EOS_ID, -1);
    }

    /**
     * Get the padding token ID declared by the model metadata.
     *
     * @return the padding token ID, or {@code -1} when the model does not declare one
     */
    public int getPadTokenId() {
        return getMetadataInt(KEY_TOKENIZER_PAD_ID, -1);
    }

    /**
     * Get the chat template (Jinja2 format) from GGUF metadata.
     * Returns null if not present.
     */
    public String getChatTemplate() {
        return getMetadataString(KEY_TOKENIZER_CHAT_TEMPLATE);
    }

    /**
     * Get the explicit attention key dimension (head_dim for K projections).
     * Returns 0 if not specified in metadata.
     */
    public int getAttentionKeyLength() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_ATTENTION_KEY_LENGTH, 0);
    }

    /**
     * Get the explicit attention value dimension (head_dim for V projections).
     * Returns 0 if not specified in metadata.
     */
    public int getAttentionValueLength() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_ATTENTION_VALUE_LENGTH, 0);
    }

    /**
     * Get the number of experts (for MoE models).
     * Returns 0 if not specified.
     */
    public int getExpertCount() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_EXPERT_COUNT, 0);
    }

    /**
     * Get the number of experts used per token (for MoE models).
     * Returns 0 if not specified.
     */
    public int getExpertUsedCount() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_EXPERT_USED_COUNT, 0);
    }

    /**
     * Get the per-layer type array from metadata (e.g., ["linear_attention", "full_attention", ...]).
     * Returns an empty list if not specified.
     */
    @SuppressWarnings("unchecked")
    public List<String> getLayerTypes() {
        String arch = getArchitecture();
        if (arch == null) return Collections.emptyList();
        Object value = metadata.get(arch + KEY_LAYER_TYPES);
        if (value instanceof List) {
            return (List<String>) value;
        }
        return Collections.emptyList();
    }

    /**
     * Get the RoPE dimension count (number of dimensions to apply rotary embedding to).
     * Returns 0 if not specified (meaning rotate all head dimensions).
     */
    public int getRopeDimensionCount() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_ROPE_DIMENSION_COUNT, 0);
    }

    /**
     * Get the full attention interval (e.g., every Nth layer is full attention, rest are linear/GDN).
     * Returns 0 if not specified (meaning all layers use the same attention type).
     */
    public int getFullAttentionInterval() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_FULL_ATTENTION_INTERVAL, 0);
    }

    /**
     * Get SSM (State Space Model) convolution kernel size.
     * Returns 0 if not specified.
     */
    public int getSsmConvKernel() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_SSM_CONV_KERNEL, 0);
    }

    /**
     * Get SSM inner size (recurrence dimension).
     * Returns 0 if not specified.
     */
    public int getSsmInnerSize() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_SSM_INNER_SIZE, 0);
    }

    /**
     * Get SSM group count (number of recurrence heads).
     * Returns 0 if not specified.
     */
    public int getSsmGroupCount() {
        String arch = getArchitecture();
        if (arch == null) return 0;
        return getMetadataInt(arch + KEY_SSM_GROUP_COUNT, 0);
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
