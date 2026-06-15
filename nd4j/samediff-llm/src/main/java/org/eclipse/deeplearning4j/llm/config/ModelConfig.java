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

package org.eclipse.deeplearning4j.llm.config;

import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Configuration loaded from HuggingFace config.json.
 *
 * This configuration contains model architecture information including:
 * - Model dimensions (hidden size, layers, heads)
 * - Vocabulary size
 * - Special token IDs
 * - Model type and architecture
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * ModelConfig config = ModelConfig.fromFile(new File("config.json"));
 * int hiddenSize = config.getHiddenSize();
 * int numLayers = config.getNumHiddenLayers();
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @JsonProperty("architectures")
    private List<String> architectures;

    @JsonProperty("model_type")
    private String modelType;

    @JsonProperty("hidden_size")
    private Integer hiddenSize;

    @JsonProperty("num_hidden_layers")
    private Integer numHiddenLayers;

    @JsonProperty("num_attention_heads")
    private Integer numAttentionHeads;

    @JsonProperty("num_key_value_heads")
    private Integer numKeyValueHeads;

    @JsonProperty("intermediate_size")
    private Integer intermediateSize;

    @JsonProperty("vocab_size")
    private Integer vocabSize;

    @JsonProperty("max_position_embeddings")
    private Integer maxPositionEmbeddings;

    @JsonProperty("hidden_act")
    private String hiddenAct;

    @JsonProperty("rms_norm_eps")
    private Double rmsNormEps;

    @JsonProperty("layer_norm_eps")
    private Double layerNormEps;

    @JsonProperty("rope_theta")
    private Double ropeTheta;

    @JsonProperty("rope_scaling")
    private Object ropeScaling;

    @JsonProperty("tie_word_embeddings")
    private Boolean tieWordEmbeddings;

    @JsonProperty("use_cache")
    private Boolean useCache;

    @JsonProperty("bos_token_id")
    private Integer bosTokenId;

    @JsonProperty("eos_token_id")
    private Object eosTokenId; // Can be int or List<Integer>

    @JsonProperty("pad_token_id")
    private Integer padTokenId;

    @JsonProperty("torch_dtype")
    private String torchDtype;

    @JsonProperty("attention_dropout")
    private Double attentionDropout;

    @JsonProperty("hidden_dropout")
    private Double hiddenDropout;

    @JsonProperty("attention_bias")
    private Boolean attentionBias;

    @JsonProperty("mlp_bias")
    private Boolean mlpBias;

    // Vision-language model specific
    @JsonProperty("image_token_id")
    private Integer imageTokenId;

    @JsonProperty("vision_config")
    private Object visionConfig;

    @JsonProperty("text_config")
    private Object textConfig;

    /**
     * Load config from a config.json file.
     *
     * @param file the config file
     * @return the parsed configuration
     * @throws IOException if reading fails
     */
    public static ModelConfig fromFile(File file) throws IOException {
        return MAPPER.readValue(file, ModelConfig.class);
    }

    /**
     * Load config from a JSON string.
     *
     * @param json the JSON string
     * @return the parsed configuration
     * @throws IOException if parsing fails
     */
    public static ModelConfig fromJson(String json) throws IOException {
        return MAPPER.readValue(json, ModelConfig.class);
    }

    /**
     * Load config from a model directory.
     *
     * @param modelDir the model directory containing config.json
     * @return the parsed configuration
     * @throws IOException if reading fails
     */
    public static ModelConfig fromDirectory(File modelDir) throws IOException {
        File configFile = new File(modelDir, "config.json");
        if (!configFile.exists()) {
            throw new IOException("config.json not found in: " + modelDir.getAbsolutePath());
        }
        return fromFile(configFile);
    }

    /**
     * Get the primary EOS token ID.
     * Handles both single integer and list formats.
     *
     * @return the EOS token ID, or null
     */
    @SuppressWarnings("unchecked")
    public Integer getEosTokenIdSingle() {
        if (eosTokenId == null) {
            return null;
        }
        if (eosTokenId instanceof Integer) {
            return (Integer) eosTokenId;
        }
        if (eosTokenId instanceof Number) {
            return ((Number) eosTokenId).intValue();
        }
        if (eosTokenId instanceof List) {
            List<?> list = (List<?>) eosTokenId;
            if (!list.isEmpty() && list.get(0) instanceof Number) {
                return ((Number) list.get(0)).intValue();
            }
        }
        return null;
    }

    /**
     * Get all EOS token IDs.
     *
     * @return list of EOS token IDs
     */
    @SuppressWarnings("unchecked")
    public List<Integer> getEosTokenIds() {
        if (eosTokenId == null) {
            return List.of();
        }
        if (eosTokenId instanceof Integer) {
            return List.of((Integer) eosTokenId);
        }
        if (eosTokenId instanceof Number) {
            return List.of(((Number) eosTokenId).intValue());
        }
        if (eosTokenId instanceof List) {
            List<?> list = (List<?>) eosTokenId;
            return list.stream()
                    .filter(o -> o instanceof Number)
                    .map(o -> ((Number) o).intValue())
                    .collect(java.util.stream.Collectors.toList());
        }
        return List.of();
    }

    /**
     * Get the number of key-value heads for grouped-query attention.
     * Falls back to numAttentionHeads if not specified (standard MHA).
     *
     * @return the number of KV heads
     */
    public int getEffectiveNumKeyValueHeads() {
        return numKeyValueHeads != null ? numKeyValueHeads :
               (numAttentionHeads != null ? numAttentionHeads : 1);
    }

    /**
     * Check if the model uses grouped-query attention (GQA).
     *
     * @return true if using GQA
     */
    public boolean usesGroupedQueryAttention() {
        return numKeyValueHeads != null && numAttentionHeads != null
               && numKeyValueHeads < numAttentionHeads;
    }

    /**
     * Get the head dimension.
     *
     * @return hidden_size / num_attention_heads
     */
    public int getHeadDim() {
        if (hiddenSize != null && numAttentionHeads != null && numAttentionHeads > 0) {
            return hiddenSize / numAttentionHeads;
        }
        return 0;
    }

    /**
     * Check if this is a vision-language model.
     *
     * @return true if the model has vision components
     */
    public boolean isVisionLanguageModel() {
        return visionConfig != null || imageTokenId != null;
    }

    /**
     * Get the primary architecture name.
     *
     * @return the first architecture, or null
     */
    public String getPrimaryArchitecture() {
        return architectures != null && !architectures.isEmpty() ? architectures.get(0) : null;
    }
}
