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

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.File;
import java.io.IOException;

/**
 * Configuration loaded from tokenizer_config.json.
 *
 * This configuration contains metadata about the tokenizer including:
 * - Special tokens (BOS, EOS, PAD, UNK)
 * - Chat template for instruction-following models
 * - Model max length
 * - Tokenizer class type
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * TokenizerConfig config = TokenizerConfig.fromFile(new File("tokenizer_config.json"));
 * String eosToken = config.getEosToken();
 * String chatTemplate = config.getChatTemplate();
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class TokenizerConfig {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @JsonProperty("tokenizer_class")
    private String tokenizerClass;

    @JsonProperty("model_max_length")
    private Integer modelMaxLength;

    @JsonProperty("bos_token")
    private Object bosTokenRaw;

    @JsonProperty("eos_token")
    private Object eosTokenRaw;

    @JsonProperty("unk_token")
    private Object unkTokenRaw;

    @JsonProperty("pad_token")
    private Object padTokenRaw;

    @JsonProperty("sep_token")
    private Object sepTokenRaw;

    @JsonProperty("cls_token")
    private Object clsTokenRaw;

    @JsonProperty("mask_token")
    private Object maskTokenRaw;

    @JsonProperty("chat_template")
    private String chatTemplate;

    @JsonProperty("add_bos_token")
    private Boolean addBosToken;

    @JsonProperty("add_eos_token")
    private Boolean addEosToken;

    @JsonProperty("clean_up_tokenization_spaces")
    private Boolean cleanUpTokenizationSpaces;

    @JsonProperty("legacy")
    private Boolean legacy;

    @JsonProperty("use_default_system_prompt")
    private Boolean useDefaultSystemPrompt;

    /**
     * Load config from a tokenizer_config.json file.
     *
     * @param file the config file
     * @return the parsed configuration
     * @throws IOException if reading fails
     */
    public static TokenizerConfig fromFile(File file) throws IOException {
        return MAPPER.readValue(file, TokenizerConfig.class);
    }

    /**
     * Load config from a JSON string.
     *
     * @param json the JSON string
     * @return the parsed configuration
     * @throws IOException if parsing fails
     */
    public static TokenizerConfig fromJson(String json) throws IOException {
        return MAPPER.readValue(json, TokenizerConfig.class);
    }

    /**
     * Get the BOS token as a string.
     * Handles both string and object ({"content": "..."}) formats.
     *
     * @return the BOS token string, or null
     */
    public String getBosToken() {
        return extractTokenString(bosTokenRaw);
    }

    /**
     * Get the EOS token as a string.
     *
     * @return the EOS token string, or null
     */
    public String getEosToken() {
        return extractTokenString(eosTokenRaw);
    }

    /**
     * Get the UNK token as a string.
     *
     * @return the UNK token string, or null
     */
    public String getUnkToken() {
        return extractTokenString(unkTokenRaw);
    }

    /**
     * Get the PAD token as a string.
     *
     * @return the PAD token string, or null
     */
    public String getPadToken() {
        return extractTokenString(padTokenRaw);
    }

    /**
     * Get the SEP token as a string.
     *
     * @return the SEP token string, or null
     */
    public String getSepToken() {
        return extractTokenString(sepTokenRaw);
    }

    /**
     * Get the CLS token as a string.
     *
     * @return the CLS token string, or null
     */
    public String getClsToken() {
        return extractTokenString(clsTokenRaw);
    }

    /**
     * Get the MASK token as a string.
     *
     * @return the MASK token string, or null
     */
    public String getMaskToken() {
        return extractTokenString(maskTokenRaw);
    }

    /**
     * Check if a chat template is available.
     *
     * @return true if chat template is defined
     */
    public boolean hasChatTemplate() {
        return chatTemplate != null && !chatTemplate.isEmpty();
    }

    @SuppressWarnings("unchecked")
    private String extractTokenString(Object tokenRaw) {
        if (tokenRaw == null) {
            return null;
        }
        if (tokenRaw instanceof String) {
            return (String) tokenRaw;
        }
        if (tokenRaw instanceof java.util.Map) {
            java.util.Map<String, Object> map = (java.util.Map<String, Object>) tokenRaw;
            Object content = map.get("content");
            if (content instanceof String) {
                return (String) content;
            }
        }
        return tokenRaw.toString();
    }
}
