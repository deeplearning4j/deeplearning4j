/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.pipeline;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import lombok.Builder;
import lombok.Data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Represents tokenizer_config.json for tokenizer settings.
 * Contains tokenizer class, special tokens, chat templates, and other settings.
 */
@Data
@Builder
public class TokenizerConfig {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final String TOKENIZER_CONFIG_FILE = "tokenizer_config.json";

    // Tokenizer class info
    private String tokenizerClass;
    private String autoMap;

    // Token strings
    private String bosToken;
    private String eosToken;
    private String unkToken;
    private String padToken;
    private String sepToken;
    private String clsToken;
    private String maskToken;

    // Additional tokens
    private List<String> additionalSpecialTokens;

    // Settings
    private Integer modelMaxLength;
    private Boolean addBosToken;
    private Boolean addEosToken;
    private Boolean addPrefixSpace;
    private Boolean useDefaultSystemPrompt;
    private Boolean cleanUpTokenizationSpaces;
    private String paddingSide;
    private String truncationSide;

    // Chat template (Jinja2 format)
    private String chatTemplate;

    // Raw JSON for any additional fields
    private JsonObject rawConfig;

    public static TokenizerConfig fromFile(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            JsonObject json = GSON.fromJson(reader, JsonObject.class);
            return fromJson(json);
        }
    }

    public static TokenizerConfig fromJson(JsonObject json) {
        TokenizerConfigBuilder builder = TokenizerConfig.builder();
        builder.rawConfig(json);

        // Tokenizer class
        if (json.has("tokenizer_class")) builder.tokenizerClass(json.get("tokenizer_class").getAsString());
        if (json.has("auto_map")) {
            JsonElement autoMap = json.get("auto_map");
            if (autoMap.isJsonObject()) {
                builder.autoMap(autoMap.toString());
            } else {
                builder.autoMap(autoMap.getAsString());
            }
        }

        // Special tokens - handle both string and object formats
        builder.bosToken(extractTokenString(json, "bos_token"));
        builder.eosToken(extractTokenString(json, "eos_token"));
        builder.unkToken(extractTokenString(json, "unk_token"));
        builder.padToken(extractTokenString(json, "pad_token"));
        builder.sepToken(extractTokenString(json, "sep_token"));
        builder.clsToken(extractTokenString(json, "cls_token"));
        builder.maskToken(extractTokenString(json, "mask_token"));

        // Additional special tokens
        if (json.has("additional_special_tokens")) {
            List<String> additionalTokens = new ArrayList<>();
            JsonElement addTokens = json.get("additional_special_tokens");
            if (addTokens.isJsonArray()) {
                for (JsonElement el : addTokens.getAsJsonArray()) {
                    if (el.isJsonPrimitive()) {
                        additionalTokens.add(el.getAsString());
                    } else if (el.isJsonObject()) {
                        JsonObject tokenObj = el.getAsJsonObject();
                        if (tokenObj.has("content")) {
                            additionalTokens.add(tokenObj.get("content").getAsString());
                        }
                    }
                }
            }
            builder.additionalSpecialTokens(additionalTokens);
        }

        // Settings
        if (json.has("model_max_length")) {
            try {
                builder.modelMaxLength(json.get("model_max_length").getAsInt());
            } catch (NumberFormatException e) {
                // Some models have very large values like 1e30
            }
        }
        if (json.has("add_bos_token")) builder.addBosToken(json.get("add_bos_token").getAsBoolean());
        if (json.has("add_eos_token")) builder.addEosToken(json.get("add_eos_token").getAsBoolean());
        if (json.has("add_prefix_space")) builder.addPrefixSpace(json.get("add_prefix_space").getAsBoolean());
        if (json.has("use_default_system_prompt")) builder.useDefaultSystemPrompt(json.get("use_default_system_prompt").getAsBoolean());
        if (json.has("clean_up_tokenization_spaces")) builder.cleanUpTokenizationSpaces(json.get("clean_up_tokenization_spaces").getAsBoolean());
        if (json.has("padding_side")) builder.paddingSide(json.get("padding_side").getAsString());
        if (json.has("truncation_side")) builder.truncationSide(json.get("truncation_side").getAsString());

        // Chat template
        if (json.has("chat_template")) {
            JsonElement ct = json.get("chat_template");
            if (ct.isJsonPrimitive()) {
                builder.chatTemplate(ct.getAsString());
            } else if (ct.isJsonArray()) {
                // Some models have multiple chat templates as array
                for (JsonElement el : ct.getAsJsonArray()) {
                    if (el.isJsonObject()) {
                        JsonObject tmpl = el.getAsJsonObject();
                        if (tmpl.has("template")) {
                            builder.chatTemplate(tmpl.get("template").getAsString());
                            break;
                        }
                    }
                }
            }
        }

        return builder.build();
    }

    private static String extractTokenString(JsonObject json, String key) {
        if (!json.has(key)) return null;
        JsonElement el = json.get(key);
        if (el.isJsonNull()) return null;
        if (el.isJsonPrimitive()) return el.getAsString();
        if (el.isJsonObject()) {
            JsonObject obj = el.getAsJsonObject();
            if (obj.has("content")) return obj.get("content").getAsString();
        }
        return null;
    }

    public static Optional<TokenizerConfig> loadIfExists(File dir) {
        File configFile = new File(dir, TOKENIZER_CONFIG_FILE);
        if (!configFile.exists()) return Optional.empty();
        try {
            return Optional.of(fromFile(configFile));
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    public static boolean exists(File dir) {
        return new File(dir, TOKENIZER_CONFIG_FILE).exists();
    }

    /**
     * Check if this tokenizer has a chat template defined.
     */
    public boolean hasChatTemplate() {
        return chatTemplate != null && !chatTemplate.isEmpty();
    }

    /**
     * Get the effective max length, with a sensible default.
     */
    public int getEffectiveMaxLength() {
        return modelMaxLength != null && modelMaxLength > 0 && modelMaxLength < Integer.MAX_VALUE
                ? modelMaxLength : 2048;
    }

    /**
     * Get all special tokens as a set.
     */
    public Set<String> getAllSpecialTokens() {
        Set<String> tokens = new LinkedHashSet<>();
        if (bosToken != null) tokens.add(bosToken);
        if (eosToken != null) tokens.add(eosToken);
        if (unkToken != null) tokens.add(unkToken);
        if (padToken != null) tokens.add(padToken);
        if (sepToken != null) tokens.add(sepToken);
        if (clsToken != null) tokens.add(clsToken);
        if (maskToken != null) tokens.add(maskToken);
        if (additionalSpecialTokens != null) tokens.addAll(additionalSpecialTokens);
        return tokens;
    }
}
