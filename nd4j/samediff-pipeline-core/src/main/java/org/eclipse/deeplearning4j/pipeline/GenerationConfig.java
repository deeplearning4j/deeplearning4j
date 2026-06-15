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
import com.google.gson.JsonObject;
import lombok.Builder;
import lombok.Data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Represents generation_config.json for LLM text generation parameters.
 * Contains default generation settings like temperature, top_p, max_length, etc.
 */
@Data
@Builder
public class GenerationConfig {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final String GENERATION_CONFIG_FILE = "generation_config.json";

    // Sampling parameters
    private Double temperature;
    private Double topP;
    private Double topK;
    private Double typicalP;
    private Double repetitionPenalty;
    private Double lengthPenalty;
    private Boolean doSample;

    // Length parameters
    private Integer maxLength;
    private Integer maxNewTokens;
    private Integer minLength;
    private Integer minNewTokens;

    // Token IDs
    private Integer bosTokenId;
    private Integer eosTokenId;
    private Integer padTokenId;
    private Integer decoderStartTokenId;
    private List<Integer> eosTokenIds;

    // Beam search
    private Integer numBeams;
    private Integer numBeamGroups;
    private Double diversityPenalty;
    private Boolean earlyStop;

    // Other
    private Integer numReturnSequences;
    private Boolean outputScores;
    private Boolean returnDictInGenerate;
    private String transformersVersion;

    // Raw JSON for any additional fields
    private JsonObject rawConfig;

    public static GenerationConfig fromFile(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            JsonObject json = GSON.fromJson(reader, JsonObject.class);
            return fromJson(json);
        }
    }

    public static GenerationConfig fromJson(JsonObject json) {
        GenerationConfigBuilder builder = GenerationConfig.builder();
        builder.rawConfig(json);

        // Sampling parameters
        if (json.has("temperature")) builder.temperature(json.get("temperature").getAsDouble());
        if (json.has("top_p")) builder.topP(json.get("top_p").getAsDouble());
        if (json.has("top_k")) builder.topK(json.get("top_k").getAsDouble());
        if (json.has("typical_p")) builder.typicalP(json.get("typical_p").getAsDouble());
        if (json.has("repetition_penalty")) builder.repetitionPenalty(json.get("repetition_penalty").getAsDouble());
        if (json.has("length_penalty")) builder.lengthPenalty(json.get("length_penalty").getAsDouble());
        if (json.has("do_sample")) builder.doSample(json.get("do_sample").getAsBoolean());

        // Length parameters
        if (json.has("max_length")) builder.maxLength(json.get("max_length").getAsInt());
        if (json.has("max_new_tokens")) builder.maxNewTokens(json.get("max_new_tokens").getAsInt());
        if (json.has("min_length")) builder.minLength(json.get("min_length").getAsInt());
        if (json.has("min_new_tokens")) builder.minNewTokens(json.get("min_new_tokens").getAsInt());

        // Token IDs
        if (json.has("bos_token_id") && !json.get("bos_token_id").isJsonNull()) {
            builder.bosTokenId(json.get("bos_token_id").getAsInt());
        }
        if (json.has("eos_token_id")) {
            if (json.get("eos_token_id").isJsonArray()) {
                List<Integer> eosIds = new ArrayList<>();
                for (var el : json.getAsJsonArray("eos_token_id")) {
                    eosIds.add(el.getAsInt());
                }
                builder.eosTokenIds(eosIds);
                if (!eosIds.isEmpty()) builder.eosTokenId(eosIds.get(0));
            } else if (!json.get("eos_token_id").isJsonNull()) {
                builder.eosTokenId(json.get("eos_token_id").getAsInt());
            }
        }
        if (json.has("pad_token_id") && !json.get("pad_token_id").isJsonNull()) {
            builder.padTokenId(json.get("pad_token_id").getAsInt());
        }
        if (json.has("decoder_start_token_id") && !json.get("decoder_start_token_id").isJsonNull()) {
            builder.decoderStartTokenId(json.get("decoder_start_token_id").getAsInt());
        }

        // Beam search
        if (json.has("num_beams")) builder.numBeams(json.get("num_beams").getAsInt());
        if (json.has("num_beam_groups")) builder.numBeamGroups(json.get("num_beam_groups").getAsInt());
        if (json.has("diversity_penalty")) builder.diversityPenalty(json.get("diversity_penalty").getAsDouble());
        if (json.has("early_stopping")) builder.earlyStop(json.get("early_stopping").getAsBoolean());

        // Other
        if (json.has("num_return_sequences")) builder.numReturnSequences(json.get("num_return_sequences").getAsInt());
        if (json.has("output_scores")) builder.outputScores(json.get("output_scores").getAsBoolean());
        if (json.has("return_dict_in_generate")) builder.returnDictInGenerate(json.get("return_dict_in_generate").getAsBoolean());
        if (json.has("transformers_version")) builder.transformersVersion(json.get("transformers_version").getAsString());

        return builder.build();
    }

    public static Optional<GenerationConfig> loadIfExists(File dir) {
        File configFile = new File(dir, GENERATION_CONFIG_FILE);
        if (!configFile.exists()) return Optional.empty();
        try {
            return Optional.of(fromFile(configFile));
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    public static boolean exists(File dir) {
        return new File(dir, GENERATION_CONFIG_FILE).exists();
    }

    /**
     * Get effective max length, preferring max_new_tokens over max_length.
     */
    public int getEffectiveMaxLength(int inputLength) {
        if (maxNewTokens != null) {
            return inputLength + maxNewTokens;
        }
        return maxLength != null ? maxLength : 2048;
    }

    /**
     * Get the effective temperature, defaulting to 1.0 if not set.
     */
    public double getEffectiveTemperature() {
        return temperature != null ? temperature : 1.0;
    }

    /**
     * Get effective top_p, defaulting to 1.0 (no nucleus sampling).
     */
    public double getEffectiveTopP() {
        return topP != null ? topP : 1.0;
    }

    /**
     * Get effective top_k, defaulting to 50.
     */
    public int getEffectiveTopK() {
        return topK != null ? topK.intValue() : 50;
    }

    /**
     * Check if sampling is enabled.
     */
    public boolean isSamplingEnabled() {
        if (doSample != null) return doSample;
        return (temperature != null && temperature > 0) || (topP != null && topP < 1.0);
    }

    /**
     * Get all EOS token IDs as a list.
     */
    public List<Integer> getAllEosTokenIds() {
        if (eosTokenIds != null && !eosTokenIds.isEmpty()) {
            return eosTokenIds;
        }
        if (eosTokenId != null) {
            return Collections.singletonList(eosTokenId);
        }
        return Collections.emptyList();
    }
}
