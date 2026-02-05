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
 * Represents special_tokens_map.json for special token mappings.
 * Maps token roles (bos, eos, pad, etc.) to their string representations.
 */
@Data
@Builder
public class SpecialTokensMap {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final String SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json";

    private TokenInfo bosToken;
    private TokenInfo eosToken;
    private TokenInfo unkToken;
    private TokenInfo padToken;
    private TokenInfo sepToken;
    private TokenInfo clsToken;
    private TokenInfo maskToken;
    private List<TokenInfo> additionalSpecialTokens;

    private JsonObject rawConfig;

    @Data
    @Builder
    public static class TokenInfo {
        private String content;
        private Boolean lstrip;
        private Boolean rstrip;
        private Boolean singleWord;
        private Boolean normalized;
        private Boolean special;

        public static TokenInfo fromString(String content) {
            return TokenInfo.builder().content(content).special(true).build();
        }

        public static TokenInfo fromJson(JsonElement el) {
            if (el.isJsonPrimitive()) {
                return fromString(el.getAsString());
            }
            if (el.isJsonObject()) {
                JsonObject obj = el.getAsJsonObject();
                TokenInfoBuilder builder = TokenInfo.builder();
                if (obj.has("content")) builder.content(obj.get("content").getAsString());
                if (obj.has("lstrip")) builder.lstrip(obj.get("lstrip").getAsBoolean());
                if (obj.has("rstrip")) builder.rstrip(obj.get("rstrip").getAsBoolean());
                if (obj.has("single_word")) builder.singleWord(obj.get("single_word").getAsBoolean());
                if (obj.has("normalized")) builder.normalized(obj.get("normalized").getAsBoolean());
                if (obj.has("special")) builder.special(obj.get("special").getAsBoolean());
                return builder.build();
            }
            return null;
        }
    }

    public static SpecialTokensMap fromFile(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            JsonObject json = GSON.fromJson(reader, JsonObject.class);
            return fromJson(json);
        }
    }

    public static SpecialTokensMap fromJson(JsonObject json) {
        SpecialTokensMapBuilder builder = SpecialTokensMap.builder();
        builder.rawConfig(json);

        if (json.has("bos_token")) builder.bosToken(TokenInfo.fromJson(json.get("bos_token")));
        if (json.has("eos_token")) builder.eosToken(TokenInfo.fromJson(json.get("eos_token")));
        if (json.has("unk_token")) builder.unkToken(TokenInfo.fromJson(json.get("unk_token")));
        if (json.has("pad_token")) builder.padToken(TokenInfo.fromJson(json.get("pad_token")));
        if (json.has("sep_token")) builder.sepToken(TokenInfo.fromJson(json.get("sep_token")));
        if (json.has("cls_token")) builder.clsToken(TokenInfo.fromJson(json.get("cls_token")));
        if (json.has("mask_token")) builder.maskToken(TokenInfo.fromJson(json.get("mask_token")));

        if (json.has("additional_special_tokens")) {
            List<TokenInfo> additionalTokens = new ArrayList<>();
            for (JsonElement el : json.getAsJsonArray("additional_special_tokens")) {
                TokenInfo info = TokenInfo.fromJson(el);
                if (info != null) additionalTokens.add(info);
            }
            builder.additionalSpecialTokens(additionalTokens);
        }

        return builder.build();
    }

    public static Optional<SpecialTokensMap> loadIfExists(File dir) {
        File mapFile = new File(dir, SPECIAL_TOKENS_MAP_FILE);
        if (!mapFile.exists()) return Optional.empty();
        try {
            return Optional.of(fromFile(mapFile));
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    public static boolean exists(File dir) {
        return new File(dir, SPECIAL_TOKENS_MAP_FILE).exists();
    }

    /**
     * Get the BOS token string if available.
     */
    public String getBosTokenString() {
        return bosToken != null ? bosToken.getContent() : null;
    }

    /**
     * Get the EOS token string if available.
     */
    public String getEosTokenString() {
        return eosToken != null ? eosToken.getContent() : null;
    }

    /**
     * Get the UNK token string if available.
     */
    public String getUnkTokenString() {
        return unkToken != null ? unkToken.getContent() : null;
    }

    /**
     * Get the PAD token string if available.
     */
    public String getPadTokenString() {
        return padToken != null ? padToken.getContent() : null;
    }

    /**
     * Get all special token strings.
     */
    public Set<String> getAllSpecialTokenStrings() {
        Set<String> tokens = new LinkedHashSet<>();
        if (bosToken != null && bosToken.getContent() != null) tokens.add(bosToken.getContent());
        if (eosToken != null && eosToken.getContent() != null) tokens.add(eosToken.getContent());
        if (unkToken != null && unkToken.getContent() != null) tokens.add(unkToken.getContent());
        if (padToken != null && padToken.getContent() != null) tokens.add(padToken.getContent());
        if (sepToken != null && sepToken.getContent() != null) tokens.add(sepToken.getContent());
        if (clsToken != null && clsToken.getContent() != null) tokens.add(clsToken.getContent());
        if (maskToken != null && maskToken.getContent() != null) tokens.add(maskToken.getContent());
        if (additionalSpecialTokens != null) {
            for (TokenInfo ti : additionalSpecialTokens) {
                if (ti.getContent() != null) tokens.add(ti.getContent());
            }
        }
        return tokens;
    }
}
