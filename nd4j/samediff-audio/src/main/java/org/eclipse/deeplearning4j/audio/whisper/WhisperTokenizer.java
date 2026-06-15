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

package org.eclipse.deeplearning4j.audio.whisper;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Whisper-specific tokenizer wrapping HuggingFace's tokenizer with
 * Whisper special tokens and language/task tokens.
 * <p>
 * Implements the {@link Tokenizer} interface so it can be used directly with
 * the native decode op via GenerationPipeline.
 * <p>
 * Special token layout:
 * <ul>
 *   <li>50257: EOT (end of transcript)</li>
 *   <li>50258: SOT (start of transcript)</li>
 *   <li>50259-50357: Language tokens (99 languages)</li>
 *   <li>50358: TRANSLATE task</li>
 *   <li>50359: TRANSCRIBE task</li>
 *   <li>50360-50362: Reserved</li>
 *   <li>50363: NO_TIMESTAMPS</li>
 *   <li>50364+: Timestamp tokens (each = 0.02 seconds)</li>
 * </ul>
 */
@Slf4j
public class WhisperTokenizer implements Tokenizer {

    // Special token IDs
    public static final int EOT = 50257;
    public static final int SOT = 50258;
    public static final int TRANSLATE = 50358;
    public static final int TRANSCRIBE = 50359;
    public static final int NO_TIMESTAMPS = 50363;
    public static final int TIMESTAMP_BEGIN = 50364;

    // Language token range
    public static final int LANGUAGE_TOKEN_START = 50259;
    public static final int LANGUAGE_TOKEN_END = 50357;

    // Seconds per timestamp token
    public static final double TIMESTAMP_RESOLUTION = 0.02;

    @Getter
    private final HuggingFaceTokenizer hfTokenizer;

    private static final Map<String, Integer> LANGUAGE_CODES = new HashMap<>();

    static {
        String[] languages = {
                "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
                "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
                "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
                "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
                "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
                "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
                "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
                "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
                "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        };
        for (int i = 0; i < languages.length; i++) {
            LANGUAGE_CODES.put(languages[i], LANGUAGE_TOKEN_START + i);
        }
    }

    private WhisperTokenizer(HuggingFaceTokenizer hfTokenizer) {
        this.hfTokenizer = hfTokenizer;
    }

    /**
     * Load a WhisperTokenizer from a tokenizer.json file.
     */
    public static WhisperTokenizer fromFile(File tokenizerJson) throws IOException {
        HuggingFaceTokenizer hfTokenizer = HuggingFaceTokenizer.fromFile(tokenizerJson);
        return new WhisperTokenizer(hfTokenizer);
    }

    /**
     * Load a WhisperTokenizer from a model directory containing tokenizer.json.
     */
    public static WhisperTokenizer fromDirectory(File modelDir) throws IOException {
        File tokenizerJson = new File(modelDir, "tokenizer.json");
        if (!tokenizerJson.exists()) {
            throw new IOException("tokenizer.json not found in " + modelDir.getAbsolutePath());
        }
        return fromFile(tokenizerJson);
    }

    // ========== Tokenizer interface implementation ==========

    @Override
    public Encoding encode(String text, boolean addSpecialTokens) {
        return hfTokenizer.encode(text, addSpecialTokens);
    }

    @Override
    public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
        return hfTokenizer.encodeBatch(texts, addSpecialTokens);
    }

    @Override
    public String decode(int[] ids, boolean skipSpecialTokens) {
        return hfTokenizer.decode(ids, skipSpecialTokens);
    }

    @Override
    public List<String> decodeBatch(List<int[]> idsBatch, boolean skipSpecialTokens) {
        return hfTokenizer.decodeBatch(idsBatch, skipSpecialTokens);
    }

    @Override
    public int getVocabSize() {
        return hfTokenizer.getVocabSize();
    }

    @Override
    public Integer getTokenId(String token) {
        return hfTokenizer.getTokenId(token);
    }

    @Override
    public String getToken(int id) {
        return hfTokenizer.getToken(id);
    }

    @Override
    public Map<String, Integer> getVocab() {
        return hfTokenizer.getVocab();
    }

    @Override
    public int getPadTokenId() {
        return hfTokenizer.getPadTokenId();
    }

    @Override
    public int getBosTokenId() {
        return SOT; // Whisper's SOT is the BOS equivalent
    }

    @Override
    public int getEosTokenId() {
        return EOT; // Whisper's EOT is the EOS equivalent
    }

    @Override
    public int getUnkTokenId() {
        return hfTokenizer.getUnkTokenId();
    }

    @Override
    public boolean isValid() {
        return hfTokenizer.isValid();
    }

    @Override
    public void close() throws Exception {
        hfTokenizer.close();
    }

    // ========== Whisper-specific methods ==========

    /**
     * Encode text to token IDs (convenience method).
     */
    public int[] encodeToIds(String text) {
        return hfTokenizer.encode(text, false).getIds();
    }

    /**
     * Decode token IDs to text, filtering out special and timestamp tokens.
     */
    public String decodeSkippingSpecial(int[] tokenIds) {
        int[] filtered = Arrays.stream(tokenIds)
                .filter(id -> id < EOT)
                .toArray();
        return hfTokenizer.decode(filtered, true);
    }

    /**
     * Create the prompt token sequence for Whisper decoding.
     *
     * @param language   Language code (e.g., "en") or null for auto-detect
     * @param task       "transcribe" or "translate"
     * @param timestamps Whether to include timestamp tokens
     * @return Prompt token IDs: [SOT, lang?, task, (no_timestamps?)]
     */
    public int[] createPromptTokens(String language, String task, boolean timestamps) {
        int size = 1; // SOT
        if (language != null) size++;
        size++; // task
        if (!timestamps) size++;

        int[] prompt = new int[size];
        int idx = 0;

        prompt[idx++] = SOT;

        if (language != null) {
            Integer langToken = LANGUAGE_CODES.get(language.toLowerCase());
            if (langToken == null) {
                throw new IllegalArgumentException("Unsupported language: " + language
                        + ". Supported: " + LANGUAGE_CODES.keySet());
            }
            prompt[idx++] = langToken;
        }

        if ("translate".equalsIgnoreCase(task)) {
            prompt[idx++] = TRANSLATE;
        } else {
            prompt[idx++] = TRANSCRIBE;
        }

        if (!timestamps) {
            prompt[idx++] = NO_TIMESTAMPS;
        }

        return prompt;
    }

    /**
     * Check if a token ID is a timestamp token.
     */
    public static boolean isTimestampToken(int tokenId) {
        return tokenId >= TIMESTAMP_BEGIN;
    }

    /**
     * Convert a timestamp token ID to seconds.
     */
    public static double timestampToSeconds(int tokenId) {
        if (!isTimestampToken(tokenId)) {
            throw new IllegalArgumentException("Token " + tokenId + " is not a timestamp token");
        }
        return (tokenId - TIMESTAMP_BEGIN) * TIMESTAMP_RESOLUTION;
    }

    /**
     * Get the language token ID for a language code.
     */
    public static int getLanguageToken(String languageCode) {
        Integer token = LANGUAGE_CODES.get(languageCode.toLowerCase());
        if (token == null) {
            throw new IllegalArgumentException("Unsupported language: " + languageCode);
        }
        return token;
    }

    /**
     * Get the language code from a language token ID.
     */
    public static String getLanguageCode(int tokenId) {
        for (Map.Entry<String, Integer> entry : LANGUAGE_CODES.entrySet()) {
            if (entry.getValue() == tokenId) {
                return entry.getKey();
            }
        }
        return null;
    }

    /**
     * Get all supported language codes.
     */
    public static java.util.Set<String> getSupportedLanguages() {
        return LANGUAGE_CODES.keySet();
    }
}
