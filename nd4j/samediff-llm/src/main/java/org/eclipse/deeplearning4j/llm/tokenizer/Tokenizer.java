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

package org.eclipse.deeplearning4j.llm.tokenizer;

import java.util.List;
import java.util.Map;

/**
 * Interface for text tokenizers used in LLM applications.
 *
 * This interface provides methods for encoding text into token IDs,
 * decoding token IDs back to text, and accessing vocabulary information.
 * Implementations may use HuggingFace tokenizers, SentencePiece, or other
 * tokenization backends.
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * Tokenizer tokenizer = HuggingFaceTokenizer.fromFile("tokenizer.json");
 *
 * // Encode text
 * Encoding encoding = tokenizer.encode("Hello, world!", true);
 * int[] ids = encoding.getIds();
 *
 * // Decode back to text
 * String text = tokenizer.decode(ids, true);
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public interface Tokenizer extends AutoCloseable {

    /**
     * Encode text into tokens.
     *
     * @param text the input text to encode
     * @param addSpecialTokens whether to add special tokens (e.g., [CLS], [SEP])
     * @return the encoding result containing token IDs and metadata
     */
    Encoding encode(String text, boolean addSpecialTokens);

    /**
     * Encode text with default special token handling (adds special tokens).
     *
     * @param text the input text to encode
     * @return the encoding result
     */
    default Encoding encode(String text) {
        return encode(text, true);
    }

    /**
     * Encode multiple texts in batch.
     *
     * @param texts the list of texts to encode
     * @param addSpecialTokens whether to add special tokens
     * @return list of encoding results
     */
    List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens);

    /**
     * Decode token IDs back to text.
     *
     * @param ids the token IDs to decode
     * @param skipSpecialTokens whether to skip special tokens in output
     * @return the decoded text
     */
    String decode(int[] ids, boolean skipSpecialTokens);

    /**
     * Decode token IDs with default handling (skips special tokens).
     *
     * @param ids the token IDs to decode
     * @return the decoded text
     */
    default String decode(int[] ids) {
        return decode(ids, true);
    }

    /**
     * Decode multiple token sequences in batch.
     *
     * @param idsBatch list of token ID arrays to decode
     * @param skipSpecialTokens whether to skip special tokens
     * @return list of decoded texts
     */
    List<String> decodeBatch(List<int[]> idsBatch, boolean skipSpecialTokens);

    /**
     * Get the vocabulary size.
     *
     * @return the number of tokens in the vocabulary
     */
    int getVocabSize();

    /**
     * Get the token ID for a given token string.
     *
     * @param token the token string
     * @return the token ID, or null if not found
     */
    Integer getTokenId(String token);

    /**
     * Get the token string for a given token ID.
     *
     * @param id the token ID
     * @return the token string, or null if not found
     */
    String getToken(int id);

    /**
     * Get the configured chat template for this tokenizer, if one was loaded
     * from tokenizer_config.json or model metadata.
     *
     * @return the chat template string, or null if none is configured
     */
    default String getChatTemplate() {
        return null;
    }

    /**
     * Get the BOS token text, if available.
     *
     * @return the BOS token text, or an empty string if it cannot be resolved
     */
    default String getBosToken() {
        try {
            int id = getBosTokenId();
            return id >= 0 ? getToken(id) : "";
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * Get the EOS token text, if available.
     *
     * @return the EOS token text, or an empty string if it cannot be resolved
     */
    default String getEosToken() {
        try {
            int id = getEosTokenId();
            return id >= 0 ? getToken(id) : "";
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * Apply the tokenizer's configured chat template to messages.
     *
     * @param messages the conversation messages
     * @param addGenerationPrompt whether to append the assistant generation prompt
     * @return formatted chat prompt
     */
    default String applyChatTemplate(List<ChatTemplate.Message> messages, boolean addGenerationPrompt) {
        String template = getChatTemplate();
        if (template == null || template.isBlank()) {
            throw new IllegalStateException("Tokenizer does not provide a chat template");
        }
        return new ChatTemplate(template, getBosToken(), getEosToken())
                .apply(messages, addGenerationPrompt);
    }

    /**
     * Get the full vocabulary mapping.
     *
     * @return map from token strings to token IDs
     */
    Map<String, Integer> getVocab();

    /**
     * Get the padding token ID.
     *
     * @return the PAD token ID, or -1 if not defined
     */
    int getPadTokenId();

    /**
     * Get the beginning-of-sequence token ID.
     *
     * @return the BOS token ID, or -1 if not defined
     */
    int getBosTokenId();

    /**
     * Get the end-of-sequence token ID.
     *
     * @return the EOS token ID, or -1 if not defined
     */
    int getEosTokenId();

    /**
     * Get the unknown token ID.
     *
     * @return the UNK token ID, or -1 if not defined
     */
    int getUnkTokenId();

    /**
     * Check if the tokenizer is still valid and usable.
     *
     * @return true if the tokenizer can be used
     */
    boolean isValid();

    /**
     * Validate that all token IDs in an encoding are within the vocabulary range.
     * Throws {@link TokenizerException} if any token ID is out of bounds.
     *
     * @param encoding the encoding to validate
     * @param vocabSize the vocabulary size to validate against (typically from the model's embedding table)
     * @throws TokenizerException if any token ID >= vocabSize or < 0
     */
    default void validateTokenIds(Encoding encoding, int vocabSize) {
        if (encoding == null || encoding.getIds() == null) return;
        int[] ids = encoding.getIds();
        for (int i = 0; i < ids.length; i++) {
            if (ids[i] < 0 || ids[i] >= vocabSize) {
                throw new TokenizerException(
                        "Token ID " + ids[i] + " at position " + i + " is out of vocabulary range [0, " +
                        vocabSize + "). This typically means the tokenizer vocabulary does not match the model's " +
                        "embedding table, or the input text contains characters not supported by this model.");
            }
        }
    }

    /**
     * Check if the encoding has an unusually high proportion of unknown tokens,
     * which indicates the input text is likely in a language not supported by this model.
     *
     * @param encoding the encoding to check
     * @param threshold the maximum acceptable ratio of UNK tokens (0.0 to 1.0). Recommended: 0.5
     * @throws TokenizerException if UNK ratio exceeds the threshold
     */
    default void validateLanguageSupport(Encoding encoding, double threshold) {
        if (encoding == null || encoding.getIds() == null || encoding.getIds().length == 0) return;
        int unkId = getUnkTokenId();
        if (unkId < 0) return; // No UNK token defined, can't check

        int[] ids = encoding.getIds();
        int unkCount = 0;
        for (int id : ids) {
            if (id == unkId) unkCount++;
        }

        double unkRatio = (double) unkCount / ids.length;
        if (unkRatio > threshold) {
            throw new TokenizerException(
                    "Input text produced " + unkCount + "/" + ids.length + " unknown tokens (" +
                    String.format("%.1f%%", unkRatio * 100) + "), exceeding threshold of " +
                    String.format("%.0f%%", threshold * 100) + ". " +
                    "This model's tokenizer likely does not support the language of the input text. " +
                    "Consider using a multilingual model instead.");
        }
    }
}
