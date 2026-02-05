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
}
