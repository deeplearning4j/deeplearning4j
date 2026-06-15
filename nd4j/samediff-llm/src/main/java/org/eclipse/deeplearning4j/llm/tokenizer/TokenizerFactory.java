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

import java.io.File;
import java.io.IOException;

/**
 * Factory for creating tokenizer instances from various file formats.
 *
 * <p>Supports automatic detection of tokenizer type based on files present:</p>
 * <ul>
 *   <li>tokenizer.json - HuggingFace fast tokenizer</li>
 *   <li>tokenizer.model / *.spiece - SentencePiece</li>
 *   <li>vocab.json + merges.txt - BPE (GPT-2, CLIP)</li>
 *   <li>vocab.txt - WordPiece (BERT)</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * // Auto-detect tokenizer type
 * Tokenizer tokenizer = TokenizerFactory.fromDirectory(new File("model"));
 *
 * // Explicit type
 * Tokenizer tokenizer = TokenizerFactory.createSentencePiece(new File("tokenizer.model"));
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class TokenizerFactory {

    /**
     * Tokenizer types.
     */
    public enum TokenizerType {
        /**
         * HuggingFace fast tokenizer (tokenizer.json).
         */
        HUGGINGFACE,

        /**
         * SentencePiece tokenizer (.model, .spiece).
         */
        SENTENCEPIECE,

        /**
         * CLIP BPE tokenizer (vocab.json + merges.txt).
         */
        CLIP,

        /**
         * WordPiece tokenizer (vocab.txt).
         */
        WORDPIECE
    }

    private TokenizerFactory() {
        // Utility class
    }

    /**
     * Load a tokenizer from a directory, auto-detecting the type.
     *
     * @param directory the model directory
     * @return the tokenizer
     * @throws IOException if loading fails
     * @throws TokenizerException if no tokenizer files found
     */
    public static Tokenizer fromDirectory(File directory) throws IOException, TokenizerException {
        if (!directory.isDirectory()) {
            throw new TokenizerException("Not a directory: " + directory);
        }

        // Check for HuggingFace tokenizer.json
        File tokenizerJson = new File(directory, "tokenizer.json");
        if (tokenizerJson.exists()) {
            return HuggingFaceTokenizer.fromFile(tokenizerJson);
        }

        // Check for SentencePiece
        File tokenizerModel = new File(directory, "tokenizer.model");
        if (tokenizerModel.exists()) {
            return SentencePieceTokenizer.fromFile(tokenizerModel);
        }

        // Check for .spiece file
        File[] spieceFiles = directory.listFiles((dir, name) -> name.endsWith(".spiece"));
        if (spieceFiles != null && spieceFiles.length > 0) {
            return SentencePieceTokenizer.fromFile(spieceFiles[0]);
        }

        // Check for CLIP-style tokenizer
        File vocabJson = new File(directory, "vocab.json");
        File mergesTxt = new File(directory, "merges.txt");
        if (vocabJson.exists() && mergesTxt.exists()) {
            return CLIPTokenizer.fromFiles(vocabJson, mergesTxt);
        }

        // Check for vocab file (could be SentencePiece vocab)
        File vocabTxt = new File(directory, "vocab.txt");
        if (vocabTxt.exists()) {
            return SentencePieceTokenizer.fromVocabFile(vocabTxt);
        }

        throw new TokenizerException("No tokenizer files found in: " + directory);
    }

    /**
     * Load a HuggingFace tokenizer from a tokenizer.json file.
     *
     * @param tokenizerJson the tokenizer.json file
     * @return the tokenizer
     * @throws IOException if loading fails
     */
    public static Tokenizer createHuggingFace(File tokenizerJson) throws IOException {
        return HuggingFaceTokenizer.fromFile(tokenizerJson);
    }

    /**
     * Load a SentencePiece tokenizer from a .model file.
     *
     * @param modelFile the .model or .spiece file
     * @return the tokenizer
     * @throws IOException if loading fails
     */
    public static Tokenizer createSentencePiece(File modelFile) throws IOException {
        return SentencePieceTokenizer.fromFile(modelFile);
    }

    /**
     * Load a CLIP tokenizer from vocab and merges files.
     *
     * @param vocabFile the vocab.json file
     * @param mergesFile the merges.txt file
     * @return the tokenizer
     * @throws IOException if loading fails
     */
    public static Tokenizer createCLIP(File vocabFile, File mergesFile) throws IOException {
        return CLIPTokenizer.fromFiles(vocabFile, mergesFile);
    }

    /**
     * Load a CLIP tokenizer from a directory.
     *
     * @param directory the directory containing vocab.json and merges.txt
     * @return the tokenizer
     * @throws IOException if loading fails
     */
    public static Tokenizer createCLIP(File directory) throws IOException {
        return CLIPTokenizer.fromDirectory(directory);
    }

    /**
     * Detect the tokenizer type from a directory.
     *
     * @param directory the model directory
     * @return the detected tokenizer type
     */
    public static TokenizerType detectType(File directory) {
        if (new File(directory, "tokenizer.json").exists()) {
            return TokenizerType.HUGGINGFACE;
        }
        if (new File(directory, "tokenizer.model").exists()) {
            return TokenizerType.SENTENCEPIECE;
        }
        File[] spieceFiles = directory.listFiles((dir, name) -> name.endsWith(".spiece"));
        if (spieceFiles != null && spieceFiles.length > 0) {
            return TokenizerType.SENTENCEPIECE;
        }
        if (new File(directory, "vocab.json").exists() && new File(directory, "merges.txt").exists()) {
            return TokenizerType.CLIP;
        }
        if (new File(directory, "vocab.txt").exists()) {
            return TokenizerType.WORDPIECE;
        }
        return null;
    }

    /**
     * Load a tokenizer with explicit type.
     *
     * @param type the tokenizer type
     * @param directory the model directory
     * @return the tokenizer
     * @throws IOException if loading fails
     * @throws TokenizerException if the specified type cannot be loaded
     */
    public static Tokenizer create(TokenizerType type, File directory) throws IOException, TokenizerException {
        switch (type) {
            case HUGGINGFACE:
                File tokenizerJson = new File(directory, "tokenizer.json");
                if (!tokenizerJson.exists()) {
                    throw new TokenizerException("tokenizer.json not found in: " + directory);
                }
                return HuggingFaceTokenizer.fromFile(tokenizerJson);

            case SENTENCEPIECE:
                File tokenizerModel = new File(directory, "tokenizer.model");
                if (!tokenizerModel.exists()) {
                    File[] spieceFiles = directory.listFiles((dir, name) -> name.endsWith(".spiece"));
                    if (spieceFiles != null && spieceFiles.length > 0) {
                        tokenizerModel = spieceFiles[0];
                    } else {
                        throw new TokenizerException("SentencePiece model not found in: " + directory);
                    }
                }
                return SentencePieceTokenizer.fromFile(tokenizerModel);

            case CLIP:
                File vocabJson = new File(directory, "vocab.json");
                File mergesTxt = new File(directory, "merges.txt");
                if (!vocabJson.exists() || !mergesTxt.exists()) {
                    throw new TokenizerException("CLIP tokenizer files not found in: " + directory);
                }
                return CLIPTokenizer.fromFiles(vocabJson, mergesTxt);

            case WORDPIECE:
                File vocabTxt = new File(directory, "vocab.txt");
                if (!vocabTxt.exists()) {
                    throw new TokenizerException("vocab.txt not found in: " + directory);
                }
                return SentencePieceTokenizer.fromVocabFile(vocabTxt);

            default:
                throw new TokenizerException("Unknown tokenizer type: " + type);
        }
    }
}
