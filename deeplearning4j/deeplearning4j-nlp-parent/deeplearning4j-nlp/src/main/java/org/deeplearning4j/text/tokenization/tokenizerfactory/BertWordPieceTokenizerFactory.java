/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.text.tokenization.tokenizerfactory;

import lombok.NonNull;
import org.deeplearning4j.text.tokenization.tokenizer.BertWordPieceStreamTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.BertWordPieceTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.io.*;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

/**
 * Bert WordPiece tokenizer
 * @author Paul Dubs
 */
public class BertWordPieceTokenizerFactory implements TokenizerFactory {

    private final NavigableMap<String, Integer> vocab;
    private TokenPreProcess tokenPreProcess;
    private boolean lowerCaseOnly = false;

    public BertWordPieceTokenizerFactory(NavigableMap<String, Integer> vocab) {
        this.vocab = vocab;
    }

    public BertWordPieceTokenizerFactory(File pathToVocab, @NonNull Charset charset) throws IOException {
        this(loadVocab(pathToVocab, charset));
    }

    public BertWordPieceTokenizerFactory(InputStream vocabInputStream, @NonNull Charset charset) throws IOException {
        this(loadVocab(vocabInputStream, charset));
    }

    @Override
    public Tokenizer create(String toTokenize) {
        Tokenizer t = new BertWordPieceTokenizer(toTokenize, vocab, lowerCaseOnly);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        Tokenizer t = new BertWordPieceStreamTokenizer(toTokenize, vocab, lowerCaseOnly);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess preProcessor) {
        this.tokenPreProcess = preProcessor;
    }

    /**
     * Returns TokenPreProcessor set for this TokenizerFactory instance
     *
     * @return TokenPreProcessor instance, or null if no preprocessor was defined
     */
    @Override
    public TokenPreProcess getTokenPreProcessor() {
        return tokenPreProcess;
    }


    public boolean isLowerCaseOnly() {
        return lowerCaseOnly;
    }

    public void setLowerCaseOnly(boolean lowerCaseOnly) {
        this.lowerCaseOnly = lowerCaseOnly;
    }

    public Map<String,Integer> getVocab(){
        return Collections.unmodifiableMap(vocab);
    }

    /**
     * The expected format is a \n seperated list of tokens for examples
     *
     * <code>
     *     foo
     *     bar
     *     baz
     * </code>
     *
     * the tokens should <b>not</b> have any whitespace on either of their sides
     *
     * @param is InputStream
     * @return A vocab map with the popper sort order for fast traversal
     */
    public static NavigableMap<String, Integer> loadVocab(InputStream is, Charset charset) throws IOException {
        final TreeMap<String, Integer> map = new TreeMap<>(Collections.reverseOrder());

        try (final BufferedReader reader = new BufferedReader(new InputStreamReader(is, charset))) {
            String token;
            int i = 0;
            while ((token = reader.readLine()) != null) {
                map.put(token, i++);
            }
        }

        return map;
    }

    public static NavigableMap<String, Integer> loadVocab(File vocabFile, Charset charset) throws IOException {
        return loadVocab(new FileInputStream(vocabFile), charset);
    }

}
