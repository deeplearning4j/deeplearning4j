/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

import org.deeplearning4j.text.tokenization.tokenizer.BertWordPieceStreamTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.BertWordPieceTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.io.*;
import java.util.Collections;
import java.util.NavigableMap;
import java.util.TreeMap;

/**
 * Bert WordPiece tokenizer
 * @author Paul Dubs
 */
public class BertWordPieceTokenizerFactory implements TokenizerFactory {

    private final NavigableMap<String, Integer> vocab;
    private TokenPreProcess tokenPreProcess;

    public BertWordPieceTokenizerFactory(NavigableMap<String, Integer> vocab) {
        this.vocab = vocab;
    }

    public BertWordPieceTokenizerFactory(File pathToVocab) throws IOException {
        this(loadVocab(pathToVocab));
    }

    @Override
    public Tokenizer create(String toTokenize) {
        Tokenizer t = new BertWordPieceTokenizer(toTokenize, vocab);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        Tokenizer t = new BertWordPieceStreamTokenizer(toTokenize, vocab);
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


    public static NavigableMap<String, Integer> loadVocab(File pathToVocab) throws IOException {
        final TreeMap<String, Integer> map = new TreeMap<>(Collections.reverseOrder());

        try(final BufferedReader reader =  new BufferedReader(new InputStreamReader(new FileInputStream(pathToVocab)))){
            String token;
            int i = 0;
            while((token = reader.readLine()) != null){
                map.put(token, i++);
            }
        }

        return map;
    }
}
