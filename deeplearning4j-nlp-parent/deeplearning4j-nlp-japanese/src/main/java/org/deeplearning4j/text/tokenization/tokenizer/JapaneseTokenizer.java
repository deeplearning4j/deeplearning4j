/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.text.tokenization.tokenizer;

import com.atilika.kuromoji.ipadic.Token;
import com.atilika.kuromoji.ipadic.Tokenizer;

import java.util.*;

/**
 * modified by kepricon on 16. 10. 28.
 * A thin wrapper for Japanese Morphological Analyzer Kuromoji (ver.0.9.0),
 * it tokenizes texts which is written in languages
 * that words are not separated by whitespaces.
 *
 * In thenory, Kuromoji is a language-independent Morphological Analyzer library,
 * so if you want to tokenize non-Japanese texts (Chinese, Korean etc.),
 * you can do it with MeCab style dictionary for each languages.
 */
public class JapaneseTokenizer implements org.deeplearning4j.text.tokenization.tokenizer.Tokenizer {

    private final List<Token> tokens;
    private final boolean useBaseForm;
    private final int tokenCount;
    private int currentToken;
    private TokenPreProcess preProcessor;

    public JapaneseTokenizer(Tokenizer kuromoji, String toTokenize, boolean useBaseForm) {
        this.useBaseForm = useBaseForm;
        this.tokens = kuromoji.tokenize(toTokenize);
        this.tokenCount = this.tokens.size();
        this.currentToken = 0;
    }

    @Override
    public boolean hasMoreTokens() {
        return currentToken < tokenCount;
    }

    @Override
    public int countTokens() {
        return tokenCount;
    }

    private String getToken(int i) {
        Token t = this.tokens.get(i);
        String ret = (useBaseForm) ? t.getBaseForm() : t.getSurface();
        return (preProcessor == null) ? ret : preProcessor.preProcess(ret);
    }

    @Override
    public String nextToken() {
        if (!hasMoreTokens()) {
            throw new NoSuchElementException();
        }
        return getToken(currentToken++);
    }

    @Override
    public List<String> getTokens() {
        ArrayList<String> ret = new ArrayList<>(this.tokenCount);

        for (int i = 0; i < this.tokenCount; i++) {
            ret.add(getToken(i));
        }

        return ret;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.preProcessor = tokenPreProcessor;
    }
}

