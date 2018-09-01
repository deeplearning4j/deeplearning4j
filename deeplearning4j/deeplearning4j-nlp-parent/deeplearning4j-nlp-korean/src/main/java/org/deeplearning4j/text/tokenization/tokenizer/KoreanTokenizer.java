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

package org.deeplearning4j.text.tokenization.tokenizer;

import com.twitter.penguin.korean.KoreanTokenJava;
import com.twitter.penguin.korean.TwitterKoreanProcessorJava;
import scala.collection.Seq;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by kepricon on 16. 10. 20.
 * KoreanTokenizer using KoreanTwitterText (<a href="https://github.com/twitter/twitter-korean-text">https://github.com/twitter/twitter-korean-text</a>)
 */
public class KoreanTokenizer implements Tokenizer {
    private Iterator<String> tokenIter;
    private List<String> tokenList;

    private TokenPreProcess preProcess;

    public KoreanTokenizer(String toTokenize) {

        // need normalize?

        // Tokenize
        Seq<com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken> tokens =
                        TwitterKoreanProcessorJava.tokenize(toTokenize);
        tokenList = new ArrayList<>();
        Iterator<KoreanTokenJava> iter = TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(tokens).iterator();

        while (iter.hasNext()) {
            tokenList.add(iter.next().getText());
        }
        tokenIter = tokenList.iterator();
    }

    @Override
    public boolean hasMoreTokens() {
        return tokenIter.hasNext();
    }

    @Override
    public int countTokens() {
        return tokenList.size();
    }

    @Override
    public String nextToken() {
        if (hasMoreTokens() == false) {
            throw new NoSuchElementException();
        }
        return this.preProcess != null ? this.preProcess.preProcess(tokenIter.next()) : tokenIter.next();
    }

    @Override
    public List<String> getTokens() {
        return tokenList;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcess) {
        this.preProcess = tokenPreProcess;
    }
}
