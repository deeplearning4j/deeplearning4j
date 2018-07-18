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

import org.deeplearning4j.text.tokenization.tokenizer.NGramTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.io.InputStream;

/**
 * @author sonali
 */
public class NGramTokenizerFactory implements TokenizerFactory {
    private TokenPreProcess preProcess;
    private Integer minN = 1;
    private Integer maxN = 1;
    private TokenizerFactory tokenizerFactory;

    public NGramTokenizerFactory(TokenizerFactory tokenizerFactory, Integer minN, Integer maxN) {
        this.tokenizerFactory = tokenizerFactory;
        this.minN = minN;
        this.maxN = maxN;
    }

    @Override
    public Tokenizer create(String toTokenize) {
        if (toTokenize == null || toTokenize.isEmpty()) {
            throw new IllegalArgumentException("Unable to proceed; no sentence to tokenize");
        }

        Tokenizer t1 = tokenizerFactory.create(toTokenize);
        t1.setTokenPreProcessor(preProcess);
        Tokenizer ret = new NGramTokenizer(t1, minN, maxN);
        return ret;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess preProcessor) {
        this.preProcess = preProcessor;
    }

    /**
     * Returns TokenPreProcessor set for this TokenizerFactory instance
     *
     * @return TokenPreProcessor instance, or null if no preprocessor was defined
     */
    @Override
    public TokenPreProcess getTokenPreProcessor() {
        return preProcess;
    }
}
