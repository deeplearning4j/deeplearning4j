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


import org.apache.uima.analysis_engine.AnalysisEngine;
import org.deeplearning4j.text.annotator.PoStagger;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.StemmerAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.tokenization.tokenizer.PosUimaTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.io.InputStream;
import java.util.Collection;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngine;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

/**
 * Creates a tokenizer that filters by 
 * part of speech tags
 * @see {org.deeplearning4j.text.tokenization.tokenizer.PosUimaTokenizer}
 * @author Adam Gibson
 *
 */
public class PosUimaTokenizerFactory implements TokenizerFactory {

    private AnalysisEngine tokenizer;
    private Collection<String> allowedPoSTags;
    private TokenPreProcess tokenPreProcess;
    private boolean stripNones = false;

    public PosUimaTokenizerFactory(Collection<String> allowedPoSTags, boolean stripNones) {
        this(defaultAnalysisEngine(), allowedPoSTags);
        this.stripNones = stripNones;
    }

    public PosUimaTokenizerFactory(Collection<String> allowedPoSTags) {
        this(allowedPoSTags, false);
    }

    public PosUimaTokenizerFactory(AnalysisEngine tokenizer, Collection<String> allowedPosTags) {
        this.tokenizer = tokenizer;
        this.allowedPoSTags = allowedPosTags;
    }


    public static AnalysisEngine defaultAnalysisEngine() {
        try {
            return createEngine(createEngineDescription(SentenceAnnotator.getDescription(),
                            TokenizerAnnotator.getDescription(), PoStagger.getDescription("en"),
                            StemmerAnnotator.getDescription("English")));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public Tokenizer create(String toTokenize) {
        PosUimaTokenizer t = new PosUimaTokenizer(toTokenize, tokenizer, allowedPoSTags, stripNones);
        if (tokenPreProcess != null)
            t.setTokenPreProcessor(tokenPreProcess);
        return t;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        throw new UnsupportedOperationException();
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


}
