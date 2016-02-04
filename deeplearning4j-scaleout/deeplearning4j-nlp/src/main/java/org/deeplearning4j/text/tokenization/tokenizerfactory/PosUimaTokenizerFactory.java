/*
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

package org.deeplearning4j.text.tokenization.tokenizerfactory;


import static  org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static  org.apache.uima.fit.factory.AnalysisEngineFactory.createEngine;

import java.io.InputStream;
import java.util.Collection;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.deeplearning4j.text.annotator.PoStagger;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.StemmerAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.tokenization.tokenizer.PosUimaTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

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


    public PosUimaTokenizerFactory(Collection<String> allowedPoSTags) {
        this(defaultAnalysisEngine(),allowedPoSTags);
    }

    public PosUimaTokenizerFactory(AnalysisEngine tokenizer,Collection<String> allowedPosTags) {
        this.tokenizer = tokenizer;
        this.allowedPoSTags = allowedPosTags;
    }


    public static AnalysisEngine defaultAnalysisEngine()  {
        try {
            return createEngine(
                    createEngineDescription(SentenceAnnotator.getDescription(),
                            TokenizerAnnotator.getDescription(),
                            PoStagger.getDescription("en"),
                            StemmerAnnotator.getDescription("English")));
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public Tokenizer create(String toTokenize) {
        PosUimaTokenizer t =  new PosUimaTokenizer(toTokenize,tokenizer,allowedPoSTags);
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
