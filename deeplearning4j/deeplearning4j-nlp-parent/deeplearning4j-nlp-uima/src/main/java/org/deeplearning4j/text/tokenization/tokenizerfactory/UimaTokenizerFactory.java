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
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.UimaTokenizer;
import org.deeplearning4j.text.uima.UimaResource;

import java.io.InputStream;


/**
 * Uses a uima {@link AnalysisEngine} to 
 * tokenize text.
 *
 *
 * @author Adam Gibson
 *
 */
public class UimaTokenizerFactory implements TokenizerFactory {

    private UimaResource uimaResource;
    private boolean checkForLabel;
    private static AnalysisEngine defaultAnalysisEngine;
    private TokenPreProcess preProcess;

    public UimaTokenizerFactory() throws ResourceInitializationException {
        this(defaultAnalysisEngine(), true);
    }

    public UimaTokenizerFactory(UimaResource resource) {
        this(resource, true);
    }

    public UimaTokenizerFactory(AnalysisEngine tokenizer) {
        this(tokenizer, true);
    }

    public UimaTokenizerFactory(UimaResource resource, boolean checkForLabel) {
        this.uimaResource = resource;
        this.checkForLabel = checkForLabel;
    }

    public UimaTokenizerFactory(boolean checkForLabel) throws ResourceInitializationException {
        this(defaultAnalysisEngine(), checkForLabel);
    }

    public UimaTokenizerFactory(AnalysisEngine tokenizer, boolean checkForLabel) {
        super();
        this.checkForLabel = checkForLabel;
        try {
            this.uimaResource = new UimaResource(tokenizer);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Tokenizer create(String toTokenize) {
        if (toTokenize == null)
            throw new IllegalArgumentException("Unable to proceed; on sentence to tokenize");
        Tokenizer ret = new UimaTokenizer(toTokenize, uimaResource, checkForLabel);
        ret.setTokenPreProcessor(preProcess);
        return ret;
    }

    public UimaResource getUimaResource() {
        return uimaResource;
    }


    /**
     * Creates a tokenization,/stemming pipeline
     * @return a tokenization/stemming pipeline
     */
    public static AnalysisEngine defaultAnalysisEngine() {
        try {
            if (defaultAnalysisEngine == null)
                defaultAnalysisEngine = AnalysisEngineFactory.createEngine(
                                AnalysisEngineFactory.createEngineDescription(SentenceAnnotator.getDescription(),
                                                TokenizerAnnotator.getDescription()));

            return defaultAnalysisEngine;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
