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

package org.datavec.nlp.annotator;


import opennlp.uima.tokenize.TokenizerModelResourceImpl;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.ExternalResourceFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.opennlp.tools.Tokenizer;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;
import org.datavec.nlp.movingwindow.Util;
import org.datavec.nlp.tokenization.tokenizer.ConcurrentTokenizer;


/**
 * Overrides OpenNLP tokenizer to be thread safe
 */
public class TokenizerAnnotator extends Tokenizer {

    static {
        //UIMA logging
        Util.disableLogging();
    }

    public static AnalysisEngineDescription getDescription(String languageCode) throws ResourceInitializationException {
        String modelPath = String.format("/models/%s-token.bin", languageCode);
        return AnalysisEngineFactory.createEngineDescription(ConcurrentTokenizer.class,
                        opennlp.uima.util.UimaUtil.MODEL_PARAMETER,
                        ExternalResourceFactory.createExternalResourceDescription(TokenizerModelResourceImpl.class,
                                        ConcurrentTokenizer.class.getResource(modelPath).toString()),
                        opennlp.uima.util.UimaUtil.SENTENCE_TYPE_PARAMETER, Sentence.class.getName(),
                        opennlp.uima.util.UimaUtil.TOKEN_TYPE_PARAMETER, Token.class.getName());
    }



    public static AnalysisEngineDescription getDescription() throws ResourceInitializationException {
        String modelPath = String.format("/models/%s-token.bin", "en");
        return AnalysisEngineFactory.createEngineDescription(ConcurrentTokenizer.class,
                        opennlp.uima.util.UimaUtil.MODEL_PARAMETER,
                        ExternalResourceFactory.createExternalResourceDescription(TokenizerModelResourceImpl.class,
                                        ConcurrentTokenizer.class.getResource(modelPath).toString()),
                        opennlp.uima.util.UimaUtil.SENTENCE_TYPE_PARAMETER, Sentence.class.getName(),
                        opennlp.uima.util.UimaUtil.TOKEN_TYPE_PARAMETER, Token.class.getName());
    }



}
