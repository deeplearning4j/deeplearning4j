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

package org.deeplearning4j.text.annotator;

import opennlp.uima.tokenize.TokenizerModelResourceImpl;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.fit.factory.ExternalResourceFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.opennlp.tools.Tokenizer;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.text.tokenization.tokenizer.ConcurrentTokenizer;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;


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
        return createEngineDescription(ConcurrentTokenizer.class, opennlp.uima.util.UimaUtil.MODEL_PARAMETER,
                        ExternalResourceFactory.createExternalResourceDescription(TokenizerModelResourceImpl.class,
                                        ConcurrentTokenizer.class.getResource(modelPath).toString()),
                        opennlp.uima.util.UimaUtil.SENTENCE_TYPE_PARAMETER, Sentence.class.getName(),
                        opennlp.uima.util.UimaUtil.TOKEN_TYPE_PARAMETER, Token.class.getName());
    }



    public static AnalysisEngineDescription getDescription() throws ResourceInitializationException {
        String modelPath = String.format("/models/%s-token.bin", "en");
        return createEngineDescription(ConcurrentTokenizer.class, opennlp.uima.util.UimaUtil.MODEL_PARAMETER,
                        ExternalResourceFactory.createExternalResourceDescription(TokenizerModelResourceImpl.class,
                                        ConcurrentTokenizer.class.getResource(modelPath).toString()),
                        opennlp.uima.util.UimaUtil.SENTENCE_TYPE_PARAMETER, Sentence.class.getName(),
                        opennlp.uima.util.UimaUtil.TOKEN_TYPE_PARAMETER, Token.class.getName());
    }



}
