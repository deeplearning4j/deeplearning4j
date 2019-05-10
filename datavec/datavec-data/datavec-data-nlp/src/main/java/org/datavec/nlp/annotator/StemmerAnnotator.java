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

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.snowball.SnowballStemmer;
import org.cleartk.token.type.Token;


public class StemmerAnnotator extends SnowballStemmer<Token> {

    public static AnalysisEngineDescription getDescription() throws ResourceInitializationException {
        return getDescription("English");
    }


    public static AnalysisEngineDescription getDescription(String language) throws ResourceInitializationException {
        return AnalysisEngineFactory.createEngineDescription(StemmerAnnotator.class, SnowballStemmer.PARAM_STEMMER_NAME,
                        language);
    }


    @SuppressWarnings("unchecked")
    @Override
    public synchronized void process(JCas jCas) throws AnalysisEngineProcessException {
        super.process(jCas);
    }



    @Override
    public void setStem(Token token, String stem) {
        token.setStem(stem);
    }

}
