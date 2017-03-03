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

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.util.ParamUtil;
import org.deeplearning4j.text.movingwindow.Util;

public class SentenceAnnotator extends org.cleartk.opennlp.tools.SentenceAnnotator {

    static {
        //UIMA logging
        Util.disableLogging();
    }

    public static AnalysisEngineDescription getDescription() throws ResourceInitializationException {
        return AnalysisEngineFactory.createPrimitiveDescription(SentenceAnnotator.class, PARAM_SENTENCE_MODEL_PATH,
                        ParamUtil.getParameterValue(PARAM_SENTENCE_MODEL_PATH, "/models/en-sent.bin"),
                        PARAM_WINDOW_CLASS_NAMES, ParamUtil.getParameterValue(PARAM_WINDOW_CLASS_NAMES, null));
    }


    @Override
    public synchronized void process(JCas jCas) throws AnalysisEngineProcessException {
        super.process(jCas);
    }



}
