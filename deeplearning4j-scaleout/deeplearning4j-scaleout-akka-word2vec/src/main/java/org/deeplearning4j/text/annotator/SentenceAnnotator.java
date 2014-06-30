package org.deeplearning4j.text.annotator;

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.util.ParamUtil;
import org.deeplearning4j.word2vec.util.Util;

public class SentenceAnnotator extends org.cleartk.opennlp.tools.SentenceAnnotator {

    static {
        //UIMA logging
        Util.disableLogging();
    }
	public static AnalysisEngineDescription getDescription() throws ResourceInitializationException {
	    return AnalysisEngineFactory.createPrimitiveDescription(
                SentenceAnnotator.class,
                PARAM_SENTENCE_MODEL_PATH,
                ParamUtil.getParameterValue(PARAM_SENTENCE_MODEL_PATH, "/models/en-sent.bin"),
                PARAM_WINDOW_CLASS_NAMES,
                ParamUtil.getParameterValue(PARAM_WINDOW_CLASS_NAMES, null));
	  }

	
	@Override
	public synchronized void process(JCas jCas) throws AnalysisEngineProcessException {
		super.process(jCas);
	}

	

}
