package org.deeplearning4j.text.annotator;

import opennlp.uima.tokenize.TokenizerModelResourceImpl;

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;
import org.deeplearning4j.text.tokenizer.ConcurrentTokenizer;
import org.uimafit.factory.AnalysisEngineFactory;
import org.uimafit.factory.ExternalResourceFactory;



public class TokenizerAnnotator extends org.cleartk.opennlp.Tokenizer {
	
	
	public static AnalysisEngineDescription getDescription(String languageCode)
		      throws ResourceInitializationException {
		    String modelPath = String.format("/models/%s-token.bin", languageCode);
		    return AnalysisEngineFactory.createPrimitiveDescription(
		    		ConcurrentTokenizer.class,
		        opennlp.uima.util.UimaUtil.MODEL_PARAMETER,
		        ExternalResourceFactory.createExternalResourceDescription(
		            TokenizerModelResourceImpl.class,
		            ConcurrentTokenizer.class.getResource(modelPath).toString()),
		        opennlp.uima.util.UimaUtil.SENTENCE_TYPE_PARAMETER,
		        Sentence.class.getName(),
		        opennlp.uima.util.UimaUtil.TOKEN_TYPE_PARAMETER,
		        Token.class.getName());
		  }

	
	
	public static AnalysisEngineDescription getDescription()
		      throws ResourceInitializationException {
		    String modelPath = String.format("/models/%s-token.bin", "en");
		    return AnalysisEngineFactory.createPrimitiveDescription(
		    		ConcurrentTokenizer.class,
		        opennlp.uima.util.UimaUtil.MODEL_PARAMETER,
		        ExternalResourceFactory.createExternalResourceDescription(
		            TokenizerModelResourceImpl.class,
		            ConcurrentTokenizer.class.getResource(modelPath).toString()),
		        opennlp.uima.util.UimaUtil.SENTENCE_TYPE_PARAMETER,
		        Sentence.class.getName(),
		        opennlp.uima.util.UimaUtil.TOKEN_TYPE_PARAMETER,
		        Token.class.getName());
		  }

	

	
	
	
	
	
	
	 
}
