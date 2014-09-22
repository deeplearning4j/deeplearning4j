package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.CasPool;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.StemmerAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.tokenization.tokenizer.UimaTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.uima.UimaResource;


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


	public UimaTokenizerFactory() throws ResourceInitializationException {
		this(defaultAnalysisEngine(),true);
	}


	public UimaTokenizerFactory(UimaResource resource) {
		this(resource,true);
	}


	public UimaTokenizerFactory(AnalysisEngine tokenizer) {
		this(tokenizer,true);
	}



	public UimaTokenizerFactory(UimaResource resource,boolean checkForLabel) {
		this.uimaResource = resource;
		this.checkForLabel = checkForLabel;
	}

	public UimaTokenizerFactory(boolean checkForLabel) throws ResourceInitializationException {
		this(defaultAnalysisEngine(),checkForLabel);
	}



	public UimaTokenizerFactory(AnalysisEngine tokenizer,boolean checkForLabel) {
		super();
		this.checkForLabel = checkForLabel;
		try {
			this.uimaResource = new UimaResource(tokenizer);


		}catch(Exception e) {
			throw new RuntimeException(e);
		}
	}



	@Override
	public  Tokenizer create(String toTokenize) {
		if(toTokenize == null || toTokenize.isEmpty())
			throw new IllegalArgumentException("Unable to proceed; on sentence to tokenize");
		return new UimaTokenizer(toTokenize,uimaResource,checkForLabel);
	}


	public UimaResource getUimaResource() {
		return uimaResource;
	}


	/**
	 * Creates a tokenization,/stemming pipeline
	 * @return a tokenization/stemming pipeline
	 */
	public static AnalysisEngine defaultAnalysisEngine()  {
		try {
			if(defaultAnalysisEngine == null)

				defaultAnalysisEngine =  AnalysisEngineFactory.createEngine(
						AnalysisEngineFactory.createEngineDescription(
								SentenceAnnotator.getDescription(),
								TokenizerAnnotator.getDescription()));

			return defaultAnalysisEngine;
		}catch(Exception e) {
			throw new RuntimeException(e);
		}
	}


}
