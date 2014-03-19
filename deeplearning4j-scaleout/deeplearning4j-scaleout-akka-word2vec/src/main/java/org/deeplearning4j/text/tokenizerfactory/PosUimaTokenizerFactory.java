package org.deeplearning4j.text.tokenizerfactory;

import java.util.Collection;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.deeplearning4j.text.tokenizer.PosUimaTokenizer;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;

/**
 * Creates a tokenizer that filters by 
 * part of speech tags
 * @see {org.deeplearning4j.text.tokenizer.PosUimaTokenizer}
 * @author agibsonccc
 *
 */
public class PosUimaTokenizerFactory implements TokenizerFactory {

	private AnalysisEngine tokenizer;
	private Collection<String> allowedPoSTags;


	public PosUimaTokenizerFactory(AnalysisEngine tokenizer,Collection<String> allowedPosTags) {
		this.tokenizer = tokenizer;
		this.allowedPoSTags = allowedPosTags;
	}



	@Override
	public Tokenizer create(String toTokenize) {
		return new PosUimaTokenizer(toTokenize,tokenizer,allowedPoSTags);
	}


}
