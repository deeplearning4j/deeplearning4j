package org.deeplearning4j.word2vec.tokenizer;

public class DefaultTokenizerFactory implements TokenizerFactory {

	@Override
	public Tokenizer create(String toTokenize) {
		return new DefaultTokenizer(toTokenize);
	}

	

}
