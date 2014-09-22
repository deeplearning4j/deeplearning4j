package org.deeplearning4j.text.tokenization.tokenizerfactory;

import java.io.InputStream;

import org.deeplearning4j.text.tokenization.tokenizer.DefaultStreamTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.DefaultTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class DefaultTokenizerFactory implements TokenizerFactory {

	@Override
	public Tokenizer create(String toTokenize) {
		return new DefaultTokenizer(toTokenize);
	}

	@Override
	public Tokenizer create(InputStream toTokenize) {
		return new DefaultStreamTokenizer(toTokenize);
	}

	

}
