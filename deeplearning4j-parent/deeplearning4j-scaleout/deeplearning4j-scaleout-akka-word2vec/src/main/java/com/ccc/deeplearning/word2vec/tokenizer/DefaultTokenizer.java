package com.ccc.deeplearning.word2vec.tokenizer;

import java.util.StringTokenizer;

public class DefaultTokenizer implements Tokenizer {

	public DefaultTokenizer(String tokens) {
		tokenizer = new StringTokenizer(tokens);
	}
	
	private StringTokenizer tokenizer;
	
	
	@Override
	public boolean hasMoreTokens() {
		return tokenizer.hasMoreTokens();
	}

	@Override
	public int countTokens() {
		return tokenizer.countTokens();
	}

	@Override
	public String nextToken() {
		return tokenizer.nextToken();
	}

	
}
