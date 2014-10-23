package org.deeplearning4j.text.tokenization.tokenizer;

import java.io.StreamTokenizer;
import java.util.ArrayList;
import java.util.List;
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

	@Override
	public List<String> getTokens() {
		List<String> tokens = new ArrayList<String>();
		while(hasMoreTokens()) {
			tokens.add(nextToken());
		}
		return tokens;
	}

	@Override
	public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
		// TODO Auto-generated method stub
		
	}
	
	

	
}
