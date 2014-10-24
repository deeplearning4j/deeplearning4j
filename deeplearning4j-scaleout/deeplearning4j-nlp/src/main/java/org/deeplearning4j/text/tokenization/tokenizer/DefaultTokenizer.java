package org.deeplearning4j.text.tokenization.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Default tokenizer
 * @author Adam Gibson
 */
public class DefaultTokenizer implements Tokenizer {

	public DefaultTokenizer(String tokens) {
		tokenizer = new StringTokenizer(tokens);
	}
	
	private StringTokenizer tokenizer;
	private TokenPreProcess tokenPreProcess;
	
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
		String base =  tokenizer.nextToken();
        if(tokenPreProcess != null)
            base = tokenPreProcess.preProcess(base);
        return base;
	}

	@Override
	public List<String> getTokens() {
		List<String> tokens = new ArrayList<>();
		while(hasMoreTokens()) {
			tokens.add(nextToken());
		}
		return tokens;
	}

	@Override
	public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
		this.tokenPreProcess = tokenPreProcessor;
		
	}
	
	

	
}
