package org.deeplearning4j.text.tokenization.tokenizerfactory;

import java.io.InputStream;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

/**
 * Generates a tokenizer for a given string
 * @author Adam Gibson
 *
 */
public interface TokenizerFactory {

	/**
	 * The tokenizer to createComplex
	 * @param toTokenize the string to createComplex the tokenizer with
	 * @return the new tokenizer
	 */
      Tokenizer create(String toTokenize);
	
	/**
	 * Create a tokenizer based on an input stream
	 * @param toTokenize
	 * @return
	 */
	Tokenizer create(InputStream toTokenize);

    /**
     * Sets a token pre processor to be used
     * with every tokenizer
     * @param preProcessor the token pre processor to use
     */
	void setTokenPreProcessor(TokenPreProcess preProcessor);
	
	
	
}
