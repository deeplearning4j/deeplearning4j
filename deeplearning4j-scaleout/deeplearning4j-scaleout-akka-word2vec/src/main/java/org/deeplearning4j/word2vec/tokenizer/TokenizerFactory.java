package org.deeplearning4j.word2vec.tokenizer;

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
	
	
}
