package org.deeplearning4j.word2vec.tokenizer;

import java.util.List;

/**
 * A representation of a tokenizer.
 * Different applications may require 
 * different kind of tokenization (say rules based vs more formal NLP approaches)
 * @author Adam Gibson
 *
 */
public interface Tokenizer {

	/**
	 * An iterator for tracking whether
	 * more tokens are left in the iterator not
	 * @return whether there is anymore tokens
	 * to iterate over
	 */
	boolean hasMoreTokens();
	/**
	 * The number of tokens in the tokenizer
	 * @return the number of tokens
	 */
	int countTokens();
	/**
	 * The next token (word usually) in the string
	 * @return the next token in the string if any
	 */
	String nextToken();
	
	/**
	 * Returns a list of all the tokens
	 * @return a list of all the tokens
	 */
	List<String> getTokens();
	
	
	
	
	
}
