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

	
	boolean hasMoreTokens();
	
	int countTokens();
	
	String nextToken();
	
	List<String> getTokens();
	
	
	
	
	
}
