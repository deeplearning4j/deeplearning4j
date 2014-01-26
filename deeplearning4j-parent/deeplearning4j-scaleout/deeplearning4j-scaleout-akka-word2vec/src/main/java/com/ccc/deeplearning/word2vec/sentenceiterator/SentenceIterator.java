package com.ccc.deeplearning.word2vec.sentenceiterator;

public interface SentenceIterator {

	String nextSentence();
	boolean hasNext();
	void reset();
	
}
