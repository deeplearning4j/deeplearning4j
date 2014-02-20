package org.deeplearning4j.word2vec.sentenceiterator;
/**
 * A sentence iterator that knows how to iterate over sentence.
 * This can be used in conjunction with more advanced NLP techniques
 * to clearly separate sentences out, or be simpler when as much
 * complexity is not needed.
 * @author Adam Gibson
 *
 */
public interface SentenceIterator {

	/**
	 * Gets the next sentence or null
	 * if there's nothing left (Do yourself a favor and
	 * check hasNext() )
	 * 
	 * @return the next sentence in the iterator
	 */
	String nextSentence();
	/**
	 * Same idea as {@link java.util.Iterator}
	 * @return whether there's anymore sentences left
	 */
	boolean hasNext();
	/**
	 * Resets the iterator to the beginning
	 */
	void reset();
	
	/**
	 * Allows for any finishing (closing of input streams or the like)
	 */
	void finish();
	
	
	SentencePreProcessor getPreProcessor();
	void setPreProcessor(SentencePreProcessor preProcessor);
	
	
}
