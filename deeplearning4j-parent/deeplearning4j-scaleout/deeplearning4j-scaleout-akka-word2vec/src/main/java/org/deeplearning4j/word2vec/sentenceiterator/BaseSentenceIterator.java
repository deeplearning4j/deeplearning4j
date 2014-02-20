package org.deeplearning4j.word2vec.sentenceiterator;

/**
 * Creates a baseline default.
 * This includes the sentence pre processor
 * and a no op finish for iterators
 * with no i/o streams or other finishing steps.
 * @author Adam Gibson
 *
 */
public abstract class BaseSentenceIterator implements SentenceIterator {

	protected SentencePreProcessor preProcessor;

	
	
	
	public BaseSentenceIterator(SentencePreProcessor preProcessor) {
		super();
		this.preProcessor = preProcessor;
	}

	public BaseSentenceIterator() {
		super();
	}

	public SentencePreProcessor getPreProcessor() {
		return preProcessor;
	}

	public void setPreProcessor(SentencePreProcessor preProcessor) {
		this.preProcessor = preProcessor;
	}

	@Override
	public void finish() {
		//No-op
	}
	
	

}
