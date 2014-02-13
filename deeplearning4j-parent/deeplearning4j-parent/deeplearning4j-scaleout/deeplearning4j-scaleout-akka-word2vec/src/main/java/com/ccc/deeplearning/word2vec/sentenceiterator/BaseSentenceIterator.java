package com.ccc.deeplearning.word2vec.sentenceiterator;

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
	
	

}
