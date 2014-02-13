package com.ccc.deeplearning.word2vec.sentenceiterator;

import java.util.Collection;
import java.util.Iterator;

public class CollectionSentenceIterator extends BaseSentenceIterator {

	private Iterator<String> iter;
	private Collection<String> coll;
	
	public CollectionSentenceIterator(SentencePreProcessor preProcessor,Collection<String> coll) {
		super(preProcessor);
		this.coll = coll;
		iter = coll.iterator();
	}
	
	public CollectionSentenceIterator(Collection<String> coll) {
		this(null,coll);
	}
	@Override
	public String nextSentence() {
		String ret = iter.next();
		if(this.getPreProcessor() != null)
			ret =this.getPreProcessor().preProcess(ret);
		return ret;
	}

	@Override
	public boolean hasNext() {
		return iter.hasNext();
	}


	@Override
	public void reset() {
		iter = coll.iterator();
	}

	

}
