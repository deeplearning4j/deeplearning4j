package com.ccc.deeplearning.word2vec.sentenceiterator;

import java.util.Collection;
import java.util.Iterator;

public class CollectionSentenceIterator implements SentenceIterator {

	private Iterator<String> iter;
	private Collection<String> coll;
	
	public CollectionSentenceIterator(Collection<String> coll) {
		this.coll = coll;
		iter = coll.iterator();
	}
	
	
	@Override
	public String nextSentence() {
		return iter.next();
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
