package com.ccc.deeplearning.word2vec.sentenceiterator;

import java.util.Collection;
import java.util.Iterator;

public class CollectionSentenceIterator implements SentenceIterator {

	private Iterator<String> iter;

	
	public CollectionSentenceIterator(Collection<String> coll) {
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

	

}
