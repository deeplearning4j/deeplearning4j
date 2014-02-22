package org.deeplearning4j.word2vec.actor;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.word2vec.VocabWord;
import org.deeplearning4j.word2vec.viterbi.Index;


public class VocabMessage implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6474030463892664765L;
	private List<String> tokens;
	private AtomicLong changeTracker;
	
	public VocabMessage(List<String> tokens,AtomicLong changeTracker) {
		super();
		this.tokens = tokens;
		this.changeTracker = changeTracker;
	}
	
	



	public AtomicLong getChangeTracker() {
		return changeTracker;
	}





	public void setChangeTracker(AtomicLong changeTracker) {
		this.changeTracker = changeTracker;
	}


	
	public List<String> getTokens() {
		return tokens;
	}
	public void setTokens(List<String> tokens) {
		this.tokens = tokens;
	}
	
	
	
}
