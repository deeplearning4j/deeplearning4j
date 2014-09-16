package org.deeplearning4j.models.word2vec.actor;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.berkeley.Counter;


public class SentenceMessage implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8132989189483837222L;
	private String sentence;
	private  AtomicLong changed;
	public SentenceMessage(String sentence,AtomicLong changed) {
		super();
		this.sentence = sentence;
		this.changed = changed;
	}

    public String getSentence() {
		return sentence;
	}
	public void setSentence(String sentence) {
		this.sentence = sentence;
	}
	public AtomicLong getChanged() {
		return changed;
	}
	public void setChanged(AtomicLong changed) {
		this.changed = changed;
	}
	
	

}
