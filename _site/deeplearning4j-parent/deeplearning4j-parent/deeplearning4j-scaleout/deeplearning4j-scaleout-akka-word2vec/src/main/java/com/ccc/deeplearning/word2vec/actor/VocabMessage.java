package com.ccc.deeplearning.word2vec.actor;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.word2vec.VocabWord;
import com.ccc.deeplearning.word2vec.viterbi.Index;

public class VocabMessage implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6474030463892664765L;
	private Counter<String> rawVocab;
	private List<String> tokens,stopWords;
	private int minWordFrequency;
	private Index wordIndex;
	private Map<String,VocabWord> vocab;
	private int layerSize;
	private AtomicLong changeTracker;
	
	public VocabMessage(Counter<String> rawVocab,List<String> tokens,List<String> stopWords,int minWordFrequency,Index wordIndex,Map<String,VocabWord> vocab,int layerSize,AtomicLong changeTracker) {
		super();
		this.rawVocab = rawVocab;
		this.tokens = tokens;
		this.stopWords = stopWords;
		this.minWordFrequency = minWordFrequency;
		this.wordIndex = wordIndex;
		this.vocab = vocab;
		this.layerSize = layerSize;
		this.changeTracker = changeTracker;
	}
	
	



	public AtomicLong getChangeTracker() {
		return changeTracker;
	}





	public void setChangeTracker(AtomicLong changeTracker) {
		this.changeTracker = changeTracker;
	}





	public int getLayerSize() {
		return layerSize;
	}





	public void setLayerSize(int layerSize) {
		this.layerSize = layerSize;
	}





	public Index getWordIndex() {
		return wordIndex;
	}





	public void setWordIndex(Index wordIndex) {
		this.wordIndex = wordIndex;
	}





	public Map<String, VocabWord> getVocab() {
		return vocab;
	}





	public void setVocab(Map<String, VocabWord> vocab) {
		this.vocab = vocab;
	}





	public int getMinWordFrequency() {
		return minWordFrequency;
	}


	public void setMinWordFrequency(int minWordFrequency) {
		this.minWordFrequency = minWordFrequency;
	}


	public List<String> getStopWords() {
		return stopWords;
	}


	public void setStopWords(List<String> stopWords) {
		this.stopWords = stopWords;
	}


	public Counter<String> getRawVocab() {
		return rawVocab;
	}
	public void setRawVocab(Counter<String> rawVocab) {
		this.rawVocab = rawVocab;
	}
	
	
	public List<String> getTokens() {
		return tokens;
	}
	public void setTokens(List<String> tokens) {
		this.tokens = tokens;
	}
	
	
	
}
