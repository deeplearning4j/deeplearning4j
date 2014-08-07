package org.deeplearning4j.topicmodeling;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.util.Index;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Given a {@link SentenceIterator} and a {@link TokenizerFactory}
 * and an {@link Index} constructs a word count vector
 * @author Adam Gibson
 *
 */
public class CountVectorizer {

	private SentenceIterator iter;
	private TokenizerFactory tokenizerFactory;
	private Index wordsToCount;
	private static Logger log = LoggerFactory.getLogger(CountVectorizer.class);
	
	public CountVectorizer(SentenceIterator iter,
			TokenizerFactory tokenizerFactory, Index wordsToCount) {
		super();
		this.iter = iter;
		this.tokenizerFactory = tokenizerFactory;
		this.wordsToCount = wordsToCount;
	}
	

	public CountVectorizer(SentenceIterator iter,
			TokenizerFactory tokenizerFactory) {
		super();
		this.iter = iter;
		this.tokenizerFactory = tokenizerFactory;
		this.wordsToCount = new Index();
	}
	
	
	public DoubleMatrix toNormalizedVector() {
		return MatrixUtil.normalizeByRowSums(toVector());
	}
	
	public DoubleMatrix toVector() {
		DoubleMatrix d = new DoubleMatrix(1,wordsToCount.size());
		Counter<String> wordFrequencies = new Counter<String>();
		while(iter.hasNext()) {
			String sentence = iter.nextSentence();
			if(sentence == null)
				continue;
			Tokenizer t = tokenizerFactory.create(sentence);
			while(t.hasMoreTokens()) {
				String token = t.nextToken();
				log.info("Token " + token);
				if(wordsToCount.indexOf(token) >= 0 || wordsToCount.size() < 1) 
					wordFrequencies.incrementCount(token, 1.0);
					
			}
		}
		
		for(int i = 0; i < wordsToCount.size(); i++) {
			d.put(i,wordFrequencies.getCount(wordsToCount.get(i).toString()));
		}
		
		return d;
	}
	
	public DoubleMatrix toBinaryVector() {
		DoubleMatrix d = new DoubleMatrix(1,wordsToCount.size());
		Counter<String> wordFrequencies = new Counter<String>();
		while(iter.hasNext()) {
			String sentence = iter.nextSentence();
			if(sentence == null)
				continue;
			Tokenizer t = tokenizerFactory.create(sentence);
			while(t.hasMoreTokens()) {
				String token = t.nextToken();
				log.info("Token " + token);
				if(wordsToCount.indexOf(token) >= 0 || wordsToCount.size() < 1) 
					wordFrequencies.incrementCount(token, 1.0);
					
			}
		}
		
		for(int i = 0; i < wordsToCount.size(); i++) {
			double count = wordFrequencies.getCount(wordsToCount.get(i).toString());
			d.put(i,count > 0 ? 1 : 0);
		}
		
		return d;
	}



}
