package com.ccc.deeplearning.word2vec.similarity;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.util.SetUtils;
import com.ccc.deeplearning.word2vec.Word2Vec;

public class Word2VecSimilarity {


	private Word2Vec vec;
	private String words1;
	private String words2;
	private double distance;
	private static Logger log = LoggerFactory.getLogger(Word2VecSimilarity.class);
	public Word2VecSimilarity(String words1,String words2,Word2Vec vec) {
		this.words1 = words1;
		this.words2 = words2;
		this.vec = vec;
	}

	public void calc() {

		/*
		 * Merely an experiment: you mainly want to use a full blown CDBN with this
		 * with a logistic output compressed.
		 * Take a CRBM and use input reconstruction on one document and use it to reconstruct the other.
		 * The distance is magnified on the reconstructed input.
		 */
		WordMetaData data = new WordMetaData(vec,words1);
		data.calc();
		WordMetaData d2 = new WordMetaData(vec,words2);
		d2.calc();
		List<String> vocab = new ArrayList<String>(SetUtils.union(new HashSet<String>(data.getWordList()), new HashSet<String>(d2.getWordList())));
		
		DoubleMatrix m1 = matrixFor(data,vocab);
		DoubleMatrix m2 = matrixFor(d2,vocab).mini(vocab.size());
		
		distance = m1.distance1(m2);
	} 





	public double getDistance() {
		return distance;
	}

	private DoubleMatrix matrixFor(WordMetaData d,List<String> vocab) {
		List<String> validWords = new ArrayList<String>();
		for(String word : vocab) {
			validWords.add(word);
		}

		DoubleMatrix m1 = new DoubleMatrix(validWords.size(),vec.getLayerSize());
		for(int i = 0; i < validWords.size(); i++) {
			if(d.getWordCounts().getCount(validWords.get(i)) > 0)
				m1.putRow(i, d.getVectorForWord(validWords.get(i)));
			else
				m1.putRow(i,DoubleMatrix.zeros(vec.getLayerSize()));
		}

		return m1;
	}

	

}
