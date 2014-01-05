package com.ccc.deeplearning.word2vec.similarity;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.rbm.matrix.jblas.CRBM;
import com.ccc.deeplearning.util.MatrixUtil;
import com.ccc.deeplearning.util.SetUtils;
import com.ccc.deeplearning.word2vec.Word2Vec;

public class Word2VecSimilarity {


	private Word2Vec vec;
	private String words1;
	private String words2;
	private double distance;
	private static Logger log = LoggerFactory.getLogger(Word2VecSimilarity.class);
	private CRBM crbm;
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
		WordMetaData data = new WordMetaData(words1);
		data.calc();
		WordMetaData d2 = new WordMetaData(words2);
		d2.calc();
		List<String> vocab = new ArrayList<String>(SetUtils.union(new HashSet<String>(data.wordList), new HashSet<String>(d2.wordList)));
		DoubleMatrix m1 = MatrixUtil.unitVec(matrixFor(data,vocab));
		DoubleMatrix m2 = MatrixUtil.unitVec(matrixFor(d2,vocab));
		crbm = new CRBM.Builder().withRandom(new MersenneTwister(123)).numberOfVisible(m1.columns).numHidden(m1.columns).build();
		DoubleMatrix c1 = matrixFor(data,vocab);
		DoubleMatrix c2 = matrixFor(d2,vocab);

		for(int i = 0; i < 5; i++)
			crbm.trainTillConvergence(0.1, 1, c1);

		DoubleMatrix reconstruct1 = crbm.reconstruct(c1);
		DoubleMatrix reconstruct2 = crbm.reconstruct(c2);


		log.info("Un transformed distance " + m1.distance1(m2));
		distance = reconstruct1.distance1(reconstruct2);
		log.info("Reconstructed distance " + distance);
	} 





	public double getDistance() {
		return distance;
	}

	private DoubleMatrix matrixFor(WordMetaData d,List<String> vocab) {
		DoubleMatrix m1 = new DoubleMatrix(vocab.size(),vec.getLayerSize());
		for(int i = 0; i < vocab.size(); i++) {
			m1.putRow(i, d.getVectorForWord(vocab.get(i)));
		}

		return m1;
	}


	private class WordMetaData {
		private String words;
		private Counter<String> wordCounts;
		private List<String> wordList;

		public WordMetaData(String words) {
			this.words = words;
			this.wordCounts = new Counter<String>();
			wordList = new ArrayList<String>();
		}

		public DoubleMatrix getVectorForWord(String word) {
			return vec.getWordVectorMatrix(word).mul(wordCounts.getCount(word) * 100);
		}

		private void addWords(String words) {
			StringTokenizer t1 = new StringTokenizer(words);
			while(t1.hasMoreTokens()) {
				String next = t1.nextToken();
				if(!wordList.contains(next) && vec.hasWord(next)) {
					wordList.add(next);
				}
				if(vec.hasWord(next))
					wordCounts.incrementCount(next, 1.0);
			}
		}

		public void calc() {
			addWords(words);
		}

	}




}
