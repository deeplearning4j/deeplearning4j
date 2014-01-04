package com.ccc.deeplearning.word2vec.similarity;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.util.SetUtils;
import com.ccc.deeplearning.word2vec.Word2Vec;

public class Word2VecSimilarity {


	private Word2Vec vec;
	private String words1;
	private String words2;
	private double distance;

	public Word2VecSimilarity(String words1,String words2,Word2Vec vec) {
		this.words1 = words1;
		this.words2 = words2;
		this.vec = vec;
	}

	public void calc() {
		WordMetaData data = new WordMetaData(words1);
		data.calc();
		WordMetaData d2 = new WordMetaData(words2);
		d2.calc();
		List<String> vocab = new ArrayList<String>(SetUtils.union(new HashSet<String>(data.wordList), new HashSet<String>(d2.wordList)));
		DoubleMatrix m1 = matrixFor(data,vocab);
		DoubleMatrix m2 = matrixFor(d2,vocab);
		distance = m1.distance1(m2);
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
			return vec.getWordVectorMatrix(word);
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
