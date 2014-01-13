package com.ccc.deeplearning.word2vec.util;

import java.util.List;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.ner.InputHomogenization;


public class WindowConverter {
	
	public static double[] asExample(Window window,Word2Vec vec) {
		int length = vec.getLayerSize();
		List<String> words = window.getWords();
		int windowSize = window.getWindowSize();
		
		double[] example = new double[ length * windowSize];
		int count = 0;
		for(int i = 0; i < words.size(); i++) {
			String word = new InputHomogenization(words.get(i)).transform();
			double[] vec2 = vec.getWordVector(word);
			if(vec2 == null)
				vec2 = vec.getOob();
			System.arraycopy(vec2, 0, example, count, length);
			count += length;
		}

		return example;
	}

	

	
	
	public static DoubleMatrix asExampleMatrix(Window window,Word2Vec vec) {
		return new DoubleMatrix(asExample(window,vec));
	}

}
