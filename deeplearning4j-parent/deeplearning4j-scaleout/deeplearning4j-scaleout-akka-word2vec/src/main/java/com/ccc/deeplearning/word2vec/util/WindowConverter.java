package com.ccc.deeplearning.word2vec.util;

import java.util.List;

import com.ccc.deeplearning.word2vec.Word2Vec;


public class WindowConverter {
	
	public static double[] asExample(Window window,Word2Vec vec) {
		int length = vec.getLayerSize();
		List<String> words = window.getWords();
		int windowSize = window.getWindowSize();
		
		double[] example = new double[ length * windowSize];
		int count = 0;
		for(int i = 0; i < words.size(); i++) {
			double[] vec2 = vec.getWordVector(words.get(i));
			if(vec2 == null)
				vec2 = vec.getOob();
			System.arraycopy(vec2, 0, example, count, length);
			count += length;
		}

		return example;
	}



}
