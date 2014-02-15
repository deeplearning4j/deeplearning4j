package com.ccc.deeplearning.word2vec.util;

import java.util.List;
import java.util.Map;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.berkeley.MapFactory;

public class Util {

	/**
	 * Returns a thread safe counter
	 * @return
	 */
	public static Counter<String> parallelCounter() {
		MapFactory<String,Double> factory = new MapFactory<String,Double>() {

			private static final long serialVersionUID = 5447027920163740307L;

			@Override
			public Map<String, Double> buildMap() {
				return new java.util.concurrent.ConcurrentHashMap<String,Double>();
			}

		};

		Counter<String> totalWords = new Counter<String>(factory);
		return totalWords;
	}
	
	public static boolean matchesAnyStopWord(List<String> stopWords,String word) {
		for(String s : stopWords)
			if(s.equalsIgnoreCase(word))
				return true;
		return false;
	}


}
