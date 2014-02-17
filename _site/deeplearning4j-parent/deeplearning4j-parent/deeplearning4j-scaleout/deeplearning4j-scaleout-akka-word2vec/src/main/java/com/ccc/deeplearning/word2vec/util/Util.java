package com.ccc.deeplearning.word2vec.util;

import java.util.List;

public class Util {

	public static boolean matchesAnyStopWord(List<String> stopWords,String word) {
		for(String s : stopWords)
			if(s.equalsIgnoreCase(word))
				return true;
		return false;
	}


}
