package com.ccc.deeplearning.word2vec.util;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

public class Windows {
	/**
	 * Constructs a list of window of size windowSize.
	 * Note that padding for each window is created as well.
	 * @param words the words to tokenize and construct windows from
	 * @return the list of windows for the tokenized string
	 */
	public static List<Window> windows(String words) {
		StringTokenizer tokenizer = new StringTokenizer(words);
		List<String> list = new ArrayList<String>();
		while(tokenizer.hasMoreTokens())
			list.add(tokenizer.nextToken());
		return windows(list,5);
	}

	/**
	 * Creates a sliding window from text
	 * @param windowSize the window size to use
	 * @param wordPos the position of the word to center
	 * @param sentence the sentence to create a window for
	 * @return a window based on the given sentence
	 */
	public static Window windowForWordInPosition(int windowSize,int wordPos,List<String> sentence) {
		List<String> window = new ArrayList<String>();
		int contextSize = (int) Math.floor((windowSize - 1 ) / 2);
		for (int i =  wordPos - contextSize; i <= wordPos + contextSize;i++){
			if(i < 0)
				window.add("<s>");
			else if(i >= sentence.size())
				window.add("</s>");
			else 
				window.add(sentence.get(i));
		}
		return new Window(window);

	}


	/**
	 * Constructs a list of window of size windowSize
	 * @param words the words to  construct windows from
	 * @return the list of windows for the tokenized string
	 */
	public static List<Window> windows(List<String> words,int windowSize) {

		List<Window> ret = new ArrayList<Window>();

		for(int i = 0; i < words.size(); i++) 
		   ret.add(windowForWordInPosition(windowSize,i,words));
		

		return ret;
	}

}
