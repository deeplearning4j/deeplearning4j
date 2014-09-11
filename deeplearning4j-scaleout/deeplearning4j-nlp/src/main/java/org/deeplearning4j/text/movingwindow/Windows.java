package org.deeplearning4j.text.movingwindow;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class Windows {


	
	
	/**
	 * Constructs a list of window of size windowSize.
	 * Note that padding for each window is created as well.
	 * @param words the words to tokenize and construct windows from
	 * @param windowSize the window size to generate
	 * @return the list of windows for the tokenized string
	 */
	public static List<Window> windows(String words,int windowSize) {
		StringTokenizer tokenizer = new StringTokenizer(words);
		List<String> list = new ArrayList<String>();
		while(tokenizer.hasMoreTokens())
			list.add(tokenizer.nextToken());
		return windows(list,windowSize);
	}
	
	/**
	 * Constructs a list of window of size windowSize.
	 * Note that padding for each window is created as well.
	 * @param words the words to tokenize and construct windows from
	 * @param tokenizerFactory tokenizer factory to use
	 * @param windowSize the window size to generate
	 * @return the list of windows for the tokenized string
	 */
	public static List<Window> windows(String words,TokenizerFactory tokenizerFactory,int windowSize) {
		Tokenizer tokenizer = tokenizerFactory.create(words);
		List<String> list = new ArrayList<>();
		while(tokenizer.hasMoreTokens())
			list.add(tokenizer.nextToken());

        if(list.isEmpty())
            throw new IllegalStateException("No tokens found for windows");

        return windows(list,windowSize);
	}
	
	
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
	 * Constructs a list of window of size windowSize.
	 * Note that padding for each window is created as well.
	 * @param words the words to tokenize and construct windows from
	 * @param tokenizerFactory tokenizer factory to use
	 * @return the list of windows for the tokenized string
	 */
	public static List<Window> windows(String words,TokenizerFactory tokenizerFactory) {
		Tokenizer tokenizer = tokenizerFactory.create(words);
		List<String> list = new ArrayList<>();
		while(tokenizer.hasMoreTokens())
			list.add(tokenizer.nextToken());
		return windows(list,5);
	}


	/**
	 * Creates a sliding window from text
	 * @param windowSize the window size to use
	 * @param wordPos the position of the word to center
	 * @param sentence the sentence to createComplex a window for
	 * @return a window based on the given sentence
	 */
	public static Window windowForWordInPosition(int windowSize,int wordPos,List<String> sentence) {
		List<String> window = new ArrayList<>();
        List<String> onlyTokens = new ArrayList<>();
		int contextSize = (int) Math.floor((windowSize - 1 ) / 2);

        for (int i =  wordPos - contextSize; i <= wordPos + contextSize;i++){
			if(i < 0)
				window.add("<s>");
			else if(i >= sentence.size())
				window.add("</s>");
			else  {
                onlyTokens.add(sentence.get(i));
                window.add(sentence.get(i));

            }
		}

        String wholeSentence = StringUtils.join(sentence);
        String window2 = StringUtils.join(onlyTokens);
        int begin = wholeSentence.indexOf(window2);
        int end =   begin + window2.length();
        return new Window(window,begin,end);

	}


	/**
	 * Constructs a list of window of size windowSize
	 * @param words the words to  construct windows from
	 * @return the list of windows for the tokenized string
	 */
	public static List<Window> windows(List<String> words,int windowSize) {

		List<Window> ret = new ArrayList<>();

		for(int i = 0; i < words.size(); i++) 
		   ret.add(windowForWordInPosition(windowSize,i,words));
		

		return ret;
	}

}
