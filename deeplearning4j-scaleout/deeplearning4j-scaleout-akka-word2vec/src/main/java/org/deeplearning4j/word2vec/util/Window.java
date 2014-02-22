package org.deeplearning4j.word2vec.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.lang3.StringUtils;


/**
 * A representation of a sliding window.
 * This is used for creating training examples.
 * @author Adam Gibson
 *
 */
public class Window implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6359906393699230579L;
	private List<String> words;
	private String label = "NONE";
	private boolean beginLabel;
	private boolean endLabel;
	private int windowSize;
	private int median;
	private static String BEGIN_LABEL = "<[A-Z]+>";
	private static String END_LABEL = "</[A-Z]+>";

	/**
	 * Creates a window with a context of size 3
	 * @param words a collection of strings of size 3
	 */
	public Window(Collection<String> words) {
		this(words,5);

	}

	public String asTokens() {
		return StringUtils.join(words, " ");
	}


	/**
	 * Initialize a window with the given size
	 * @param words the words to use 
	 * @param windowSize the size of the window
	 */
	public Window(Collection<String> words, int windowSize) {
		if(words == null)
			throw new IllegalArgumentException("Words must be a list of size 3");

		this.words = new ArrayList<String>(words);
		this.windowSize = windowSize;
		initContext();
	}


	private void initContext() {
		int median = (int) Math.floor(words.size() / 2);
		List<String> begin = words.subList(0, median);
		List<String> after = words.subList(median + 1,words.size());


		for(String s : begin) {
			if(s.matches(BEGIN_LABEL)) {
				this.label = s.replaceAll("(<|>)","").replace("/","");
				beginLabel = true;
			}
			else if(s.matches(END_LABEL)) {
				endLabel = true;
				this.label = s.replaceAll("(<|>|/)","").replace("/","");

			}

		}

		for(String s1 : after) {
			if(s1.matches(END_LABEL)) {
				endLabel = true;
				this.label = s1.replaceAll("(<|>)","");

			}
		}
		this.median = median;

	}


	
	
	
	@Override
	public String toString() {
		return words.toString();
	}

	public List<String> getWords() {
		return words;
	}

	public void setWords(List<String> words) {
		this.words = words;
	}

	public String getWord(int i) {
		return words.get(i);
	}

	public String getFocusWord() {
		return words.get(median);
	}

	public boolean isBeginLabel() {
		return !label.equals("NONE") && beginLabel;
	}

	public boolean isEndLabel() {
		return !label.equals("NONE") && endLabel;
	}

	public String getLabel() {
		return label.replace("/","");
	}

	public int getWindowSize() {
		return words.size();
	}

	public int getMedian() {
		return median;
	}

	public void setLabel(String label) {
		this.label = label;
	}




}
