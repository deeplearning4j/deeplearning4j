package org.deeplearning4j.word2vec.inputsanitation;

import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.List;

/**
 * Performs some very basic textual transformations 
 * such as word shape, lower casing, and stripping of punctuation
 * @author Adam Gibson
 *
 */
public class InputHomogenization {
	private String input;
	private List<String> ignoreCharactersContaining;
	private boolean preserveCase;
	
	/**
	 * Input text to transform
	 * @param input the input text to transform, 
	 * equivalent to calling this(input,false)
	 * wrt preserving case
	 */
	public InputHomogenization(String input) {
		this(input,false);
	}
	
	/**
	 * 
	 * @param input the input to transform
	 * @param preserveCase whether to preserve case
	 */
	public InputHomogenization(String input,boolean preserveCase) {
		this.input = input;
		this.preserveCase = preserveCase;
	}
	
	/**
	 * 
	 * @param input the input to transform
	 * @param ignoreCharactersContaining ignore transformation of words
	 * containigng specified strings
	 */
	public InputHomogenization(String input,List<String> ignoreCharactersContaining) {
		this.input = input;
		this.ignoreCharactersContaining = ignoreCharactersContaining;
	}
	/**
	 * Returns the normalized text passed in via constructor
	 * @return the normalized text passed in via constructor
	 */
	public String transform() {
		StringBuffer sb = new StringBuffer();
		for(int i = 0; i < input.length(); i++) {
			if(ignoreCharactersContaining != null && ignoreCharactersContaining.contains(String.valueOf(input.charAt(i))))
				sb.append(input.charAt(i));
			else if(Character.isDigit(input.charAt(i)))
				sb.append("d");
			else if(Character.isUpperCase(input.charAt(i)) && !preserveCase)
					sb.append(Character.toLowerCase(input.charAt(i)));

			else 
				sb.append(input.charAt(i));

		}
		
		String normalized = Normalizer.normalize(sb.toString(),Form.NFD);
		normalized = normalized.replace(".","");
		normalized = normalized.replace(",","");
		normalized = normalized.replaceAll("\"","");
		normalized = normalized.replace("'","");
		normalized = normalized.replace("(","");
		normalized = normalized.replace(")","");
		normalized = normalized.replace("“","");
		normalized = normalized.replace("”","");
		normalized = normalized.replace("…","");
		normalized = normalized.replace("|","");
		normalized = normalized.replace("/","");
		normalized = normalized.replace("\\", "");
		normalized = normalized.replace("[", "");
		normalized = normalized.replace("]", "");
		normalized = normalized.replace("‘","");
		normalized = normalized.replace("’","");
        normalized = normalized.replaceAll("[!]+","!");
		return normalized;
	}

}
