package org.deeplearning4j.word2vec.ner;

import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.List;

import org.apache.commons.lang3.StringUtils;

import bsh.StringUtil;

public class InputHomogenization {
	private String input;
	private List<String> ignoreCharactersContaining;
	private boolean preserveCase;
	public InputHomogenization(String input) {
		this(input,false);
	}
	
	public InputHomogenization(String input,boolean preserveCase) {
		this.input = input;
		this.preserveCase = preserveCase;
	}
	
	public InputHomogenization(String input,List<String> ignoreCharactersContaining) {
		this.input = input;
		this.ignoreCharactersContaining = ignoreCharactersContaining;
	}
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
		return normalized;
	}

}
