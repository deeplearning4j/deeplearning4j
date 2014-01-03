package com.ccc.deeplearning.word2vec;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.regex.Pattern;


public class Word2VecQuizzer {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		if(args.length < 1) {
			System.err.println("Usage input file");
		}
		
		Word2Vec vec = new Word2Vec();
		vec.loadModel(new File(args[0]));
		BufferedReader br = 
				new BufferedReader(new InputStreamReader(System.in));

		String line = null;
		Pattern lsSub = Pattern.compile("ls");
		while(true) {
			System.out.println("ENTER QUERY");
			line = br.readLine();
			java.util.regex.Matcher m = lsSub.matcher(line);
			
			if(line == null || line.equals("quit"))
				break;
			else if(line.equals("ls")) {
				for(String key : vec.getVocab().keySet())
					System.out.println(key);
				continue;
			}
			
			else if(m.find()) {
				String[] split = line.split(" ");
				for(int i = 1; i < split.length; i++) {
					boolean found = vec.getWord(split[i]) != null;
					System.out.println(split[i] + " exists " + found);
				}
				continue;
			}
			
			StringTokenizer tokenizer = new StringTokenizer(line);
			List<String> list = new ArrayList<String>();
			while(tokenizer.hasMoreTokens()) {
				list.add(tokenizer.nextToken());
			}
			if(list.size() < 2) {
				System.out.println("PLEASE ENTER MORE TOKENS");
				continue;
			}
			for(String s : list)
				for(String s2 : list)
					System.out.println("SIMILARITY " + s  + " to " + s2 + " is " + vec.similarity(s, s2));
		}
	}

}
