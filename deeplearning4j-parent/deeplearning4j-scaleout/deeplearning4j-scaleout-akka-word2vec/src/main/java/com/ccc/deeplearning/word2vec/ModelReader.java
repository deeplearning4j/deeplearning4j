package com.ccc.deeplearning.word2vec;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

public class ModelReader {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException {
		Word2Vec vec = new Word2Vec();
		vec.loadBinary(new File(args[0]));
		vec.train();
		BufferedReader br = 
				new BufferedReader(new InputStreamReader(System.in));

		String line = null;
		while(true) {
			System.out.println("ENTER QUERY");
			line = br.readLine();
			if(line == null || line.equals("quit"))
				break;
			else if(line.equals("ls")) {
				for(String key : vec.getVocab().keySet())
					System.out.println(key);
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
