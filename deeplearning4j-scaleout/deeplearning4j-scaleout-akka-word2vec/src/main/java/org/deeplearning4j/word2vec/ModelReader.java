package org.deeplearning4j.word2vec;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.deeplearning4j.word2vec.loader.Word2VecLoader;


public class ModelReader {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		Word2Vec vec = Word2VecLoader.loadModel(new File(args[0]));
		vec.fit();
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
