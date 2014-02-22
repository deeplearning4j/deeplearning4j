package org.deeplearning4j.apps;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Scanner;

import org.deeplearning4j.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Word2VecQuizzer {

	
	private static Logger log = LoggerFactory.getLogger(Word2VecQuizzer.class);
	
	/**
	 * @param args
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException {
		Word2Vec vec = new Word2Vec();
		vec.load(new BufferedInputStream(new FileInputStream(new File(args[0]))));
		Scanner scan = new Scanner(System.in);
		String line = null;
		log.info("Enter a word for similarity ");

		while((line = scan.nextLine()) != null && !line.equals("quit")) {
			log.info("Enter 3 words for similarity ");
			String[] split  = line.split(" ");
			log.info("Score " + vec.similarity(split[0], split[1]));
			
		}
		
		scan.close();
		
		
		
	}

}
