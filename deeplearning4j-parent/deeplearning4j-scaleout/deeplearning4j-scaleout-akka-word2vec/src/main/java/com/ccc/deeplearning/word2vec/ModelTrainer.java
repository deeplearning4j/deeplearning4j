package com.ccc.deeplearning.word2vec;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.word2vec.loader.Word2VecSaver;

public class ModelTrainer {

	private static Logger log = LoggerFactory.getLogger(ModelTrainer.class);
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		if(args.length < 2) {
			log.info("Usage input output");
			System.exit(1);
		}
		
		@SuppressWarnings("unchecked")
		Iterator<File> files = FileUtils.iterateFiles(new File(args[0]), null	,true);
		log.info("Iterating over corpora at " + args[0]);
		List<String> sentences = new ArrayList<String>();
		while(files.hasNext()) {
			File next = files.next();
			@SuppressWarnings("unchecked")
			Collection<String> lines = FileUtils.readLines(next);
			sentences.addAll(lines);
		}
		
		log.info("Training on " + sentences.size() + " sentences");
		
		Word2Vec vec = new Word2Vec(sentences,1);
		vec.setWindow(3);
		vec.train();
		Word2VecSaver.saveModel(vec,new File(args[1]));
		log.info("Saved to " + args[1]);
	}

}
