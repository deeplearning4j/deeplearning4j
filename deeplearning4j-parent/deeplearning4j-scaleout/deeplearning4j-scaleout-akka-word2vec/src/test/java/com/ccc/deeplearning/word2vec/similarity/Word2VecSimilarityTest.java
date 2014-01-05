package com.ccc.deeplearning.word2vec.similarity;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataFetcher;

public class Word2VecSimilarityTest {
	private Word2VecDataFetcher fetcher;
	private static Logger log = LoggerFactory.getLogger(Word2VecDataFetcher.class);
	@Before
	public void setup() {
		/*if(fetcher == null)
			fetcher = new Word2VecDataFetcher("src/test/resources/articles",1);*/
	}


	@Test
	public void testSimilarity() throws Exception {
		String venturebeat = FileUtils.readFileToString(new File("src/test/resources/articles/venturebeat/-2013-12-02-hmmm-apple-buys-twitter-consumer-sentiment-company-topsy-for-200m-"));
		String techcrunch = FileUtils.readFileToString(new File("src/test/resources/articles/techcrunch/-2013-12-02-apple-buys-topsy-for-a-reported-200m-could-use-social-signals-to-bolster-app-store-relevance-"));
		String other = FileUtils.readFileToString(new File("src/test/resources/articles/gigaom/-2009-12-02-bing-becoming-search-in-name-only-"));
		
		File dir = new File("src/test/resources/articles");
		Iterator<File> files = FileUtils.iterateFiles(dir, null, true);
		List<String> sentences = new ArrayList<String>();
		while(files.hasNext()) {
			LineIterator iter2 = FileUtils.lineIterator(files.next());
			while(iter2.hasNext()) {
				sentences.add(iter2.nextLine());
			}
		}
		
		
		Word2Vec vec = new Word2Vec(sentences);
		vec.train();
		
		log.info("Similarity " + vec.similarity("Apple", "Twitter"));
		
		Word2VecSimilarity sim = new Word2VecSimilarity(techcrunch, venturebeat, vec);
		sim.calc();
		log.info(String.valueOf(sim.getDistance()));
		Word2VecSimilarity sim2 = new Word2VecSimilarity(techcrunch, other, vec);
		sim2.calc();
		log.info(String.valueOf(sim2.getDistance()));

	}


}
