package com.ccc.deeplearning.word2vec.similarity;

import static org.junit.Assert.*;


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
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataFetcher;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;

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
		
		
		
		
		Word2Vec vec = Word2VecLoader.loadModel(new ClassPathResource("/word2vecmodel-techblogs.bin").getFile());
		
		log.info("Similarity " + vec.similarity("Apple", "Twitter"));
		
	
		Word2VecSimilarity sim = new Word2VecSimilarity(techcrunch, venturebeat, vec);
		sim.calc();
		log.info(String.valueOf(sim.getDistance()));
		Word2VecSimilarity sim2 = new Word2VecSimilarity(techcrunch, other, vec);
		sim2.calc();
		log.info(String.valueOf(sim2.getDistance()));
		assertEquals(true,sim.getDistance() < sim2.getDistance());
	}
	
	


}
