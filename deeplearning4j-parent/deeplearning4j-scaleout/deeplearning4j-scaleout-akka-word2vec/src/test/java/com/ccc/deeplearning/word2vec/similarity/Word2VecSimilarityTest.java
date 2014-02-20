package com.ccc.deeplearning.word2vec.similarity;

import static org.junit.Assert.*;


import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.iterator.Word2VecDataFetcher;
import org.deeplearning4j.word2vec.loader.Word2VecLoader;
import org.deeplearning4j.word2vec.similarity.Word2VecSimilarity;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;


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


		String techcrunch2 = FileUtils.readFileToString(new File("src/test/resources/articles/techcrunch/-2013-12-09-facebook-artificial-intelligence-lab-lecun-"));
		String venturebeat2 = FileUtils.readFileToString(new File("src/test/resources/articles/venturebeat/-2013-12-09-zuckerberg-to-talk-about-facebooks-artificial-intelligence-plans-"));


		Word2Vec vec = Word2VecLoader.loadModel(new ClassPathResource("/word2vecmodel-techblogs.bin").getFile());

	

		eval(vec,techcrunch,venturebeat,other);
		log.info("Next test");
		eval(vec,techcrunch2,venturebeat2,other);
		log.info("Next test 2");

		String techcrunchSnapChat = FileUtils.readFileToString(new File("src/test/resources/articles/techcrunch/-2013-12-31-hackers-claim-to-publish-list-of-4-6m-snapchat-usernames-and-numbers-"));
		String venturebeatSnapChat = FileUtils.readFileToString(new File("src/test/resources/articles/venturebeat/-2013-12-31-snapchat-cracked-4-6-million-usernames-and-phone-numbers-published-"));
		eval(vec,techcrunchSnapChat,venturebeatSnapChat,other);

		String tcJaja = FileUtils.readFileToString(new File("src/test/resources/articles/techcrunch/-2013-12-01-sold-to-telefonica-for-207m-in-2009-now-neglected-voip-provider-jajah-is-shutting-down-"));
		String vbJaja = FileUtils.readFileToString(new File("src/test/resources/articles/venturebeat/-2013-12-01-telecom-giant-telefonica-shutters-voip-provider-jajah-after-skype-and-others-drive-prices-to-bottom-"));
		eval(vec,tcJaja,vbJaja,other);
	}


	private void eval(Word2Vec vec,String good1,String good2,String bad) {
		Word2VecSimilarity sim2 = new Word2VecSimilarity(good1, good2, vec);
		sim2.calc();
		Word2VecSimilarity badDistance2 = new Word2VecSimilarity(good1, bad, vec);
		badDistance2.calc();
		log.info("Good " + sim2.getDistance() + " bad " + badDistance2.getDistance());
		assertEquals(true,sim2.getDistance() < badDistance2.getDistance());

	}



}
