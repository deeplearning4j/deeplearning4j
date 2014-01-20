package com.ccc.deeplearning.word2vec.iterator;

import static org.junit.Assert.*;
import static org.junit.Assume.*;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataFetcherTest {

	private static Logger log = LoggerFactory.getLogger(DataFetcherTest.class);
	@Test
	public void testIterateArticles() {
		Word2VecDataFetcher fetcher = new Word2VecDataFetcher("src/test/resources/articles",1);
		fetcher.fetch(10);
		assertEquals(10,fetcher.getBatch());
		assumeNotNull(fetcher.getVec());
		assumeTrue(fetcher.hasMore());
		assumeTrue(fetcher.getVec().getWordIndex().size() > 0);
		log.info("Examples " + fetcher.totalExamples());
		log.info("Number of words " + fetcher.getVec().getWordIndex().size());
		assertEquals(3663349,fetcher.totalExamples());
		assertEquals(878005,fetcher.getVec().getWordIndex().size());
	}

}
