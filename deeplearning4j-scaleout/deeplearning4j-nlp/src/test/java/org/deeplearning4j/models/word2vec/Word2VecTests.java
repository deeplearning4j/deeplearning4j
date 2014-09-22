package org.deeplearning4j.models.word2vec;

import static org.junit.Assert.*;

import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 8/31/14.
 */
public class Word2VecTests {

	private static Logger log = LoggerFactory.getLogger(Word2VecTests.class);




	@Test
	public void testWord2VecRunThrough() throws Exception {
		ClassPathResource resource = new ClassPathResource("/basic/word2vec.txt");
		File file = resource.getFile().getParentFile();
		DocumentIterator iter = new FileDocumentIterator(file);


		TokenizerFactory t = new DefaultTokenizerFactory();

		InMemoryLookupCache cache = new InMemoryLookupCache(100,true,0.025f);
		Word2Vec vec = new Word2Vec.Builder()
		.minWordFrequency(1).layerSize(100)
				.vocabCache(cache)
				.windowSize(5).iterate(iter).tokenizerFactory(t).build();
		vec.fit();
		double sim = vec.similarity("This","test.");
		assertTrue(vec.getCache().numWords() > 0);




	}

}
