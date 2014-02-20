package com.ccc.deeplearning.word2vec.sentenceiterator;
import static org.junit.Assert.*;
import java.io.File;

import org.deeplearning4j.word2vec.sentenceiterator.FileSentenceIterator;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

public class FileSentenceIteratorTest {

	@Test
	public void testIter() throws Exception {
		File f = new ClassPathResource("/itertest").getFile();

		FileSentenceIterator iter = new FileSentenceIterator(f);
		for(int i = 0; i < 9; i++) {
			String sentence = iter.nextSentence();
			assertEquals("line" + i,sentence);
		}

		assertEquals(false,iter.hasNext());

	}

}
