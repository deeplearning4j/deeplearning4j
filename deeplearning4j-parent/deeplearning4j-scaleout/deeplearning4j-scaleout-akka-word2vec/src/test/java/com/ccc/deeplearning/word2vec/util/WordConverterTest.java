package com.ccc.deeplearning.word2vec.util;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.loader.Word2VecLoader;
import org.deeplearning4j.word2vec.util.Window;
import org.deeplearning4j.word2vec.util.Windows;
import org.deeplearning4j.word2vec.util.WordConverter;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;


public class WordConverterTest {

	@Test
	public void testExampleConversion() throws IOException, Exception {
		List<String> labels = Arrays.asList("NONE","ADDRESS");
		List<String> sentences = Arrays.asList("I work at <ADDRESS> 4820 1st street  P.O Box 94540952 mZcpm  Fatehpur	  marshall Islands  mBysky ls'm  mLfm 33397-5443 </ADDRESS> that I am going to try.");
		Word2Vec vec = Word2VecLoader.loadModel(new ClassPathResource("/word2vec-address.bin").getFile());
		List<Window> windows = Windows.windows(sentences.get(0));
		
		WordConverter converter = new WordConverter(sentences,vec);
		DoubleMatrix inputs = converter.toInputMatrix();
		assertEquals(inputs.rows,windows.size());
		
		DoubleMatrix labelMatrix = converter.toLabelMatrix(labels);
		assertEquals(labels.size(),labelMatrix.columns);
		
	}

}
