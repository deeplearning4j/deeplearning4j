package com.ccc.deeplearning.word2vec.ner;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.viterbi.Viterbi;


public class NERClassifierTest {
	private static Logger log = LoggerFactory.getLogger(NERClassifier.class);
	@Test
	public void testNERClassifier() throws Exception {
		Word2Vec vec = Word2VecLoader.loadModel(new ClassPathResource("/word2vec-address.bin").getFile());
		Viterbi v = Viterbi.load(new ClassPathResource("/viterbi-addresses-model.bin").getFile().getAbsolutePath());
		BaseMultiLayerNetwork dbn = BaseMultiLayerNetwork.loadFromFile(new BufferedInputStream(new ClassPathResource("/nn-model.bin").getInputStream()));
		NERClassifier classifier = new NERClassifier((Word2VecMultiLayerNetwork) dbn,vec,v,Arrays.asList("NONE","ADDRESS"));
		List<String> classified = classifier.predict("4820 1st street  P.O Box 94540952 mZcpm  Fatehpur marshall Islands  mBysky ls'm  mLfm 33397-5443");
		log.info(classified.toString());
	}

}
