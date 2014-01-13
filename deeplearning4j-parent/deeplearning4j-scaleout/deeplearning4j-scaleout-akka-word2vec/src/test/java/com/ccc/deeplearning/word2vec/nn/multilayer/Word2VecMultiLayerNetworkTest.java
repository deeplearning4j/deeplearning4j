package com.ccc.deeplearning.word2vec.nn.multilayer;


import java.io.BufferedInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.nn.activation.HardTanh;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.Windows;

public class Word2VecMultiLayerNetworkTest {

	private Word2VecMultiLayerNetwork network;
	private static Logger log = LoggerFactory.getLogger(Word2VecMultiLayerNetworkTest.class);

	@Before
	public void setup() throws Exception {
	
		if(network == null) {
			network = (Word2VecMultiLayerNetwork) BaseMultiLayerNetwork.loadFromFile(new BufferedInputStream(new ClassPathResource("/nn-model.bin").getInputStream()));
		}

	}




	@Test
	public void testMultiLayer() throws IOException {
		@SuppressWarnings("unchecked")
		List<String> example = IOUtils.readLines(new ClassPathResource("/deeplearning/ADDRESS.split").getInputStream());		
		example = example.subList(0, 500);
		
		Evaluation eval = new Evaluation();

		/*for(int i = 0; i < example.size(); i++) {
			printPredictFor(example.get(i));
		}*/
		
		printPredictFor("sadcasdfoijasdfasdf.");
		printPredictFor(example.get(0));
	}
	
	private void printPredictFor(String example) {
		if(example.isEmpty()) {
			return;
		}
		log.info("Predicting " + example);
		DoubleMatrix m = network.predict(example);
		List<Window> windows = Windows.windows(example);
		StringBuffer print = new StringBuffer();

		for(int w = 0; w < windows.size(); w++) {
			print.append("Classification for " + windows.get(w).asTokens() + " was " + m.getRow(w) + "\n");
		}
		log.info(print.toString());
	}

}
