package com.ccc.deeplearning.word2vec.nn.multilayer;


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
import com.ccc.deeplearning.nn.activation.HardTanh;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.Windows;

public class Word2VecMultiLayerNetworkTest {

	private Word2VecMultiLayerNetwork network;
	private Word2Vec vec;
	private static Logger log = LoggerFactory.getLogger(Word2VecMultiLayerNetworkTest.class);

	@Before
	public void setup() throws Exception {
		if(vec == null) {
			vec = Word2VecLoader.loadModel(new ClassPathResource("/word2vec-address.bin").getFile());
		}

		if(network == null) {
			int inputs = vec.getLayerSize() * vec.getWindow();
			int[] hiddenLayerSizes = {inputs * 4,inputs *2 ,inputs};
			network = new Word2VecMultiLayerNetwork.Builder().withWord2Vec(vec)
					.withRng(new MersenneTwister(123)).numberOfInputs(inputs).withActivation(new HardTanh())
					.numberOfOutPuts(2).hiddenLayerSizes(hiddenLayerSizes).build();
		}

	}




	@Test
	public void testMultiLayer() throws IOException {
		@SuppressWarnings("unchecked")
		List<String> example = IOUtils.readLines(new ClassPathResource("/deeplearning/ADDRESS.split").getInputStream());		
		example = example.subList(0, 500);
		network.pretrain(example, 1, 0.001,500);
		network.finetune(0.001, 1000, example, Arrays.asList("NONE","ADDRESS"));
		Evaluation eval = new Evaluation();

		for(int i = 0; i < example.size(); i++) {
			if(example.get(i).isEmpty()) {
				continue;
			}
			log.info("Predicting " + example.get(i));
			DoubleMatrix m = network.predict(example.get(i));
			List<Window> windows = Windows.windows(example.get(i));
			StringBuffer print = new StringBuffer();

			for(int w = 0; w < windows.size(); w++) {
				print.append("Classification for " + windows.get(w).asTokens() + " was " + m.getRow(w) + "\n");
			}
			log.info(print.toString());
		}

	}

}
