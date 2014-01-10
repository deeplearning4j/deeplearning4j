package com.ccc.deeplearning.word2vec.nn.multilayer;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.nn.activation.HardTanh;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;

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
			int[] hiddenLayerSizes = {inputs,inputs * 2};
			network = new Word2VecMultiLayerNetwork.Builder().withWord2Vec(vec)
					.withRng(new MersenneTwister(123)).numberOfInputs(inputs).withActivation(new HardTanh())
					.numberOfOutPuts(2).hiddenLayerSizes(hiddenLayerSizes).build();
		}

	}




	@Test
	public void testMultiLayer() throws IOException {
		@SuppressWarnings("unchecked")
		List<String> example = IOUtils.readLines(new ClassPathResource("/deeplearning/ADDRESS-small.split").getInputStream());		
		example = example.subList(0, 20);
		network.pretrain(example, 1, 0.01, 1000);
		network.finetune(0.01, 1000, example, Arrays.asList("NONE","ADDRESS"));
		for(int i = 0; i < example.size(); i++) {
			if(example.get(i).isEmpty()) {
				continue;
			}
			log.info("Predicting " + example.get(i));
			DoubleMatrix m = network.predict(example.get(i));
			log.info(m.toString());
		}

	}

}
