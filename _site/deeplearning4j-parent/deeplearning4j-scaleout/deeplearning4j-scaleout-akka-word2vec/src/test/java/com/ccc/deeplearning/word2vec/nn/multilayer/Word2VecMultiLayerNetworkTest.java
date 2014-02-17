package com.ccc.deeplearning.word2vec.nn.multilayer;


import java.io.BufferedInputStream;
import java.io.File;
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

import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.nn.activation.HardTanh;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataSetIterator;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataSetIteratorImpl;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.Windows;
import com.ccc.deeplearning.word2vec.util.WordConverter;

public class Word2VecMultiLayerNetworkTest {

	private Word2VecMultiLayerNetwork network;
	private static Logger log = LoggerFactory.getLogger(Word2VecMultiLayerNetworkTest.class);
	private List<String> labels = Arrays.asList("NONE","ADDRESS");
	private Word2Vec vec;
	@Before
	public void setup() throws Exception {

		if(network == null) {
			vec = Word2VecLoader.loadModel(new ClassPathResource("/word2vec-address.bin").getFile());
			int size = vec.getWindow() * vec.getLayerSize();
			network = new Word2VecMultiLayerNetwork.Builder().withWord2Vec(vec).withLabels(labels)
					.numberOfInputs(size)
					.numberOfOutPuts(2).hiddenLayerSizes(new int[]{100,100,100}).useRegularization(false)
					.withActivation(new HardTanh()).build();
		}

	}




	@Test
	public void testMultiLayer() throws IOException {
		Word2VecDataSetIterator iter = new Word2VecDataSetIteratorImpl(new ClassPathResource("/deeplearning/").getFile().getAbsolutePath(), labels, 10, vec);

		network.pretrain(iter, 1, 0.01, 1000);
		network.finetune(0.01, 1000, iter, labels);
		Evaluation eval = new Evaluation();
		iter.reset();
		while(iter.hasNext()) {
			List<Window> next = iter.next();
			DoubleMatrix actual = WordConverter.toLabelMatrix(labels, next);
			DoubleMatrix prediction = network.predict(next);
			eval.eval(actual, prediction);
		}
		log.info(eval.stats());
	}

	
}
