package com.ccc.deeplearning.word2vec.iterativereduce.actor.multilayer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.junit.Before;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;
import com.ccc.deeplearning.scaleout.conf.ExtraParamsBuilder;
import com.ccc.deeplearning.util.MatrixUtil;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.conf.Conf;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataSetIterator;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataSetIteratorImpl;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;

public class ActorRunnerTest implements DeepLearningConfigurable {

	private Conf conf;
	private ActorNetworkRunner runner;
	private Word2VecDataSetIterator iter;
	private Word2Vec vec;
	private List<String> labels;
	
	@Before
	public void setup() throws IOException {
		labels = Arrays.asList("NONE","ADDRESS");

		try {
			vec = Word2VecLoader.loadModel(new ClassPathResource("/word2vec-address.bin").getFile());
			vec.setSyn0(MatrixUtil.normalizeByColumnSums(vec.getSyn0()));
			//vec.saveAsCsv(new File("/home/agibsonccc/word2vec.arff"));
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
		
		this.iter = new Word2VecDataSetIteratorImpl("/home/agibsonccc/workspace/deeplearning4j-parent/deeplearning4j-scaleout/deeplearning4j-scaleout-akka-word2vec/src/test/resources/addresstraining", labels,200,vec);
		if(conf == null)
			setupConf();
		
		
		
		
		
	}
	
	private void setupConf() {
		conf = new Conf();
		int[] hiddenLayerSizes = { vec.getLayerSize() * 2,vec.getLayerSize() * 2, vec.getLayerSize()};
		int pretrainEpochs = 1000;
		int finetuneEpochs = 1000;
		long rngSeed = 123;
		double pretrainLearningRate = 0.001;
		double corruptionLevel = 0.3;
		int split = 100;
		int k = 1;
		double finetuneLearningRate = 0.001;
		int numPasses = 3;
		String algorithm = "wordcdbn";
		
		conf.put(NUM_PASSES, String.valueOf(numPasses));
		conf.put(LAYER_SIZES, Arrays.toString(hiddenLayerSizes).replace("[","").replace("]","").replace(" ",""));
		conf.put(SPLIT,String.valueOf(10));
		conf.put(N_IN, String.valueOf(iter.inputColumns()));
		conf.put(OUT, String.valueOf(iter.totalOutcomes()));
		conf.put(PRE_TRAIN_EPOCHS, String.valueOf(pretrainEpochs));
		conf.put(FINE_TUNE_EPOCHS, String.valueOf(finetuneEpochs));
		conf.put(SEED, String.valueOf(rngSeed));
		conf.put(LEARNING_RATE,String.valueOf(pretrainLearningRate));

		conf.put(LAYER_SIZES, Arrays.toString(hiddenLayerSizes).replace("[","").replace("]","").replace(" ",""));
		conf.put(CORRUPTION_LEVEL,corruptionLevel);
		conf.put(SPLIT, String.valueOf(split));
		conf.put(PARAMS, new ExtraParamsBuilder().algorithm(algorithm).corruptionlevel(corruptionLevel).finetuneEpochs(finetuneEpochs)
				.k(k)
				.finetuneLearningRate(finetuneLearningRate).learningRate(pretrainLearningRate).epochs(pretrainEpochs).build());

		
		
		runner = new ActorNetworkRunner("master",iter,vec,labels);
		runner.setup(conf);
	}

	
	
	@Test
	public void testOnOneFile() throws Exception {
		runner.train();
		
		while(true) {
			Thread.sleep(10000);
		}
		
		
		
	}
	

	

	@Override
	public void setup(com.ccc.deeplearning.scaleout.conf.Conf conf) {
		
	}

}
