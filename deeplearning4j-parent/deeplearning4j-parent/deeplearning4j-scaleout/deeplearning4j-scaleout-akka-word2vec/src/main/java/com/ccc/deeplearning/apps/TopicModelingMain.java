package com.ccc.deeplearning.apps;

import java.io.File;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.multilayer.ActorNetworkRunner;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.topicmodeling.TopicModelingDataSetIterator;

public class TopicModelingMain {
	private static Logger log = LoggerFactory.getLogger(TopicModelingMain.class);
	
	public static void main(String[] args) throws Exception {
		int numWords = Integer.parseInt(args[1]);
		int batchSize = Integer.parseInt(args[2]);
		int numOuts = Integer.parseInt(args[3]);
		int split = Integer.parseInt(args[4]);
		TopicModelingDataSetIterator iter = new TopicModelingDataSetIterator(new File(args[0]), numOuts, numWords,batchSize);
		int vocabSize = iter.inputColumns();
		log.info("Training with vocab size of " + vocabSize);
		
		Conf c = new Conf();
		c.setLayerSizes(new int[]{numWords / 4, numWords / 8,numOuts });
		c.setnIn(vocabSize);
		c.setnOut(iter.totalOutcomes());
		c.setFinetuneLearningRate(0.1);
		c.setPretrainLearningRate(0.1);
		c.setPretrainEpochs(1000);
		c.setFinetuneEpochs(1000);
		c.setNumPasses(1);
		c.setUseRegularization(false);
		c.setSplit(split);
		c.setDeepLearningParams( new Object[]{1,0.1,1000});
		c.setMultiLayerClazz(CDBN.class);
		
		ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
		runner.setup(c);
		runner.train();
		
		while(true)
			Thread.sleep(10000);

	}

}
