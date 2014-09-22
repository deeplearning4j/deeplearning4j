package org.deeplearning4j.multicore;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.test.TestDataSetIterator;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.scaleout.conf.Conf;

public class MnistMultiThreadedExample {


	public static void main(String[] args) throws Exception {
		//5 batches of 100: 20 each
		MnistDataSetIterator mnist = new MnistDataSetIterator(20, 100);
		TestDataSetIterator iter = new TestDataSetIterator(mnist);
		ActorNetworkRunner runner = new ActorNetworkRunner(iter);


		NeuralNetConfiguration conf2 = new NeuralNetConfiguration.Builder()
		.nIn(784).nOut(10).build();

		Conf conf = new Conf();
		conf.setConf(conf2);
		conf.getConf().setFinetuneEpochs(1000);
		conf.getConf().setPretrainLearningRate(0.0001f);
		conf.setLayerSizes(new int[]{500,250,100});
		conf.setMultiLayerClazz(DBN.class);
		conf.getConf().setnOut(10);
		conf.getConf().setFinetuneLearningRate(0.0001f);
		conf.getConf().setnIn(784);
		conf.getConf().setL2(0.001f);
		conf.getConf().setMomentum(0);
		conf.setSplit(10);
		//conf.setRenderWeightEpochs(100);
		conf.getConf().setUseRegularization(false);
		conf.setDeepLearningParams(new Object[]{1,0.0001,1000});
		runner.setup(conf);

		runner.train();

	}


}
