package org.deeplearning4j.sda.matrix.jblas.iterativereduce.runner;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.base.DeepLearningTest;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.matrix.jblas.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.conf.ExtraParamsBuilder;
import org.deeplearning4j.sda.StackedDenoisingAutoEncoder;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class AkkaSdaRunnerTest extends DeepLearningTest implements DeepLearningConfigurable {
	private static Logger log = LoggerFactory.getLogger(AkkaSdaRunnerTest.class);
	private ActorNetworkRunner runner;
	private Conf conf;
	Integer[] hidden_layer_sizes_arr = {300, 300,300};
	double pretrain_lr = 0.1;
	double corruption_level = 0.3;
	int n_layers = hidden_layer_sizes_arr.length;
	Random rng = new Random(123);

	int pretraining_epochs = 1000;
	double finetune_lr = 0.1;
	int finetune_epochs = 500;
	double[][] train_X_arr;
	DoubleMatrix train_X_matrix;
	double[][] train_Y_arr;
	int n_ins = 28;

	DoubleMatrix train_Y_matrix;
	@Before
	public void setup() {
		conf = new Conf();

		train_X_arr = new double[][] {
				{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1}
		};


		train_Y_arr = new double[][] {
				{1, 0},
				{1, 0},
				{1, 0},
				{1, 0},
				{1, 0},
				{0, 1},
				{0, 1},
				{0, 1},
				{0, 1},
				{0, 1}
		};


		train_X_matrix = new DoubleMatrix(train_X_arr);
		train_Y_matrix = new DoubleMatrix(train_Y_arr);


		conf.setPretrainEpochs(100);
		conf.setFinetuneEpochs(100);
		conf.setLayerSizes(hidden_layer_sizes_arr);
		conf.setnIn(n_ins);
		conf.setSplit(1);
		conf.setFinetuneEpochs(finetune_epochs);
		conf.setFinetuneLearningRate(finetune_lr);
		conf.setnOut(2);
		conf.setMultiLayerClazz(StackedDenoisingAutoEncoder.class);






	}

	@Test
	@Ignore
	public void testOutput() {
		runner.train(this.train_X_matrix,this.train_Y_matrix);
		BaseMultiLayerNetwork m = runner.getResult().get();
		log.info(m.getLogLayer().getW().toString());
	}

	@Test
	public void testMnist() throws Exception {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(60,600);
		runner = new ActorNetworkRunner("master",fetcher);

		conf.setPretrainEpochs(100);
		conf.setSplit(10);
		conf.setFinetuneEpochs(100);
		conf.setLayerSizes(hidden_layer_sizes_arr);
		conf.setnIn(784);
		conf.setSplit(1);
		conf.setFinetuneEpochs(finetune_epochs);
		conf.setFinetuneLearningRate(finetune_lr);
		conf.setnOut(10);
		conf.setMultiLayerClazz(StackedDenoisingAutoEncoder.class);
		conf.setDeepLearningParams(Conf.getDefaultDenoisingAutoEncoderParams());
		
		runner.setup(conf);
		Thread.sleep(10000);


		ActorNetworkRunner worker = new ActorNetworkRunner("worker",runner.getMasterAddress().toString());
		worker.setup(conf);

		DataSet first = fetcher.next();

		runner.train(first.getFirst(), first.getSecond());  
		
		/*
		BaseMultiLayerNetwork trained = runner.getResult().get();
		Evaluation eval = new Evaluation();
		DoubleMatrix predicted = trained.predict(first.getFirst());
		eval.eval(first.getSecond(), predicted);
		log.info(eval.stats());*/
	}



	@Override
	public void setup(Conf conf) {

	}

}

