package org.deeplearning4j.sda.matrix.jblas.iterativereduce.runner;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.base.DeepLearningTest;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterativereduce.akka.runner.NetworkRunner;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.conf.ExtraParamsBuilder;
import org.deeplearning4j.sda.StackedDenoisingAutoEncoder;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class SdaRunnerTest extends DeepLearningTest implements DeepLearningConfigurable {
	private static Logger log = LoggerFactory.getLogger(SdaRunnerTest.class);
	private NetworkRunner runner;
	private Conf conf;
	Integer[] hidden_layer_sizes_arr = {300, 300,300};
	double pretrain_lr = 0.1;
	double corruption_level = 0.3;
	int n_layers = hidden_layer_sizes_arr.length;
	Random rng = new Random(123);

	int pretraining_epochs = 50;
	double finetune_lr = 0.1;
	int finetune_epochs = 50;
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




		runner = new NetworkRunner();
		conf.setPretrainEpochs(100);
		conf.setFinetuneEpochs(100);

		conf.setnIn(n_ins);
		conf.setLayerSizes(hidden_layer_sizes_arr);
		conf.setSplit(1);
		conf.setnOut(2);
		conf.setMultiLayerClazz(StackedDenoisingAutoEncoder.class);
			
		
		
		
		runner.setup(conf);
		
		
	}

	@Test
	public void testOutput() {
		BaseMultiLayerNetwork m = runner.train(this.train_X_matrix,this.train_Y_matrix);
		log.info(m.getLogLayer().getW().toString());
	}

	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(600);
		Pair<DoubleMatrix,DoubleMatrix> first = fetcher.next();
		int numIns = first.getFirst().columns;
		int numLabels = first.getSecond().columns;
		int[] layerSizes = new int[3];
		Arrays.fill(layerSizes,1000);
		runner = new NetworkRunner();
		conf.setSplit(600);
		conf.setPretrainEpochs(100);
		conf.setFinetuneEpochs(100);

		conf.setnIn(numIns);
		conf.setLayerSizes(hidden_layer_sizes_arr);
		conf.setnOut(numLabels);
		conf.setMultiLayerClazz(StackedDenoisingAutoEncoder.class);
			
			
        runner.setup(conf);
    	runner.train(first.getFirst(), first.getSecond());

      
        
        
        BaseMultiLayerNetwork trained = runner.result();
        Evaluation eval = new Evaluation();
        DoubleMatrix predicted = trained.predict(first.getFirst());
        eval.eval(first.getSecond(), predicted);
        log.info(eval.stats());
	}



	@Override
	public void setup(Conf conf) {

	}

}
