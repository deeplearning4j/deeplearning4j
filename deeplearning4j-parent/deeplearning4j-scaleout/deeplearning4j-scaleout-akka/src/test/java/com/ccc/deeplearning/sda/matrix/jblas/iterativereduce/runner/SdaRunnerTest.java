package com.ccc.deeplearning.sda.matrix.jblas.iterativereduce.runner;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang3.StringUtils;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.fetchers.MnistDataFetcher;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.iterativereduce.akka.runner.NetworkRunner;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;
import com.ccc.deeplearning.scaleout.conf.ExtraParamsBuilder;


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
		conf.put(PRE_TRAIN_EPOCHS, 100);
		conf.put(FINE_TUNE_EPOCHS, 100);

		conf.put(ROWS,train_X_arr.length);
		conf.put(LAYER_SIZES, hidden_layer_sizes_arr);
		conf.put(LEARNING_RATE, 0.1);
		conf.put(N_IN, n_ins);
		conf.put(SEED, 1);
		conf.put(LAYER_SIZES, StringUtils.join(hidden_layer_sizes_arr,","));
		conf.put(CORRUPTION_LEVEL, 0.3);
		conf.put(SPLIT, 1);
		conf.put(OUT, 2);
		conf.put(CLASS, "com.ccc.deeplearning.sda.StackedDenoisingAutoEncoder");
		conf.put(PARAMS, new ExtraParamsBuilder().algorithm(PARAM_SDA).corruptionlevel(0.3).finetuneEpochs(finetune_epochs)
				.finetuneLearningRate(finetune_lr).learningRate(pretrain_lr).epochs(10).build());
				
		
		
		
		runner.setup(conf);
		
		
	}

	@Test
	public void testOutput() {
		BaseMultiLayerNetwork m = runner.train(this.train_X_matrix,this.train_Y_matrix);
		log.info(m.logLayer.W.toString());
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
		conf.put(SPLIT, String.valueOf(600));
		conf.put(CLASS, "com.ccc.deeplearning.sda.StackedDenoisingAutoEncoder");
		conf.put(LAYER_SIZES, Arrays.toString(hidden_layer_sizes_arr).replace("[","").replace("]","").replace(" ",""));

		conf.put(PARAMS, new ExtraParamsBuilder().algorithm(PARAM_SDA).corruptionlevel(0.5).finetuneEpochs(finetune_epochs)
				.finetuneLearningRate(finetune_lr).learningRate(pretrain_lr).epochs(pretraining_epochs).build());
			
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
