package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.runner;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang3.StringUtils;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.deeplearning.eval.Evaluation;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor.ActorNetworkRunner;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.DeepLearningConfigurable;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.ExtraParamsBuilder;

public class AkkaSdaRunnerTest extends DeepLearningTest implements DeepLearningConfigurable {
	private static Logger log = LoggerFactory.getLogger(SdaRunnerTest.class);
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




		runner = new ActorNetworkRunner();
		conf.put(PRE_TRAIN_EPOCHS, 10);
		conf.put(FINE_TUNE_EPOCHS, 10);

		conf.put(ROWS,train_X_arr.length);
		conf.put(LAYER_SIZES, hidden_layer_sizes_arr);
		conf.put(LEARNING_RATE, 0.1);
		conf.put(N_IN, n_ins);
		conf.put(SEED, 1);
		conf.put(LAYER_SIZES, StringUtils.join(hidden_layer_sizes_arr,","));
		conf.put(CORRUPTION_LEVEL, 0.3);
		conf.put(SPLIT, 1);
		conf.put(OUT, 2);
		conf.put(CLASS, "com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix");
		conf.put(PARAMS, new ExtraParamsBuilder().algorithm(PARAM_SDA).corruptionlevel(0.3).finetuneEpochs(finetune_epochs)
				.finetuneLearningRate(finetune_lr).learningRate(pretrain_lr).epochs(10).build());
				
		
		
		
		runner.setup(conf);
		
		
	}

	@Test
	public void testOutput() {
		runner.train(this.train_X_matrix,this.train_Y_matrix);
		BaseMultiLayerNetwork m = runner.getResult().get();
		log.info(m.logLayer.W.toString());
		runner.shutdown();
	}

	@Test
	public void testMnist() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> mnist = this.getMnistExampleBatch(1);
		runner = new ActorNetworkRunner();
		conf.put(CLASS, "com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix");
		conf.put(LAYER_SIZES, Arrays.toString(hidden_layer_sizes_arr).replace("[","").replace("]","").replace(" ",""));
		conf.put(N_IN, String.valueOf(mnist.getFirst().columns));
		conf.put(OUT, String.valueOf(mnist.getSecond().columns));
		conf.put(PARAMS, new ExtraParamsBuilder().algorithm(PARAM_SDA).corruptionlevel(0.5).finetuneEpochs(finetune_epochs)
				.finetuneLearningRate(finetune_lr).learningRate(pretrain_lr).epochs(pretraining_epochs).build());
			
        runner.setup(conf);
    	runner.train(mnist.getFirst(), mnist.getSecond());

      
        
        
        BaseMultiLayerNetwork trained = runner.getResult().get();
        Evaluation eval = new Evaluation();
        DoubleMatrix predicted = trained.predict(mnist.getFirst());
        eval.eval(mnist.getSecond(), predicted);
        log.info(eval.stats());
        runner.shutdown();
	}



	@Override
	public void setup(Conf conf) {

	}

}

