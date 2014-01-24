package com.ccc.deeplearning.sda.jblas;
import static org.junit.Assert.*;


import java.io.IOException;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.IrisUtils;
import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.nn.activation.HardTanh;
import com.ccc.deeplearning.sda.StackedDenoisingAutoEncoder;

public class SdaTest {

	double pretrain_lr = 0.1;
	double corruption_level = 0.8;
	int pretraining_epochs = 1000;
	double finetune_lr = 0.1;
	int finetune_epochs = 500;
	int test_N = 4;
	RandomGenerator rng = new JDKRandomGenerator();
	private static Logger log = LoggerFactory.getLogger(SdaTest.class);
	int train_N = 10;
	int n_ins = 20;
	int n_outs = 2;
	int[] hidden_layer_sizes_arr = {15, 15,10};
	int n_layers = hidden_layer_sizes_arr.length;

	int seed = 123;
	// training data
	double[][] train_X_arr;
	DoubleMatrix train_X_matrix;
	double[][] train_Y_arr;
	DoubleMatrix train_Y_matrix;
	// test data
	double[][] test_X_arr;
	double[][] test_Y_arr;
	DoubleMatrix test_Y_matrix;
	// test data
	DoubleMatrix test_X_matrix;
	@Before
	public void init() {
		rng.setSeed(seed);
		train_X_arr = new double[][] {
				{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}
		};

		test_X_arr = new double[][] {
				{1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1},
				{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1}
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


		test_Y_arr = new double[][]{
				{0.0,1.0},
				{0.0,1.0},
				{1.0,0.0},
				{1.0,0.0}
		};


		test_X_matrix = new DoubleMatrix(test_X_arr);
		test_Y_matrix = new DoubleMatrix(test_Y_arr);
	}





	@Test
	public void testOutput() {
		StackedDenoisingAutoEncoder sda = new StackedDenoisingAutoEncoder.Builder()
		.withActivation(new HardTanh()).hiddenLayerSizes(hidden_layer_sizes_arr)
		.numberOfInputs(n_ins).numberOfOutPuts(n_outs).renderWeights(10)
		.useRegularization(true).withMomentum(0).withRng(rng).build();		
		sda.pretrain(train_X_matrix,pretrain_lr, corruption_level, pretraining_epochs);
		// finetune
		
		sda.finetune(this.train_Y_matrix,finetune_lr, finetune_epochs);
		log.info("OUTPUT TEST");
		DoubleMatrix predicted = sda.predict(train_X_matrix);
		
		Evaluation eval = new Evaluation();
		eval.eval(train_Y_matrix, predicted);
		log.info(eval.stats());
	

	}



	




}
