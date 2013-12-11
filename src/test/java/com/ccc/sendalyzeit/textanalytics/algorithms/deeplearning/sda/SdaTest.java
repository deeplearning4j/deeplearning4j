package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda;

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

import com.ccc.sendalyzeit.deeplearning.berkeley.Counter;
import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.IrisUtils;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix;

public class SdaTest {

	double pretrain_lr = 0.1;
	double corruption_level = 0.3;
	int pretraining_epochs = 1000;
	double finetune_lr = 0.1;
	int finetune_epochs = 500;
	int test_N = 4;
	RandomGenerator rng = new JDKRandomGenerator();
	private static Logger log = LoggerFactory.getLogger(SdaTest.class);
	int train_N = 10;
	int n_ins = 20;
	int n_outs = 2;
	int[] hidden_layer_sizes_arr = {15, 15};
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


		test_Y_arr = new double[test_N][n_outs];


		test_X_matrix = new DoubleMatrix(test_X_arr);
		test_Y_matrix = new DoubleMatrix(test_Y_arr);
	}





	@Test
	@Ignore
	public void testOutput() {
		SdAMatrix sda = sdamatrix();
		sda.pretrain( pretrain_lr, corruption_level, pretraining_epochs);
		// finetune
		sda.finetune(finetune_lr, finetune_epochs);
		log.info("OUTPUT TEST");
		log.info(sda.predict(test_X_matrix).toString());

	}


	public SdAMatrix sdamatrix() {
		// construct SdA
		SdAMatrix sda = new SdAMatrix( n_ins, hidden_layer_sizes_arr, n_outs, n_layers, rng,train_X_matrix,train_Y_matrix);
		return sda;
	}

	



	@Test
	public void testIris() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> pair = IrisUtils.loadIris(1000);
		DoubleMatrix input = pair.getFirst();
		DoubleMatrix y = pair.getSecond();
	
		int[] hidden_layer_sizes_arr = {300, 300};

		

		SdAMatrix sdaMatrix = new SdAMatrix(input.columns,hidden_layer_sizes_arr,y.columns,n_layers,rng,input,train_Y_matrix);
		sdaMatrix.pretrain( pretrain_lr, 0.5, 1000);
		sdaMatrix.finetune(finetune_lr, 1000);

		DoubleMatrix output = sdaMatrix.predict(input);
		int numCorrect = 0;
		Counter<Integer> yIndices = new Counter<Integer>();
		Counter<Integer> guessIndices = new Counter<Integer>();
		for(int i = 0; i < y.rows; i++) {
			int yMax = SimpleBlas.iamax(y.getRow(i));
			int outputMax = SimpleBlas.iamax(output.getRow(i));
			yIndices.incrementCount(yMax,1.0);
			guessIndices.incrementCount(outputMax,1.0);
			if(yMax == outputMax) {
				numCorrect++;
			}
		}
		log.info("Y " + yIndices.toString());
		log.info("Guesses " + guessIndices.toString());
		log.info("Correct was " + numCorrect);
		
		//log.info("Prediction was " + sdaMatrix.predict(input).toString().replace(";","\n")  + " \n Y was " + y.toString().replace(";","\n"));

	}





}
