package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda;

import java.io.IOException;
import java.util.Random;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.IrisUtils;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.DenoisingAutoEncoderMatrix;


public class DenoisingTests {

	private static Logger log = LoggerFactory.getLogger(DenoisingTests.class);



	// training data
	double[][] train_X_arr;
	DoubleMatrix train_X_matrix;
	// test data
	double[][] test_X_arr;

	// test data
	DoubleMatrix test_X_matrix;
	@Before
	public void init() {
		org.jblas.util.Random.seed(1234);
		train_X_arr = new double[][] {
				{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
				{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
				{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0},
				{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0},
				{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0}
		};

		test_X_arr = new double[][] {
				{1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0},
				{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0}};




		train_X_matrix = new DoubleMatrix(train_X_arr);




		test_X_matrix = new DoubleMatrix(test_X_arr);
	}



	@Test
	public void testOther() {
		MersenneTwister rand = new MersenneTwister(123);
		for(int j = 0; j < 5; j++) {
			DenoisingAutoEncoderMatrix da = new DenoisingAutoEncoderMatrix(train_X_matrix, 20, 5, null, null, null, rand);
			double lr = 0.1;
			for(int i = 0; i < 50; i++) {
					da.train(train_X_matrix, lr, 0.3);
					lr *= 0.95;
			}
			log.info(da.reconstruct(test_X_matrix).toString());
		}
		
	}


	@Test
	@Ignore
	public void testIris() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> pair = IrisUtils.loadIris();
		DoubleMatrix input = pair.getFirst();
		DoubleMatrix y = pair.getSecond();

		DenoisingAutoEncoderMatrix da = new DenoisingAutoEncoderMatrix(input, input.columns, 1000, null, null, null, null);
		for(int i = 0; i < input.rows; i++) {
			da.train(input.getRow(i), 0.1, 0.9);
		}
		log.info(da.W.transpose().toString());
		double p = 1 - 0.3;

		for(int i = 0; i < input.rows; i++) {
			DoubleMatrix tilde_x = da.get_corrupted_input(input.getRow(i), p);
			DoubleMatrix y2 = da.get_hidden_values(tilde_x);
			DoubleMatrix z = da.get_reconstructed_input(y2);		
			log.info(z.toString());
		}
	}

}
