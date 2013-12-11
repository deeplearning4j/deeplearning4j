package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.LogisticRegressionMatrix;


public class LogisticTest {
	double[][] x = new double[][] {
			{1,1,1,0,0,0},
			{1,0,1,0,0,0},
			{1,1,1,0,0,0},
			{0,0,1,1,1,0},
			{0,0,1,1,0,0},
			{0,0,1,1,1,0}};
	double[][] y = new double[][] {{1, 0},
			{1, 0},
			{1, 0},
			{0, 1},
			{0, 1},
			{0, 1}};
	DoubleMatrix xMatrix = new DoubleMatrix(x);
	DoubleMatrix yMatrix = new DoubleMatrix(y);
	private static Logger log = LoggerFactory.getLogger(LogisticTest.class);
	double[][] xTest = new double[][] {
			{1, 1, 0, 0, 0, 0},
			{0, 0, 0, 1, 1, 0},
			{1, 1, 1, 1, 1, 0}
	};
	DoubleMatrix xTestMatrix = new DoubleMatrix(xTest);

	@Test
	public void testLogistic() {
		LogisticRegressionMatrix log2 = new LogisticRegressionMatrix(xTestMatrix,x[0].length,2);
		double learningRate = 0.01;
		for(int i = 0; i < 1000; i++) {
			log2.train(xMatrix, yMatrix, learningRate);
			learningRate *= 0.95;
		}
		log.info(log2.predict(xTestMatrix).toString());


	}


}
