package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda;

import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.LogisticRegressionMatrix;


public class LogisticTest extends DeepLearningTest {

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
	@Test
	public void testIris() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> iris = getIris();
		LogisticRegressionMatrix classifier = new LogisticRegressionMatrix.Builder().numberOfInputs(4).numberOfOutputs(3)
				.build();
		for(int i = 0; i < 1000; i++)
			classifier.train(iris.getFirst(), iris.getSecond(), 0.1);
		DoubleMatrix predicted = classifier.predict(iris.getFirst());		
		int numCorrect = 0;
		for(int i = 0; i < predicted.rows; i++) {
			DoubleMatrix predictedRow = predicted.getRow(i);
			DoubleMatrix yRow = iris.getSecond().getRow(i);
			int max = SimpleBlas.iamax(predictedRow);
			int actual = SimpleBlas.iamax(yRow);
			if(max == actual) 
				numCorrect++;
		}
		log.info("Number correct for logistic regression " + numCorrect);

	}


}
