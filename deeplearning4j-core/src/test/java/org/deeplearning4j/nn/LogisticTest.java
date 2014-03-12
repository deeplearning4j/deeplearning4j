package org.deeplearning4j.nn;


import java.io.IOException;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.LogisticRegression;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class LogisticTest {
	private static Logger log = LoggerFactory.getLogger(LogisticTest.class);

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

	double[][] xTest = new double[][] {
			{1, 1, 0, 0, 0, 0},
			{0, 0, 0, 1, 1, 0},
			{1, 1, 1, 1, 1, 0}
	};
	DoubleMatrix xTestMatrix = new DoubleMatrix(xTest);




	@Test
	public void testLogistic() {
		LogisticRegression log2 = new LogisticRegression(xTestMatrix,x[0].length,2);
		for(int i = 0; i < 1000; i++) {
			log2.train(xMatrix, yMatrix);
		}
		
		
		log.info(log2.predict(xTestMatrix).toString());


	}


	@Test
	public void testIris() throws IOException {
		IrisDataFetcher fetcher = new IrisDataFetcher();
		fetcher.fetch(110);

		DataSet iris = fetcher.next();
		LogisticRegression classifier = new LogisticRegression.Builder().numberOfInputs(4).numberOfOutputs(3)
				.build();
		classifier.trainTillConvergence(iris.getFirst(), iris.getSecond(), 10000);
		fetcher.fetch(40);
		iris = fetcher.next();

		DoubleMatrix predicted = classifier.predict(iris.getFirst());		


		Evaluation eval = new Evaluation();
		eval.eval(iris.getSecond(), predicted);

		log.info(eval.stats());

	}


	@Test
	public void testIrisCg() throws IOException {
		IrisDataFetcher fetcher = new IrisDataFetcher();
		fetcher.fetch(110);

		DataSet iris = fetcher.next();
		LogisticRegression classifier = new LogisticRegression.Builder().numberOfInputs(4).numberOfOutputs(3)
				.build();
		
		classifier.trainTillConvergence(iris.getFirst(), iris.getSecond(), 1000);
		
		fetcher.fetch(40);
		iris = fetcher.next();

		DoubleMatrix predicted = classifier.predict(iris.getFirst());		


		Evaluation eval = new Evaluation();
		eval.eval(iris.getSecond(), predicted);

		log.info(eval.stats());

	}
	
}
