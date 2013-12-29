package com.ccc.deeplearning.nn.matrix.jblas;


import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.fetchers.IrisDataFetcher;
import com.ccc.deeplearning.datasets.fetchers.MnistDataFetcher;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.nn.matrix.jblas.LogisticRegression;


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
	public void testMnist() throws IOException {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(1,60000);
		DataSet d = fetcher.next();


		int inputColumns = d.getFirst().columns;
		int outputs = d.numOutcomes();
		LogisticRegression l = new LogisticRegression.Builder()
		.numberOfInputs(inputColumns).numberOfOutputs(outputs).build();
		Evaluation e = new Evaluation();

		while(fetcher.hasNext()) {
			d = fetcher.next();
			for(int i = 0; i < 1000; i++) {
				l.train(d.getFirst(), d.getSecond(), 0.1);
			}

			log.info("Loss " + l.negativeLogLikelihood());

			DoubleMatrix predict = l.predict(d.getFirst());
			e.eval(d.getSecond(), predict);
			log.info(e.stats());

		}
	}

	@Test
	public void testLogistic() {
		LogisticRegression log2 = new LogisticRegression(xTestMatrix,x[0].length,2);
		double learningRate = 0.01;
		for(int i = 0; i < 1000; i++) {
			log2.train(xMatrix, yMatrix, learningRate);
			learningRate *= 0.95;
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
		for(int i = 0; i < 1000; i++)
			classifier.train(iris.getFirst(), iris.getSecond(), 0.1);

		fetcher.fetch(40);
		iris = fetcher.next();

		DoubleMatrix predicted = classifier.predict(iris.getFirst());		


		Evaluation eval = new Evaluation();
		eval.eval(iris.getSecond(), predicted);

		log.info(eval.stats());

	}


}
