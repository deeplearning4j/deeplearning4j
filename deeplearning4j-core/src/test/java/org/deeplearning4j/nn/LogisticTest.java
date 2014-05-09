package org.deeplearning4j.nn;


import java.io.IOException;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.eval.Evaluation;
import org.jblas.DoubleMatrix;
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
	public void testIris() throws IOException {
		IrisDataFetcher fetcher = new IrisDataFetcher();
		fetcher.fetch(110);

		DataSet iris = fetcher.next();
		OutputLayer classifier = new OutputLayer.Builder().numberOfInputs(4).numberOfOutputs(3)
				.build();
		iris.normalizeZeroMeanZeroUnitVariance();

		classifier.trainTillConvergence(iris.getFirst(), iris.getSecond(),1e-1, 40000);

		DoubleMatrix predicted = classifier.output(iris.getFirst());


		Evaluation eval = new Evaluation();
		eval.eval(iris.getSecond(), predicted);

		log.info(eval.stats());

	}
	
}
