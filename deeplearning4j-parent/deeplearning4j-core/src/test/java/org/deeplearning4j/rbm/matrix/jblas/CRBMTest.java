package org.deeplearning4j.rbm.matrix.jblas;


import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.deeplearning4j.base.DeepLearningTest;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.rbm.CRBM;
import org.jblas.DoubleMatrix;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class CRBMTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(CRBMTest.class);
	@Test
	public void testBasic() {
		DoubleMatrix input = new DoubleMatrix(new double[][]{
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.5, 0.3,  0.5, 0.,  0.,  0.},
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.,  0.,  0.5, 0.3, 0.5, 0.},
				{0.,  0.,  0.5, 0.4, 0.5, 0.},
				{0.,  0.,  0.5, 0.5, 0.5, 0.}});

		RandomGenerator g = new MersenneTwister(123);

		CRBM r = new CRBM.Builder().numberOfVisible(input.getRow(0).columns).numHidden(10).withRandom(g).build();



		for(int i = 0; i < 1000; i++) {
			r.contrastiveDivergence(0.1, 1, input);
			log.info("Entropy " + r.getReConstructionCrossEntropy());
		}

		DoubleMatrix test = new DoubleMatrix(new double[][]
				{{0.5, 0.5, 0., 0., 0., 0.},
				{0., 0., 0., 0.5, 0.5, 0.}});



	}

	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(1000);
		DataSet pair = fetcher.next();
		pair.roundToTheNearest(100);
		int numVisible = pair.getFirst().columns;
		RandomGenerator g = new MersenneTwister(123);

		CRBM r = new CRBM.Builder().numberOfVisible(numVisible)
				.numHidden(1000).withRandom(g)
				.build();
		DoubleMatrix input = pair.getFirst();

		for(int i = 0; i < 1000; i++) {
			r.contrastiveDivergence(0.1, 1, input);
			log.info("Entropy " + r.getReConstructionCrossEntropy());
		}


	}



}
