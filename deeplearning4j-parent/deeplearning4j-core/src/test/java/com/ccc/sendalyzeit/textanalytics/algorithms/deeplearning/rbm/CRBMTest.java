package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm;


import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers.MnistDataFetcher;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas.CRBM;

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

		log.info(r.reconstruct(test).toString());


	}

	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(10);
		Pair<DoubleMatrix,DoubleMatrix> pair = fetcher.next();
		int numVisible = pair.getFirst().columns;
		RandomGenerator g = new MersenneTwister(123);

		CRBM r = new CRBM.Builder().numberOfVisible(numVisible)
				.numHidden(100).withRandom(g)
				.build();
		DoubleMatrix input = pair.getFirst();

		for(int i = 0; i < 1000; i++) {
			r.contrastiveDivergence(0.1, 1, input);
			log.info("Entropy " + r.getReConstructionCrossEntropy());
		}


	}



}
