package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.jblas.DoubleMatrix;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.deeplearning.eval.Evaluation;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers.MnistDataFetcher;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas.RBM;

public class RBMTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(RBMTest.class);


	@Test
	public void testBasic() {
		double[][] data = new double[][]
				{{1,1,1,0,0,0},
				{1,0,1,0,0,0},
				{1,1,1,0,0,0},
				{0,0,1,1,1,0},
				{0,0,1,1,0,0},
				{0,0,1,1,1,0}};

		DoubleMatrix d = new DoubleMatrix(data);
		RandomGenerator g = new MersenneTwister(123);

		RBM r = new RBM.Builder().numberOfVisible(6).numHidden(2).withRandom(g).build();



		for(int i = 0; i < 1000; i++) {
			r.contrastiveDivergence(0.1, 1, d);
			log.info("Cross entropy " + r.getReConstructionCrossEntropy());
		}
		DoubleMatrix v = new DoubleMatrix(new double[][]
				{{1, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 1, 0}});	


		log.info(r.reconstruct(v).toString());

	}


	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(100);
		Pair<DoubleMatrix,DoubleMatrix> pair = fetcher.next();
		int numVisible = pair.getFirst().columns;
		RandomGenerator g = new MersenneTwister(123);

		RBM r = new RBM.Builder().numberOfVisible(numVisible)
				.numHidden(100).withRandom(g)
				.build();
		DoubleMatrix input = pair.getFirst();
		DoubleMatrix reconstructed = r.reconstruct(input);

		for(int i = 0; i < 3000; i++) {
			r.contrastiveDivergence(0.1, 1, input);
			log.info("Entropy " + r.getReConstructionCrossEntropy());
		}


	}

}
