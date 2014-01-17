package com.ccc.deeplearning.rbm.matrix.jblas;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.jblas.DoubleMatrix;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.fetchers.MnistDataFetcher;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.nn.NeuralNetwork;
import com.ccc.deeplearning.rbm.RBM;

public class RBMTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(RBMTest.class);


	@Test
	public void testBasic() {
		double[][] data = new double[][]
				{
				{1,1,1,0,0,0},
				{1,0,1,0,0,0},
				{1,1,1,0,0,0},
				{0,0,1,1,1,0},
				{0,0,1,1,0,0},
				{0,0,1,1,1,0},
				{0,0,1,1,1,0}
			};

		DoubleMatrix d = new DoubleMatrix(data);
		RandomGenerator g = new MersenneTwister(123);

		RBM r = new RBM.Builder().numberOfVisible(6).numHidden(2).withRandom(g).build();

		

		for(int i = 0; i < 10; i++) {
			r.contrastiveDivergence(0.1, 1, d);
			log.info("Cross entropy " + r.getReConstructionCrossEntropy());
		}
		DoubleMatrix v = new DoubleMatrix(new double[][]
				{{1, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 1, 0}});	

		log.info(r.reconstruct(v).toString());

		NeuralNetwork r2 = r.clone();
		assertEquals(r2.getnVisible(),r.nVisible);
		assertEquals(r2.getnHidden(),r.nHidden);
		assertEquals(r2.getW(),r.W);
		assertEquals(r2.gethBias(),r.hBias);
		assertEquals(r2.getvBias(),r.vBias);
		for(int i = 0; i < 10; i++) {
			r2.trainTillConvergence(d, 0.1, new Object[]{1,0.1});
			log.info("Cross entropy " + r.getReConstructionCrossEntropy());
		}

	}


	@Test
	@Ignore
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(100);
		DataSet pair = fetcher.next();
		pair.roundToTheNearest(1200);
		int numVisible = pair.getFirst().columns;
		RandomGenerator g = new MersenneTwister(123);
		MnistDataSetIterator iter = new MnistDataSetIterator(100,1200);
		RBM r = new RBM.Builder().numberOfVisible(numVisible)
				.numHidden(1000).withRandom(g)
				.build();

		while(iter.hasNext()) {
			pair = iter.next();
			r.trainTillConvergence(0.1, 1, pair.getFirst());
			
			
		}



	}

}
