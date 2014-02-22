package org.deeplearning4j.rbm;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.deeplearning4j.base.DeepLearningTest;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.rbm.RBM;
import org.jblas.DoubleMatrix;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


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

		
		r.trainTillConvergence(d, 0.01, new Object[]{1,0.01,1000});
		
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
			r2.trainTillConvergence(d, 0.1, new Object[]{1,0.01,1000});
			log.info("Cross entropy " + r.getReConstructionCrossEntropy());
		}

	}


}
