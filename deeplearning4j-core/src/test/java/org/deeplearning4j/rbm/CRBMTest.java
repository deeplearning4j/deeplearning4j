package org.deeplearning4j.rbm;


import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.base.DeepLearningTest;
import org.jblas.DoubleMatrix;
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

		CRBM r = new CRBM.Builder().renderWeights(200)
		.numberOfVisible(input.getRow(0).columns).withSparsity(0.01)
		.numHidden(4).withRandom(g).build();
		r.getW().muli(1000);
	
		r.trainTillConvergence(input, 0.0001, new Object[]{1,0.0001,1000});
		
		
		DoubleMatrix test = new DoubleMatrix(new double[][]
				{{0.5, 0.5, 0., 0., 0., 0.},
				{0., 0., 0., 0.5, 0.5, 0.}});


		log.info(r.reconstruct(test).toString());
		log.info("Cross entropy " + r.getReConstructionCrossEntropy());
	}

	


}
