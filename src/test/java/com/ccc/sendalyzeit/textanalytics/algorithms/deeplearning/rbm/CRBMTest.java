package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm;


import java.io.IOException;


import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas.CRBM;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas.RBM;

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

		CRBM r = new CRBM(input, 6, 2, null, null, null, g);



		for(int i = 0; i < 1000; i++)
			r.contrastiveDivergence(0.1, 1, null);
		
		DoubleMatrix test = new DoubleMatrix(new double[][]
				{{0.5, 0.5, 0., 0., 0., 0.},
                     {0., 0., 0., 0.5, 0.5, 0.}});
		
		log.info(r.reconstruct(test).toString());


	}
	
	@Test
	public void testIris() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> iris = this.getIris();

		RandomGenerator g = new MersenneTwister(123);

		CRBM r = new CRBM(iris.getFirst(), iris.getFirst().columns, 3, null, null, null, g);



		for(int i = 0; i < 1000; i++)
			r.contrastiveDivergence(0.1, 1, null);
		
		DoubleMatrix guess = r.reconstruct(iris.getFirst());
		DoubleMatrix y = iris.getSecond();
		
		log.info("GUESS");
		log.info(guess.toString());
		log.info("Y");
		
		log.info(y.toString());
		
		
	}

}
