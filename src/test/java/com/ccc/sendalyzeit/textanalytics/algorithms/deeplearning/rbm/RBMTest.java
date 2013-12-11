package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
		
		RBM r = new RBM(d, 6, 2, null, null, null, g);
		
		
		for(int i = 0; i < 1000; i++)
			r.contrastiveDivergence(0.1, 1, null);
		
		DoubleMatrix v = new DoubleMatrix(new double[][]{{1, 1, 0, 0, 0, 0},
                     {0, 0, 0, 1, 1, 0}});	
		
		
		log.info(r.reconstruct(v).toString());
		
	}

}
