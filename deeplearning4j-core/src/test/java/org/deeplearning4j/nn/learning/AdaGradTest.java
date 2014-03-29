package org.deeplearning4j.nn.learning;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.distributions.Distributions;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AdaGradTest {

	private static Logger log = LoggerFactory.getLogger(AdaGradTest.class);
	
	@Test
	public void testAdaGrad() {
		int rows = 10;
		int cols = 2;
		
		
		AdaGrad grad = new AdaGrad(rows,cols,0.1);
		DoubleMatrix W = DoubleMatrix.zeros(rows,cols);
		RealDistribution dist = Distributions.normal(new MersenneTwister(123));
		for(int i = 0; i < W.rows; i++) 
			W.putRow(i,new DoubleMatrix(dist.sample(W.columns)));
		
		for(int i = 0; i < 5; i++) {
			String learningRates = String.valueOf("\nAdagrad\n " + grad.getLearningRates(W)).replaceAll(";","\n");
			log.info(learningRates);
			W.addi(DoubleMatrix.randn(rows, cols));
		}

	}
}
