package org.deeplearning4j.nn.learning;

import static org.junit.Assert.*;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.distributions.Distributions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AdaGradTest {

	private static Logger log = LoggerFactory.getLogger(AdaGradTest.class);
	
	
	@Test
	public void testAdaGrad1() {
		int rows = 1;
		int cols = 1;
		
		
		AdaGrad grad = new AdaGrad(rows,cols,1e-3);
		INDArray W = Nd4j.ones(rows,cols);
	    assertEquals(1e-1,grad.getLearningRates(W).get(0),1e-1);

		

	}
	
	@Test
	public void testAdaGrad() {
		int rows = 10;
		int cols = 2;
		
		/*
		Project for tomorrow:

         BaseElementWiseOp is having issues with the reshape (which produces inconsistent results) the test case for this  was adagrad
		 */
		AdaGrad grad = new AdaGrad(rows,cols,0.1);
		INDArray W = Nd4j.zeros(rows, cols);
		RealDistribution dist = Distributions.normal(new MersenneTwister(123),1);
		for(int i = 0; i < W.rows(); i++)
			W.putRow(i,Nd4j.create(dist.sample(W.columns())));
		
		for(int i = 0; i < 5; i++) {
			String learningRates = String.valueOf("\nAdagrad\n " + grad.getLearningRates(W)).replaceAll(";","\n");
			log.info(learningRates);
			W.addi(Nd4j.randn(rows, cols));
		}

	}
}
