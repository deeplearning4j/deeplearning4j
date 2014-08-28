package org.deeplearning4j.linalg.learning;


import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
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
		INDArray W = NDArrays.ones(rows,cols);
	
		log.info("Learning rates for 1 " + grad.getLearningRates(W));
		
		

	}
	
	@Test
	public void testAdaGrad() {
		int rows = 10;
		int cols = 2;
		
		
		AdaGrad grad = new AdaGrad(rows,cols,0.1);
		INDArray W = NDArrays.zeros(rows, cols);

	}
}
