package org.nd4j.linalg.jcublas.rng.distribution;

import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * RngKernelTest
 * 
 * @author bam4d
 *
 */
public class RngKernelTest {
	
	double max = 0.4;
	double min = -max;

	@Test
	public void testUniformSample() {
		Nd4j.dtype = DataBuffer.FLOAT;
		for(int x = 0; x<100; x++) {
			INDArray arr = Nd4j.rand(200,200,min,max,Nd4j.getRandom());
			
			// Assert the values are within range
			assertMaxMin(arr);
		}
	}
	
	@Test
	public void testUniformSampleDouble() {
		Nd4j.dtype = DataBuffer.DOUBLE;
		for(int x = 0; x<100; x++) {
			
			INDArray arr = Nd4j.rand(200,200,min,max,Nd4j.getRandom());
			
			// Assert the values are within range
			assertMaxMin(arr);
		}
	}
	
	private void assertMaxMin(INDArray arr) {
		
		boolean nonZero = false;
		for(Double d: arr.data().asDouble()) {
			assertTrue("Returned value is above the maximum", max > d);
			assertTrue("Returned value is below the minimum", min < d);
			
			if(d != 0.0) {
				nonZero = true;
			}
		}
		
		assertTrue("The entire array is zeros", nonZero);
		
	}
}
