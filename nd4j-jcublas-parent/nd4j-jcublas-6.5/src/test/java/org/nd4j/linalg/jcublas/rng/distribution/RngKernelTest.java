package org.nd4j.linalg.jcublas.rng.distribution;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RngKernelTest {
	
	@Before
	public void before() {
		
	}
	
	@Test
	public void testRandomUniform() {
		INDArray arr = Nd4j.rand(200,200,-0.4,0.4,Nd4j.getRandom());
		
		// Just test that we get the shape and there is no execptions thrown
		assertEquals(Nd4j.create(200,200).shape()[0],arr.shape()[0]);
    	assertEquals(Nd4j.create(200,200).shape()[1],arr.shape()[1]);
	}
	
	@Test
	public void testRandomUniformDouble() {
		
	}

}
