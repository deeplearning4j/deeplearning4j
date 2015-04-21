package jcublas.rng.distribution;

import static org.junit.Assert.assertEquals;

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
	

	@Test
	public void testUniformSample() {
		Nd4j.dtype = DataBuffer.FLOAT;
		for(int x = 0; x<100; x++) {
			INDArray arr = Nd4j.rand(200,200,-0.4,0.4,Nd4j.getRandom());
			
			// Just test that we get the shape and there is no execptions thrown
			assertEquals(Nd4j.create(200,200).shape()[0],arr.shape()[0]);
	    	assertEquals(Nd4j.create(200,200).shape()[1],arr.shape()[1]);
		}
	}
	
	@Test
	public void testUniformSampleDouble() {
		Nd4j.dtype = DataBuffer.DOUBLE;
		for(int x = 0; x<100; x++) {
			INDArray arr = Nd4j.rand(200,200,-0.4,0.4,Nd4j.getRandom());
			
			// Just test that we get the shape and there is no execptions thrown
			assertEquals(Nd4j.create(200,200).shape()[0],arr.shape()[0]);
	    	assertEquals(Nd4j.create(200,200).shape()[1],arr.shape()[1]);
		}
	}

}
