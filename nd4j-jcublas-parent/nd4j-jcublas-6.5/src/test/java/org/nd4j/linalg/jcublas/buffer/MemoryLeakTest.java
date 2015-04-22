package org.nd4j.linalg.jcublas.buffer;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MemoryLeakTest {
	
	@Test
	public void testAllocateAndRemove() throws InterruptedException {
		for(int x = 0; x<100; x++) {
			INDArray arr = Nd4j.create(30000,3000);
		}
		
		Thread.sleep(20000);
	}

}
