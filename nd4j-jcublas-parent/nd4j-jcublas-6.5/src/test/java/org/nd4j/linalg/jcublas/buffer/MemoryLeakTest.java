package org.nd4j.linalg.jcublas.buffer;

import java.io.IOException;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MemoryLeakTest {
	
	@Test
	public void testAllocateAndRemove() throws Exception {
		for(int x = 0; x<1000; x++) {
			try(INDArray arr = Nd4j.create(40000,2000)) {
				//Thread.sleep(50);
			}
			
//			arr.cleanup();
		}
		
		Thread.sleep(20000);
	}

}
