package org.nd4j.linalg.jcublas.buffer;

import static jcuda.driver.JCudaDriver.cuMemGetInfo;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MemoryLeakTest {
	
	@Test
	public void testAllocateAndRemove() throws Exception {
		for(int x = 0; x<1000; x++) {
			INDArray arr = Nd4j.rand(8000,8000);
			INDArray arr2 = Nd4j.rand(8000,8000);
			arr2.divi(arr);
			Thread.sleep(100);
		}

	}
	
	@Test
	public void testKernelFunction() throws Exception {
					
		for(int x = 0; x<10000; x++) {
			//INDArray arr2 = Nd4j.rand(4000,4000);
			INDArray arr2 = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f});
			
			INDArray norm2 = arr2.norm2(1);
			assertEquals(5.477f, norm2.getFloat(0), 0.01);
			//INDArray eps = arr2.eps(arr2);
			Thread.sleep(10);
			
		}
					
	}
}
