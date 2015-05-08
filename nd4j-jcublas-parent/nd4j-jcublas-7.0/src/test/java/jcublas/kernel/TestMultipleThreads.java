package jcublas.kernel;

import static org.junit.Assert.assertEquals;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestMultipleThreads {
	
	@Test
	public void testMultipleThreads() throws InterruptedException {
		int numThreads = 50;
		final INDArray array = Nd4j.rand(3000, 3000);
		final INDArray expected = array.dup().mmul(array).mmul(array).div(array).div(array);
		final AtomicInteger correct = new AtomicInteger();
		final CountDownLatch latch = new CountDownLatch(numThreads);
		
		ExecutorService executors = Executors.newCachedThreadPool();
		
		for(int x = 0; x< numThreads; x++) {
			executors.execute(new Runnable() {
				@Override
				public void run() {
					try
					{
						int total = 10000;
						int right = 0;
						for(int x = 0; x<total; x++) {
							INDArray actual = array.dup().mmul(array).mmul(array).div(array).div(array);
							if(expected.equals(actual)) right++;						
						}
						
						if(total == right)
							correct.incrementAndGet();
					} finally {
						latch.countDown();
					}
					
				}
			});
		}
		
		latch.await();
		
		assertEquals(numThreads, correct.get());
		
	}

}
