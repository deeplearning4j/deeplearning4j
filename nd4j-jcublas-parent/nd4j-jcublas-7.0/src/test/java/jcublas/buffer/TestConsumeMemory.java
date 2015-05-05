package jcublas.buffer;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestConsumeMemory {

	@Test
	@Ignore
	public void testCreateFloatArrays() throws InterruptedException {
		for(int x= 0; x<1000000; x++) {
			INDArray arr = Nd4j.create(200, 200);
		}
		
		//Thread.sleep(50000);
	}
}
