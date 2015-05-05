package jcublas;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

public class CublasPointerTests {
	
	@Test
	public void testAllocateAndCopyBackToHost() throws Exception {
		
		INDArray test = Nd4j.rand(5,5).linearView();
		
		CublasPointer p = new CublasPointer(test);
		CublasPointer p1 = new CublasPointer((JCudaBuffer)test.data());
		
		p.copyToHost();
		p1.copyToHost();
		
		assertEquals(p.getBuffer(), p1.getBuffer());
		assertArrayEquals(p.getBuffer().asBytes(), p1.getBuffer().asBytes());
		
		p.close();
		p1.close();
	}
	
	/** 
	 * Test that when using offsets, the data is not corrupted
	 * @throws Exception
	 */
	@Test
	public void testColumnOffsettingCopyBackToHost() throws Exception {
		for(int i = 0; i<100; i++) {
			INDArray test = Nd4j.rand(i,i);
			
			INDArray testDupe = test.dup();
			
			// Create an offsetted set of pointers and copy to and from device, this should copy back to the same offset it started at.
			for(int x = 0; x< i; x++) {
				INDArray test2 = test.getRow(x);
				CublasPointer p1 = new CublasPointer(test2);
				p1.copyToHost();
				p1.close();
			}
		
		
			assertArrayEquals(testDupe.data().asBytes(), test.data().asBytes());
		}
	}

}
