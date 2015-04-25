package org.nd4j.linalg.jcublas.buffer;

import java.io.IOException;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MemoryLeakTest {
	
	@Test
	public void testAllocateAndRemove() throws Exception {
		for(int x = 0; x<1000; x++) {
			try(INDArray arr = Nd4j.rand(8000,8000)) {
				try(INDArray arr2 = Nd4j.rand(8000,8000)) {
					arr2.divi(arr);
					Thread.sleep(100);
				}
			}
			
//			arr.cleanup();
		}

	}
	
	@Test
	public void testAllocateRemoveNative() {
		for(int x = 0; x<1000; x++) {
			//Pointer p1 = allocate(10000000, 1.0f);
			//Pointer p2 = allocate(10000000, 1.0f);
			
			
			//deallocate(p1);
			//deallocate(p2);
		}
	}

	private void deallocate(Pointer p) {
		// TODO Auto-generated method stub
		
	}

	private Pointer allocate(int i) {
		Pointer p = new Pointer();
		
		//JCuda.cudaHostAlloc(ptr, size, flags)
		
		return null;
	}

}
