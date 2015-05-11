/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package jcuda.jcublas.rng.distribution;

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
	
	Number max = 0.4;
	Number min = -0.4;

	@Test
	public void testUniformSample() {
		Nd4j.dtype = DataBuffer.Type.FLOAT;
		for(int x = 0; x<100; x++) {
			INDArray arr = Nd4j.rand(200,200,min.floatValue(),max.floatValue(),Nd4j.getRandom());
			
			// Assert the values are within range
			assertMaxMinF(arr);
		}
	}
	
	@Test
	public void testUniformSampleDouble() {
		Nd4j.dtype = DataBuffer.Type.DOUBLE;
		for(int x = 0; x<100; x++) {
			
			INDArray arr = Nd4j.rand(200,200,min.doubleValue(),max.doubleValue(),Nd4j.getRandom());
			
			// Assert the values are within range
			assertMaxMinD(arr);
		}
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testBadMaxMin() {
		Nd4j.rand(200,200,2,1,Nd4j.getRandom());
	}
	
	private void assertMaxMinF(INDArray arr) {
		
		boolean nonZero = false;

		for(Float d: arr.data().asFloat()) {
			assertTrue("Returned value is above the maximum", max.floatValue() >= d);
			assertTrue("Returned value is below the minimum", min.floatValue() <= d);
			
			if(d != 0.0) {
				nonZero = true;
			}
		}
		assertTrue("The entire array is zeros", nonZero);

	}
	
	private void assertMaxMinD(INDArray arr) {
		
		boolean nonZero = false;

		for(Double d: arr.data().asDouble()) {
			assertTrue("Returned value is above the maximum", max.doubleValue() >= d);
			assertTrue("Returned value is below the minimum", min.doubleValue() <= d);
			
			if(d != 0.0) {
				nonZero = true;
			}
		}
		assertTrue("The entire array is zeros", nonZero);

	}
}
