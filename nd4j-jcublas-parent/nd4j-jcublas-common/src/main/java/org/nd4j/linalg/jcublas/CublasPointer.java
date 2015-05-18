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

package org.nd4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;

/**
 * Wraps the allocation
 * and freeing of resources on a cuda device
 * @author bam4d
 *
 */
public class CublasPointer extends Pointer implements AutoCloseable {

	/**
	 * The underlying cuda buffer that contains the host and device memory
	 */
	final JCudaBuffer buffer;
	
	/**
	 * frees the underlying device memory allocated for this pointer
	 */
	@Override
	public void close() throws Exception {
		buffer.freeDevicePointer();
	}

	public JCudaBuffer getBuffer() {
		return buffer;
	}
	
	/**
	 * copies the result to the host buffer
	 */
	public void copyToHost() {
		buffer.copyToHost();
	}
	
	/**
	 * Creates a CublasPointer for a given JCudaBuffer
	 * @param buffer
	 */
	public CublasPointer(JCudaBuffer buffer) {
		super(buffer.getDevicePointer());
		this.buffer = buffer;
		SimpleJCublas.checkResult(
				JCuda.cudaMemcpyAsync(
                        buffer.getDevicePointer()
                        , buffer.getHostPointer()
                        , buffer.length() * buffer.getElementSize()
                        , cudaMemcpyKind.cudaMemcpyHostToDevice, ContextHolder.getInstance().getCudaStream()));
	}
	
	/**
	 * Creates a CublasPointer for a given INDArray.
	 * 
	 * This wrapper makes sure that the INDArray offset, stride and memory pointers are accurate to the data being copied to and from the device.
	 * 
	 * If the copyToHost function is used in in this class, the host buffer offset and data length is taken care of automatically
	 * @param array
	 */
	public CublasPointer(INDArray array) {
		super( ((JCudaBuffer)array.data()).getDevicePointer().withByteOffset(array.offset() * (array.data()).getElementSize()));
		buffer = (JCudaBuffer)array.data();
		
		// Copy the data to the device
		SimpleJCublas.checkResult(
                JCuda.cudaMemcpyAsync(
                buffer.getDevicePointer()
                , buffer.getHostPointer()
                , buffer.getElementSize() * buffer.length()
                , cudaMemcpyKind.cudaMemcpyHostToDevice, ContextHolder.getInstance().getCudaStream()));
	}

}