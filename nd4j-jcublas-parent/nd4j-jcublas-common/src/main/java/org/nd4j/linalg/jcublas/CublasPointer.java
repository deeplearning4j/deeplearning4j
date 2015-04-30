package org.nd4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

/**
 * Wraps the allocation and freeing of resources on a cuda device
 * @author bam4d
 *
 */
public class CublasPointer extends Pointer implements AutoCloseable {

	/**
	 * The underlying cuda buffer that contains the host and device memory
	 */
	final JCudaBuffer buffer;
	
	/**
	 * the byte offset of the result in the host
	 */
	final long hostByteOffset;
	
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
		buffer.copyToHost(hostByteOffset);
	}
	
	/**
	 * Creates a CublasPointer for a given JCudaBuffer
	 * @param buffer
	 */
	public CublasPointer(JCudaBuffer buffer) {
		super(buffer.getDevicePointer());
		this.buffer = buffer;
		hostByteOffset = 0;
		SimpleJCublas.checkResult(JCuda.cudaMemcpy(buffer.getDevicePointer(), buffer.getHostPointer(), buffer.getLength()*buffer.getElementSize(), cudaMemcpyKind.cudaMemcpyHostToDevice));
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
		super( ((JCudaBuffer)array.data()).getDevicePointer(array.length()*((JCudaBuffer)array.data()).getElementSize()*array.majorStride()));
		buffer = (JCudaBuffer)array.data();
		hostByteOffset = array.offset()*array.data().getElementSize();
		
		// Calculate the minimum amount of bytes we need to copy to the device to perform this operation
		long dataCount = array.length()*buffer.getElementSize()*array.majorStride();
		// Copy the data to the device
		SimpleJCublas.checkResult(JCuda.cudaMemcpy(buffer.getDevicePointer(), buffer.getHostPointer().withByteOffset(hostByteOffset), dataCount, cudaMemcpyKind.cudaMemcpyHostToDevice));
	}

}