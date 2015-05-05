package org.nd4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.complex.IComplexNDArray;
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
		SimpleJCublas.checkResult(JCuda.cudaMemcpy(buffer.getDevicePointer(), buffer.getHostPointer(), buffer.length()*buffer.getElementSize(), cudaMemcpyKind.cudaMemcpyHostToDevice));
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
		super( ((JCudaBuffer)array.data()).getDevicePointer().withByteOffset(array.offset()*((JCudaBuffer)array.data()).getElementSize()));
		buffer = (JCudaBuffer)array.data();
		
		// Copy the data to the device
		SimpleJCublas.checkResult(JCuda.cudaMemcpy(buffer.getDevicePointer(), buffer.getHostPointer(), buffer.getElementSize()*buffer.length(), cudaMemcpyKind.cudaMemcpyHostToDevice));
	}

}