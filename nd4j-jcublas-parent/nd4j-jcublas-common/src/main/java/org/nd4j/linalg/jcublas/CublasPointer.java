package org.nd4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

public class CublasPointer extends Pointer implements AutoCloseable {

	final JCudaBuffer buffer;
	
	@Override
	public void close() throws Exception {
		buffer.freeDevicePointer();
	}
	
	public void copyToHost() {
		buffer.copyToHost();
	}
	
	public CublasPointer(JCudaBuffer buffer) {
		super(buffer.getDevicePointer());
		this.buffer = buffer;
		copyToDevice();
	}
	
	public CublasPointer(INDArray array) {
		super( ((JCudaBuffer)array.data()).getDevicePointer().withByteOffset(array.offset()*array.data().getElementSize()));
		buffer = (JCudaBuffer)array.data();
		copyToDevice();
	}

	private void copyToDevice() {
		// Push the data to the device
		SimpleJCublas.checkResult(JCuda.cudaMemcpy(buffer.getDevicePointer(), buffer.getHostPointer(), buffer.getLength()*buffer.getElementSize(), cudaMemcpyKind.cudaMemcpyHostToDevice));
	}
}