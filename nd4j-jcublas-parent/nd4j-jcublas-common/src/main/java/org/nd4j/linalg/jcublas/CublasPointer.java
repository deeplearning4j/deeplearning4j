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
		SimpleJCublas.checkResult(JCuda.cudaMemcpy(buffer.getDevicePointer(), buffer.getHostPointer(), buffer.getLength()*buffer.getElementSize(), cudaMemcpyKind.cudaMemcpyHostToDevice));
	}
	
	public CublasPointer(INDArray array) {
		super( ((JCudaBuffer)array.data()).getDevicePointer(((JCudaBuffer)array.data()).getLength()*((JCudaBuffer)array.data()).getElementSize() - array.offset()*array.data().getElementSize()));
		buffer = (JCudaBuffer)array.data();
		
		long dataCount = buffer.getLength()*buffer.getElementSize() - array.offset()*array.data().getElementSize();
		SimpleJCublas.checkResult(JCuda.cudaMemcpy(buffer.getDevicePointer(), buffer.getHostPointer().withByteOffset(array.offset()*array.data().getElementSize()), dataCount, cudaMemcpyKind.cudaMemcpyHostToDevice));
	}

}