package org.nd4j.linalg.jcublas.util;

import static jcuda.driver.JCudaDriver.cuMemGetInfo;

import java.util.HashSet;
import java.util.Set;

import jcuda.CudaException;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.ops.executioner.JCudaExecutioner;

/**
 * Wraps the generation of kernel parameters, creating, copying and destroying any cuda device allocations
 * @author bam4d
 *
 */
public class KernelParamsWrapper implements AutoCloseable {
	final public Object[] kernelParameters;
	private Set<JCudaBuffer> toFree;
	
	public Object[] getKernelParameters() {
		return kernelParameters;
	}
	
	public KernelParamsWrapper(Object... kernelParams) {
		kernelParameters = new Object[kernelParams.length];
		toFree = new HashSet<>();
		for(int i = 0; i<kernelParams.length; i++) {
			Object arg = kernelParams[i];
			
			// If the instance is a JCudaBuffer we should assign it to the device
			if(arg instanceof JCudaBuffer) {
				
				JCudaBuffer bufferToFree = (JCudaBuffer)arg;
				kernelParameters[i] = bufferToFree.getDevicePointer();
				copyToDevice(bufferToFree);
				
			// If we have an INDArray we should assign the buffer to the device and set an appropriate pointer
			} else if(arg instanceof INDArray) {
            	
            	INDArray array = (INDArray)arg;
				JCudaBuffer bufferToFree = (JCudaBuffer)array.data();
				kernelParameters[i] = bufferToFree.getDevicePointer().withByteOffset(array.offset()*array.data().getElementSize());
            	
				copyToDevice(bufferToFree);
				
			// If we don't need to copy anything to the device just copy it to the parameters
            } else {
            	kernelParameters[i] = arg;
            }
		}
	}

	/**
	 * Copy the buffer to the device memory and remeber to free it when this object is closed
	 * @param bufferToFree
	 */
	private void copyToDevice(JCudaBuffer bufferToFree) {
		JCudaExecutioner.checkResult(JCuda.cudaMemcpy(bufferToFree.getDevicePointer(), bufferToFree.getHostPointer(), bufferToFree.getLength()*bufferToFree.getElementSize(), cudaMemcpyKind.cudaMemcpyHostToDevice));
		toFree.add(bufferToFree);
	}
	
	

	/**
	 * Free all the buffers from this kernel's parameters
	 */
	@Override
	public void close() throws Exception {
		for(JCudaBuffer buffer : toFree) {
        	buffer.freeDevicePointer();
        }
        
        long[] free = new long[1];
        long[] total = new long[1];
        JCudaExecutioner.checkResult(cuMemGetInfo(free, total));
	}
	
}