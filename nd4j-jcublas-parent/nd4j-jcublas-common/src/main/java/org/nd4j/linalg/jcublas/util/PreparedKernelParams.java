package org.nd4j.linalg.jcublas.util;

import static jcuda.driver.JCudaDriver.cuMemGetInfo;

import java.util.HashSet;
import java.util.Set;

import jcuda.CudaException;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.ops.executioner.JCudaExecutioner;

/**
 * Wraps the generation of kernel parameters, creating, copying and destroying any cuda device allocations
 * @author bam4d
 *
 */
public class PreparedKernelParams implements AutoCloseable {
	final public Object[] kernelParameters;
	private Set<JCudaBuffer> toFree;
	
	public Object[] getKernelParameters() {
		return kernelParameters;
	}
	
	public PreparedKernelParams(Object... kernelParams) {
		kernelParameters = new Object[kernelParams.length];
		toFree = new HashSet<>();
		for(int i = 0; i<kernelParams.length; i++) {
			Object arg = kernelParams[i];
    		if(arg instanceof JCudaBuffer) {
            	
            	JCudaBuffer bufferToFree = (JCudaBuffer)arg;
				kernelParameters[i] = bufferToFree.getDevicePointer();
            	
				JCudaExecutioner.checkResult(JCuda.cudaMemcpy(bufferToFree.getDevicePointer(), bufferToFree.getHostPointer(), bufferToFree.getLength()*bufferToFree.getElementSize(), cudaMemcpyKind.cudaMemcpyHostToDevice));
            	
            	toFree.add(bufferToFree);
            } else {
            	kernelParameters[i] = arg;
            }
		}
	}

	@Override
	public void close() throws Exception {
		for(JCudaBuffer buffer : toFree) {
        	buffer.freeDevicePointer();
        }
        
        long[] free = new long[1];
        long[] total = new long[1];
        JCudaExecutioner.checkResult(cuMemGetInfo(free, total));
	}
	
	private static int checkResult(int result)
    {
        if (result != cudaError.cudaSuccess)
        {
            throw new CudaException(cudaError.stringFor(result));
        }
        return result;
    }

	
}