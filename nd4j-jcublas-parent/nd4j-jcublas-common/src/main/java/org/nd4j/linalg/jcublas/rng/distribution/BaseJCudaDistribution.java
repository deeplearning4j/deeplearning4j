package org.nd4j.linalg.jcublas.rng.distribution;


import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.jcublas.JCublas;
import jcuda.jcurand.JCurand;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.utils.KernelLauncher;

import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.rng.JcudaRandom;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;

/**
 * Base JCuda distribution
 * Mainly handles calls to cuda
 *
 * @author Adam Gibson
 */
public abstract class BaseJCudaDistribution implements Distribution {


    protected JcudaRandom random;
    
    public BaseJCudaDistribution(JcudaRandom random) {
        this.random = random;
    }

    @Override
    public void reseedRandomGenerator(long seed) {
        random.setSeed(seed);
    }


    protected void doBinomial(INDArray p, JCudaBuffer out, int n, int len) {
        String functionName = "binomial";
        int blocks = PointerUtil.getNumBlocks(len, 128, 64);
        int threads = PointerUtil.getNumThreads(len, 64);
        JCudaBuffer randomNumbers = new CudaFloatDataBuffer(len * n);
        JCudaBuffer probBuffer = (JCudaBuffer) p.data();
       
        Object[] kernelParams = new Object[]{len, n, probBuffer, randomNumbers, out, random.generator()};

        try(KernelParamsWrapper params = new KernelParamsWrapper(kernelParams)) {
	        //int len,int n,double *ps,double *result, curandState *s
	        KernelFunctions.invoke(
	                blocks,
	                threads,
	                functionName,
	                "float"
	                , params.getKernelParameters());
	        
            //copy the result to the house before it gets destroyed
	        out.copyToHost();
        } catch(Exception e) {
        	throw new RuntimeException("Cannot run kernel", e);
        }
        
    }


    protected void doBinomialDouble(INDArray p, JCudaBuffer out, int n, int len) {
        String functionName = "binomial";
        int blocks = PointerUtil.getNumBlocks(len, 128, 64);
        int threads = PointerUtil.getNumThreads(len, 64);
        JCudaBuffer randomNumbers = new CudaDoubleDataBuffer(len);
        JCudaBuffer probBuffer = (JCudaBuffer) p.data();

        Object[] kernelParams = new Object[]{len, n, probBuffer, randomNumbers, out, random.generator()};
        
        try(KernelParamsWrapper params = new KernelParamsWrapper(kernelParams)) {
	        //int len,int n,double *ps,double *result, curandState *s
	        KernelFunctions.invoke(
	                blocks,
	                threads,
	                functionName,
	                "double"
	                , params.getKernelParameters());
	        
            //copy the result to the house before it gets destroyed
	        out.copyToHost();
	        
        } catch(Exception e) {
        	throw new RuntimeException("Cannot run kernel", e);
        }

    }

    protected void doBinomial(float p, JCudaBuffer out, int n, int len) {
        String functionName = "binomial_scalar";
        int blocks = PointerUtil.getNumBlocks(len, KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(len, KernelFunctions.THREADS);
        JCudaBuffer randomNumbers = new CudaFloatDataBuffer(len * n);

        Object[] kernelParams = new Object[]{len, n, p, randomNumbers, out, random.generator()};
        
        try(KernelParamsWrapper params = new KernelParamsWrapper(kernelParams)) {

        	//int len,int n,double *ps,double *result, curandState *s
            KernelFunctions.invoke(
                    blocks,
                    threads,
                    functionName,
                    "float"
                    , params.getKernelParameters());
            
            //copy the result to the house before it gets destroyed
	        out.copyToHost();
	        
        } catch(Exception e) {
        	throw new RuntimeException("Cannot run kernel", e);
        }
        
    }

    @Override
    public double sample() {
    	return inverseCumulativeProbability(random.nextDouble());
    }

    protected void doBinomialDouble(double p, JCudaBuffer out, int n, int len) {
        String functionName = "binomial_scalar";
        int blocks = PointerUtil.getNumBlocks(len, KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(len, KernelFunctions.THREADS);
        JCudaBuffer randomNumbers = new CudaDoubleDataBuffer(len);
       
        Object[] kernelParams = new Object[]{len, n, p, randomNumbers, out, Pointer.to(random.generator())};
        
        try(KernelParamsWrapper params = new KernelParamsWrapper(kernelParams)) {

		    //int len,int n,double *ps,double *result, curandState *s
		    KernelFunctions.invoke(
		            blocks,
		            threads,
		            functionName,
		            "double",
		            params.getKernelParameters());
		    
		    //copy the result to the house before it gets destroyed
	        out.copyToHost();
        
        } catch(Exception e) {
        	throw new RuntimeException("Cannot run kernel", e);
        }

    }

    
    /**
     * Generate floats from a uniform distribution
     * @param out
     * @param min
     * @param max
     * @param n
     */
    protected void doSampleUniform(JCudaBuffer out, float min, float max, int n) {
        String functionName = "uniform";
        int blocks = PointerUtil.getNumBlocks(n, KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(n, KernelFunctions.THREADS);
        
        //In the future these random numbers could be generated within the kernel
        JCudaBuffer randomNumbers = new CudaFloatDataBuffer(n);
        JCurand.curandGenerateUniform(random.generator(), randomNumbers.getDevicePointer(), n);
        //Generate the kernel parameters
        Object[] kernelParams = new Object[]{ n, min, max, randomNumbers.getDevicePointer(), out };
        
        try(KernelParamsWrapper params = new KernelParamsWrapper(kernelParams)) {

	        KernelFunctions.invoke(
	                blocks,
	                threads,
	                functionName,"float"
	                , params.getKernelParameters());
	        
	        //copy the result to the house before it gets destroyed
	        out.copyToHost();
	        randomNumbers.freeDevicePointer();
        } catch(Exception e) {
        	throw new RuntimeException("Cannot run kernel", e);
        }

    }

    /**
     * Generate doubles from a uniform distribution
     * @param out
     * @param min
     * @param max
     * @param n
     */
    protected void doSampleUniformDouble(JCudaBuffer out, double min, double max, int n) {
        String functionName = "uniform";
        int blocks = PointerUtil.getNumBlocks(n, 128, 64);
        int threads = PointerUtil.getNumThreads(n, 64);
        
        //In the future these random numbers could be generated within the kernel
        JCudaBuffer randomNumbers = new CudaDoubleDataBuffer(n);
	    JCurand.curandGenerateUniformDouble(random.generator(), randomNumbers.getDevicePointer(), n);
        //Generate the kernel parameters
        Object[] kernelParams = new Object[]{ n, min, max, randomNumbers.getDevicePointer(), out };
        
        try(KernelParamsWrapper params = new KernelParamsWrapper(kernelParams)) {
        	
	        KernelFunctions.invoke(
	                blocks,
	                threads,
	                functionName,
	                "double", 
	                params.getKernelParameters());
	        
	        //copy the result to the house before it gets destroyed
	        out.copyToHost();
	        randomNumbers.freeDevicePointer();
        } catch(Exception e) {
        	throw new RuntimeException("Cannot run kernel", e);
        }
    }

    protected void doSampleNormal(Pointer out, INDArray means, float std) {
        float[] means2 = means.data().asFloat();
        for (int i = 0; i < means.length(); i++) {
            JCudaBuffer dummy = new CudaFloatDataBuffer(2);

            JCurand.curandGenerateNormal(random.generator(), dummy.getDevicePointer(), 2, means2[i], std);
            JCuda.cudaMemcpy(
                    out.withByteOffset(Sizeof.FLOAT * i)
                    , dummy.getDevicePointer()
                    , Sizeof.FLOAT
                    , cudaMemcpyKind.cudaMemcpyDeviceToDevice);
            
            dummy.freeDevicePointer();
        } 


    }

    protected void doSampleNormalDouble(Pointer out, INDArray means, double std) {
        double[] means2 = means.data().asDouble();
        for (int i = 0; i < means.length(); i++) {
            JCudaBuffer dummy = new CudaDoubleDataBuffer(2);

            JCurand.curandGenerateNormalDouble(random.generator(), dummy.getDevicePointer(), 2, means2[i], std);
            JCuda.cudaMemcpy(
                    out.withByteOffset(Sizeof.DOUBLE * i), 
                    dummy.getDevicePointer(),
                    Sizeof.DOUBLE, 
                    cudaMemcpyKind.cudaMemcpyDeviceToDevice);
            
            dummy.freeDevicePointer();
        }

    }


    protected void doSampleNormal(float mean, float std, Pointer out, int n) {
        JCurand.curandGenerateNormal(random.generator(), out, n, mean, std);
    }

    protected void doSampleNormal(double mean, double std, Pointer out, int n) {
        JCurand.curandGenerateNormalDouble(random.generator(), out, n, mean, std);
    }


    public abstract double probability(double x0,
                                       double x1)
            throws NumberIsTooLargeException;
}
