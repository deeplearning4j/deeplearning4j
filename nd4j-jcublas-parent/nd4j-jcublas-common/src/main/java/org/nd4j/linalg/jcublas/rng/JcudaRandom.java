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

package org.nd4j.linalg.jcublas.rng;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.utils.KernelLauncher;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SetRange;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static org.nd4j.linalg.jcublas.SimpleJCublas.sync;

/**
 * Jcuda random number generator
 *
 * @author Adam Gibson
 */
public class JcudaRandom implements Random {
    private curandGenerator generator = new curandGenerator();

    /**
     * Initialize the random generator on the gpu
     */
    public JcudaRandom() {
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator, 1234);
        JCurand.setExceptionsEnabled(true);

    }

    public curandGenerator generator() {
        return generator;
    }


    @Override
    public void setSeed(int seed) {
       
        curandSetPseudoRandomGeneratorSeed(generator, seed);
    }

    @Override
    public void setSeed(int[] seed) {
    }

    @Override
    public void setSeed(long seed) {
       
        curandSetPseudoRandomGeneratorSeed(generator, seed);

    }

    @Override
    public void nextBytes(byte[] bytes) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int nextInt() {
       
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerate(generator, buffer.getDevicePointer(), 2);
        buffer.copyToHost();
        double[] data = buffer.asDouble();
        int ret = (int) data[0];
        
        buffer.freeDevicePointer();
        return ret;
    }

    @Override
    public int nextInt(int n) {
       
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerateUniformDouble(generator, buffer.getDevicePointer(), 2);
        buffer.copyToHost();
        double[] data = buffer.asDouble();
        int ret = (int) data[0];
        buffer.freeDevicePointer();
        return ret;
    }

    @Override
    public long nextLong() {
       
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerate(generator, buffer.getDevicePointer(), 2);
        buffer.copyToHost();
        double[] data = buffer.asDouble();
        long ret = (long) data[0];
        buffer.freeDevicePointer();
        return ret;
    }

    @Override
    public boolean nextBoolean() {
        return nextGaussian() > 0.5;
    }

    @Override
    public float nextFloat() {
       
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerate(generator, buffer.getDevicePointer(), 2);
        buffer.copyToHost();
        double[] data = buffer.asDouble();
        float ret = (float) data[0];
        buffer.freeDevicePointer();
        return ret;
    }

    @Override
    public double nextDouble() {
       
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerate(generator, buffer.getDevicePointer(), 2);
        buffer.copyToHost();
        double[] data = buffer.asDouble();
        buffer.freeDevicePointer();
        return data[0];
    }

    @Override
    public double nextGaussian() {
       
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerateUniformDouble(generator, buffer.getDevicePointer(), 2);
        buffer.copyToHost();
        double[] data = buffer.asDouble();
        buffer.freeDevicePointer();
        return data[0];
    }

    @Override
    public INDArray nextGaussian(int[] shape) {
    	sync();
        INDArray create = Nd4j.create(shape);
        try(CublasPointer p = new CublasPointer(create)) {
            
	        if (p.getBuffer().dataType() == DataBuffer.Type.FLOAT)
	            checkResult(curandGenerateUniform(generator, p, create.length()));
	        else if (p.getBuffer().dataType() == DataBuffer.Type.DOUBLE)
	        	checkResult(curandGenerateUniformDouble(generator, p, create.length()));
	        else
	            throw new IllegalStateException("Illegal data type discovered");
	        
	        p.copyToHost();
	        return create;
        } catch(Exception e) {
        	throw new RuntimeException("Could not allocate resources", e);
        }
        
    }

    @Override
    public INDArray nextDouble(int[] shape) {
       
    	INDArray create = Nd4j.create(shape);
        try(CublasPointer p = new CublasPointer(create)) {
        
	        if (p.getBuffer().dataType() == DataBuffer.Type.FLOAT)
	            checkResult(curandGenerateUniform(generator, p, create.length()));
	        else if (p.getBuffer().dataType() == DataBuffer.Type.DOUBLE)
	        	checkResult(curandGenerateUniformDouble(generator, p, create.length()));
	        else
	            throw new IllegalStateException("Illegal data type discovered");
	        
	        p.copyToHost();
	        return create;
        } catch(Exception e) {
        	throw new RuntimeException("Could not allocate resources", e);
        }
    }

    @Override
    public INDArray nextFloat(int[] shape) {
       
    	INDArray create = Nd4j.create(shape);
        try(CublasPointer p = new CublasPointer(create)) {
        
	        if (p.getBuffer().dataType() == DataBuffer.Type.FLOAT)
	            checkResult(curandGenerateUniform(generator, p, create.length()));
	        else if (p.getBuffer().dataType() == DataBuffer.Type.DOUBLE)
	        	checkResult(curandGenerateUniformDouble(generator, p, create.length()));
	        else
	            throw new IllegalStateException("Illegal data type discovered");
	        
	        p.copyToHost();
	        return create;
        } catch(Exception e) {
        	throw new RuntimeException("Could not allocate resources", e);
        }
    }

    @Override
    public INDArray nextInt(int[] shape) {
       
        INDArray create = Nd4j.create(shape);
        try(CublasPointer p = new CublasPointer(create)) {
	        if (p.getBuffer().dataType() == DataBuffer.Type.FLOAT)
	            curandGenerateUniform(generator, p, create.length());
	        else if (p.getBuffer().dataType() == DataBuffer.Type.DOUBLE)
	            curandGenerateUniformDouble(generator, p, create.length());
	        else
	            throw new IllegalStateException("Illegal data type discovered");
	
	        Nd4j.getExecutioner().exec(new SetRange(create, 0, 1));
	        
	        p.copyToHost();
	        return create;
        } catch(Exception e) {
        	throw new RuntimeException("Could not allocate resources", e);
        }
    }

    @Override
    public INDArray nextInt(int n, int[] shape) {
       
    	INDArray create = Nd4j.create(shape);
        try(CublasPointer p = new CublasPointer(create)) {
	        if (p.getBuffer().dataType() == DataBuffer.Type.FLOAT)
	            curandGenerateUniform(generator, p, create.length());
	        else if (p.getBuffer().dataType() == DataBuffer.Type.DOUBLE)
	            curandGenerateUniformDouble(generator, p, create.length());
	        else
	            throw new IllegalStateException("Illegal data type discovered");
	
	        Nd4j.getExecutioner().exec(new SetRange(create, 0, 1));
	        
	        p.copyToHost();
	        return create;
        } catch(Exception e) {
        	throw new RuntimeException("Could not allocate resources", e);
        }
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
