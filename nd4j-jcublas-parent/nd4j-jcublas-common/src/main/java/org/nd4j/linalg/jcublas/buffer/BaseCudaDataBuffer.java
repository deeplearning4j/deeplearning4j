/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jcublas.buffer;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.complex.CudaComplexConversion;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for a data buffer
 *
 * @author Adam Gibson
 */
public abstract class BaseCudaDataBuffer implements JCudaBuffer {

	protected transient Pointer hostPointer;
	protected transient ByteBuffer hostBuffer;
	protected transient Map<String,DevicePointerInfo> pointersToContexts = new ConcurrentHashMap<>();
    protected AtomicBoolean modified = new AtomicBoolean(false);
    protected int length;
    protected int elementSize;
    protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    protected transient WeakReference<DataBuffer> ref;
    protected boolean isPersist = false;
    protected AtomicBoolean freed = new AtomicBoolean(false);
    static AtomicLong allocated = new AtomicLong();
    static AtomicLong totalAllocated = new AtomicLong();
    
    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);
    
    /**
     * Provides information about a device pointer 
     * @author bam4d
     *
     */
    private static class DevicePointerInfo {
    	final private CUdeviceptr pointer;
    	final private long length;
    	
    	public CUdeviceptr getPointer() { 
    		return pointer;
    	}
    	
    	public long getLength() {
    		return length;
    	}
    	
    	public DevicePointerInfo(CUdeviceptr pointer, long length) {
    		this.pointer = pointer;
    		this.length = length;
    	}
    }
    
    @Override
    public void setHostBuffer(ByteBuffer hostBuffer) {
    	this.hostBuffer = hostBuffer;
    	this.hostPointer = Pointer.to(hostBuffer);
    	hostBuffer.order(ByteOrder.nativeOrder());
    }
    
    @Override
    public ByteBuffer getHostBuffer() {
    	return hostBuffer;
    }
    
    @Override
    public Pointer getHostPointer() {
    	return hostPointer;
    }

    @Override
    public void persist() {
        isPersist = true;
    }

    @Override
    public boolean isPersist() {
        return isPersist;
    }

    /**
     * Base constructor
     *
     * @param length      the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(int length, int elementSize) {
        this.length = length;
        this.elementSize = elementSize;
        hostBuffer = ByteBuffer.allocate(length*elementSize);
        hostBuffer.order(ByteOrder.nativeOrder());
        hostPointer = Pointer.to(hostBuffer);
    }

    @Override
    public void removeReferencing(String id) {
        referencing.remove(id);
    }

    @Override
    public Collection<String> references() {
        return referencing;
    }

    @Override
    public void addReferencing(String id) {
        referencing.add(id);
    }

    @Override
    public void put(int i, IComplexNumber result) {
        
        modified.set(true);
        if (dataType() == DataBuffer.Type.FLOAT) {
            JCublas.cublasSetVector(
                    length(),
                    new cuComplex[]{CudaComplexConversion.toComplex(result.asFloat())}
                    , i
                    , 1
                    , hostPointer
                    , 1);
        } else {
            JCublas.cublasSetVector(
                    length(),
                    new cuDoubleComplex[]{CudaComplexConversion.toComplexDouble(result.asDouble())}
                    , i
                    , 1
                    , hostPointer
                    , 1);
        }
    }

    @Override
    public float[] asFloat() {
        return hostBuffer.asFloatBuffer().array();
    }

    @Override
    public double[] asDouble() {
        //
    	double[] ret = new double[length()];
        DoubleBuffer buf = getDoubleBuffer();
        for(int i = 0; i < length(); i++) {
            ret[i] = buf.get(i);
        }
        return ret;
    }

    @Override
    public int[] asInt() {
        return hostBuffer.asIntBuffer().array();
    }

    @Override
	public CUdeviceptr getDevicePointer() {
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName());

    	if(devicePointerInfo == null) {
    		CUdeviceptr devicePointer = new CUdeviceptr();
    		int devicePointerLength = getElementSize() * length();
    		allocated.addAndGet(devicePointerLength);
    		totalAllocated.addAndGet(devicePointerLength);
    		log.trace("Allocating {} bytes, total: {}, overall: {}", devicePointerLength, allocated.get(), totalAllocated);
    		
    		devicePointerInfo = new DevicePointerInfo(devicePointer, devicePointerLength);
    		checkResult(JCuda.cudaMalloc(devicePointer, devicePointerLength));
            pointersToContexts.put(Thread.currentThread().getName(),devicePointerInfo);
    		freed.set(false);
    	}
    	
    	return devicePointerInfo.getPointer();
    }


    @Override
    public void set(Pointer pointer) {
        
        modified.set(true);

        if (dataType() == DataBuffer.Type.DOUBLE) {
            JCublas.cublasDcopy(
                    length(),
                    pointer,
                    1,
                    hostPointer,
                    1
            );
        } else {
            JCublas.cublasScopy(
                    length(),
                    pointer,
                    1,
                    hostPointer,
                    1
            );
        }


    }


    /**
     * Copy the data of this buffer to another buffer on the gpu
     *
     * @param to the buffer to copy data to
     */
    protected void copyTo(JCudaBuffer to) {
        
        for(int i = 0; i < length(); i++) {
            to.put(i,getDouble(i));
        }
    }

    @Override
    public void assign(Number value) {
        assign(value, 0);
    }





    @Override
    public IComplexFloat getComplexFloat(int i) {
        return Nd4j.createFloat(getFloat(i), getFloat(i + 1));
    }

    @Override
    public IComplexDouble getComplexDouble(int i) {
        return Nd4j.createDouble(getDouble(i), getDouble(i + 1));
    }

    @Override
    public IComplexNumber getComplex(int i) {
        return dataType() == DataBuffer.Type.FLOAT ? getComplexFloat(i) : getComplexDouble(i);
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    protected void set(int index, int length, Pointer from, int inc) {
        
        modified.set(true);

        int offset = getElementSize() * index;
        if (offset >= length() * getElementSize())
            throw new IllegalArgumentException("Illegal offset " + offset + " with index of " + index + " and length " + length());
        JCublas.cublasSetVector(
                length
                , getElementSize()
                , from
                , inc
                , hostPointer.withByteOffset(offset)
                , 1);

    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    protected void set(int index, int length, Pointer from) {
        
        set(index, length, from, 1);
    }

    @Override
    public void assign(DataBuffer data) {
        
        JCudaBuffer buf = (JCudaBuffer) data;
        set(0, buf.getHostPointer());
    }
    protected ByteBuffer getBuffer() {
        return getBuffer(0);
    }

    protected java.nio.FloatBuffer getFloatBuffer(long offset) {
        return getHostBuffer(offset*Sizeof.FLOAT).asFloatBuffer();
    }

    protected java.nio.FloatBuffer getFloatBuffer() {
        return getFloatBuffer(0);
    }

    protected java.nio.DoubleBuffer getDoubleBuffer(long offset) {
        return getHostBuffer(offset*Sizeof.DOUBLE).asDoubleBuffer();
    }

    protected java.nio.DoubleBuffer getDoubleBuffer() {
        return getDoubleBuffer(0);
    }

    protected ByteBuffer getBuffer(long offset) {
        //ByteBuffer buf = pinnedPointer.getByteBuffer(offset * elementSize(),elementSize() * length() - offset * elementSize());
        // Set the byte order of the ByteBuffer
        hostBuffer.order(ByteOrder.nativeOrder());
        return hostBuffer;
    }
    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    protected void set(int index, Pointer from) {
        
        set(index, 1, from);
    }
    
    public static void checkResult(int cuResult)
    {
        if (cuResult != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(CUresult.stringFor(cuResult));
        }
    }
    
    @Override
    public boolean freeDevicePointer() {
    	DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName());
    	if(devicePointerInfo != null && !freed.get()) {
			allocated.addAndGet(-devicePointerInfo.getLength());
	        log.trace("freeing {} bytes, total: {}", devicePointerInfo.getLength(), allocated.get());
	        checkResult(JCuda.cudaFree(devicePointerInfo.getPointer()));
	        devicePointerInfo = null;
            freed.set(true);
            pointersToContexts.remove(Thread.currentThread().getName());
			return true;	
    	} 
    	return false;
    }
    
    @Override
    public void copyToHost() {
    	DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName());
        if(devicePointerInfo != null)
    		checkResult(JCuda.cudaMemcpy(hostPointer, devicePointerInfo.getPointer(), devicePointerInfo.getLength(), cudaMemcpyKind.cudaMemcpyDeviceToHost));
    }


    @Override
    public double[] getDoublesAt(int offset, int length) {
        return getDoublesAt(offset, 1, length);
    }

    @Override
    public float[] getFloatsAt(int offset, int length) {
        return getFloatsAt(offset, 1, length);
    }

    @Override
    public int getElementSize() {
        return elementSize;
    }

    @Override
    public int length() {
        return length;
    }




    @Override
    public void put(int i, float element) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void put(int i, double element) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void put(int i, int element) {
        throw new UnsupportedOperationException();
    }


    @Override
    public int getInt(int ix) {
        

        return 0;
    }



    @Override
    public void flush() {
        throw new UnsupportedOperationException();
    }


    @Override
    public void assign(int[] indices, float[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }
    
    private ByteBuffer getHostBuffer(long byteOffset)
    {
        if (hostBuffer == null)
        {
            return null;
        }
        if (!(hostBuffer instanceof ByteBuffer))
        {
            return null;
        }
        //hostBuffer.limit((int)(byteOffset + hostBuffer.capacity()));
        hostBuffer.position((int)byteOffset);
        return hostBuffer;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        BaseCudaDataBuffer that = (BaseCudaDataBuffer) o;

        if (length != that.length) return false;
        if (elementSize != that.elementSize) return false;
        if (isPersist != that.isPersist) return false;
        if (hostPointer != null ? !hostPointer.equals(that.hostPointer) : that.hostPointer != null) return false;
        if (hostBuffer != null ? !hostBuffer.equals(that.hostBuffer) : that.hostBuffer != null) return false;
        if (pointersToContexts != null ? !pointersToContexts.equals(that.pointersToContexts) : that.pointersToContexts != null)
            return false;
        if (modified != null ? !modified.equals(that.modified) : that.modified != null) return false;
        if (referencing != null ? !referencing.equals(that.referencing) : that.referencing != null) return false;
        if (ref != null ? !ref.equals(that.ref) : that.ref != null) return false;
        return !(freed != null ? !freed.equals(that.freed) : that.freed != null);

    }

    @Override
    public int hashCode() {
        int result = (int) (hostPointer != null ? hostPointer.hashCode() : 0);
        result = 31 * result + (hostBuffer != null ? hostBuffer.hashCode() : 0);
        result = 31 * result + (pointersToContexts != null ? pointersToContexts.hashCode() : 0);
        result = 31 * result + (modified != null ? modified.hashCode() : 0);
        result = 31 * result + length;
        result = 31 * result + elementSize;
        result = 31 * result + (referencing != null ? referencing.hashCode() : 0);
        result = 31 * result + (ref != null ? ref.hashCode() : 0);
        result = 31 * result + (isPersist ? 1 : 0);
        result = 31 * result + (freed != null ? freed.hashCode() : 0);
        return result;
    }

    @Override
    public void assign(int[] offsets, int[] strides, int n, DataBuffer... buffers) {
        //

        int count = 0;
        for (int i = 0; i < buffers.length; i++) {
            DataBuffer buffer = buffers[i];
            if (buffer instanceof JCudaBuffer) {
                JCudaBuffer buff = (JCudaBuffer) buffer;
                
                try(CublasPointer buffPointer = new CublasPointer(buff)) {
                
	                if (buff.dataType() == DataBuffer.Type.DOUBLE) {
	                	
	                    JCublas.cublasDcopy(
	                            buff.length()
	                            , buffPointer.withByteOffset(buff.getElementSize() * offsets[i])
	                            , strides[i]
	                            , getDevicePointer().withByteOffset(count * buff.getElementSize())
	                            , 1);
	                    
	                    
	                    count += (buff.length() - 1 - offsets[i]) / strides[i] + 1;
	                } else {
	                    JCublas.cublasScopy(buff.length()
	                            , buffPointer.withByteOffset(buff.getElementSize() * offsets[i])
	                            , strides[i]
	                            , getDevicePointer().withByteOffset(count * buff.getElementSize())
	                            , 1);
	                    
	                    count += (buff.length() - 1 - offsets[i]) / strides[i] + 1;
	                }
                } catch (Exception e) {
            		throw new RuntimeException("Could not run cublas command", e);
            	}
            } else
                throw new IllegalArgumentException("Only jcuda data buffers allowed");
        }
        copyToHost();
        freeDevicePointer();
    }

    @Override
    public void assign(DataBuffer... buffers) {
        

        int[] offsets = new int[buffers.length];
        int[] strides = new int[buffers.length];
        for (int i = 0; i < strides.length; i++)
            strides[i] = 1;
        assign(offsets, strides, buffers);
    }

    @Override
    public void assign(int[] offsets, int[] strides, DataBuffer... buffers) {
        assign(offsets, strides, length(), buffers);
    }
    
    @Override
	public void destroy() {
		freeDevicePointer();
		hostBuffer = null;
		hostPointer = null;
		
	}

    private void writeObject(java.io.ObjectOutputStream stream)
            throws IOException {
        stream.writeInt(length);
        stream.writeInt(elementSize);
        stream.writeBoolean(isPersist);
        if(dataType() == DataBuffer.Type.DOUBLE) {
            double[] d = asDouble();
            for(int i = 0; i < d.length; i++)
                stream.writeDouble(d[i]);
        }
        else if(dataType() == DataBuffer.Type.FLOAT) {
            float[] f = asFloat();
            for(int i = 0; i < f.length; i++)
                stream.writeFloat(f[i]);
        }


    }


    private void readObject(java.io.ObjectInputStream stream)
            throws IOException, ClassNotFoundException {
        length = stream.readInt();
        elementSize = stream.readInt();
        isPersist = stream.readBoolean();
        pointersToContexts = new ConcurrentHashMap<>();
        referencing = Collections.synchronizedSet(new HashSet<String>());
        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        freed = new AtomicBoolean(false);
        if(dataType() == DataBuffer.Type.DOUBLE) {
            double[] d = new double[length];
            for(int i = 0; i < d.length; i++)
                d[i] = stream.readDouble();
            BaseCudaDataBuffer  buf = (BaseCudaDataBuffer) KernelFunctions.alloc(d);
            hostPointer = buf.hostPointer;
        }
        else if(dataType() == DataBuffer.Type.FLOAT) {
            float[] f = new float[length];
            for(int i = 0; i < f.length; i++)
                f[i] = stream.readFloat();
            BaseCudaDataBuffer  buf = (BaseCudaDataBuffer) KernelFunctions.alloc(f);
            hostPointer = buf.hostPointer;
        }
    }


}
