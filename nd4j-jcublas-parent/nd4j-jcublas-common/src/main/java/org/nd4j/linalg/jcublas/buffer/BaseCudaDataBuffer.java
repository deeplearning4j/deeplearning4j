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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import jcuda.CudaException;
import jcuda.Pointer;
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

	protected transient CUdeviceptr devicePointer;
	protected transient Pointer hostPointer;
	protected transient ByteBuffer hostBuffer;
    //protected transient Pointer pinnedPointer;
    protected int length;
    protected int elementSize;
    protected AtomicBoolean freed = new AtomicBoolean(false);
    protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    protected transient WeakReference<DataBuffer> ref;
    protected boolean isPersist = false;
    
    static AtomicLong allocated = new AtomicLong();
    static AtomicLong totalAllocated = new AtomicLong();
    
    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);
    
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
        ensureNotFreed();


        if (dataType() == DataBuffer.FLOAT) {
            JCublas.cublasSetVector(
                    getLength(),
                    new cuComplex[]{CudaComplexConversion.toComplex(result.asFloat())}
                    , i
                    , 1
                    , hostPointer
                    , 1);
        } else {
            JCublas.cublasSetVector(
                    getLength(),
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
        //ensureNotFreed();
    	double[] ret = new double[getLength()];
        DoubleBuffer buf = getDoubleBuffer();
        for(int i = 0; i < getLength(); i++) {
            ret[i] = buf.get(i);
        }
        return ret;
    }

    @Override
    public int[] asInt() {
        ensureNotFreed();
        return hostBuffer.asIntBuffer().array();
    }
    
    @Override
	public CUdeviceptr getDevicePointer() {
    	if(devicePointer == null) {
	    	devicePointer = new CUdeviceptr();
	    	allocated.addAndGet(elementSize * length);
	        totalAllocated.addAndGet(elementSize * length);
	        log.debug("Allocating {} bytes, total: {}, overall: {}", elementSize * length, allocated.get(), totalAllocated);
	        checkResult(JCuda.cudaMalloc(devicePointer, length*elementSize));
    	}
    	
    	return devicePointer;
    }
    
    @Override
    public void set(Pointer pointer) {
        ensureNotFreed();


        if (dataType() == DOUBLE) {
            JCublas.cublasDcopy(
                    getLength(),
                    pointer,
                    1,
                    hostPointer,
                    1
            );
        } else {
        	JCublas.cublasScopy(
                    getLength(),
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
        ensureNotFreed();
        for(int i = 0; i < getLength(); i++) {
            to.put(i,getDouble(i));
        }
    }

    @Override
    public void assign(Number value) {
        assign(value, 0);
    }





    @Override
    public IComplexFloat getComplexFloat(int i) {
        return Nd4j.createFloat(getFloat(i), getFloat(i) + 1);
    }

    @Override
    public IComplexDouble getComplexDouble(int i) {
        return Nd4j.createDouble(getDouble(i), getDouble(i + 1));
    }

    @Override
    public IComplexNumber getComplex(int i) {
        ensureNotFreed();
        return dataType() == DataBuffer.FLOAT ? getComplexFloat(i) : getComplexDouble(i);
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    protected void set(int index, int length, Pointer from, int inc) {
        ensureNotFreed();


        int offset = getElementSize() * index;
        if (offset >= getLength() * getElementSize())
            throw new IllegalArgumentException("Illegal offset " + offset + " with index of " + index + " and length " + getLength());
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
        ensureNotFreed();
        set(index, length, from, 1);
    }

    @Override
    public void assign(DataBuffer data) {
        ensureNotFreed();
        JCudaBuffer buf = (JCudaBuffer) data;
        set(0, buf.getHostPointer());
    }
    protected ByteBuffer getBuffer() {
        return getBuffer(0);
    }

    protected java.nio.FloatBuffer getFloatBuffer(long offset) {
        return getHostBuffer(offset).asFloatBuffer();
    }

    protected java.nio.FloatBuffer getFloatBuffer() {
        return getFloatBuffer(0);
    }

    protected java.nio.DoubleBuffer getDoubleBuffer(long offset) {
        return getHostBuffer(offset).asDoubleBuffer();
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
        ensureNotFreed();
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
    	if(!freed.get()) {
    		
    		allocated.addAndGet(-getElementSize() * getLength());
            log.debug("freeing {} bytes, total: {}", getElementSize() * getLength(), allocated.get());
            checkResult(JCuda.cudaFree(devicePointer));
            freed.set(true);
    		devicePointer = null;
    		return true;	
    	}
    	return false;
    }
    
    @Override
    public void copyToHost() {
    	if(devicePointer!=null)
    		checkResult(JCuda.cudaMemcpy(hostPointer, devicePointer, length*elementSize, cudaMemcpyKind.cudaMemcpyDeviceToHost));
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
    public int getLength() {
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
        ensureNotFreed();

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
        hostBuffer.limit((int)(byteOffset + hostBuffer.capacity()));
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
        for(int i = 0; i < getLength(); i++) {
            if(getDouble(i) != that.getDouble(i))
                return false;
        }

        return true;

    }

    @Override
    public int hashCode() {
        int result = devicePointer != null ? devicePointer.hashCode() : 0;
        result = 31 * result + length;
        result = 31 * result + elementSize;
        result = 31 * result + (freed != null ? freed.hashCode() : 0);
        result = 31 * result + (referencing != null ? referencing.hashCode() : 0);
        result = 31 * result + (ref != null ? ref.hashCode() : 0);
        result = 31 * result + (isPersist ? 1 : 0);
        return result;
    }

    @Override
    public void assign(int[] offsets, int[] strides, int n, DataBuffer... buffers) {
        ensureNotFreed();


        int count = 0;
        for (int i = 0; i < buffers.length; i++) {
            DataBuffer buffer = buffers[i];
            if (buffer instanceof JCudaBuffer) {
                JCudaBuffer buff = (JCudaBuffer) buffer;
                if (buff.dataType() == DataBuffer.DOUBLE) {
                    JCublas.cublasDcopy(
                            buff.getLength()
                            , buff.getHostPointer().withByteOffset(buff.getElementSize() * offsets[i])
                            , strides[i]
                            , hostPointer.withByteOffset(count * buff.getElementSize())
                            , 1);
                    count += (buff.getLength() - 1 - offsets[i]) / strides[i] + 1;
                } else {
                    JCublas.cublasScopy(buff.getLength()
                            , buff.getHostPointer().withByteOffset(buff.getElementSize() * offsets[i])
                            , strides[i]
                            , hostPointer.withByteOffset(count * buff.getElementSize())
                            , 1);
                    count += (buff.getLength() - 1 - offsets[i]) / strides[i] + 1;
                }
            } else
                throw new IllegalArgumentException("Only jcuda data buffers allowed");
        }
    }

    @Override
    public void assign(DataBuffer... buffers) {
        ensureNotFreed();

        int[] offsets = new int[buffers.length];
        int[] strides = new int[buffers.length];
        for (int i = 0; i < strides.length; i++)
            strides[i] = 1;
        assign(offsets, strides, buffers);
    }

    @Override
    public void assign(int[] offsets, int[] strides, DataBuffer... buffers) {
        ensureNotFreed();
        assign(offsets, strides, getLength(), buffers);
    }

    protected void ensureNotFreed() {
        if (freed.get())
            throw new IllegalStateException("Unable to do operation, buffer already freed");
    }

    private void writeObject(java.io.ObjectOutputStream stream)
            throws IOException {
        stream.writeInt(length);
        stream.writeInt(elementSize);
        stream.writeBoolean(isPersist);
        if(dataType() == DataBuffer.DOUBLE) {
            double[] d = asDouble();
            for(int i = 0; i < d.length; i++)
                stream.writeDouble(d[i]);
        }
        else if(dataType() == DataBuffer.FLOAT) {
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
        referencing = Collections.synchronizedSet(new HashSet<String>());
        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        freed  = new AtomicBoolean(false);
        if(dataType() == DataBuffer.DOUBLE) {
            double[] d = new double[length];
            for(int i = 0; i < d.length; i++)
                d[i] = stream.readDouble();
            BaseCudaDataBuffer  buf = (BaseCudaDataBuffer) KernelFunctions.alloc(d);
            hostPointer = buf.hostPointer;
        }
        else if(dataType() == DataBuffer.FLOAT) {
            float[] f = new float[length];
            for(int i = 0; i < f.length; i++)
                f[i] = stream.readFloat();
            BaseCudaDataBuffer  buf = (BaseCudaDataBuffer) KernelFunctions.alloc(f);
            hostPointer = buf.hostPointer;
        }
    }


}
