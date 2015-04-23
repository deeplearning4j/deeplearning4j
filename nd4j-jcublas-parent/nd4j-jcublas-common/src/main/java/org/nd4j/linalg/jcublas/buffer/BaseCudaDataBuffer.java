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
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import jcuda.Pointer;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.instrumentation.Instrumentation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.complex.CudaComplexConversion;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;

/**
 * Base class for a data buffer
 *
 * @author Adam Gibson
 */
public abstract class BaseCudaDataBuffer implements JCudaBuffer {

    protected transient Pointer pinnedPointer;
    protected int length;
    protected int elementSize;
    protected AtomicBoolean freed = new AtomicBoolean(false);
    protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    protected transient WeakReference<DataBuffer> ref;
    protected boolean isPersist = false;

    static {
        SimpleJCublas.init();
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
        if (pinnedPointer == null)
            alloc();
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
                    length(),
                    new cuComplex[]{CudaComplexConversion.toComplex(result.asFloat())}
                    , i
                    , 1
                    , pointer()
                    , 1);
        } else {
            JCublas.cublasSetVector(
                    length(),
                    new cuDoubleComplex[]{CudaComplexConversion.toComplexDouble(result.asDouble())}
                    , i
                    , 1
                    , pointer()
                    , 1);
        }
    }

    @Override
    public float[] asFloat() {
        ensureNotFreed();
        ByteBuffer buf = pinnedPointer.getByteBuffer(0, length() * elementSize());
        return buf.asFloatBuffer().array();
    }

    @Override
    public double[] asDouble() {
        ensureNotFreed();
        java.nio.DoubleBuffer doubleBuffer = getDoubleBuffer();
        double[] ret = new double[length()];
        for(int i = 0; i < length(); i++) {
            ret[i] = doubleBuffer.get(i);
        }
        return ret;
    }

    @Override
    public int[] asInt() {
        ensureNotFreed();
        ByteBuffer buf = pinnedPointer.getByteBuffer(0, length() * elementSize());
        return buf.asIntBuffer().array();
    }
    
    private void doCuda(int result) {
    	if(result!=0) {
        	//System.out.printf("getPointer %d\n",result);
        }
    }


    @Override
    public Pointer pointer() {
        ensureNotFreed();
        Pointer pointer = new Pointer();
        doCuda(JCuda.cudaHostGetDevicePointer(pointer, pinnedPointer, 0));
        return pointer;
    }
    
    @Override
    public void alloc() {


        pinnedPointer = new Pointer();

        // Check if the device supports mapped host memory
        cudaDeviceProp deviceProperties = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(deviceProperties, 0);
        if (deviceProperties.canMapHostMemory == 0) {
            System.err.println("This device can not map host memory");
            System.err.println(deviceProperties.toFormattedString());
            return;
        }

        // Set the flag indicating that mapped memory will be used
        JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceMapHost);
        doCuda(JCuda.cudaHostAlloc(pinnedPointer,elementSize() * length(),JCuda.cudaHostAllocMapped));
        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        Nd4j.getResourceManager().incrementCurrentAllocatedMemory(elementSize() * length());
        freed.set(false);

    }


    @Override
    public void set(Pointer pointer) {
        ensureNotFreed();


        if (dataType() == DOUBLE) {
            JCublas.cublasDcopy(
                    length(),
                    pointer,
                    1,
                    pointer(),
                    1
            );
        } else {
        	JCublas.cublasScopy(
                    length(),
                    pointer,
                    1,
                    pointer(),
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
        for(int i = 0; i < length(); i++) {
            to.put(i,getDouble(i));
        }
    }

 /*   @Override
    protected void finalize() throws Throwable {
        super.finalize();
        destroy();
    }
*/
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


        int offset = elementSize() * index;
        if (offset >= length() * elementSize())
            throw new IllegalArgumentException("Illegal offset " + offset + " with index of " + index + " and length " + length());
        JCublas.cublasSetVector(
                length
                , elementSize()
                , from
                , inc
                , pointer().withByteOffset(offset)
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
        set(0, buf.pointer());
    }
    protected ByteBuffer getBuffer() {
        return getBuffer(0);
    }

    protected java.nio.FloatBuffer getFloatBuffer(long offset) {
        if(pinnedPointer == null)
            throw new IllegalStateException("Pinned pointer uninitialized");

        ByteBuffer buf = pinnedPointer.getByteBuffer(offset * elementSize(),elementSize() * length() - offset * elementSize());
        if(buf == null)
            throw new IllegalStateException("Unable to obtain buffer: was null");
        // Set the byte order of the ByteBuffer
        buf.order(ByteOrder.nativeOrder());
        return buf.asFloatBuffer();
    }

    protected java.nio.FloatBuffer getFloatBuffer() {
        return getFloatBuffer(0);
    }

    protected java.nio.DoubleBuffer getDoubleBuffer(long offset) {
        if(pinnedPointer == null)
            throw new IllegalStateException("Pinned pointer uninitialized");
        ByteBuffer buf = pinnedPointer.getByteBuffer(offset * elementSize(),elementSize() * length() - offset * elementSize());

        if(buf == null)
            throw new IllegalStateException("Unable to obtain buffer: was null");
        // Set the byte order of the ByteBuffer
        buf.order(ByteOrder.nativeOrder());
        return buf.asDoubleBuffer();
    }

    protected java.nio.DoubleBuffer getDoubleBuffer() {
        return getDoubleBuffer(0);
    }

    protected ByteBuffer getBuffer(long offset) {
        ByteBuffer buf = pinnedPointer.getByteBuffer(offset * elementSize(),elementSize() * length() - offset * elementSize());
        // Set the byte order of the ByteBuffer
        buf.order(ByteOrder.nativeOrder());
        return buf;
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


    @Override
    public  void destroy() {
        try {
        	

            if(!freed.get()) {
                if (Nd4j.shouldInstrument)
                    Nd4j.getInstrumentation().log(this, Instrumentation.DESTROYED);

                
                
                JCuda.cudaFreeHost(pinnedPointer);
                freed.set(true);
                Nd4j.getResourceManager().decrementCurrentAllocatedMemory(elementSize() * length());
                references().clear();
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
    public int elementSize() {
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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        BaseCudaDataBuffer that = (BaseCudaDataBuffer) o;

        if (length != that.length) return false;
        if (elementSize != that.elementSize) return false;
        if (isPersist != that.isPersist) return false;
        for(int i = 0; i < length(); i++) {
            if(getDouble(i) != that.getDouble(i))
                return false;
        }

        return true;

    }

    @Override
    public int hashCode() {
        int result = pinnedPointer != null ? pinnedPointer.hashCode() : 0;
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
                            buff.length()
                            , buff.pointer().withByteOffset(buff.elementSize() * offsets[i])
                            , strides[i]
                            , pointer().withByteOffset(count * buff.elementSize())
                            , 1);
                    count += (buff.length() - 1 - offsets[i]) / strides[i] + 1;
                } else {
                    JCublas.cublasScopy(buff.length()
                            , buff.pointer().withByteOffset(buff.elementSize() * offsets[i])
                            , strides[i]
                            , pointer().withByteOffset(count * buff.elementSize())
                            , 1);
                    count += (buff.length() - 1 - offsets[i]) / strides[i] + 1;
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
        assign(offsets, strides, length(), buffers);
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
            pinnedPointer = buf.pinnedPointer;
        }
        else if(dataType() == DataBuffer.FLOAT) {
            float[] f = new float[length];
            for(int i = 0; i < f.length; i++)
                f[i] = stream.readFloat();
            BaseCudaDataBuffer  buf = (BaseCudaDataBuffer) KernelFunctions.alloc(f);
            pinnedPointer = buf.pinnedPointer;
        }
    }


}
