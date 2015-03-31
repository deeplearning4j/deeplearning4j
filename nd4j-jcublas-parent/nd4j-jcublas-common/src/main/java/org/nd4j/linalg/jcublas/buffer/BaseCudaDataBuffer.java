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

import jcuda.Pointer;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.instrumentation.Instrumentation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.complex.CudaComplexConversion;

import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Base class for a data buffer
 *
 * @author Adam Gibson
 */
public abstract class BaseCudaDataBuffer implements JCudaBuffer {

    protected transient Pointer pointer;
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
        if (pointer() == null)
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
    public Pointer pointer() {
        ensureNotFreed();
        return pointer;
    }

    @Override
    public void alloc() {
        pointer = new Pointer();
        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        Nd4j.getResourceManager().incrementCurrentAllocatedMemory(elementSize() * length());
        //allocate memory for the pointer
        JCuda.cudaMalloc(pointer(), elementSize() * length());

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

        if (to.dataType() != dataType())
            throw new IllegalArgumentException("Unable to copy buffer, mis matching data types.");
        JCuda.cudaMemcpy(to.pointer(), pointer(), length() * elementSize(), cudaMemcpyKind.cudaMemcpyDeviceToDevice);

    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
       destroy();
    }

    @Override
    public void assign(Number value) {
        assign(value, 0);
    }


    /**
     * Get element with the specified index
     *
     * @param index  the index of the element to get
     * @param inc    the increment step when getting data
     * @param length the length to iterate for
     * @param init   the initialized pointer
     */
    protected void get(int index, int inc, int length, Pointer init) {
        ensureNotFreed();

        //cuda memcpy is a little more stable
        if (inc == 1) {
            JCuda.cudaMemcpy(init, pointer().withByteOffset(elementSize() * index), length * elementSize(), cudaMemcpyKind.cudaMemcpyDeviceToHost);
        } else {
            JCublas.cublasGetVector(
                    length
                    , elementSize(),
                    pointer().withByteOffset(index * elementSize())
                    ,
                    inc,
                    init
                    , 1);

        }

    }

    /**
     * Get element with the specified index
     *
     * @param index the index of the element to get
     * @param init  the initialized pointer
     */
    protected void get(int index, int length, Pointer init) {
        get(index, 1, length, init);
    }

    /**
     * Get element with the specified index
     *
     * @param index the index of the element to get
     * @param init  the initialized pointer
     */
    protected void get(int index, Pointer init) {
        get(index, 1, init);
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
                JCuda.cudaFree(pointer);
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
    public float[] asFloat() {
        return new float[0];
    }

    @Override
    public double[] asDouble() {
        return new double[0];
    }

    @Override
    public int[] asInt() {
        return new int[0];
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
        if (!(o instanceof BaseCudaDataBuffer)) return false;

        BaseCudaDataBuffer that = (BaseCudaDataBuffer) o;

        if (elementSize != that.elementSize) return false;
        if (length != that.length) return false;
        if (dataType() != that.dataType()) return false;
        if (dataType() == DataBuffer.DOUBLE) {
            double[] data = asDouble();
            double[] other = that.asDouble();
            if (!Arrays.equals(data, other))
                return false;
        } else if (dataType() == DataBuffer.FLOAT) {
            float[] data = asFloat();
            float[] other = that.asFloat();
            if (!Arrays.equals(data, other))
                return false;

        }

        return true;
    }

    @Override
    public int hashCode() {
        int result = pointer != null ? pointer.hashCode() : 0;
        result = 31 * result + length;
        result = 31 * result + elementSize;
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


}
