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

package org.nd4j.linalg.jcublas.buffer;

import io.netty.buffer.ByteBuf;

import io.netty.buffer.Unpooled;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.*;
import java.util.*;

/**
 * Base class for a data buffer
 *
 * CUDA implementation for DataBuffer always uses JavaCPP as allocationMode, and device access is masked by appropriate allocator mover implementation.
 *
 * Memory allocation/deallocation is strictly handled by allocator, since JavaCPP alloc/dealloc has nothing to do with CUDA. But besides that, host pointers obtained from CUDA are 100% compatible with CPU
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
public abstract class BaseCudaDataBuffer extends BaseDataBuffer implements JCudaBuffer {

    private AllocationPoint allocationPoint;

    private static AtomicAllocator allocator = AtomicAllocator.getInstance();

    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);

    public BaseCudaDataBuffer() {
        throw new UnsupportedOperationException("You can't instantiate undefined buffer");
    }

    public BaseCudaDataBuffer(ByteBuf buf, int length) {
        super(buf, length);
        // TODO: to be implemented, using ByteBuf.memoryAddress() and memcpy
    }

    public BaseCudaDataBuffer(ByteBuf buf, int length, int offset) {
        super(buf, length, offset);
        // TODO: to be implemented, using ByteBuf.memoryAddress() and memcpy
    }

    public BaseCudaDataBuffer(float[] data, boolean copy) {
        //super(data, copy);
        this(data, copy, 0);
    }

    public BaseCudaDataBuffer(float[] data, boolean copy, int offset) {
        this(data.length - offset, 4);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, 0, offset);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy) {
        this(data, copy, 0);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy, int offset) {
        this(data.length - offset, 8);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, 0, offset);
    }

    public BaseCudaDataBuffer(int[] data, boolean copy) {
        this(data, copy, 0);
    }

    public BaseCudaDataBuffer(int[] data, boolean copy, int offset) {
        this(data.length - offset, 4);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, 0, offset);
    }

    /**
     * Base constructor. It's used within all constructors internally
     *
     * @param length      the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(int length, int elementSize) {
        allocationPoint = AtomicAllocator.getInstance().allocateMemory(new AllocationShape(length, elementSize));
        this.length = length;
        this.offset = 0;
        this.originalOffset = 0;
        allocationPoint.attachBuffer(this);
        this.elementSize = elementSize;
        this.trackingPoint = allocationPoint.getObjectId();

        // TODO: probably this one could be reconsidered
        AtomicAllocator.getInstance().pickupSpan(allocationPoint);
    }

    public BaseCudaDataBuffer(int length, int elementSize, int offset) {
        this(length, elementSize);
        this.offset = offset;
        this.originalOffset = offset;
    }

    public BaseCudaDataBuffer(DataBuffer underlyingBuffer, int length, int offset) {
        //this(length, underlyingBuffer.getElementSize(), offset);
        this.wrappedDataBuffer = underlyingBuffer;
        this.originalBuffer = underlyingBuffer.originalDataBuffer();
        this.length = length;
        this.offset = offset;
        this.originalOffset = underlyingBuffer.originalOffset() + offset;

        // TODO: make sure we're getting pointer with offset at allocator
    }

    public BaseCudaDataBuffer(int length) {
        this(length, Sizeof.FLOAT);
    }

    public BaseCudaDataBuffer(float[] data) {
        //super(data);
        this(data.length, Sizeof.FLOAT);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(int[] data) {
        //super(data);
        this(data.length, Sizeof.INT);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(double[] data) {
       // super(data);
        this(data.length, Sizeof.DOUBLE);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(byte[] data, int length) {
        this(Unpooled.wrappedBuffer(data),length);
    }

    public BaseCudaDataBuffer(ByteBuffer buffer, int length) {
        super(buffer,length);
    }

    public BaseCudaDataBuffer(ByteBuffer buffer, int length, int offset) {
        super(buffer, length, offset);
    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(int[] data, int length, int srcOffset, int dstOffset) {
        // TODO: make sure getPointer returns proper pointer
        Pointer dstPtr = dstOffset > 0 ? new Pointer(allocator.getPointer(this).address()).withByteOffset(dstOffset * 4 ) : new Pointer(allocator.getPointer(this).address());
        Pointer srcPtr = srcOffset > 0 ? Pointer.to(data).withByteOffset(srcOffset * 4) : Pointer.to(data);
        memcpyAsync(dstPtr, srcPtr, length * 4);
    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(float[] data, int length, int srcOffset, int dstOffset) {
        // TODO: make sure getPointer returns proper pointer
        Pointer dstPtr = dstOffset > 0 ? new Pointer(allocator.getPointer(this).address()).withByteOffset(dstOffset * 4 ) : new Pointer(allocator.getPointer(this).address());
        Pointer srcPtr = srcOffset > 0 ? Pointer.to(data).withByteOffset(srcOffset * 4) : Pointer.to(data);
        memcpyAsync(dstPtr, srcPtr, length * 4);
    }

    @Override
    public void setData(int[] data) {
        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(float[] data) {
        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(double[] data) {
        set(data, data.length, 0, 0);
    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(double[] data, int length, int srcOffset, int dstOffset) {
        // TODO: make sure getPointer returns proper pointer
        Pointer dstPtr = dstOffset > 0 ? new Pointer(allocator.getPointer(this).address()).withByteOffset(dstOffset * 8 ) : new Pointer(allocator.getPointer(this).address());
        Pointer srcPtr = srcOffset > 0 ? Pointer.to(data).withByteOffset(srcOffset * 8) : Pointer.to(data);
        memcpyAsync(dstPtr, srcPtr, length * 8);
    }

    public void memcpyBlocking(Pointer dstPtr, Pointer srcPtr, long byteLen) {
        CudaContext context = allocator.getCudaContext();

        JCuda.cudaMemcpyAsync(
                dstPtr,
                srcPtr,
                byteLen,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
                context.getOldStream()
        );

        context.syncOldStream();
    }

    public void memcpyAsync(Pointer dstPtr, Pointer srcPtr, long byteLen) {
        CudaContext context = allocator.getCudaContext();

        JCuda.cudaMemcpyAsync(
                dstPtr,
                srcPtr,
                byteLen,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
                context.getOldStream()
        );

        // TODO: remove sync later
        context.syncOldStream();
    }

    @Override
    protected void setNioBuffer() {
        throw new UnsupportedOperationException("setNioBuffer() is not supported for CUDA backend");
        /*
        wrappedBuffer = ByteBuffer.allocateDirect(elementSize * length);
        wrappedBuffer.order(ByteOrder.nativeOrder());
        */
    }

    @Override
    @Deprecated
    public void copyAtStride(DataBuffer buf, int n, int stride, int yStride, int offset, int yOffset) {
        super.copyAtStride(buf, n, stride, yStride, offset, yOffset);
    }
    @Override
    public AllocationMode allocationMode() {
        return allocationMode;
    }

    @Override
    public ByteBuffer getHostBuffer() {
        return wrappedBuffer;
    }

    @Override
    public Pointer getHostPointer() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pointer getHostPointer(int offset) {
        throw new UnsupportedOperationException();
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
    public int getElementSize() {
        return elementSize;
    }


    @Override
    public void addReferencing(String id) {
        referencing.add(id);
    }

    @Override
    public void put(int i, IComplexNumber result) {
        throw new UnsupportedOperationException("ComplexNumbers are not supported yet");
        /*
        modified.set(true);
        if (dataType() == DataBuffer.Type.FLOAT) {
            JCublas2.cublasSetVector(
                    (int) length(),
                    getElementSize(),
                    PointerUtil.getPointer(CudaComplexConversion.toComplex(result.asFloat()))
                    , 1
                    , getHostPointer()
                    , 1);
        }
        else {
            JCublas2.cublasSetVector(
                    (int) length(),
                    getElementSize(),
                    PointerUtil.getPointer(CudaComplexConversion.toComplexDouble(result.asDouble()))
                    , 1
                    , getHostPointer()
                    , 1);
        }
        */
    }


    @Deprecated
    public Pointer getHostPointer(INDArray arr,int stride, int offset,int length) {
        throw new UnsupportedOperationException("This method is deprecated");
    }

    @Deprecated
    public void set(Pointer pointer) {
        throw new UnsupportedOperationException("set(Pointer) is not supported");
        //modified.set(true);

        /*
        if (dataType() == DataBuffer.Type.DOUBLE) {
            JCublas2.cublasDcopy(
                    ContextHolder.getInstance().getHandle(),
                    length(),
                    pointer,
                    1,
                    getHostPointer(),
                    1
            );
        } else {
            JCublas2.cublasScopy(
                    ContextHolder.getInstance().getHandle(),
                    length(),
                    pointer,
                    1,
                    getHostPointer(),
                    1
            );
        }
        */
    }

    @Override
    public void put(int i, float element) {
        float[] tmp = new float[] {element};
        Pointer dstPtr = i > 0 ? new Pointer(allocator.getPointer(this).address()).withByteOffset(i * 4 ) : new Pointer(allocator.getPointer(this).address());
        Pointer srcPtr = Pointer.to(tmp);
        memcpyAsync(dstPtr, srcPtr, length * 4);
    }

    @Override
    public void put(int i, double element) {
        double[] tmp = new double[] {element};
        Pointer dstPtr = i > 0 ? new Pointer(allocator.getPointer(this).address()).withByteOffset(i * 4 ) : new Pointer(allocator.getPointer(this).address());
        Pointer srcPtr = Pointer.to(tmp);
        memcpyAsync(dstPtr, srcPtr, length * 4);
    }

    @Override
    public void put(int i, int element) {
        int[] tmp = new int[] {element};
        Pointer dstPtr = i > 0 ? new Pointer(allocator.getPointer(this).address()).withByteOffset(i * 4 ) : new Pointer(allocator.getPointer(this).address());
        Pointer srcPtr = Pointer.to(tmp);
        memcpyAsync(dstPtr, srcPtr, length * 4);
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
        return dataType() == Type.FLOAT ? getComplexFloat(i) : getComplexDouble(i);
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(int index, int length, Pointer from, int inc) {


        int offset = getElementSize() * index;
        if (offset >= length() * getElementSize())
            throw new IllegalArgumentException("Illegal offset " + offset + " with index of " + index + " and length " + length());

        JCublas2.cublasSetVectorAsync(
                length
                , getElementSize()
                , from
                , inc
                , getHostPointer().withByteOffset(offset)
                , 1, ContextHolder.getInstance().getCudaStream());
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(int index, int length, Pointer from) {
        set(index, length, from, 1);
    }

    @Override
    public void assign(DataBuffer data) {
        /*JCudaBuffer buf = (JCudaBuffer) data;
        set(0, buf.getHostPointer());
        */
        memcpyAsync(
                new Pointer(allocator.getPointer(this).address()),
                new Pointer(allocator.getPointer(data).address()),
                data.length()
        );
    }





    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(int index, Pointer from) {
        set(index, 1, from);
    }

    @Override
    public void flush() {
        //
    }


    @Override
    public void destroy() {
    }

    private void writeObject(java.io.ObjectOutputStream stream)
            throws IOException {
        stream.defaultWriteObject();
        write(stream);
    }

    private void readObject(java.io.ObjectInputStream stream)
            throws IOException, ClassNotFoundException {
        doReadObject(stream);
        // TODO: to be implemented
        /*
        copied = new HashMap<>();
        pointersToContexts = HashBasedTable.create();
        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        freed = new AtomicBoolean(false);
        */
    }

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("[");
        for(int i = 0; i < length(); i++) {
            sb.append(getDouble(i));
            if(i < length() - 1)
                sb.append(",");
        }
        sb.append("]");
        return sb.toString();

    }

    @Override
    public boolean sameUnderlyingData(DataBuffer buffer) {
        if(allocationMode() != buffer.allocationMode())
            return false;
        if(allocationMode() == AllocationMode.HEAP) {
            return array() == buffer.array();
        }
        else if(allocationMode() == AllocationMode.JAVACPP)
            return pointer() == buffer.pointer();
        else {
            return buffer.originalDataBuffer() == originalDataBuffer();
        }
    }

    /**
     * PLEASE NOTE: this method implies STRICT equality only.
     * I.e: this == object
     *
     * @param o
     * @return
     */
    @Override
    public boolean equals(Object o) {
        if (o == null) return false;
        if (this == o) return true;

        return false;
    }

    @Override
    public byte[] asBytes() {
        allocator.synchronizeHostData(this);
        return super.asBytes();
    }

    @Override
    public double[] asDouble() {
        allocator.synchronizeHostData(this);
        return super.asDouble();
    }

    @Override
    public float[] asFloat() {
        allocator.synchronizeHostData(this);
        return super.asFloat();
    }

    @Override
    public int[] asInt() {
        allocator.synchronizeHostData(this);
        return super.asInt();
    }

    @Override
    public ByteBuf asNetty() {
        allocator.trySynchronizeHostData(this);
        return super.asNetty();
    }

    @Override
    public ByteBuffer asNio() {
        allocator.trySynchronizeHostData(this);
        return super.asNio();
    }

    @Override
    public DoubleBuffer asNioDouble() {
        allocator.trySynchronizeHostData(this);
        return super.asNioDouble();
    }

    @Override
    public FloatBuffer asNioFloat() {
        allocator.trySynchronizeHostData(this);
        return super.asNioFloat();
    }

    @Override
    public IntBuffer asNioInt() {
        allocator.trySynchronizeHostData(this);
        return super.asNioInt();
    }

    @Override
    public DataBuffer dup() {
        allocator.synchronizeHostData(this);
        return super.dup();
    }

    @Override
    public Number getNumber(int i) {
        allocator.synchronizeHostData(this);
        return super.getNumber(i);
    }

    @Override
    public double getDouble(int i) {
        allocator.synchronizeHostData(this);
        return super.getDouble(i);
    }

    @Override
    public double[] getDoublesAt(int offset, int inc, int length) {
        allocator.synchronizeHostData(this);
        return super.getDoublesAt(offset, inc, length);
    }

    @Override
    public double[] getDoublesAt(int offset, int length) {
        allocator.synchronizeHostData(this);
        return super.getDoublesAt(offset, length);
    }

    @Override
    public float getFloat(int i) {
        allocator.synchronizeHostData(this);
        return super.getFloat(i);
    }

    @Override
    public float[] getFloatsAt(int offset, int inc, int length) {
        allocator.synchronizeHostData(this);
        return super.getFloatsAt(offset, inc, length);
    }

    @Override
    public float[] getFloatsAt(int offset, int length) {
        allocator.synchronizeHostData(this);
        return super.getFloatsAt(offset, length);
    }

    @Override
    public int getInt(int ix) {
        allocator.synchronizeHostData(this);
        return super.getInt(ix);
    }
}