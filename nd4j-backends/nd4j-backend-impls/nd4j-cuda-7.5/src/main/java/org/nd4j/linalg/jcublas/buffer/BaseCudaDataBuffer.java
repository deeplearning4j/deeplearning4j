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

import com.google.common.collect.Table;
import io.netty.buffer.ByteBuf;

import jcuda.Pointer;
import jcuda.jcublas.JCublas2;
import org.apache.commons.lang3.tuple.Triple;
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

    private static AtomicAllocator allocator = AtomicAllocator.getInstance();

    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);

    public BaseCudaDataBuffer() {

    }

    public BaseCudaDataBuffer(ByteBuf buf, int length) {
        super(buf, length);
    }

    public BaseCudaDataBuffer(ByteBuf buf, int length, int offset) {
        super(buf, length, offset);
    }

    public BaseCudaDataBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public BaseCudaDataBuffer(float[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public BaseCudaDataBuffer(int[] data, boolean copy) {
        super(data, copy);

    }

    public BaseCudaDataBuffer(int[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    /**
     * Base constructor
     *
     * @param length      the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(int length, int elementSize) {
        super(length,elementSize);

    }

    public BaseCudaDataBuffer(int length, int elementSize, int offset) {
        super(length, elementSize, offset);
    }

    public BaseCudaDataBuffer(DataBuffer underlyingBuffer, int length, int offset) {
        super(underlyingBuffer, length, offset);
    }

    public BaseCudaDataBuffer(int length) {
        super(length);

    }

    public BaseCudaDataBuffer(float[] data) {
        super(data);

    }

    public BaseCudaDataBuffer(int[] data) {
        super(data);

    }

    public BaseCudaDataBuffer(double[] data) {
        super(data);

    }

    public BaseCudaDataBuffer(byte[] data, int length) {
        super(data,length);
    }

    public BaseCudaDataBuffer(ByteBuffer buffer, int length) {
        super(buffer,length);
    }

    public BaseCudaDataBuffer(ByteBuffer buffer, int length, int offset) {
        super(buffer, length, offset);
    }

    @Override
    protected void setNioBuffer() {
        wrappedBuffer = ByteBuffer.allocateDirect(elementSize * length);
        wrappedBuffer.order(ByteOrder.nativeOrder());
    }

    @Override
    @Deprecated
    public void copyAtStride(DataBuffer buf, int n, int stride, int yStride, int offset, int yOffset) {
        super.copyAtStride(buf, n, stride, yStride, offset, yOffset);
    }

    @Override
    @Deprecated
    public boolean copied(String name) {
        throw new UnsupportedOperationException("Not supported atm");
    }

    @Override
    @Deprecated
    public void setCopied(String name) {
        throw new UnsupportedOperationException("Not supported atm");
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
    public void setHostBuffer(ByteBuffer hostBuffer) {
        this.wrappedBuffer = hostBuffer;
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




    @Override
    @Deprecated
    public Pointer getDevicePointer(int stride, int offset,int length) {
        throw new UnsupportedOperationException("getPointer(stride, offset, length) shouldn't be used anymore");
    }



    @Deprecated
    public Pointer getHostPointer(INDArray arr,int stride, int offset,int length) {
        return null;
    }

    @Override
    @Deprecated
    public Pointer getDevicePointer(INDArray arr,int stride, int offset,int length) {
        throw new UnsupportedOperationException("getPointer(INDArray, stride, offset, length) shouldn't be used");
    }

    @Override
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


    @Deprecated
    private void copyOneElement(int i,double val) {
        throw new UnsupportedOperationException("copyOneElement() isn't supported atm");
        /*
        if(pointersToContexts != null)
            for(DevicePointerInfo info : pointersToContexts.values()) {
                if(dataType() == Type.FLOAT)
                    JCublas2.cublasSetVector(1,getElementSize(),Pointer.to(new float[]{(float) val}),1,info.getPointers().getDevicePointer().withByteOffset(getElementSize() * i),1);
                else
                    JCublas2.cublasSetVector(1,getElementSize(), Pointer.to(new double[]{val}),1,info.getPointers().getDevicePointer().withByteOffset(getElementSize() * i),1);

            }
         */
    }


    @Override
    public void put(int i, float element) {
        super.put(i, element);
        copyOneElement(i, element);
    }

    @Override
    public void put(int i, double element) {
        super.put(i, element);
        //        copyOneElement(i, element);
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
    protected void set(int index, int length, Pointer from) {
        set(index, length, from, 1);
    }

    @Override
    public void assign(DataBuffer data) {
        JCudaBuffer buf = (JCudaBuffer) data;
        set(0, buf.getHostPointer());
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

    @Override
    @Deprecated
    public boolean freeDevicePointer(int offset, int length, int stride) {
        /*
            actually this method should do nothing, since memory deallocation is handled with Allocator implementations

         */
        return true;
    }

    @Override
    @Deprecated
    public synchronized void copyToHost(CudaContext context, int offset, int length, int stride) {
        throw new UnsupportedOperationException("copyToHost() isn't supported atm");
        /*
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,length,stride));
        if(devicePointerInfo == null)
            throw new IllegalStateException("No pointer found for offset " + offset);
        //prevent inconsistent pointers
        if (devicePointerInfo.getOffset() != offset)
            throw new IllegalStateException("Device pointer offset didn't match specified offset in pointer map");

        if (devicePointerInfo != null) {
            ContextHolder.getInstance().getMemoryStrategy().copyToHost(this,offset,stride,length,null,offset,stride);
        }

        else
            throw new IllegalStateException("No offset found to copy");
        //synchronize for the copy to avoid data inconsistencies
        context.syncOldStream();
        */
    }

    @Override
    @Deprecated
    public synchronized  void copyToHost(int offset,int length) {
        throw new UnsupportedOperationException("copyToHost() isn't supported atm");
        /*
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,length,1));
        if(devicePointerInfo == null)
            throw new IllegalStateException("No pointer found for offset " + offset);
        //prevent inconsistent pointers
        if (devicePointerInfo.getOffset() != offset)
            throw new IllegalStateException("Device pointer offset didn't match specified offset in pointer map");

        if (devicePointerInfo != null) {
            int deviceStride = devicePointerInfo.getStride();
            int  deviceOffset = devicePointerInfo.getOffset();
            int deviceLength = (int) devicePointerInfo.getLength();
            if(deviceOffset == 0 && length < length()) {
                ContextHolder.getInstance().getMemoryStrategy().copyToHost(this,offset,deviceStride, deviceLength,null, deviceOffset,deviceStride);
            }
            else {
                ContextHolder.getInstance().getMemoryStrategy().copyToHost(this,offset,deviceStride,deviceLength,null, deviceOffset,deviceStride);
            }

        }
        */
    }






    @Override
    public void flush() {
        throw new UnsupportedOperationException();
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
        /*
        copied = new HashMap<>();
        pointersToContexts = HashBasedTable.create();
        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        freed = new AtomicBoolean(false);
        */
    }





    @Override
    @Deprecated
    public Table<String, Triple<Integer, Integer, Integer>, DevicePointerInfo> getPointersToContexts() {
        throw new UnsupportedOperationException("getPointersToContext() isn't supported atm");
    }

    @Deprecated
    public void setPointersToContexts( Table<String, Triple<Integer, Integer, Integer>, DevicePointerInfo> pointersToContexts) {
        //
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