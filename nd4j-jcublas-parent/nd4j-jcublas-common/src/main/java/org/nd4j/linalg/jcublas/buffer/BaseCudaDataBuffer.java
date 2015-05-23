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

import io.netty.buffer.Unpooled;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.complex.CudaComplexConversion;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base class for a data buffer
 *
 * @author Adam Gibson
 */
public abstract class BaseCudaDataBuffer extends BaseDataBuffer implements JCudaBuffer {

    static AtomicLong allocated = new AtomicLong();
    static AtomicLong totalAllocated = new AtomicLong();
    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);
    protected transient Map<String,DevicePointerInfo> pointersToContexts = new ConcurrentHashMap<>();
    protected AtomicBoolean modified = new AtomicBoolean(false);
    protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    protected transient WeakReference<DataBuffer> ref;
    protected boolean isPersist = false;
    protected AtomicBoolean freed = new AtomicBoolean(false);
    private Pointer hostPointer;


    /**
     * Base constructor
     *
     * @param length      the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(int length, int elementSize) {
        super(length,elementSize);
    }


    @Override
    public AllocationMode allocationMode() {
        return allocationMode;
    }

    @Override
    public ByteBuffer getHostBuffer() {
        return dataBuffer.nioBuffer();
    }

    @Override
    public void setHostBuffer(ByteBuffer hostBuffer) {
        this.dataBuffer = Unpooled.wrappedBuffer(hostBuffer);
    }

    @Override
    public Pointer getHostPointer() {
        if(hostPointer == null) {
            hostPointer = Pointer.to(asNio());
        }
        return hostPointer;
    }

    @Override
    public Pointer getHostPointer(int offset) {
        if(hostPointer == null) {
            hostPointer = Pointer.to(dataBuffer.nioBuffer());
        }
        return hostPointer.withByteOffset(offset * getElementSize());
    }

    @Override
    public void persist() {
        isPersist = true;
    }

    @Override
    public boolean isPersist() {
        return isPersist;
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

        modified.set(true);
        if (dataType() == DataBuffer.Type.FLOAT) {
            JCublas2.cublasSetVector(
                    length(),
                    getElementSize(),
                    PointerUtil.getPointer(CudaComplexConversion.toComplex(result.asFloat()))
                    , 1
                    , getHostPointer()
                    , 1);
        }
        else {
            JCublas2.cublasSetVector(
                    length(),
                    getElementSize(),
                    PointerUtil.getPointer(CudaComplexConversion.toComplexDouble(result.asDouble()))
                    , 1
                    , getHostPointer()
                    , 1);
        }
    }




    @Override
    public Pointer getDevicePointer(int stride, int offset,int length) {
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName());

        if(devicePointerInfo == null) {
            int devicePointerLength = getElementSize() * length;
            allocated.addAndGet(devicePointerLength);
            totalAllocated.addAndGet(devicePointerLength);
            log.trace("Allocating {} bytes, total: {}, overall: {}", devicePointerLength, allocated.get(), totalAllocated);
            devicePointerInfo = (DevicePointerInfo)
                    ContextHolder.getInstance()
                            .getConf()
                            .getMemoryStrategy()
                            .alloc(this,stride,offset,length);

            pointersToContexts.put(Thread.currentThread().getName(), devicePointerInfo);
            freed.set(false);
        }

        return devicePointerInfo.getPointer();
    }


    @Override
    public void set(Pointer pointer) {

        modified.set(true);

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

        JCublas2.cublasSetVectorAsync(
                length
                , getElementSize()
                , from
                , inc
                , getHostPointer().withByteOffset(offset)
                , 1, ContextHolder.getInstance().getCudaStream());

        ContextHolder.syncStream();

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
    public boolean freeDevicePointer() {
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName());
        if (devicePointerInfo != null && !freed.get()) {
            allocated.addAndGet(-devicePointerInfo.getLength());
            log.trace("freeing {} bytes, total: {}", devicePointerInfo.getLength(), allocated.get());
            ContextHolder.getInstance().getMemoryStrategy().free(this);
            freed.set(true);
            pointersToContexts.remove(Thread.currentThread().getName());
            return true;


        }
        return false;
    }

    @Override
    public void copyToHost() {
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName());
        if (devicePointerInfo != null) {
            ContextHolder.syncStream();

            JCublas2.cublasGetVectorAsync(
                    (int) devicePointerInfo.getLength()
                    , getElementSize()
                    , devicePointerInfo.getPointer()
                    , 1
                    , getHostPointer(devicePointerInfo.getOffset())
                    , devicePointerInfo.getStride()
                    , ContextHolder.getInstance().getCudaStream());

            ContextHolder.syncStream();


        }

    }






    @Override
    public void flush() {
        throw new UnsupportedOperationException();
    }






    @Override
    public void destroy() {
        freeDevicePointer();
       dataBuffer.clear();

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
        } else if (dataType() == DataBuffer.Type.FLOAT) {
            float[] f = new float[length];
            for (int i = 0; i < f.length; i++)
                f[i] = stream.readFloat();
            BaseCudaDataBuffer buf = (BaseCudaDataBuffer) KernelFunctions.alloc(f);
            setHostBuffer(buf.getHostBuffer());
        }
    }



    @Override
    public Map<String, DevicePointerInfo> getPointersToContexts() {
        return pointersToContexts;
    }

    public void setPointersToContexts(Map<String, DevicePointerInfo> pointersToContexts) {
        this.pointersToContexts = pointersToContexts;
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

    /**
     * Provides information about a device pointer
     *
     * @author bam4d
     */
    public static class DevicePointerInfo {
        final private Pointer pointer;
        final private long length;
        final private int stride;
        final private int offset;
        private boolean freed = false;

        public DevicePointerInfo(Pointer pointer, long length,int stride,int offset) {
            this.pointer = pointer;
            this.length = length;
            this.stride = stride;
            this.offset = offset;
        }

        public boolean isFreed() {
            return freed;
        }

        public void setFreed(boolean freed) {
            this.freed = freed;
        }

        public int getOffset() {
            return offset;
        }



        public int getStride() {
            return stride;
        }

        public Pointer getPointer() {
            return pointer;
        }

        public long getLength() {
            return length;
        }
    }
}
