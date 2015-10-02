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

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import jcuda.Pointer;
import jcuda.jcublas.JCublas2;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.complex.CudaComplexConversion;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
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
    /**
     * Pointers to contexts covers this buffer on the gpu at offset 0
     * for each thread.
     *
     * The column key is for offsets. If we only have buffer one device allocation per thread
     * we will clobber anything that is already allocated on the gpu.
     *
     * This also allows us to make a simplifying assumption about how to allocate the data as follows:
     *
     * Always allocate for offset zero by default. This allows us to reuse the same pointer with an offset
     * for each extra allocations (say for row wise operations)
     *
     * This also prevents duplicate uploads to the gpu.
     */
    protected transient Table<String,Triple<Integer,Integer,Integer>,DevicePointerInfo> pointersToContexts = HashBasedTable.create();
    protected AtomicBoolean modified = new AtomicBoolean(false);
    protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    protected transient WeakReference<DataBuffer> ref;
    protected AtomicBoolean freed = new AtomicBoolean(false);
    private transient Pointer hostPointer;
    private Map<String,Boolean> copied = new ConcurrentHashMap<>();

    public BaseCudaDataBuffer(ByteBuf buf, int length) {
        super(buf, length);
    }

    public BaseCudaDataBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public BaseCudaDataBuffer(int[] data, boolean copy) {
        super(data, copy);
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

    @Override
    public boolean copied(String name) {
        Boolean copied = this.copied.get(name);
        if(copied == null)
            return false;
        return this.copied.get(name);
    }

    @Override
    public void setCopied(String name) {
        copied.put(name, true);
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
            hostPointer = Pointer.to(asNio());
        }
        return hostPointer.withByteOffset(offset * getElementSize());
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
        String name = Thread.currentThread().getName();
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(name,Triple.of(offset,length,1));
        if(devicePointerInfo == null) {
            int devicePointerLength = getElementSize() * length;
            allocated.addAndGet(devicePointerLength);
            totalAllocated.addAndGet(devicePointerLength);
            log.trace("Allocating {} bytes, total: {}, overall: {}", devicePointerLength, allocated.get(), totalAllocated);
            if(devicePointerInfo == null) {
                /**
                 * Add zero first no matter what.
                 * Allocate the whole buffer on the gpu
                 * and use offsets for any other pointers that come in.
                 * This will allow us to set device pointers with offsets
                 *
                 * with no extra allocation.
                 *
                 * Notice here we ignore the length of the actual array.
                 *
                 * We are going to allocate the whole buffer on the gpu only once.
                 *
                 */

                if(!pointersToContexts.contains(name,Triple.of(0,this.length,1))) {
                    devicePointerInfo = (DevicePointerInfo)
                            ContextHolder.getInstance()
                                    .getConf()
                                    .getMemoryStrategy()
                                    .alloc(this, 1, 0, this.length);

                    pointersToContexts.put(name, Triple.of(0, this.length, 1), devicePointerInfo);
                }

                if(offset > 0) {
                    /**
                     * Store the length for the offset of the pointer.
                     * Return the original pointer with an offset
                     * (these pointers can't be reused?)
                     *
                     * With the device pointer info,
                     * we want to store the original pointer.
                     * When retrieving the vector from the gpu later,
                     * we will use the recorded offset.
                     *
                     * Due to gpu instability (please correct me if I'm wrong here)
                     * we can't seem to reuse the pointers with the offset specified,
                     * therefore it is desirable to recreate this pointer later.
                     *
                     * This will prevent extra allocation as well
                     * as inform the length for retrieving data from the gpu
                     * for this particular offset and buffer.
                     *
                     */
                    Pointer zero = pointersToContexts.get(name,Triple.of(0,length,1)).getPointer();
                    Pointer ret = zero.withByteOffset(offset * getElementSize());
                    devicePointerInfo = new DevicePointerInfo(zero,length,stride,offset);
                    pointersToContexts.put(name, Triple.of(offset, length, 1), devicePointerInfo);
                    return ret;

                }



            }


            freed.set(false);
        }

        /**
         * Return the device pointer with the specified offset.
         * Regardless of whether the device pointer has been allocated,
         * we need to return with it respect to the specified array
         * not the array's underlying buffer.
         */
        return devicePointerInfo.getPointer().withByteOffset(offset * getElementSize());
    }




    @Override
    public Pointer getDevicePointer(INDArray arr,int stride, int offset,int length) {
        String name = Thread.currentThread().getName();
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(name,Triple.of(offset, length, stride));
        if(devicePointerInfo == null) {
            int devicePointerLength = getElementSize() * length;
            allocated.addAndGet(devicePointerLength);
            totalAllocated.addAndGet(devicePointerLength);
            log.trace("Allocating {} bytes, total: {}, overall: {}", devicePointerLength, allocated.get(), totalAllocated);
            //check its the same object
            if(arr.data() != this) {
                throw new IllegalArgumentException("Unable to get pointer for array that doesn't have this as the buffer");
            }
            int compareLength = arr instanceof IComplexNDArray ? arr.length() * 2 : arr.length();
            /**
             * Add zero first no matter what.
             * Allocate the whole buffer on the gpu
             * and use offsets for any other pointers that come in.
             * This will allow us to set device pointers with offsets
             *
             * with no extra allocation.
             *
             * Notice here we ignore the length of the actual array.
             *
             * We are going to allocate the whole buffer on the gpu only once.
             *
             */
            if(!pointersToContexts.contains(name,Triple.of(0, this.length, 1))) {
                devicePointerInfo = (DevicePointerInfo)
                        ContextHolder.getInstance()
                                .getConf()
                                .getMemoryStrategy()
                                .alloc(this, 1, 0, this.length);

                pointersToContexts.put(name, Triple.of(0,this.length,1), devicePointerInfo);
            }


            if(offset > 0) {
                /**
                 * Store the length for the offset of the pointer.
                 * Return the original pointer with an offset
                 * (these pointers can't be reused?)
                 *
                 * With the device pointer info,
                 * we want to store the original pointer.
                 * When retrieving the vector from the gpu later,
                 * we will use the recorded offset.
                 *
                 * Due to gpu instability (please correct me if I'm wrong here)
                 * we can't seem to reuse the pointers with the offset specified,
                 * therefore it is desirable to recreate this pointer later.
                 *
                 * This will prevent extra allocation as well
                 * as inform the length for retrieving data from the gpu
                 * for this particular offset and buffer.
                 *
                 */
                DevicePointerInfo info2 = pointersToContexts.get(name, Triple.of(0, this.length, 1));
                if(info2 == null)
                    throw new IllegalStateException("No pointer found for name " + name + " and offset/length " + offset + " / " + length);
                Pointer zero = info2.getPointer();
                Pointer ret =  info2.getPointer().withByteOffset(offset * getElementSize());
                devicePointerInfo = new DevicePointerInfo(zero,length,stride,offset);
                pointersToContexts.put(name, Triple.of(offset, compareLength, stride), devicePointerInfo);
                return ret;

            }

            else if(offset == 0 && compareLength < arr.data().length()) {
                DevicePointerInfo info2 = pointersToContexts.get(name, Triple.of(0, this.length, 1));
                DevicePointerInfo info3 = new DevicePointerInfo(info2.getPointer(),this.length, BlasBufferUtil.getBlasStride(arr),arr.offset());
                int compareLength2 = arr instanceof IComplexNDArray ? arr.length() * 2 : arr.length();

                /**
                 * Need a pointer that
                 * points at the buffer but doesnt extend all the way to the end.
                 * This is for data like the first row of a matrix
                 * that has zero offset but does not extend all the way to the end of the buffer.
                 */

                pointersToContexts.put(name, Triple.of(offset, compareLength2, BlasBufferUtil.getBlasStride(arr)), info3);
                return info3.getPointer();
            }






            freed.set(false);
        }

        /**
         * Return the device pointer with the specified offset.
         * Regardless of whether the device pointer has been allocated,
         * we need to return with it respect to the specified array
         * not the array's underlying buffer.
         */
        if(devicePointerInfo == null && offset == 0 && length < length()) {
            DevicePointerInfo origin = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(0, length(), stride));
            DevicePointerInfo newInfo = new DevicePointerInfo(origin.getPointer(),length,stride,0);
            return newInfo.getPointer();
        }


        return devicePointerInfo.getPointer().withByteOffset(offset * getElementSize());
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
    public boolean freeDevicePointer(int offset, int length) {
        String name = Thread.currentThread().getName();
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(name,Triple.of(offset,length,1));

        //nothing to free, there was no copy. Only the gpu pointer was reused with a different offset.
        if(offset != 0)
            pointersToContexts.remove(name,Triple.of(offset,length,1));
        else if(offset == 0 && isPersist) {
            return true;
        }
        else if (devicePointerInfo != null && !freed.get()) {
            allocated.addAndGet(-devicePointerInfo.getLength());
            log.trace("freeing {} bytes, total: {}", devicePointerInfo.getLength(), allocated.get());
            ContextHolder.getInstance().getMemoryStrategy().free(this,offset,length);
            freed.set(true);
            copied.remove(name);
            pointersToContexts.remove(name,Triple.of(offset,length,1));
            return true;


        }

        return false;
    }

    @Override
    public void copyToHost(int offset,int length) {
        DevicePointerInfo devicePointerInfo = pointersToContexts.get(Thread.currentThread().getName(),Triple.of(offset,length,1));
        if(devicePointerInfo == null)
            throw new IllegalStateException("No pointer found for offset " + offset);
        //prevent inconsistent pointers
        if (devicePointerInfo.getOffset() != offset)
            throw new IllegalStateException("Device pointer offset didn't match specified offset in pointer map");

        if (devicePointerInfo != null) {
            ContextHolder.syncStream();
            int deviceStride = devicePointerInfo.getStride();
            int  deviceOffset = devicePointerInfo.getOffset();
            long deviceLength = devicePointerInfo.getLength();
            if(deviceOffset == 0 && length < length()) {
                /**
                 * The way the data works out the stride for retrieving the data
                 * should be 1.
                 *
                 * The device stride should be used for resetting the data.
                 *
                 * This is for the edge case where the offset is zero and
                 * the length of the pointer is < the actual buffer length itself.
                 *
                 */
                JCublas2.cublasGetVectorAsync(
                        length
                        , getElementSize()
                        , devicePointerInfo.getPointer().withByteOffset(offset * getElementSize())
                        , deviceStride
                        , getHostPointer(deviceOffset)
                        , deviceStride
                        , ContextHolder.getInstance().getCudaStream());
            }
            else {
                JCublas2.cublasGetVectorAsync(
                        (int) deviceLength
                        , getElementSize()
                        , devicePointerInfo.getPointer().withByteOffset(offset * getElementSize())
                        , deviceStride
                        , getHostPointer(deviceOffset)
                        , deviceStride
                        , ContextHolder.getInstance().getCudaStream());
            }


            ContextHolder.syncStream();


        }

        else
            throw new IllegalStateException("No offset found to copy");

    }






    @Override
    public void flush() {
        throw new UnsupportedOperationException();
    }






    @Override
    public void destroy() {
        dataBuffer.clear();
    }

    private void writeObject(java.io.ObjectOutputStream stream)
            throws IOException {
        stream.defaultWriteObject();
        write(stream);

    }

    private void readObject(java.io.ObjectInputStream stream)
            throws IOException, ClassNotFoundException {
        doReadObject(stream);
        copied = new HashMap<>();
        pointersToContexts = HashBasedTable.create();
        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        freed = new AtomicBoolean(false);
    }





    @Override
    public Table<String, Triple<Integer, Integer, Integer>, DevicePointerInfo> getPointersToContexts() {
        return pointersToContexts;
    }

    public void setPointersToContexts( Table<String,Triple<Integer,Integer,Integer>,DevicePointerInfo> pointersToContexts) {
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