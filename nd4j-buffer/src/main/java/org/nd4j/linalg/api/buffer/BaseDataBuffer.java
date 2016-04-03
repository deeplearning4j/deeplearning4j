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

package org.nd4j.linalg.api.buffer;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.api.buffer.pointer.JavaCppDoublePointer;
import org.nd4j.linalg.api.buffer.pointer.JavaCppFloatPointer;
import org.nd4j.linalg.api.buffer.pointer.JavaCppIntPointer;
import org.nd4j.linalg.api.buffer.unsafe.UnsafeHolder;
import org.nd4j.linalg.api.buffer.util.AllocUtil;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.util.ArrayUtil;


import java.io.*;
import java.lang.reflect.Field;
import java.nio.*;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Base class for a data buffer
 * handling basic byte operations
 * among other things.
 *
 * @author Adam Gibson
 */
public abstract class BaseDataBuffer implements DataBuffer {

    protected long length;
    protected long underlyingLength;
    protected long offset;
    protected int elementSize;
    protected transient ByteBuffer wrappedBuffer;
    protected transient DataBuffer wrappedDataBuffer;
    protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    protected boolean isPersist = false;
    protected AllocationMode allocationMode;
    protected transient  double[] doubleData;
    protected transient  int[] intData;
    protected transient float[] floatData;
    protected transient Pointer pointer;
    protected transient Indexer indexer;
    protected AtomicBoolean dirty = new AtomicBoolean(false);

    // Allocator-related stuff. Moved down here to avoid type casting.
    protected transient DataBuffer originalBuffer;
    protected transient long originalOffset = 0;
    protected transient Long trackingPoint;

    public BaseDataBuffer() {
    }

    /**
     * Meant for creating another view of a buffer
     * @param underlyingBuffer the underlying buffer to create a view from
     * @param length the length of the view
     * @param offset the offset for the view
     */
    protected BaseDataBuffer(DataBuffer underlyingBuffer,long length,long offset) {
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        this.length = length;
        this.offset = offset;
        this.allocationMode = underlyingBuffer.allocationMode();
        this.elementSize = underlyingBuffer.getElementSize();
        this.underlyingLength = underlyingBuffer.underlyingLength();
        this.wrappedDataBuffer = underlyingBuffer;

        // Adding link to original databuffer
        if (underlyingBuffer.originalDataBuffer() == null) {
            this.originalBuffer = underlyingBuffer;
            this.originalOffset = offset;
        } else {

            this.originalBuffer = underlyingBuffer.originalDataBuffer();

            // FIXME: please don't remove this comment, since there's probably a bug in current offset() impl,
            // and this line will change originalOffset accroding to proper offset() impl
            // FIXME: raver119@gmail.com
            this.originalOffset = offset; // + underlyingBuffer.originalOffset();
        }

        if(underlyingBuffer.dataType() == Type.DOUBLE) {
            if(underlyingBuffer.allocationMode() == AllocationMode.HEAP) {
                double[] underlyingArray = (double[]) underlyingBuffer.array();
                if(underlyingArray != null)
                    this.doubleData = underlyingArray;
                else
                    this.wrappedBuffer = underlyingBuffer.asNio();
            }
            else if(underlyingBuffer.allocationMode() == AllocationMode.JAVACPP) {
                pointer = underlyingBuffer.pointer();
                indexer = DoubleIndexer.create((DoublePointer)pointer);
                this.wrappedBuffer = pointer.asByteBuffer();
            }
            else {
                ByteBuffer underlyingBuff = underlyingBuffer.asNio();
                this.wrappedBuffer = underlyingBuff;
            }
        }
        else if(underlyingBuffer.dataType() == Type.FLOAT) {
            if(underlyingBuffer.allocationMode() == AllocationMode.HEAP) {
                float[] underlyingArray = (float[]) underlyingBuffer.array();
                if(underlyingArray != null)
                    this.floatData = underlyingArray;
                else
                    this.wrappedBuffer = underlyingBuffer.asNio();
            }
            else if(underlyingBuffer.allocationMode() == AllocationMode.JAVACPP) {
                pointer = underlyingBuffer.pointer();
                indexer = FloatIndexer.create((FloatPointer)pointer);
                wrappedBuffer = underlyingBuffer.asNio();
            }
            else {
                ByteBuffer underlyingBuff = underlyingBuffer.asNio();
                this.wrappedBuffer = underlyingBuff;

            }
        }
        else if(underlyingBuffer.dataType() == Type.INT) {
            if(underlyingBuffer.allocationMode() == AllocationMode.HEAP) {
                int[] underlyingArray = (int[]) underlyingBuffer.array();
                if(underlyingArray != null)
                    this.intData = underlyingArray;
                else
                    this.wrappedBuffer = underlyingBuffer.asNio();
            }
            else if(underlyingBuffer.allocationMode() == AllocationMode.JAVACPP) {
                pointer = underlyingBuffer.pointer();
                indexer = IntIndexer.create((IntPointer)pointer);
                wrappedBuffer = underlyingBuffer.asNio();
            }
            else {
                ByteBuffer underlyingBuff = underlyingBuffer.asNio();
                this.wrappedBuffer = underlyingBuff;

            }
        }
    }

    /**
     * Original DataBuffer.
     * In case if we have a view derived from another view, derived from some other view, original DataBuffer will point to the originating DataBuffer, where all views come from.
     */
    @Override
    public DataBuffer originalDataBuffer() {
        return originalBuffer;
    }

    /**
     *
     * @param buf
     * @param length
     */
    protected BaseDataBuffer(ByteBuf buf,int length,int offset) {
        this(buf,length);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = length - offset;
        this.underlyingLength = length;
    }
    /**
     *
     * @param buf
     * @param length
     */
    protected BaseDataBuffer(ByteBuf buf,int length) {
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        allocationMode = AllocUtil.getAllocationModeFromContext();
        this.wrappedBuffer = buf.nioBuffer();
        this.length = length;
        this.underlyingLength = length;
    }
    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(float[] data, boolean copy,int offset) {
        this(data,copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;

    }
    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(float[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if(allocationMode == AllocationMode.HEAP) {
            if(copy) {
                floatData = ArrayUtil.copy(data);
            }
            else {
                this.floatData = data;
            }
        }
        else if(allocationMode == AllocationMode.JAVACPP) {
            pointer = new FloatPointer(ArrayUtil.copy(data));
            indexer = FloatIndexer.create((FloatPointer)pointer);
            wrappedBuffer = pointer.asByteBuffer();
        }
        else {
            wrappedBuffer =  ByteBuffer.allocateDirect(4 * data.length);
            wrappedBuffer.order(ByteOrder.nativeOrder());
            FloatBuffer buffer = wrappedBuffer.asFloatBuffer();
            for(int i = 0; i < data.length; i++) {
                buffer.put(i,data[i]);
            }
        }

        length = data.length;
        underlyingLength = data.length;

    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(double[] data, boolean copy,int offset) {
        this(data,copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = data.length;
        this.length = underlyingLength - offset;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(double[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if(allocationMode == AllocationMode.HEAP) {
            if(copy) {
                doubleData = ArrayUtil.copy(data);
            }
            else {
                this.doubleData = data;
            }
        }
        else if(allocationMode == AllocationMode.JAVACPP) {
            if(copy) {
                pointer = new DoublePointer(ArrayUtil.copy(data));
            }
            else {
                pointer = new DoublePointer(data);
            }
            indexer = DoubleIndexer.create((DoublePointer)pointer);
            wrappedBuffer = pointer.asByteBuffer();
        }
        else {
            wrappedBuffer =  ByteBuffer.allocateDirect(8 * data.length);
            wrappedBuffer.order(ByteOrder.nativeOrder());
            DoubleBuffer buffer = wrappedBuffer.asDoubleBuffer();
            for(int i = 0; i < data.length; i++) {
                buffer.put(i,data[i]);
            }
        }

        length = data.length;
        underlyingLength = data.length;
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(int[] data, boolean copy,int offset) {
        this(data,copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
    }
    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(int[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if(allocationMode == AllocationMode.HEAP) {
            if(copy)
                intData = ArrayUtil.copy(data);

            else
                this.intData = data;

        }
        else if(allocationMode == AllocationMode.JAVACPP) {
            if(copy) {
                pointer = new IntPointer(ArrayUtil.copy(data));
                indexer = IntIndexer.create((IntPointer)pointer);
                wrappedBuffer = pointer.asByteBuffer();
            }

        }
        else {
            wrappedBuffer =  ByteBuffer.allocateDirect(4 * data.length);
            wrappedBuffer.order(ByteOrder.nativeOrder());
            IntBuffer buffer = wrappedBuffer.asIntBuffer();
            for(int i = 0; i < data.length; i++) {
                buffer.put(i,data[i]);
            }
        }

        length = data.length;
        underlyingLength = data.length;
    }

    /**
     *
     * @param data
     */
    public BaseDataBuffer(double[] data) {
        this(data,true);
    }

    /**
     *
     * @param data
     */
    public BaseDataBuffer(int[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseDataBuffer(float[] data) {
        this(data,true);
    }
    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseDataBuffer(int length, int elementSize,int offset) {
        this(length,elementSize);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = length - offset;
        this.underlyingLength = length;
    }

    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseDataBuffer(int length, int elementSize) {
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        allocationMode = AllocUtil.getAllocationModeFromContext();
        this.length = length;
        this.underlyingLength = length;
        this.elementSize = elementSize;
        if(allocationMode() == AllocationMode.DIRECT) {
            //allows for creation of the nio byte buffer to be overridden
            setNioBuffer();
        }
        else if(dataType() == Type.DOUBLE) {
            doubleData = new double[length];
        }
        else if(dataType() == Type.FLOAT) {
            floatData = new float[length];
        }
        else if(dataType() == Type.INT)
            intData = new int[length];
    }
    /**
     * Create a data buffer from
     * the given length
     *
     * @param buffer
     * @param length
     */
    public BaseDataBuffer(ByteBuffer buffer,int length,int offset) {
        this(buffer,length);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = length;
        this.length = length - offset;

    }
    /**
     * Create a data buffer from
     * the given length
     *
     * @param buffer
     * @param length
     */
    public BaseDataBuffer(ByteBuffer buffer,int length) {
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        allocationMode = AllocUtil.getAllocationModeFromContext();
        this.length = length;
        this.underlyingLength = length;
        buffer.order(ByteOrder.nativeOrder());
        if(allocationMode() == AllocationMode.DIRECT) {
            this.wrappedBuffer = buffer;
        }
        else if(dataType() == Type.INT) {
            intData = new int[length];
            IntBuffer intBuffer = buffer.asIntBuffer();
            for(int i = 0; i < length; i++) {
                intData[i] = intBuffer.get(i);
            }
        }
        else if(dataType() == Type.DOUBLE) {
            doubleData = new double[length];
            DoubleBuffer doubleBuffer = buffer.asDoubleBuffer();
            for(int i = 0; i < length; i++) {
                doubleData[i] = doubleBuffer.get(i);
            }


        }
        else if(dataType() == Type.FLOAT) {
            floatData = new float[length];
            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            for(int i = 0; i < length; i++) {
                floatData[i] = floatBuffer.get(i);
            }
        }
    }

    //sets the nio wrapped buffer (allows to be overridden for other use cases like cuda)
    protected void setNioBuffer() {
        if(elementSize * length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create buffer of length " + length);
        indexer = null;
        wrappedBuffer = ByteBuffer.allocateDirect((int)(elementSize * length));
        wrappedBuffer.order(ByteOrder.nativeOrder());

    }


    public BaseDataBuffer(byte[] data, int length) {
        this(Unpooled.wrappedBuffer(data),length);
    }




    @Override
    public Pointer pointer() {
        return pointer;
    }

    @Override
    public DataBuffer underlyingDataBuffer() {
        return wrappedDataBuffer;
    }

    @Override
    public long offset() {
        return offset;
    }

    @Override
    public AllocationMode allocationMode() {
        return allocationMode;
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
    public void unPersist() {
        isPersist = false;
    }

    private void fillPointerWithZero() {
        Pointer.memset(this.pointer(),0,getElementSize() * length());
    }

    /**
     * Instantiate a buffer with the given length
     *
     * @param length the length of the buffer
     */
    protected BaseDataBuffer(long length) {
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if(length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        if(allocationMode == AllocationMode.HEAP) {
            if(length >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Length of data buffer can not be > Integer.MAX_VALUE for heap (array based storage) allocation");
            if(dataType() == Type.DOUBLE)
                doubleData = new double[(int)length];
            else if(dataType() == Type.FLOAT)
                floatData = new float[(int)length];
        }
        else if(allocationMode == AllocationMode.JAVACPP) {
            if(dataType() == Type.DOUBLE) {
                pointer = new DoublePointer(length());
                indexer = DoubleIndexer.create((DoublePointer)pointer);
                fillPointerWithZero();

            }
            else if(dataType() == Type.FLOAT) {
                pointer = new FloatPointer(length());
                indexer = FloatIndexer.create((FloatPointer)pointer);
                fillPointerWithZero();

            }
            else if(dataType() == Type.INT) {
                pointer = new IntPointer(length());
                indexer = IntIndexer.create((IntPointer)pointer);
                fillPointerWithZero();
            }
        }
        else {
            if(length * getElementSize() < 0)
                throw new IllegalArgumentException("Unable to create buffer of length " + length + " due to negative length specified");
            wrappedBuffer = ByteBuffer.allocateDirect((int)(getElementSize() * length)).order(ByteOrder.nativeOrder());
        }

    }

    @Override
    public void copyAtStride(DataBuffer buf, long n, long stride, long yStride, long offset, long yOffset) {
        if(dataType() == Type.FLOAT) {
            for(int i = 0; i < n; i++) {
                put(offset + i * stride,buf.getFloat(yOffset + i * yStride));
            }
        }
        else {
            for(int i = 0; i < n; i++) {
                put(offset + i * stride,buf.getDouble(yOffset + i * yStride));
            }
        }

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
    public long address() {
        switch(allocationMode) {
            case JAVACPP: {
                return pointer.address() + getElementSize() * offset();
            }
            case DIRECT:
                if(wrappedBuffer.isDirect())
                    try {
                        Field address = Buffer.class.getDeclaredField("address");
                        address.setAccessible(true);

                        return address.getLong(wrappedBuffer);
                        //return  UnsafeHolder.getUnsafe().objectFieldOffset(UnsafeHolder.getAddressField()) + getElementSize() * offset();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                else {
                    try {
                        //http://stackoverflow.com/questions/8820164/is-there-a-way-to-get-a-reference-address
                        int offset = UnsafeHolder.getUnsafe().arrayBaseOffset(array().getClass());
                        int scale = UnsafeHolder.getUnsafe().arrayIndexScale(array().getClass());
                        switch (scale) {
                            case 4:
                                long factor = UnsafeHolder.is64Bit() ? 8 : 1;
                                final long i1 = (UnsafeHolder.getUnsafe().getInt(array(), offset) & 0xFFFFFFFFL) * factor;
                                return i1;
                            case 8:
                                throw new AssertionError("Not supported");
                        }
                    }catch(Exception e) {
                        throw new IllegalStateException("Unable to get address", e);
                    }
                }
            case HEAP:
                //http://stackoverflow.com/questions/8820164/is-there-a-way-to-get-a-reference-address
                try {
                    //http://stackoverflow.com/questions/8820164/is-there-a-way-to-get-a-reference-address
                    int scale = UnsafeHolder.getUnsafe().arrayIndexScale(array().getClass());
                    switch (scale) {
                        case 4:
                            long factor = UnsafeHolder.is64Bit() ? 8 : 1;
                            final long i1 = (UnsafeHolder.getUnsafe().getInt(array(), offset) & 0xFFFFFFFFL) * factor;
                            return i1;
                        case 8:
                            long addressOfObject	= UnsafeHolder.getUnsafe().getLong(array(), offset);
                            return addressOfObject;
                    }
                }
                catch(Exception e) {
                    throw new IllegalStateException("Unable to detect pointer");

                }

        }
        throw new IllegalStateException("Illegal pointer");
    }

    @Override
    public void addReferencing(String id) {
        referencing.add(id);
    }

    @Override
    public void assign(long[] indices, float[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);
        for (int i = 0; i < indices.length; i++) {
            put(indices[i], data[i]);
        }
    }



    @Override
    public void setData(int[] data) {
        if(intData != null)
            this.intData = data;
        else {
            for (int i = 0; i < data.length; i++) {
                put(i,data[i]);
            }
        }

    }

    @Override
    public void setData(float[] data) {
        if(floatData != null) {
            this.floatData = data;
        }
        else {
            for(int i = 0; i < data.length; i++)
                put(i,data[i]);
        }

    }

    @Override
    public void setData(double[] data) {
        if(doubleData != null) {
            this.doubleData = data;
        }
        else {
            for(int i = 0; i < data.length; i++)
                put(i, data[i]);
        }
    }


    @Override
    public void assign(long[] indices, double[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);
        for (int i = 0; i < indices.length; i += inc) {
            put(indices[i], data[i]);
        }
    }

    @Override
    public void assign(DataBuffer data) {
        if (data.length() != length())
            throw new IllegalArgumentException("Unable to assign buffer of length " + data.length() + " to this buffer of length " + length());

        for (int i = 0; i < data.length(); i++) {
            put(i, data.getDouble(i));
        }
    }

    @Override
    public void assign(long[] indices, float[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public void assign(long[] indices, double[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public long underlyingLength() {
        return underlyingLength;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public void assign(Number value) {
        for(int i = 0; i < length(); i++)
            assign(value,i);
    }


    @Override
    public double[] getDoublesAt(long offset, int length) {
        return getDoublesAt(offset, 1, length);
    }

    @Override
    public float[] getFloatsAt(long offset, long inc, int length) {
        if (offset + length > length())
            length -= offset;
        float[] ret = new float[length];
        for (int i = 0; i < length; i++) {
            ret[i] = getFloat(i + offset);
        }
        return ret;
    }


    @Override
    public DataBuffer dup() {
        if(floatData != null) {
            return create(floatData);
        }
        else if(doubleData != null) {
            return create(doubleData);
        }
        else if(intData != null) {
            return create(intData);
        }

        DataBuffer ret = create(length);
        for(int i = 0; i < ret.length(); i++)
            ret.put(i, getDouble(i));

        return ret;
    }

    /**
     * Create with length
     * @param length a databuffer of the same type as
     *               this with the given length
     * @return a data buffer with the same length and datatype as this one
     */
    protected abstract  DataBuffer create(long length);


    /**
     * Create the data buffer
     * with respect to the given byte buffer
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    public abstract DataBuffer create(double[] data);
    /**
     * Create the data buffer
     * with respect to the given byte buffer
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    public abstract DataBuffer create(float[] data);

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    public abstract DataBuffer create(int[] data);

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     * @param buf the buffer to create
     * @return the data buffer based on the given buffer
     */
    public abstract DataBuffer create(ByteBuf buf,int length);

    @Override
    public double[] getDoublesAt(long offset, long inc, int length) {
        if (offset + length > length())
            length -= offset;

        double[] ret = new double[length];
        for (int i = 0; i < length; i++) {
            ret[i] = getDouble(i + offset);
        }


        return ret;
    }

    @Override
    public float[] getFloatsAt(long offset, int length) {
        return getFloatsAt(offset, 1, length);
    }

    @Override
    public abstract IComplexFloat getComplexFloat(long i);

    @Override
    public abstract IComplexDouble getComplexDouble(long i);

    @Override
    public IComplexNumber getComplex(long i) {
        return dataType() == Type.FLOAT ? getComplexFloat(i) : getComplexDouble(i);
    }


    @Override
    public void put(long i, IComplexNumber result) {
        put(i, result.realComponent().doubleValue());
        put(i + 1, result.imaginaryComponent().doubleValue());
    }


    @Override
    public void assign(long[] offsets, long[] strides, DataBuffer... buffers) {
        assign(offsets, strides, length(), buffers);
    }

    @Override
    public byte[] asBytes() {
        if(allocationMode == AllocationMode.HEAP) {
            if(getElementSize() * length() >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Unable to create array of length " + length);
            ByteArrayOutputStream bos = new ByteArrayOutputStream((int)(getElementSize() * length()));
            DataOutputStream dos = new DataOutputStream(bos);

            if(dataType() == Type.DOUBLE) {
                if(doubleData == null)
                    throw new IllegalStateException("Double array is null!");

                try {
                    for(int i = 0; i < doubleData.length; i++)
                        dos.writeDouble(doubleData[i]);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

            }
            else {
                if(floatData == null)
                    throw new IllegalStateException("Double array is null!");

                try {
                    for(int i = 0; i < floatData.length; i++)
                        dos.writeFloat(floatData[i]);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }


            }

            return bos.toByteArray();

        }

        else {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            DataOutputStream dos = new DataOutputStream(bos);
            if(dataType() == Type.DOUBLE) {
                for(int i = 0; i < length(); i++) {
                    try {
                        dos.writeDouble(getDouble(i));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            else {
                for(int i = 0; i < length(); i++) {
                    try {
                        dos.writeFloat(getFloat(i));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            return bos.toByteArray();
        }
    }

    @Override
    public float[] asFloat() {
        if(allocationMode == AllocationMode.HEAP) {
            if(floatData != null) {
                return floatData;
            }
        }

        if(length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        float[] ret = new float[(int)length];
        for(int i = 0; i < length; i++)
            ret[i] = getFloat(i);
        return ret;

    }

    @Override
    public double[] asDouble() {
        if(allocationMode == AllocationMode.HEAP) {
            if(doubleData != null) {
                return doubleData;
            }
        }


        if(length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        double[] ret = new double[(int)length];
        for(int i = 0; i < length; i++)
            ret[i] = getDouble(i);
        return ret;

    }

    @Override
    public int[] asInt() {
        if(allocationMode == AllocationMode.HEAP) {
            if(intData != null) {
                return intData;
            }
        }
        return wrappedBuffer.asIntBuffer().array();
    }

    @Override
    public double getDouble(long i) {
        if(doubleData != null) {
            if(offset() + i >= doubleData.length)
                throw new IllegalStateException("Index out of bounds " + i);
            dirty.set(false);
            return doubleData[(int)(offset() + i)];
        }
        else if(floatData != null) {
            if(offset() + i >= floatData.length)
                throw new IllegalStateException("Index out of bounds " + i);
            dirty.set(false);
            return (double) floatData[(int)(offset() + i)];
        }
        else if(intData != null) {
            if(offset() + i >= intData.length)
                throw new IllegalStateException("Index out of bounds " + i);
            dirty.set(false);
            return (double) intData[(int)(offset() + i)];
        }


        if(dataType() == Type.FLOAT) {
            dirty.set(false);
            if (indexer != null) {
                return ((FloatIndexer)indexer).get(offset() + i);
            } else {
                return wrappedBuffer.asFloatBuffer().get((int)(offset() + i));
            }
        }

        else if(dataType() == Type.INT) {
            dirty.set(false);
            if (indexer != null) {
                return ((IntIndexer)indexer).get(offset() + i);
            } else {
                return wrappedBuffer.asIntBuffer().get((int)(offset() + i));
            }
        }
        else {
            dirty.set(false);
            if (indexer != null) {
                return ((DoubleIndexer)indexer).get(offset() + i);
            } else {
                return wrappedBuffer.asDoubleBuffer().get((int)(offset() + i));
            }
        }
    }

    @Override
    public float getFloat(long i) {
        if(doubleData != null) {
            if(i >= doubleData.length)
                throw new IllegalStateException("Index out of bounds " + i);
            dirty.set(false);
            return (float) doubleData[(int)(offset() + i)];
        } else if(floatData != null) {
            if(i >= floatData.length)
                throw new IllegalStateException("Index out of bounds " + i);
            dirty.set(false);
            return floatData[(int)(offset() + i)];
        }
        else if(intData != null) {
            if(i >= intData.length)
                throw new IllegalStateException("Index out of bounds " + i);
            dirty.set(false);
            return (float) intData[(int)(offset() + i)];
        }

        if(dataType() == Type.DOUBLE) {
            dirty.set(false);
            if (indexer != null) {
                return (float)((DoubleIndexer)indexer).get(offset() + i);
            } else {
                return (float) wrappedBuffer.asDoubleBuffer().get((int)(offset() + i));
            }
        }

        dirty.getAndSet(true);
        if (indexer != null) {
            return ((FloatIndexer)indexer).get(offset() + i);
        } else {
            return wrappedBuffer.asFloatBuffer().get((int)(offset() + i));
        }
    }

    @Override
    public Number getNumber(long i) {
        if(dataType() == Type.DOUBLE)
            return getDouble(i);
        else if(dataType() == Type.INT)
            return getInt(i);
        return getFloat(i);
    }

    @Override
    public void put(long i, float element) {
        put(i,(double) element);
    }

    @Override
    public void put(long i, double element) {
        if(doubleData != null)
            doubleData[(int)(offset() + i)] = element;

        else if(floatData != null)
            floatData[(int)(offset() + i)] = (float) element;

        else if(intData != null)
            intData[(int)(offset() + i)] = (int) element;

        else {
            if(dataType() == Type.DOUBLE) {
                if (indexer != null) {
                    ((DoubleIndexer)indexer).put(offset() + i, element);
                } else {
                    wrappedBuffer.asDoubleBuffer().put((int)(offset() + i),element);
                }
            }
            else if(dataType() == Type.INT) {
                if (indexer != null) {
                    ((IntIndexer)indexer).put(offset() + i, (int)element);
                } else {
                    wrappedBuffer.asIntBuffer().put((int)(offset() + i),(int) element);
                }
            }
            else {
                if (indexer != null) {
                    ((FloatIndexer)indexer).put(offset() + i, (float)element);
                } else {
                    wrappedBuffer.asFloatBuffer().put((int)(offset() + i),(float) element);
                }
            }
        }

        dirty.set(true);
    }

    @Override
    public boolean dirty() {
        return dirty.get();
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
            return buffer.asNio() == asNio();
        }
    }

    @Override
    public IntBuffer asNioInt() {
        if(offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if(wrappedBuffer == null) {
            if(offset() > 0)
                return (IntBuffer) IntBuffer.wrap(intData).position((int)offset());
            else
                return IntBuffer.wrap(intData);
        }
        if(offset() == 0) {
            return wrappedBuffer.asIntBuffer();
        }
        else
            return (IntBuffer) wrappedBuffer.asIntBuffer().position((int)offset());
    }

    @Override
    public DoubleBuffer asNioDouble() {
        if(offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if(wrappedBuffer == null) {
            if(offset() == 0) {
                return DoubleBuffer.wrap(doubleData);
            }
            else
                return (DoubleBuffer) DoubleBuffer.wrap(doubleData).position((int)offset());
        }

        if(offset() == 0) {
            return wrappedBuffer.asDoubleBuffer();
        }
        else {
            ByteBuffer ret = (ByteBuffer) wrappedBuffer.slice().position((int)(offset() * getElementSize()));
            ByteBuffer convert =  ret.slice();
            return convert.asDoubleBuffer();
        }
    }

    @Override
    public FloatBuffer asNioFloat() {
        if(offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if(wrappedBuffer == null) {
            if(offset() == 0) {
                return FloatBuffer.wrap(floatData);
            }
            else
                return (FloatBuffer) FloatBuffer.wrap(floatData).position((int)offset());
        }
        if(offset() == 0) {
            return wrappedBuffer.asFloatBuffer();
        }
        else {
            ByteBuffer ret = (ByteBuffer) wrappedBuffer.slice().position((int)(offset() * getElementSize()));
            ByteBuffer convert =  ret.slice();
            return convert.asFloatBuffer();
        }

    }

    @Override
    public ByteBuffer asNio() {
        return wrappedBuffer;
    }

    @Override
    public ByteBuf asNetty() {
        if(wrappedBuffer != null)
            return Unpooled.wrappedBuffer(wrappedBuffer);
        else if(floatData != null)
            return Unpooled.copyFloat(floatData);
        else if(doubleData != null)
            return Unpooled.copyDouble(doubleData);
        throw new IllegalStateException("No data source defined");
    }

    @Override
    public void put(long i, int element) {
        //note here that the final put will take care of the offset
        put(i,(double) element);
    }

    @Override
    public void assign(Number value, long offset) {
        //note here that the final put will take care of the offset
        for(long i = offset; i < length(); i++)
            put(i, value.doubleValue());
    }

    @Override
    public void write(OutputStream dos) {
        if(dos instanceof DataOutputStream) {
            try {
                write((DataOutputStream) dos);
            } catch (IOException e) {
                throw new IllegalStateException("IO Exception writing buffer",e);
            }
        }
        else {
            DataOutputStream dos2 = new DataOutputStream(dos);
            try {

                write( dos2);
            } catch (IOException e) {
                throw new IllegalStateException("IO Exception writing buffer",e);
            }
        }

    }

    @Override
    public void read(InputStream is) {
        if(is instanceof DataInputStream) {
            read((DataInputStream) is);
        }

        else {
            DataInputStream dis2 = new DataInputStream(is);
            read(dis2);
        }
    }

    @Override
    public void flush() {

    }

    @Override
    public int getInt(long ix) {
        return (int) getDouble(ix);
    }

    @Override
    public void assign(long[] offsets, long[] strides, long n, DataBuffer... buffers) {
        if (offsets.length != strides.length || strides.length != buffers.length)
            throw new IllegalArgumentException("Unable to assign buffers, please specify equal lengths strides, offsets, and buffers");
        int count = 0;
        for (int i = 0; i < buffers.length; i++) {
            //note here that the final put will take care of the offset
            for (long j = offsets[i]; j < buffers[i].length(); j += strides[i]) {
                put(count++, buffers[i].getDouble(j));
            }
        }

        if (count != n)
            throw new IllegalArgumentException("Strides and offsets didn't match up to length " + n);

    }

    @Override
    public void assign(DataBuffer... buffers) {
        long[] offsets = new long[buffers.length];
        long[] strides = new long[buffers.length];
        for (int i = 0; i < strides.length; i++)
            strides[i] = 1;
        assign(offsets, strides, buffers);
    }


    @Override
    public void destroy() {

    }

    @Override
    public boolean equals(Object o) {
        // FIXME: this is BAD. it takes too long to work, and it breaks general equals contract
        if(o instanceof DataBuffer) {
            DataBuffer d = (DataBuffer) o;
            if(d.length() != length())
                return false;
            for(int i = 0; i < length(); i++) {
                double eps = Math.abs(getDouble(i) - d.getDouble(i));
                if(eps > 1e-12)
                    return false;
            }
        }

        return true;
    }

    private void readObject(ObjectInputStream s) {
        doReadObject(s);
    }

    private void writeObject(ObjectOutputStream out)
            throws IOException {
        out.defaultWriteObject();
        write(out);
    }


    protected void doReadObject(ObjectInputStream s) {
        try {
            s.defaultReadObject();
            read(s);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }



    @Override
    public void read(DataInputStream s) {
        try {
            referencing = Collections.synchronizedSet(new HashSet<String>());
            dirty = new AtomicBoolean(false);
            allocationMode = AllocationMode.valueOf(s.readUTF());
            length = s.readInt();
            Type t = Type.valueOf(s.readUTF());
            if(t == Type.DOUBLE) {
                if(allocationMode == AllocationMode.HEAP) {
                    doubleData = new double[(int)length()];
                    for(int i = 0; i < length(); i++) {
                        put(i,s.readDouble());
                    }

                }
                else {
                    indexer = null;
                    wrappedBuffer = ByteBuffer.allocateDirect((int)length() * getElementSize());
                    wrappedBuffer.order(ByteOrder.nativeOrder());
                    for(int i = 0; i < length(); i++) {
                        put(i,s.readDouble());
                    }
                }
            }
            else if(t == Type.FLOAT) {
                if(allocationMode == AllocationMode.HEAP) {
                    floatData = new float[(int)length()];
                    for(int i = 0; i < length(); i++) {
                        put(i,s.readFloat());
                    }

                }
                else {
                    indexer = null;
                    wrappedBuffer = ByteBuffer.allocateDirect((int)length() * getElementSize());
                    wrappedBuffer.order(ByteOrder.nativeOrder());
                    for(int i = 0; i < length(); i++) {
                        put(i,s.readFloat());
                    }
                }
            }
            else {
                if(allocationMode == AllocationMode.HEAP) {
                    intData = new int[(int)length()];
                    for(int i = 0; i < length(); i++) {
                        put(i,s.readInt());
                    }
                }
                else {
                    indexer = null;
                    wrappedBuffer = ByteBuffer.allocateDirect((int)length() * getElementSize());
                    wrappedBuffer.order(ByteOrder.nativeOrder());
                    for(int i = 0; i < length(); i++) {
                        put(i,s.readInt());
                    }
                }
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

    @Override
    public void write(DataOutputStream out) throws IOException {
        if(length() >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Length of data buffer can not be >= Integer.MAX_VALUE on output");

        out.writeUTF(allocationMode.name());
        out.writeInt((int)length());
        out.writeUTF(dataType().name());
        if(dataType() == Type.DOUBLE) {
            for(int i = 0; i < length(); i++)
                out.writeDouble(getDouble(i));
        }
        else if(dataType() == Type.INT) {
            for(int i = 0; i < length(); i++)
                out.writeInt(getInt(i));
        }
        else {
            for(int i = 0; i < length(); i++)
                out.writeFloat(getFloat(i));
        }
    }




    @Override
    public Object array() {
        if(floatData != null)
            return floatData;
        if(doubleData != null)
            return doubleData;
        else if (intData != null)
            return intData;
        return null;
    }

    @Override
    public String toString() {
        StringBuffer ret = new StringBuffer();
        ret.append("[");
        for(int i = 0; i < length(); i++) {
            ret.append(getNumber(i));
            if(i < length() - 1)
                ret.append(",");
        }
        ret.append("]");

        return ret.toString();
    }

    @Override
    public int hashCode() {
        int result = (int)length;
        result = 31 * result + (referencing != null ? referencing.hashCode() : 0);
        result = 31 * result + (isPersist ? 1 : 0);
        result = 31 * result + (allocationMode != null ? allocationMode.hashCode() : 0);
        return result;
    }

    /**
     * Returns the offset of the buffer relative to originalDataBuffer
     *
     * @return
     */
    @Override
    public long originalOffset() {
        return originalOffset;
    }

    /**
     * Returns tracking point for Allocator
     *
     * PLEASE NOTE: Suitable & meaningful only for specific backends
     *
     * @return
     */
    @Override
    public Long getTrackingPoint() {
        return trackingPoint;
    }

    /**
     * Sets tracking point used by Allocator
     *
     * PLEASE NOTE: Suitable & meaningful only for specific backends
     *
     * @param trackingPoint
     */
    public void setTrackingPoint(Long trackingPoint) {
        this.trackingPoint = trackingPoint;
    }
}