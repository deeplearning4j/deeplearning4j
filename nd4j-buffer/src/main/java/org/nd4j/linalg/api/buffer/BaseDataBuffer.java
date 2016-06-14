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

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
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

    protected Type type;
    protected long length;
    protected long underlyingLength;
    protected long offset;
    protected int elementSize;
    protected transient ByteBuffer wrappedBuffer;
    protected transient DataBuffer wrappedDataBuffer;
    protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    protected boolean isPersist = false;
    protected AllocationMode allocationMode;
    protected transient Pointer pointer;
    protected transient Indexer indexer;
    protected AtomicBoolean dirty = new AtomicBoolean(false);

    // Allocator-related stuff. Moved down here to avoid type casting.
    protected transient DataBuffer originalBuffer;
    protected transient long originalOffset = 0;
    protected transient Long trackingPoint;

    protected transient boolean constant = false;

    public BaseDataBuffer() {
    }

    /**
     * Initialize the type of this buffer
     */
    protected abstract void initTypeAndSize();

    @Override
    public int getElementSize() {
        return elementSize;
    }


    /**
     *
     * Meant for creating another view of a buffer
     * @param underlyingBuffer the underlying buffer to create a view from
     * @param length the length of the view
     * @param offset the offset for the view
     */
    protected BaseDataBuffer(DataBuffer underlyingBuffer,long length,long offset) {
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
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
            pointer = underlyingBuffer.pointer();
            indexer = DoubleIndexer.create((DoublePointer)pointer);
        }
        else if(underlyingBuffer.dataType() == Type.FLOAT) {
            pointer = underlyingBuffer.pointer();
            indexer = FloatIndexer.create((FloatPointer)pointer);
        }
        else if(underlyingBuffer.dataType() == Type.INT) {
            pointer = underlyingBuffer.pointer();
            indexer = IntIndexer.create((IntPointer)pointer);
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
        initTypeAndSize();

        pointer = new FloatPointer(data);
        indexer = FloatIndexer.create((FloatPointer)pointer);
        wrappedBuffer = pointer.asByteBuffer();

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
        initTypeAndSize();

        pointer = new DoublePointer(data);
        indexer = DoubleIndexer.create((DoublePointer)pointer);
        wrappedBuffer = pointer.asByteBuffer();

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
        initTypeAndSize();

        pointer = new IntPointer(data);
        indexer = IntIndexer.create((IntPointer)pointer);
        wrappedBuffer = pointer.asByteBuffer();

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
    public BaseDataBuffer(long length, int elementSize) {
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        allocationMode = AllocUtil.getAllocationModeFromContext();
        this.length = length;
        this.underlyingLength = length;
        this.elementSize = elementSize;

        if(dataType() == Type.DOUBLE) {
            pointer = new DoublePointer(length);
            indexer = DoubleIndexer.create((DoublePointer)pointer);
        }
        else if(dataType() == Type.FLOAT) {
            pointer = new FloatPointer(length);
            indexer = FloatIndexer.create((FloatPointer)pointer);
        }
        else if(dataType() == Type.INT) {
            pointer = new IntPointer(length);
            indexer = IntIndexer.create((IntPointer)pointer);
        }
    }
    /**
     * Create a data buffer from
     * the given length
     *
     * @param buffer
     * @param length
     */
    public BaseDataBuffer(ByteBuffer buffer,long length,int offset) {
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
    public BaseDataBuffer(ByteBuffer buffer,long length) {
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();

        this.length = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();

        if(dataType() == Type.DOUBLE) {
            pointer = new DoublePointer(buffer.asDoubleBuffer());
            indexer = DoubleIndexer.create((DoublePointer)pointer);
        }
        else if(dataType() == Type.FLOAT) {
            pointer = new FloatPointer(buffer.asFloatBuffer());
            indexer = FloatIndexer.create((FloatPointer)pointer);
        }
        else if(dataType() == Type.INT) {
            pointer = new IntPointer(buffer.asIntBuffer());
            indexer = IntIndexer.create((IntPointer)pointer);
        }
    }

    //sets the nio wrapped buffer (allows to be overridden for other use cases like cuda)
    protected void setNioBuffer() {
        if(elementSize * length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create buffer of length " + length);
        wrappedBuffer = pointer.asByteBuffer();

    }


    public BaseDataBuffer(byte[] data, long length) {
        this(ByteBuffer.wrap(data),length);
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
        this(length, true);
    }

    protected BaseDataBuffer(long length, boolean initialize){
        if(length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if(length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        if(dataType() == Type.DOUBLE) {
            pointer = new DoublePointer(length());
            indexer = DoubleIndexer.create((DoublePointer)pointer);
            if(initialize) fillPointerWithZero();
        }
        else if(dataType() == Type.FLOAT) {
            pointer = new FloatPointer(length());
            indexer = FloatIndexer.create((FloatPointer)pointer);
            if(initialize) fillPointerWithZero();

        }
        else if(dataType() == Type.INT) {
            pointer = new IntPointer(length());
            indexer = IntIndexer.create((IntPointer)pointer);
            if(initialize) fillPointerWithZero();
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
    public Pointer addressPointer() {
        if (offset() > 0) {
            if(dataType() == Type.DOUBLE) {
                return new DoublePointer(pointer) { { address = pointer.address() + getElementSize() * offset(); } };
            }
            else if(dataType() == Type.FLOAT) {
                return new FloatPointer(pointer) { { address = pointer.address() + getElementSize() * offset(); } };
            }
            else if(dataType() == Type.INT) {
                return new IntPointer(pointer) { { address = pointer.address() + getElementSize() * offset(); } };
            }
        }
        return pointer;
    }

    @Override
    public long address() {
        return pointer.address() + getElementSize() * offset();
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
        for (int i = 0; i < data.length; i++) {
            put(i,data[i]);
        }
    }

    @Override
    public void setData(float[] data) {
        for(int i = 0; i < data.length; i++) {
            put(i,data[i]);
        }
    }

    @Override
    public void setData(double[] data) {
        for(int i = 0; i < data.length; i++) {
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

    @Override
    public float[] asFloat() {
        if(length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        float[] ret = new float[(int)length];
        for(int i = 0; i < length; i++)
            ret[i] = getFloat(i);
        return ret;
    }

    @Override
    public double[] asDouble() {
        if(length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        double[] ret = new double[(int)length];
        for(int i = 0; i < length; i++)
            ret[i] = getDouble(i);
        return ret;
    }

    @Override
    public int[] asInt() {
        if(length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        int[] ret = new int[(int)length];
        for(int i = 0; i < length; i++)
            ret[i] = getInt(i);
        return ret;
    }

    @Override
    public double getDouble(long i) {
        if(dataType() == Type.FLOAT) {
            dirty.set(false);
            return ((FloatIndexer)indexer).get(offset() + i);
        }
        else if(dataType() == Type.INT) {
            dirty.set(false);
            return ((IntIndexer)indexer).get(offset() + i);
        }
        else {
            dirty.set(false);
            return ((DoubleIndexer)indexer).get(offset() + i);
        }
    }

    @Override
    public float getFloat(long i) {
        if(dataType() == Type.DOUBLE) {
            dirty.set(false);
            return (float)((DoubleIndexer)indexer).get(offset() + i);
        }
        else if(dataType() == Type.INT) {
            dirty.set(false);
            return ((IntIndexer)indexer).get(offset() + i);
        }
        else {
            dirty.set(false);
            return ((FloatIndexer)indexer).get(offset() + i);
        }
    }

    @Override
    public int getInt(long i) {
        if(dataType() == Type.DOUBLE) {
            dirty.set(false);
            return (int)((DoubleIndexer)indexer).get(offset() + i);
        }
        else if(dataType() == Type.INT) {
            dirty.set(false);
            return ((IntIndexer)indexer).get(offset() + i);
        }
        else {
            dirty.set(false);
            return (int)((FloatIndexer)indexer).get(offset() + i);
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
        if(dataType() == Type.DOUBLE) {
            ((DoubleIndexer)indexer).put(offset() + i, element);
        }
        else if(dataType() == Type.INT) {
            ((IntIndexer)indexer).put(offset() + i, (int)element);
        }
        else {
            ((FloatIndexer)indexer).put(offset() + i, element);
        }

        dirty.set(true);
    }

    @Override
    public void put(long i, double element) {
        if(dataType() == Type.DOUBLE) {
            ((DoubleIndexer)indexer).put(offset() + i, element);
        }
        else if(dataType() == Type.INT) {
            ((IntIndexer)indexer).put(offset() + i, (int)element);
        }
        else {
            ((FloatIndexer)indexer).put(offset() + i, (float)element);
        }

        dirty.set(true);
    }

    @Override
    public void put(long i, int element) {
        if(dataType() == Type.DOUBLE) {
            ((DoubleIndexer)indexer).put(offset() + i, element);
        }
        else if(dataType() == Type.INT) {
            ((IntIndexer)indexer).put(offset() + i, element);
        }
        else {
            ((FloatIndexer)indexer).put(offset() + i, element);
        }

        dirty.set(true);
    }

    @Override
    public boolean dirty() {
        return dirty.get();
    }

    @Override
    public boolean sameUnderlyingData(DataBuffer buffer) {
        return pointer() == buffer.pointer();
    }

    @Override
    public IntBuffer asNioInt() {
        if(offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if(wrappedBuffer == null) {
            return pointer.asByteBuffer().asIntBuffer();
        }
        else if(offset() == 0) {
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
            return pointer.asByteBuffer().asDoubleBuffer();
        }
        else if(offset() == 0) {
            return wrappedBuffer.asDoubleBuffer();
        }
        else {
            return (DoubleBuffer) wrappedBuffer.asDoubleBuffer().position((int)(offset()));
        }
    }

    @Override
    public FloatBuffer asNioFloat() {
        if(offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if(wrappedBuffer == null) {
            return pointer.asByteBuffer().asFloatBuffer();
        }
        else if(offset() == 0) {
            return wrappedBuffer.asFloatBuffer();
        }
        else {
            return (FloatBuffer) wrappedBuffer.asFloatBuffer().position((int)(offset()));
        }

    }

    @Override
    public ByteBuffer asNio() {
        if(wrappedBuffer == null) {
            return pointer.asByteBuffer();
        } else {
            return wrappedBuffer;
        }
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

    /**
     * The data type of the buffer
     *
     * @return the data type of the buffer
     */
    @Override
    public Type dataType() {
        return type;
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
            type = Type.valueOf(s.readUTF());
            if(type == Type.DOUBLE)
                elementSize = 8;
            else if(type == Type.FLOAT || type == Type.INT)
                elementSize = 4;

            if(type == Type.DOUBLE) {
                pointer = new DoublePointer(length());
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                for(int i = 0; i < length(); i++) {
                    put(i,s.readDouble());
                }
                wrappedBuffer = pointer.asByteBuffer();
            }
            else if(type == Type.FLOAT) {
                pointer = new FloatPointer(length());
                indexer = FloatIndexer.create((FloatPointer) pointer);
                for(int i = 0; i < length(); i++) {
                    put(i,s.readFloat());
                }
                wrappedBuffer = pointer.asByteBuffer();
            }
            else {
                pointer = new IntPointer(length());
                indexer = IntIndexer.create((IntPointer) pointer);
                for(int i = 0; i < length(); i++) {
                    put(i,s.readInt());
                }
                wrappedBuffer = pointer.asByteBuffer();
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

    /**
     * This method returns whether this DataBuffer is constant, or not.
     * Constant buffer means that it modified only during creation time, and then it stays the same for all lifecycle. I.e. used in shape info databuffers.
     *
     * @return
     */
    public boolean isConstant() {
        return constant;
    }

    /**
     *
     * This method allows you to mark databuffer as constant.
     *
     * PLEASE NOTE: DO NOT USE THIS METHOD, UNLESS YOU'RE 100% SURE WHAT YOU DO
     *
     * @param reallyConstant
     */
    public void setConstant(boolean reallyConstant) {
        this.constant = reallyConstant;
    }
}