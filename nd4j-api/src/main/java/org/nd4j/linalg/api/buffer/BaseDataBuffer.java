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
import io.netty.buffer.ByteBufAllocator;
import io.netty.buffer.Unpooled;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.ref.WeakReference;
import java.nio.ByteOrder;
import java.util.*;

/**
 * Base class for a data buffer
 * handling basic byte operations among other things.
 *
 * @author Adam Gibson
 */
public abstract class BaseDataBuffer implements DataBuffer {

    protected int length;
    protected ByteBuf dataBuffer;
    protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    protected transient WeakReference<DataBuffer> ref;
    protected boolean isPersist = false;
    protected AllocationMode allocationMode;
    protected double[] doubleData;
    protected int[] intData;
    protected float[] floatData;

    /**
     *
     * @param buf
     * @param length
     */
    protected BaseDataBuffer(ByteBuf buf,int length) {
        allocationMode = Nd4j.alloc;
        this.dataBuffer = buf;
        this.length = length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(float[] data, boolean copy) {
        allocationMode = Nd4j.alloc;
        if(allocationMode == AllocationMode.HEAP) {
            if(copy) {
                floatData = ArrayUtil.copy(data);
            }
            else {
                this.floatData = data;
            }
        }
        else {
            dataBuffer = Unpooled.copyFloat(data);
        }
        length = data.length;

    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(double[] data, boolean copy) {
        allocationMode = Nd4j.alloc;
        if(allocationMode == AllocationMode.HEAP) {
            if(copy) {
                doubleData = ArrayUtil.copy(data);
            }
            else {
                this.doubleData = data;
            }
        }
        else {
            dataBuffer = Unpooled.copyDouble(data);
        }
        length = data.length;

    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(int[] data, boolean copy) {
        allocationMode = Nd4j.alloc;
        if(allocationMode == AllocationMode.HEAP) {
            if(copy) {
                intData = ArrayUtil.copy(data);
            }
            else {
                this.intData = data;
            }
        }
        else {
            dataBuffer = Unpooled.copyInt(data);
        }
        length = data.length;
    }

    public BaseDataBuffer(double[] data) {
        this(data,Nd4j.copyOnOps);
    }

    public BaseDataBuffer(int[] data) {
        this(data,Nd4j.copyOnOps);
    }

    public BaseDataBuffer(float[] data) {
        this(data,Nd4j.copyOnOps);
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

    /**
     * Instantiate a buffer with the given length
     *
     * @param length the length of the buffer
     */
    protected BaseDataBuffer(int length) {
        this.length = length;
        allocationMode = Nd4j.alloc;

        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        if(allocationMode == AllocationMode.HEAP) {
            if(dataType() == Type.DOUBLE)
                doubleData = new double[length];
            else if(dataType() == Type.FLOAT)
                floatData = new float[length];
        }
        else {
            dataBuffer = allocationMode == AllocationMode.DIRECT ? Unpooled.directBuffer(length * getElementSize()) : Unpooled.buffer(length * getElementSize());
            dataBuffer.order(ByteOrder.nativeOrder());
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
    public void addReferencing(String id) {
        referencing.add(id);
    }

    @Override
    public void assign(int[] indices, float[] data, boolean contiguous, int inc) {
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
                dataBuffer.setInt(i, data[i]);
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
                dataBuffer.setFloat(i, data[i]);
        }

    }

    @Override
    public void setData(double[] data) {
        if(doubleData != null) {
            this.doubleData = data;
        }
        else {
            for(int i = 0; i < data.length; i++)
                dataBuffer.setDouble(i, data[i]);
        }
    }


    @Override
    public void assign(int[] indices, double[] data, boolean contiguous, int inc) {
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
    public void assign(int[] indices, float[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public int length() {
        return length;
    }

    @Override
    public void assign(Number value) {
        assign(value, 0);
    }


    @Override
    public double[] getDoublesAt(int offset, int length) {
        return getDoublesAt(offset, 1, length);
    }

    @Override
    public float[] getFloatsAt(int offset, int inc, int length) {
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

        return create(dataBuffer.copy(),length);
    }

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
    public double[] getDoublesAt(int offset, int inc, int length) {
        if (offset + length > length())
            length -= offset;

        double[] ret = new double[length];
        for (int i = 0; i < length; i++) {
            ret[i] = getDouble(i + offset);
        }


        return ret;
    }

    @Override
    public float[] getFloatsAt(int offset, int length) {
        return getFloatsAt(offset, 1, length);
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


    @Override
    public void put(int i, IComplexNumber result) {
        put(i, result.realComponent().doubleValue());
        put(i + 1, result.imaginaryComponent().doubleValue());
    }


    @Override
    public void assign(int[] offsets, int[] strides, DataBuffer... buffers) {
        assign(offsets, strides, length(), buffers);
    }

    @Override
    public byte[] asBytes() {
        return dataBuffer.array();
    }

    @Override
    public float[] asFloat() {
        if(allocationMode == AllocationMode.HEAP) {
            if(floatData != null) {
                return floatData;
            }
        }
        return dataBuffer.nioBuffer().asFloatBuffer().array();
    }

    @Override
    public double[] asDouble() {
        if(allocationMode == AllocationMode.HEAP) {
            if(doubleData != null) {
                return doubleData;
            }
        }
        return dataBuffer.nioBuffer().asDoubleBuffer().array();
    }

    @Override
    public int[] asInt() {
        if(allocationMode == AllocationMode.HEAP) {
            if(intData != null) {
                return intData;
            }
        }
        return dataBuffer.nioBuffer().asIntBuffer().array();
    }

    @Override
    public double getDouble(int i) {
        if(doubleData != null) {
            return doubleData[i];
        }
        else if(floatData != null) {
            return (double) floatData[i];
        }
        else if(intData != null) {
            return (double) intData[i];
        }
        return dataBuffer.getDouble(i);
    }

    @Override
    public float getFloat(int i) {
        if(doubleData != null) {
            return (float) doubleData[i];
        }
        else if(floatData != null) {
            return floatData[i];
        }
        else if(intData != null) {
            return (float) intData[i];
        }
        return dataBuffer.getFloat(i);
    }

    @Override
    public Number getNumber(int i) {
        return getDouble(i);
    }

    @Override
    public void put(int i, float element) {
        if(doubleData != null) {
            doubleData[i] = element;
        }
        else if(floatData != null) {
            floatData[i] = element;
        }
        else if(intData != null) {
            intData[i] = (int) element;
        }
        else
            dataBuffer.setFloat(i, element);
    }

    @Override
    public void put(int i, double element) {
        if(doubleData != null) {
            doubleData[i] = element;
        }
        else if(floatData != null) {
            floatData[i] = (float) element;
        }
        else if(intData != null) {
            intData[i] = (int) element;
        }
        else
            dataBuffer.setDouble(i,element);
    }

    @Override
    public void put(int i, int element) {
        dataBuffer.setIndex(i,element);
    }

    @Override
    public void assign(Number value, int offset) {
        dataBuffer.setDouble(offset,value.doubleValue());
    }

    @Override
    public void flush() {

    }

    @Override
    public int getInt(int ix) {
        return 0;
    }

    @Override
    public void assign(int[] offsets, int[] strides, int n, DataBuffer... buffers) {
        if (offsets.length != strides.length || strides.length != buffers.length)
            throw new IllegalArgumentException("Unable to assign buffers, please specify equal lengths strides, offsets, and buffers");
        int length = 0;
        for (int i = 0; i < buffers.length; i++)
            length += buffers[i].length();

        if (length != n)
            throw new IllegalArgumentException("Buffers must fill up specified length " + n);

        int count = 0;
        for (int i = 0; i < buffers.length; i++) {
            for (int j = offsets[i]; j < buffers[i].length(); j += strides[i]) {
                put(count++, buffers[i].getDouble(j));
            }
        }

        if (count != n)
            throw new IllegalArgumentException("Strides and offsets didn't match up to length " + n);

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
    public void destroy() {
        this.dataBuffer.clear();
        this.dataBuffer = null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        BaseDataBuffer that = (BaseDataBuffer) o;

        if (length != that.length) return false;
        if (isPersist != that.isPersist) return false;
        if (dataBuffer != null ? !dataBuffer.equals(that.dataBuffer) : that.dataBuffer != null) return false;
        if (referencing != null ? !referencing.equals(that.referencing) : that.referencing != null) return false;
        if (ref != null ? !ref.equals(that.ref) : that.ref != null) return false;
        return allocationMode == that.allocationMode;

    }

    @Override
    public int hashCode() {
        int result = length;
        result = 31 * result + (dataBuffer != null ? dataBuffer.hashCode() : 0);
        result = 31 * result + (referencing != null ? referencing.hashCode() : 0);
        result = 31 * result + (ref != null ? ref.hashCode() : 0);
        result = 31 * result + (isPersist ? 1 : 0);
        result = 31 * result + (allocationMode != null ? allocationMode.hashCode() : 0);
        return result;
    }
}
