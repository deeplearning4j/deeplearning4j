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
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.Collection;

/**
 * A data buffer is an interface
 * for handling storage and retrieval of data
 *
 * @author Adam Gibson
 */
public interface DataBuffer extends Serializable {

    enum Type {
        DOUBLE,
        FLOAT,
        INT
    }


    /**
     * Direct (off heap) and heap allocation
     *
     * Each has their trade offs.
     *
     * One allows for storing unlimited array sizes, faster i/o with native
     * applications
     *
     * heap is backed by an array and can be useful depending on the api
     */
    enum AllocationMode {
        DIRECT,
        HEAP,
        JAVACPP,
        CUDA
    }

    /**
     * Returns an underlying pointer if one exists
     * @return
     */
    Pointer pointer();

    /**
     * Returns the address of the pointer
     * @return the address of the pointer
     */
    long address();

    /**
     * Returns true if the underlying data source
     * is the same for both buffers (referential equals)
     * @param buffer whether the buffer is the same underlying data or not
     * @return true if both data buffers have the same
     * underlying data SOURCE
     */
    boolean sameUnderlyingData(DataBuffer buffer);

    void read(DataInputStream s);

    void write(DataOutputStream out) throws IOException;

    /**
     * Returns the backing array
     * of this buffer (if there is one)
     * @return the backing array of this buffer
     */
    Object array();

    /**
     * Returns a view of this as an
     * nio byte buffer
     * @return a view of this as an nio double buffer
     */
    java.nio.IntBuffer asNioInt();
    /**
     * Returns a view of this as an
     * nio byte buffer
     * @return a view of this as an nio double buffer
     */
    java.nio.DoubleBuffer asNioDouble();
    /**
     * Returns a view of this as an
     * nio byte buffer
     * @return a view of this as an nio float buffer
     */
    java.nio.FloatBuffer asNioFloat();

    /**
     * Returns a view of this as an
     * nio byte buffer
     * @return a view of this as an nio byte buffer
     */
    ByteBuffer asNio();

    /**
     * Whether the buffer is dirty:
     * aka has been updated
     * @return true if the buffer has been
     * updated, false otherwise
     */
    boolean dirty();


    /**
     * Returns a view of this as a
     * netty byte buffer
     * @return a reference to this as
     * a netty byte buffer
     */
    ByteBuf asNetty();

    /**
     * Underlying buffer:
     * This is meant for a data buffer
     * to be a view of another data buffer
     * @return
     */
    DataBuffer underlyingDataBuffer();


    /**
     *  Original DataBuffer.
     *  In case if we have a view derived from another view, derived from some other view, original DataBuffer will point to the originating DataBuffer, where all views come from.
     */
    DataBuffer originalDataBuffer();

    /**
     * Copies from
     * the given buffer
     * at the specified stride
     * for up to n elements
     * @param buf the data buffer to copy from
     * @param n the number of elements to copy
     * @param stride the stride to copy at
     * @param yStride
     * @param offset
     * @param yOffset
     */
    void copyAtStride(DataBuffer buf, int n, int stride, int yStride, int offset, int yOffset);

    /**
     * Allocation mode for buffers
     * @return the allocation mode for the buffer
     */
    AllocationMode allocationMode();

    /**
     * Mark this buffer as persistent
     */
    void persist();

    /**
     * Whether the buffer should be persistent.
     * This is mainly for the
     * aggressive garbage collection strategy.
     * @return whether the buffer should be persistent or not (default false)
     */
    boolean isPersist();

    /**
     * Un persist the buffer
     */
    void unPersist();

    /**
     * The number of bytes for each individual element
     *
     * @return the number of bytes for each individual element
     */
    int getElementSize();

    /**
     * Remove the referenced id if it exists
     *
     * @param id the id to remove
     */
    void removeReferencing(String id);

    /**
     * The referencers pointing to this buffer
     *
     * @return the references pointing to this buffer
     */
    Collection<String> references();

    /**
     * Add a referencing element to this buffer
     *
     * @param id the id to reference
     */
    void addReferencing(String id);

    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the indices are contiguous or not
     * @param inc        the number to increment by when assigning
     */
    void assign(int[] indices, float[] data, boolean contiguous, int inc);

    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the data is contiguous or not
     * @param inc        the number to increment by when assigning
     */
    void assign(int[] indices, double[] data, boolean contiguous, int inc);


    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the indices are contiguous or not
     */
    void assign(int[] indices, float[] data, boolean contiguous);

    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the data is contiguous or not
     */
    void assign(int[] indices, double[] data, boolean contiguous);

    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    double[] getDoublesAt(int offset, int length);


    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    float[] getFloatsAt(int offset, int length);


    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param inc    the increment to use
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    double[] getDoublesAt(int offset, int inc, int length);


    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param inc    the increment to use
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    float[] getFloatsAt(int offset, int inc, int length);


    /**
     * Assign the given value to the buffer
     *
     * @param value the value to assign
     */
    void assign(Number value);

    /**
     * Assign the given value to the buffer
     * starting at offset
     *
     * @param value  assign the value to set
     * @param offset the offset to start at
     */
    void assign(Number value, int offset);

    /**
     * Set the data for this buffer
     *
     * @param data the data for this buffer
     */
    void setData(int[] data);

    /**
     * Set the data for this buffer
     *
     * @param data the data for this buffer
     */
    void setData(float[] data);

    /**
     * Set the data for this buffer
     *
     * @param data the data for this buffer
     */
    void setData(double[] data);

    /**
     * Raw byte array storage
     *
     * @return the data represented as a raw byte array
     */
    byte[] asBytes();

    /**
     * The data type of the buffer
     *
     * @return the data type of the buffer
     */
    Type dataType();

    /**
     * Return the buffer as a float array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     *
     * @return the buffer as a float
     */
    float[] asFloat();

    /**
     * Return the buffer as a double array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     *
     * @return the buffer as a float
     */
    double[] asDouble();

    /**
     * Return the buffer as an int  array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     *
     * @return the buffer as a float
     */
    int[] asInt();

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    double getDouble(int i);

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    float getFloat(int i);

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    Number getNumber(int i);


    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(int i, float element);

    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(int i, double element);

    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(int i, int element);


    /**
     * Get the complex float
     *
     * @param i the i togete
     * @return the complex float at the specified index
     */
    IComplexFloat getComplexFloat(int i);

    /**
     * Get the complex double at the specified index
     *
     * @param i the index
     * @return the complex double
     */
    IComplexDouble getComplexDouble(int i);

    /**
     * Returns a complex number
     *
     * @param i the complex number cto get
     * @return the complex number to get
     */
    IComplexNumber getComplex(int i);


    /**
     * Returns the length of the buffer
     *
     * @return the length of the buffer
     */
    int length();

    /**
     * Returns the length of the buffer
     *
     * @return the length of the buffer
     */
    int underlyingLength();
    /**
     * Returns the offset of the buffer
     *
     * @return the offset of the buffer
     */
    int offset();

    /**
     * Returns the offset of the buffer relative to originalDataBuffer
     *
     * @return
     */
    int originalOffset();

    /**
     * Get the int at the specified index
     *
     * @param ix the int at the specified index
     * @return the int at the specified index
     */
    int getInt(int ix);

    /**
     * Return a copy of this buffer
     *
     * @return a copy of this buffer
     */
    DataBuffer dup();

    /**
     * Flush the data buffer
     */
    void flush();

    /**
     * Insert a complex number at the given index
     * @param i the index to insert
     * @param result the element to insert
     */
    void put(int i, IComplexNumber result);


    /**
     * Assign the contents of this buffer
     * to this buffer
     *
     * @param data the data to assign
     */
    void assign(DataBuffer data);


    /**
     * Assign the given buffers to this buffer
     * based on the given offsets and strides.
     * Note that the offsets and strides must be of equal
     * length to the number of buffers
     *  @param offsets the offsets to use
     * @param strides the strides to use
     * @param n       the number of elements to operate on
     * @param buffers the buffers to assign data from
     */
    void assign(int[] offsets, int[] strides, long n, DataBuffer... buffers);

    /**
     * Assign the given data buffers to this buffer
     *
     * @param buffers the buffers to assign
     */
    void assign(DataBuffer... buffers);

    /**
     * Assign the given buffers to this buffer
     * based on the given offsets and strides.
     * Note that the offsets and strides must be of equal
     * length to the number of buffers
     *
     * @param offsets the offsets to use
     * @param strides the strides to use
     * @param buffers the buffers to assign data from
     */
    void assign(int[] offsets, int[] strides, DataBuffer... buffers);


    /**
     * release all resources for this buffer
     */
    void destroy();

    /**
     * Write this buffer to the output stream
     * @param dos the output stream to write
     */
    void write(OutputStream dos);

    /**
     * Write this buffer to the input stream.
     * @param is the inpus tream to write to
     */
    void read(InputStream is);

    /**
     * Returns tracking point for Allocator
     *
     * PLEASE NOTE: Suitable & meaningful only for specific backends
     * @return
     */
    Long getTrackingPoint();

    /**
     * Sets tracking point used by Allocator
     *
     * PLEASE NOTE: Suitable & meaningful only for specific backends
     *
     * @param trackingPoint
     */
    void setTrackingPoint(Long trackingPoint);
}
