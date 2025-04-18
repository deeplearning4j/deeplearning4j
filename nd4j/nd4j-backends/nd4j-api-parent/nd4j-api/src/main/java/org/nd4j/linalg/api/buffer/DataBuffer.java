/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.buffer;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.Collection;

public interface DataBuffer extends Serializable, AutoCloseable, Deallocatable {


    enum TypeEx {

    }

    long getGenerationId();


    /**
     * Mainly used for backward compatability.
     * Note that DIRECT and HEAP modes have been deprecated asd should not be used.
     */
    enum AllocationMode {

        @Deprecated
        DIRECT,
        @Deprecated
        HEAP,
        @Deprecated
        JAVACPP,
        @Deprecated
        LONG_SHAPE, // long shapes will be used instead of int

        MIXED_DATA_TYPES, // latest generation of INDArrays support multiple data types, with information stored within shapeInfo "offset" field.
    }

    StackTraceElement[] allocationTrace();

    /**
     * Returns the underlying opaque buffer for this data buffer
     * @return
     */
    OpaqueDataBuffer opaqueBuffer();

    /**
     * Returns an underlying pointer if one exists
     * @return
     */
    Pointer pointer();


    /**
     * Returns the address of the pointer wrapped in a Pointer
     * @return the address of the pointer wrapped in a Pointer
     */
    Pointer addressPointer();

    /**
     * Returns the indexer for the buffer
     * @return
     */
    Indexer indexer();


    /**
     * Returns the address of the pointer
     * @return the address of the pointer
     */
    long address();

    /**
     * Returns the address of platform-specific pointer:
     * - for native backend that'll be host pointer
     * - for cuda backend that'll be device pointer
     * @return
     */
    long platformAddress();

    /**
     * Returns true if the underlying data source
     * is the same for both buffers (referential equals)
     * @param buffer whether the buffer is the same underlying data or not
     * @return true if both data buffers have the same
     * underlying data SOURCE
     */
    boolean sameUnderlyingData(DataBuffer buffer);

    void read(DataInputStream s, AllocationMode allocationMode, long length, DataType dataType);

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
     * @return a view of this as an nio int buffer
     */
    java.nio.IntBuffer asNioInt();

    java.nio.LongBuffer asNioLong();

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

    boolean[] asBoolean();


    void put(float[] element);

    void put(double[] element);

    void put(int[] element);

    void put(boolean[] element);

    void put(short[] element);

    void put(byte[] element);

    void put(long[] element);

    /**
     * Whether the buffer is dirty:
     * aka has been updated
     * @return true if the buffer has been
     * updated, false otherwise
     */
    boolean dirty();


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
    void copyAtStride(DataBuffer buf, long n, long stride, long yStride, long offset, long yOffset);

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
    void assign(long[] indices, float[] data, boolean contiguous, long inc);

    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the data is contiguous or not
     * @param inc        the number to increment by when assigning
     */
    void assign(long[] indices, double[] data, boolean contiguous, long inc);


    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the indices are contiguous or not
     */
    void assign(long[] indices, float[] data, boolean contiguous);

    /**
     * Assign the given elements to the given indices
     *
     * @param indices    the indices to assign
     * @param data       the data to assign
     * @param contiguous whether the data is contiguous or not
     */
    void assign(long[] indices, double[] data, boolean contiguous);

    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    double[] getDoublesAt(long offset, int length);


    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    float[] getFloatsAt(long offset, int length);

    /**
     * Get the ints at a particular offset
     *
     * @param offset the offset to start
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    int[] getIntsAt(long offset, int length);

    /**
     * Get the longs at a particular offset
     *
     * @param offset the offset to start
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    long[] getLongsAt(long offset, int length);

    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param inc    the increment to use
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    double[] getDoublesAt(long offset, long inc, int length);


    /**
     * Get the doubles at a particular offset
     *
     * @param offset the offset to start
     * @param inc    the increment to use
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    float[] getFloatsAt(long offset, long inc, int length);

    /**
     * Get the ints at a particular offset
     *
     * @param offset the offset to start
     * @param inc    the increment to use
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    int[] getIntsAt(long offset, long inc, int length);


    /**
     * Get the long at a particular offset
     *
     * @param offset the offset to start
     * @param inc    the increment to use
     * @param length the length of the array
     * @return the doubles at the specified offset and length
     */
    long[] getLongsAt(long offset, long inc, int length);


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
    void assign(Number value, long offset);

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
    void setData(long[] data);


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
    void setData(short[] data);
    void setData(byte[] data);
    void setData(boolean[] data);

    /**
     * Raw byte array storage
     *
     * @return the data represented as a raw byte array
     */
    byte[] asBytes();

    /**
     * The data opType of the buffer
     *
     * @return the data opType of the buffer
     */
    DataType dataType();

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
     * @return the buffer as a int
     */
    int[] asInt();

    /**
     * Return the buffer as an long  array
     * Relative to the datatype, this will either be a copy
     * or a reference. The reference is preferred for
     * faster access of data and no copying
     *
     * @return the buffer as a long
     */
    long[] asLong();


    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    double getDouble(long i);

    /**
     * Get element i in the buffer as long value
     * @param i
     * @return
     */
    long getLong(long i);

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    float getFloat(long i);

    /**
     * Get element i in the buffer as a double
     *
     * @param i the element to getFloat
     * @return the element at this index
     */
    Number getNumber(long i);


    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(long i, float element);

    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(long i, double element);

    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(long i, int element);


    /**
     * Assign an element in the buffer to the specified index
     *
     * @param i       the index
     * @param element the element to assign
     */
    void put(long i, short element);

    void put(long i, long element);

    void put(long i, boolean element);


    /**
     * Returns the length of the buffer
     *
     * @return the length of the buffer
     */
    long length();

    /**
     * Returns the length of the buffer
     *
     * @return the length of the buffer
     */
    long underlyingLength();

    /**
     * Get the int at the specified index
     *
     * @param ix the int at the specified index
     * @return the int at the specified index
     */
    int getInt(long ix);

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
    void assign(long[] offsets, long[] strides, long n, DataBuffer... buffers);

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
    void assign(long[] offsets, long[] strides, DataBuffer... buffers);


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
    void read(InputStream is, AllocationMode allocationMode, long length, DataType dataType);

    /**
     * This method returns whether this DataBuffer is constant, or not.
     * Constant buffer means that it modified only during creation time, and then it stays the same for all lifecycle. I.e. used in shape info databuffers.
     *
     * @return
     */
    boolean isConstant();

    /**
     *
     * This method allows you to mark databuffer as constant.
     *
     * PLEASE NOTE: DO NOT USE THIS METHOD, UNLESS YOU'RE 100% SURE WHAT YOU DO
     *
     * @param reallyConstant
     */
    void setConstant(boolean reallyConstant);

    /**
     * This method returns True, if this DataBuffer is attached to some workspace. False otherwise
     *
     * @return
     */
    boolean isAttached();

    /**
     * This method checks, if given attached INDArray is still in scope of its parent Workspace
     *
     * PLEASE NOTE: if this INDArray isn't attached to any Workspace, this method will return true
     * @return
     */
    boolean isInScope();

    /**
     * This method returns Workspace this DataBuffer is attached to
     * @return
     */
    MemoryWorkspace getParentWorkspace();

    /**
     * Reallocate the native memory of the buffer
     * @param length the new length of the buffer
     * @return this databuffer
     * */
    DataBuffer reallocate(long length);

    /**
     * @return the capacity of the databuffer
     * */
    long capacity();

    /**
     * This method checks, if this DataBuffer instalce can use close() method
     * @return true if DataBuffer can be released, false otherwise
     */
    boolean closeable();

    /**
     * This method releases exclusive off-heap resources uses by this DataBuffer instance.
     * If DataBuffer relies on shared resources, exception will be thrown instead
     *
     * PLEASE NOTE: This method is NOT safe by any means
     */
    void close();

    /**
     * This method checks if array or its buffer was closed before
     * @return true if was closed, false otherwise
     */
    boolean wasClosed();
}
