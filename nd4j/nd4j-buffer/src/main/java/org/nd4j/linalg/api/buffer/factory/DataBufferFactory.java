/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.buffer.factory;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.nio.ByteBuffer;

/**
 * DataBufferFactory: Creates the data buffer wrt
 * a specified data opType
 *
 * @author Adam Gibson
 */
public interface DataBufferFactory {

    /**
     * Setter for the allocation mode
     * @param allocationMode
     */
    void setAllocationMode(DataBuffer.AllocationMode allocationMode);

    /**
     * Allocation mode for the data buffer
     * @return
     */
    DataBuffer.AllocationMode allocationMode();

    /**
     * Create a databuffer wrapping another one
     * this allows you to create a view of a buffer
     * with a different offset and length
     * backed by the same storage
     * @param underlyingBuffer the underlying buffer to get the storage from
     * @param offset the offset to view the data as
     * @param length the length of the buffer
     * @return the databuffer as a view
     */
    DataBuffer create(DataBuffer underlyingBuffer, long offset, long length);

    /**
     * Create int buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createInt(long offset, ByteBuffer buffer, int length);

    /**
     * Create a float data buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createFloat(long offset, ByteBuffer buffer, int length);

    /**
     * Creates a double data buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createDouble(long offset, ByteBuffer buffer, int length);

    DataBuffer createLong(ByteBuffer buffer, int length);

    /**
     * Create a double data buffer
     *
     * @return the new data buffer
     */
    DataBuffer createDouble(long offset, int length);

    /**
     * This method will create new DataBuffer of the same dataType & same length
     * @param buffer
     * @return
     */
    DataBuffer createSame(DataBuffer buffer, boolean init);

    /**
     * This method will create new DataBuffer of the same dataType & same length
     * @param buffer
     * @return
     */
    DataBuffer createSame(DataBuffer buffer, boolean init, MemoryWorkspace workspace);

    /**
     * Create a float data buffer
     *
     * @param length the length of the buffer
     * @return the new data buffer
     */
    DataBuffer createFloat(long offset, int length);

    /**
     * Create an int data buffer
     *
     * @param length the length of the data buffer
     * @return the create data buffer
     */
    DataBuffer createInt(long offset, int length);


    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(long offset, int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(long offset, int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(long offset, int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(long offset, double[] data);

    DataBuffer createDouble(long offset, double[] data, MemoryWorkspace workspace);


    /**
     * Create a double buffer buffer
     * @param data
     * @param length
     * @return
     */
    DataBuffer createDouble(long offset, byte[] data, int length);

    /**
     * Create a double buffer
     * @param data
     * @param length
     * @return
     */
    DataBuffer createFloat(long offset, byte[] data, int length);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(long offset, double[] data);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(long offset, double[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(long offset, float[] data);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(long offset, float[] data);

    DataBuffer createFloat(long offset, float[] data, MemoryWorkspace workspace);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(long offset, float[] data);


    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(long offset, int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(long offset, int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(long offset, int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(long offset, double[] data, boolean copy);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(long offset, double[] data, boolean copy);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(long offset, double[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(long offset, float[] data, boolean copy);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(long offset, float[] data, boolean copy);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(long offset, float[] data, boolean copy);


    /**
     * Create int buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createInt(ByteBuffer buffer, int length);

    /**
     * Create a float data buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createFloat(ByteBuffer buffer, int length);

    /**
     * Creates a double data buffer
     * @param buffer
     * @param length
     * @return
     */
    DataBuffer createDouble(ByteBuffer buffer, int length);

    /**
     * Create a double data buffer
     *
     * @return the new data buffer
     */
    DataBuffer createDouble(long length);

    /**
     * Create a double data buffer, with optional initialization
     *
     * @param initialize If true: initialize the buffer. If false: don't initialize.
     *
     * @return the new data buffer
     */
    DataBuffer createDouble(long length, boolean initialize);

    DataBuffer createDouble(long length, boolean initialize, MemoryWorkspace workspace);

    /**
     * Create a float data buffer
     *
     * @param length the length of the buffer
     * @return the new data buffer
     */
    DataBuffer createFloat(long length);

    /**
     * Create a float data buffer, with optional initialization
     *
     * @param length the length of the buffer
     * @param initialize If true: initialize the buffer. If false: don't initialize.
     * @return the new data buffer
     */
    DataBuffer createFloat(long length, boolean initialize);

    DataBuffer createFloat(long length, boolean initialize, MemoryWorkspace workspace);

    /**
     * Create an int data buffer
     *
     * @param length the length of the data buffer
     * @return the create data buffer
     */
    DataBuffer createInt(long length);

    /**
     * Create an int data buffer, with optional initialization
     *
     * @param length the length of the data buffer
     * @param initialize If true: initialize the buffer. If false: don't initialize.
     * @return the create data buffer
     */
    DataBuffer createInt(long length, boolean initialize);

    DataBuffer createInt(long length, boolean initialize, MemoryWorkspace workspace);


    DataBuffer createLong(long[] data);

    DataBuffer createLong(long[] data, boolean copy);

    DataBuffer createLong(long length);

    /**
     * Create an int data buffer, with optional initialization
     *
     * @param length the length of the data buffer
     * @param initialize If true: initialize the buffer. If false: don't initialize.
     * @return the create data buffer
     */
    DataBuffer createLong(long length, boolean initialize);

    DataBuffer createLong(long[] data, MemoryWorkspace workspace);

    DataBuffer createLong(long length, boolean initialize, MemoryWorkspace workspace);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int[] data);

    DataBuffer createInt(int[] data, MemoryWorkspace workspace);

    DataBuffer createInt(int[] data, boolean copy, MemoryWorkspace workspace);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(double[] data);


    /**
     * Create a double buffer buffer
     * @param data
     * @param length
     * @return
     */
    DataBuffer createDouble(byte[] data, int length);

    /**
     * Create a double buffer
     * @param data
     * @param length
     * @return
     */
    DataBuffer createFloat(byte[] data, int length);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(double[] data);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(double[] data);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(float[] data);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(float[] data);

    DataBuffer createFloat(float[] data, MemoryWorkspace workspace);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(float[] data);


    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(int[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(int[] data, boolean copy);

    /**
     * Creates a long data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createLong(int[] data, boolean copy);


    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(long[] data, boolean copy);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(long[] data, boolean copy);

    /**
     * Creates a int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(long[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(double[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(double[] data, MemoryWorkspace workspace);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(double[] data, boolean copy, MemoryWorkspace workspace);



    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(double[] data, boolean copy);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(double[] data, boolean copy);

    /**
     * Creates a double data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createDouble(float[] data, boolean copy);

    /**
     * Creates a float data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createFloat(float[] data, boolean copy);

    DataBuffer createFloat(float[] data, boolean copy, MemoryWorkspace workspace);

    /**
     * Creates an int data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createInt(float[] data, boolean copy);


    /**
     * Create a data buffer based on the
     * given pointer, data buffer opType,
     * and length of the buffer
     * @param pointer the pointer to use
     * @param type the opType of buffer
     * @param length the length of the buffer
     * @param indexer
     * @return the data buffer
     * backed by this pointer with the given
     * opType and length.
     */
    DataBuffer create(Pointer pointer, DataBuffer.Type type, long length, Indexer indexer);

    /**
     *
     * @param doublePointer
     * @param length
     * @return
     */
    DataBuffer create(DoublePointer doublePointer, long length);

    /**
     *
     * @param intPointer
     * @param length
     * @return
     */
    DataBuffer create(IntPointer intPointer, long length);

    /**
     *
     * @param floatPointer
     * @param length
     * @return
     */
    DataBuffer create(FloatPointer floatPointer, long length);

    /**
     * Creates half-precision data buffer
     *
     * @param length length of new data buffer
     * @return
     */
    DataBuffer createHalf(long length);

    /**
     * Creates half-precision data buffer
     *
     * @param length length of new data buffer
     * @param initialize true if memset should be used on allocated memory, false otherwise
     * @return
     */
    DataBuffer createHalf(long length, boolean initialize);

    DataBuffer createHalf(long length, boolean initialize, MemoryWorkspace workspace);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(float[] data, boolean copy);

    DataBuffer createHalf(float[] data, MemoryWorkspace workspace);

    DataBuffer createHalf(float[] data, boolean copy, MemoryWorkspace workspace);



    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(double[] data, boolean copy);


    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, double[] data, boolean copy);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, float[] data, boolean copy);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, int[] data, boolean copy);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, double[] data);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, float[] data);

    DataBuffer createHalf(long offset, float[] data, MemoryWorkspace workspace);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, int[] data);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, byte[] data, boolean copy);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(int[] data, boolean copy);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(float[] data);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(double[] data);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(int[] data);

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, byte[] data, int length);

    /**
     * Creates a half-precision data buffer
     *
     * @return the new buffer
     */
    DataBuffer createHalf(long offset, int length);

    /**
     * Creates a half-precision data buffer
     *
     * @return the new buffer
     */
    DataBuffer createHalf(ByteBuffer buffer, int length);

    /**
     * Creates a half-precision data buffer
     *
     * @param data
     * @param length
     * @return
     */
    DataBuffer createHalf(byte[] data, int length);


    Class<? extends DataBuffer> intBufferClass();

    Class<? extends DataBuffer> longBufferClass();

    Class<? extends DataBuffer> halfBufferClass();

    Class<? extends DataBuffer> floatBufferClass();

    Class<? extends DataBuffer> doubleBufferClass();

}
