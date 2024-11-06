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

package org.nd4j.linalg.jcublas.buffer;

import lombok.NonNull;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.common.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Cuda Bool buffer implementation
 *
 * This class handles boolean data types for CUDA operations.
 * Each boolean is represented as a single byte (1 for true, 0 for false).
 *
 * @author raver119@gmail.com
 */
public class CudaBoolDataBuffer extends BaseCudaDataBuffer {

    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaBoolDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    /**
     * Meant for creating another view of a buffer with a special pointer
     *
     * @param pointer        the underlying buffer to create a view from
     * @param specialPointer the special pointer for device memory
     * @param indexer        the indexer for the pointer
     * @param length         the length of the view
     */
    public CudaBoolDataBuffer(Pointer pointer, Pointer specialPointer, Indexer indexer, long length){
        super(pointer, specialPointer, indexer, length);
    }

    /**
     * Constructor using a ByteBuffer
     *
     * @param buffer    the ByteBuffer to initialize the buffer with
     * @param dataType  the data type of the buffer
     * @param length    the length of the buffer
     * @param offset    the offset in the buffer
     */
    public CudaBoolDataBuffer(ByteBuffer buffer, DataType dataType, long length, long offset) {
        super(buffer, dataType, length, offset);
    }

    /**
     * Constructor with memory workspace
     *
     * @param length        the length of the buffer
     * @param elementSize   the size of each element in bytes
     * @param initialize    whether to initialize the buffer
     * @param workspace     the memory workspace
     */
    public CudaBoolDataBuffer(long length, int elementSize, boolean initialize, @NonNull MemoryWorkspace workspace) {
        super(length, elementSize, initialize, workspace);
    }

    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaBoolDataBuffer(long length) {
        super(length, 1, true); // elementSize for BOOL is 1 byte
        setData(new boolean[(int) length]); // Initialize with false
    }

    /**
     * Constructor with initialization flag
     *
     * @param length      the length of the buffer
     * @param initialize  whether to initialize the buffer
     */
    public CudaBoolDataBuffer(long length, boolean initialize) {
        super(length, 1, initialize);
        if (initialize) {
            setData(new boolean[(int) length]); // Initialize with false
        }
    }

    /**
     * Constructor with specified element size
     *
     * @param length        the length of the buffer
     * @param elementSize   the size of each element in bytes
     */
    public CudaBoolDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }

    /**
     * Constructor with memory workspace and initialization flag
     *
     * @param length        the length of the buffer
     * @param initialize    whether to initialize the buffer
     * @param workspace     the memory workspace
     */
    public CudaBoolDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 1, initialize, workspace);
        if (initialize) {
            setData(new boolean[(int) length]); // Initialize with false
        }
    }

    /**
     * Constructor using a ByteBuffer without offset
     *
     * @param underlyingBuffer the ByteBuffer to initialize the buffer with
     * @param dataType         the data type of the buffer
     * @param length           the length of the buffer
     */
    public CudaBoolDataBuffer(ByteBuffer underlyingBuffer, DataType dataType, long length) {
        super(underlyingBuffer, dataType, length);
    }

    /**
     * Initialize the data type and element size for BOOL
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 1;
        type = DataType.BOOL;
    }

    /**
     * Constructor for float[] data
     *
     * @param buffer the float array to initialize the buffer with
     */
    public CudaBoolDataBuffer(float[] buffer) {
        super(buffer.length, 1, true); // Initialize with length, element size 1, and initialize flag
        setData(buffer); // Set the data using the overridden setData(float[]) method
    }

    /**
     * Constructor for double[] data
     *
     * @param data the double array to initialize the buffer with
     */
    public CudaBoolDataBuffer(double[] data) {
        super(data.length, 1, true); // Initialize with length, element size 1, and initialize flag
        setData(data); // Set the data using the overridden setData(double[]) method
    }

    /**
     * Constructor for int[] data
     *
     * @param data the int array to initialize the buffer with
     */
    public CudaBoolDataBuffer(int[] data) {
        super(data.length, 1, true); // Initialize with length, element size 1, and initialize flag
        setData(data); // Set the data using the overridden setData(int[]) method
    }

    @Override
    protected DataBuffer create(long length) {
        return new CudaBoolDataBuffer(length);
    }

    @Override
    public float[] getFloatsAt(long offset, long inc, int length) {
        return super.getFloatsAt(offset, inc, length);
    }

    @Override
    public double[] getDoublesAt(long offset, long inc, int length) {
        return ArrayUtil.toDoubles(getFloatsAt(offset, inc, length));
    }

    @Override
    public void setData(float[] data) {
        if (data.length == 0)
            return;
        byte[] byteData = new byte[data.length];
        for(int i = 0; i < data.length; i++) {
            byteData[i] = (byte)(data[i] != 0.0f ? 1 : 0);
        }
        set(byteData, byteData.length, 0, 0);
    }

    @Override
    public void setData(int[] data) {
        if (data.length == 0)
            return;
        byte[] byteData = new byte[data.length];
        for(int i = 0; i < data.length; i++) {
            byteData[i] = (byte)(data[i] != 0 ? 1 : 0);
        }
        set(byteData, byteData.length, 0, 0);
    }

    @Override
    public void setData(double[] data) {
        if (data.length == 0)
            return;
        byte[] byteData = new byte[data.length];
        for(int i = 0; i < data.length; i++) {
            byteData[i] = (byte)(data[i] != 0.0 ? 1 : 0);
        }
        set(byteData, byteData.length, 0, 0);
    }

    @Override
    public DataType dataType() {
        return DataType.BOOL;
    }

    @Override
    public float[] asFloat() {
        return super.asFloat();
    }

    @Override
    public double[] asDouble() {
        return ArrayUtil.toDoubles(asFloat());
    }

    @Override
    public int[] asInt() {
        return ArrayUtil.toInts(asFloat());
    }

    @Override
    public double getDouble(long i) {
        return super.getFloat(i);
    }

    @Override
    public DataBuffer create(double[] data) {
        return new CudaBoolDataBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new CudaBoolDataBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new CudaBoolDataBuffer(data);
    }

    @Override
    public void flush() {
        // No action needed for BOOL data
    }
}
