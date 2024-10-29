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

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.common.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Cuda Byte buffer implementation
 *
 * @author raver119@gmail.com
 */
public class CudaByteDataBuffer extends BaseCudaDataBuffer {
    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaByteDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public CudaByteDataBuffer(Pointer pointer, Pointer specialPointer, Indexer indexer, long length){
        super(pointer, specialPointer, indexer, length);
    }

    public CudaByteDataBuffer(ByteBuffer buffer, DataType dataType, long length, long offset) {
        super(buffer, dataType, length, offset);
    }

    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaByteDataBuffer(long length) {
        super(length, 1);
    }

    public CudaByteDataBuffer(long length, boolean initialize) {
        super(length, 1, initialize);
    }

    public CudaByteDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }


    public CudaByteDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 1, initialize, workspace);
    }

    public CudaByteDataBuffer(ByteBuffer underlyingBuffer, DataType dataType, long length) {
        super(underlyingBuffer, dataType, length);
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 1;
        type = DataType.BYTE;
    }

    public CudaByteDataBuffer(float[] buffer) {
        super(buffer.length, 1, true);
        setData(buffer);
    }

    public CudaByteDataBuffer(double[] data) {
        super(data.length, 1, true);
        setData(data);
    }

    public CudaByteDataBuffer(int[] data) {
        super(data.length, 1, true);
        setData(data);
    }

    public CudaByteDataBuffer() {
    }

    @Override
    protected DataBuffer create(long length) {
        return new CudaByteDataBuffer(length);
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
        return DataType.BYTE;
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
        return new CudaByteDataBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new CudaByteDataBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new CudaByteDataBuffer(data);
    }

    @Override
    public void flush() {
        // No action needed for BYTE data
    }
}
