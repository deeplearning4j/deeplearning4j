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

package org.nd4j.linalg.jcublas.buffer;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Cuda int buffer
 *
 * @author Adam Gibson
 */
public class CudaIntDataBuffer extends BaseCudaDataBuffer {
    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaIntDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaIntDataBuffer(long length) {
        super(length, 4);
    }

    public CudaIntDataBuffer(long length, MemoryWorkspace workspace) {
        super(length, 4, workspace);
    }

    public CudaIntDataBuffer(long length, boolean initialize) {
        super(length, 4, initialize);
    }

    public CudaIntDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 4, initialize, workspace);
    }

    public CudaIntDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }

    public CudaIntDataBuffer(long length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    public CudaIntDataBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public CudaIntDataBuffer(int[] data) {
        this(data.length);
        setData(data);
    }

    public CudaIntDataBuffer(long[] data) {
        this(data.length);
        setData(data);
    }

    public CudaIntDataBuffer(int[] data, MemoryWorkspace workspace) {
        this(data.length, workspace);
        setData(data);
    }

    public CudaIntDataBuffer(int[] data, boolean copy) {
        super(data, copy);
    }

    public CudaIntDataBuffer(int[] data, boolean copy, MemoryWorkspace workspace) {
        super(data, copy, workspace);
    }

    public CudaIntDataBuffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }


    public CudaIntDataBuffer(byte[] data, int length) {
        super(data, length);
    }

    public CudaIntDataBuffer(double[] data) {
        super(data);
    }

    public CudaIntDataBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public CudaIntDataBuffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaIntDataBuffer(float[] data) {
        super(data);
    }

    public CudaIntDataBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public CudaIntDataBuffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaIntDataBuffer(ByteBuffer buffer, int length) {
        super(buffer, length);
    }

    public CudaIntDataBuffer(ByteBuffer buffer, int length, long offset) {
        super(buffer, length, offset);
    }

    @Override
    public void assign(long[] indices, float[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                            + length() + " where the indices are of length " + data.length);

        if (!contiguous)
            throw new UnsupportedOperationException("Non contiguous is not supported");

    }

    @Override
    public void assign(long[] indices, double[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                            + length() + " where the indices are of length " + data.length);

        if (!contiguous)
            throw new UnsupportedOperationException("Non contiguous is not supported");

    }



    @Override
    protected DataBuffer create(long length) {
        return new CudaIntDataBuffer(length);
    }

    @Override
    public DataBuffer create(double[] data) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    @Override
    public DataBuffer create(float[] data) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    @Override
    public DataBuffer create(int[] data) {
        return new CudaIntDataBuffer(data);
    }

    private void writeObject(java.io.ObjectOutputStream stream) throws java.io.IOException {
        stream.defaultWriteObject();

        if (getHostPointer() == null) {
            stream.writeInt(0);
        } else {
            int[] arr = this.asInt();

            stream.writeInt(arr.length);
            for (int i = 0; i < arr.length; i++) {
                stream.writeInt(arr[i]);
            }
        }
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 4;
        type = Type.INT;
    }



    private void readObject(java.io.ObjectInputStream stream) throws java.io.IOException, ClassNotFoundException {
        stream.defaultReadObject();

        int n = stream.readInt();
        int[] arr = new int[n];

        for (int i = 0; i < n; i++) {
            arr[i] = stream.readInt();
        }
        setData(arr);
    }
}
