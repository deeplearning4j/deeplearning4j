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


import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;

/**
 * UTF-8 buffer
 *
 * @author Adam Gibson
 */
public class CudaUtf8Buffer extends BaseCudaDataBuffer {

    protected Collection<Pointer> references = new ArrayList<>();

    @Getter
    protected long numWords = 0;

    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaUtf8Buffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public CudaUtf8Buffer(ByteBuffer buffer, DataType dataType, long length, long offset) {
        super(buffer, dataType, length, offset);
    }

    public CudaUtf8Buffer(long length) {
        super(length);
    }

    public CudaUtf8Buffer(long length, boolean initialize) {
        super((length + 1) * 8, 1, initialize);
        numWords = length;
    }

    public CudaUtf8Buffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super((length + 1) * 8, 1, initialize, workspace);
        numWords = length;
    }

    public CudaUtf8Buffer(int[] ints, boolean copy, MemoryWorkspace workspace) {
        super(ints, copy, workspace);
    }

    public CudaUtf8Buffer(byte[] data, long numWords) {
        super(data.length, 1, false);

        lazyAllocateHostPointer();

        val bp = (BytePointer) pointer;
        bp.put(data);
        this.numWords = numWords;
    }

    public CudaUtf8Buffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public CudaUtf8Buffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaUtf8Buffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public CudaUtf8Buffer(long[] data, boolean copy) {
        super(data, copy);
    }

    public CudaUtf8Buffer(long[] data, boolean copy, MemoryWorkspace workspace) {
        super(data, copy);
    }

    public CudaUtf8Buffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaUtf8Buffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaUtf8Buffer(int length, int elementSize) {
        super(length, elementSize);
    }

    public CudaUtf8Buffer(int length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    public CudaUtf8Buffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
        this.numWords = length;

        Preconditions.checkArgument(((CudaUtf8Buffer) underlyingBuffer).numWords == numWords, "String array can't be a view");
    }

    public CudaUtf8Buffer(@NonNull Collection<String> strings) {
        super(CudaUtf8Buffer.stringBufferRequiredLength(strings), 1, false);
        lazyAllocateHostPointer();

        // at this point we should have fully allocated buffer, time to fill length
        val headerLength = (strings.size() + 1) * 8;
        val headerPointer = new LongPointer(this.pointer);
        val dataPointer = new BytePointer(this.pointer);

        numWords = strings.size();

        long cnt = 0;
        long currentLength = 0;
        for (val s: strings) {
            headerPointer.put(cnt++, currentLength);
            val length = s.length();
            val chars = s.toCharArray();

            // putting down chars
            for (int e = 0; e < length; e++) {
                val b = (byte) chars[e];
                val idx = headerLength + currentLength + e;
                dataPointer.put(idx, b);
            }

            currentLength += length;
        }
        headerPointer.put(cnt, currentLength);
        allocationPoint.tickHostWrite();
    }

    public String getString(long index) {
        if (index > numWords)
            throw new IllegalArgumentException("Requested index [" + index + "] is above actual number of words stored: [" + numWords + "]");

        val headerPointer = new LongPointer(this.pointer);
        val dataPointer = (BytePointer) (this.pointer);

        val start = headerPointer.get(index);
        val end = headerPointer.get(index+1);

        if (end - start > Integer.MAX_VALUE)
            throw new IllegalStateException("Array is too long for Java");

        val dataLength = (int) (end - start);
        val bytes = new byte[dataLength];

        val headerLength = (numWords + 1) * 8;

        for (int e = 0; e < dataLength; e++) {
            val idx = headerLength + start + e;
            bytes[e] = dataPointer.get(idx);
        }

        try {
            return new String(bytes, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected DataBuffer create(long length) {
        return new CudaUtf8Buffer(length);
    }

    @Override
    public DataBuffer create(double[] data) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer create(float[] data) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataBuffer create(int[] data) {
        throw new UnsupportedOperationException();
    }

    private static long stringBufferRequiredLength(@NonNull Collection<String> strings) {
        // header size first
        long size = (strings.size() + 1) * 8;

        for (val s:strings)
            size += s.length();

        return size;
    }

    public void put(long index, Pointer pointer) {
        throw new UnsupportedOperationException();
        //references.add(pointer);
        //((LongIndexer) indexer).put(index, pointer.address());
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 1;
        type = DataType.UTF8;
    }


}
