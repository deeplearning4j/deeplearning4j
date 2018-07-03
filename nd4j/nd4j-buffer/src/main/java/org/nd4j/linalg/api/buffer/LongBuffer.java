/*-
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


import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.nio.ByteBuffer;

/**
 * Int buffer
 *
 * @author Adam Gibson
 */
public class LongBuffer extends BaseDataBuffer {

    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public LongBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public LongBuffer(long length) {
        super(length);
    }

    public LongBuffer(long length, boolean initialize) {
        super(length, initialize);
    }

    public LongBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, initialize, workspace);
    }

    public LongBuffer(int[] ints, boolean copy, MemoryWorkspace workspace) {
        super(ints, copy, workspace);
    }

    public LongBuffer(ByteBuffer buffer, int length, long offset) {
        super(buffer, length, offset);
    }

    public LongBuffer(byte[] data, int length) {
        super(data, length);
    }

    public LongBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public LongBuffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public LongBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public LongBuffer(long[] data, boolean copy) {
        super(data, copy);
    }

    public LongBuffer(long[] data, boolean copy, MemoryWorkspace workspace) {
        super(data, copy, workspace);
    }

    public LongBuffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public LongBuffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public LongBuffer(int length, int elementSize) {
        super(length, elementSize);
    }

    public LongBuffer(int length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    public LongBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public LongBuffer(ByteBuffer buffer, int length) {
        super(buffer, length);
    }

    @Override
    protected DataBuffer create(long length) {
        return new LongBuffer(length);
    }

    public LongBuffer(int[] data) {
        super(data);
    }

    public LongBuffer(double[] data) {
        super(data);
    }

    public LongBuffer(float[] data) {
        super(data);
    }

    public LongBuffer(long[] data) {
        super(data, true);
    }

    @Override
    public DataBuffer create(double[] data) {
        return new LongBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new LongBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new LongBuffer(data);
    }

    @Override
    public IComplexFloat getComplexFloat(long i) {
        return null;
    }

    @Override
    public IComplexDouble getComplexDouble(long i) {
        throw new UnsupportedOperationException();

    }

    public LongBuffer(int[] data, boolean copy) {
        super(data, copy);
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 8;
        type = Type.LONG;
    }


}
