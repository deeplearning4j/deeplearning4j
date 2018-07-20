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

package org.nd4j.linalg.api.buffer;


import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.nio.ByteBuffer;

/**
 * Double buffer implementation of data buffer
 *
 * @author Adam Gibson
 */
public class DoubleBuffer extends BaseDataBuffer {
    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public DoubleBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public DoubleBuffer(long length) {
        super(length);
    }

    public DoubleBuffer(long length, boolean initialize) {
        super(length, initialize);
    }

    public DoubleBuffer(int length, int elementSize) {
        super(length, elementSize);
    }

    public DoubleBuffer(int length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    public DoubleBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public DoubleBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, initialize, workspace);
    }

    public DoubleBuffer(double[] data, MemoryWorkspace workspace) {
        this(data, true, workspace);
    }

    public DoubleBuffer(double[] floats, boolean copy, MemoryWorkspace workspace) {
        super(floats, copy, workspace);
    }

    public DoubleBuffer(double[] data) {
        super(data);
    }

    public DoubleBuffer(int[] data) {
        this(data, true);
    }

    public DoubleBuffer(int[] data, boolean copyOnOps) {
        super(data, copyOnOps);
    }

    public DoubleBuffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public DoubleBuffer(float[] data) {
        this(data, true);
    }

    public DoubleBuffer(float[] data, boolean copyOnOps) {
        super(data, copyOnOps);
    }

    public DoubleBuffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public DoubleBuffer(ByteBuffer buffer, int length) {
        super(buffer, length);
    }

    public DoubleBuffer(ByteBuffer buffer, int length, long offset) {
        super(buffer, length, offset);
    }

    public DoubleBuffer(byte[] data, int length) {
        super(data, length);
    }

    @Override
    public IComplexFloat getComplexFloat(long i) {
        return null;
    }

    @Override
    public IComplexDouble getComplexDouble(long i) {
        return null;
    }

    public DoubleBuffer(double[] doubles, boolean copy) {
        super(doubles, copy);
    }

    public DoubleBuffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public DoubleBuffer(double[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        super(data, copy, offset, workspace);
    }

    @Override
    public float getFloat(long i) {
        return (float) getDouble(i);
    }

    @Override
    public Number getNumber(long i) {
        return getDouble(i);
    }

    @Override
    public DataBuffer create(double[] data) {
        return new DoubleBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new DoubleBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new DoubleBuffer(data);
    }

    @Override
    protected DataBuffer create(long length) {
        return new DoubleBuffer(length);
    }


    @Override
    public void flush() {}

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 8;
        type = Type.DOUBLE;
    }



}
