/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.cpu.nativecpu.buffer;


import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.nio.ByteBuffer;

/**
 * Data buffer for floats
 *
 * @author Adam Gibson
 */
public class UInt64Buffer extends BaseCpuDataBuffer {

    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public UInt64Buffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    /**
     * Create a float buffer with the given length
     * @param length the float buffer with the given length
     */
    public UInt64Buffer(long length) {
        super(length);
    }

    public UInt64Buffer(ByteBuffer buffer, DataType dataType, long length, long offset) {
        super(buffer, dataType, length, offset);
    }

    public UInt64Buffer(long length, boolean initialize) {
        super(length, initialize);
    }

    public UInt64Buffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, initialize, workspace);
    }

    public UInt64Buffer(int length, int elementSize) {
        super(length, elementSize);
    }

    public UInt64Buffer(int length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        type = DataType.UINT64;
        elementSize = 8;
    }

    public UInt64Buffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public UInt64Buffer(float[] data) {
        this(data, true);
    }

    public UInt64Buffer(float[] data, MemoryWorkspace workspace) {
        this(data, true, workspace);
    }

    public UInt64Buffer(int[] data) {
        this(data, true);
    }

    public UInt64Buffer(double[] data) {
        this(data, true);
    }

    public UInt64Buffer(int[] data, boolean copyOnOps) {
        super(data, copyOnOps);
    }

    public UInt64Buffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public UInt64Buffer(double[] data, boolean copyOnOps) {
        super(data, copyOnOps);
    }

    public UInt64Buffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public UInt64Buffer(float[] floats, boolean copy) {
        super(floats, copy);
    }

    public UInt64Buffer(float[] floats, boolean copy, MemoryWorkspace workspace) {
        super(floats, copy, workspace);
    }

    public UInt64Buffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public UInt64Buffer(float[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        super(data, copy, offset, workspace);
    }

    @Override
    protected DataBuffer create(long length) {
        return new UInt64Buffer(length);
    }


    @Override
    public DataBuffer create(double[] data) {
        return new UInt64Buffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new UInt64Buffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new UInt64Buffer(data);
    }


}
