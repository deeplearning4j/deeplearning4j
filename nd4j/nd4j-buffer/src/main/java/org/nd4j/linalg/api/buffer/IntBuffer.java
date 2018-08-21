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
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.nio.ByteBuffer;

/**
 * Int buffer
 *
 * @author Adam Gibson
 */
public class IntBuffer extends BaseDataBuffer {

    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public IntBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public IntBuffer(long length) {
        super(length);
    }

    public IntBuffer(long length, boolean initialize) {
        super(length, initialize);
    }

    public IntBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, initialize, workspace);
    }

    public IntBuffer(int[] ints, boolean copy, MemoryWorkspace workspace) {
        super(ints, copy, workspace);
    }

    public IntBuffer(ByteBuffer buffer, int length, long offset) {
        super(buffer, length, offset);
    }

    public IntBuffer(byte[] data, int length) {
        super(data, length);
    }

    public IntBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public IntBuffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public IntBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public IntBuffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public IntBuffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public IntBuffer(int length, int elementSize) {
        super(length, elementSize);
    }

    public IntBuffer(int length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    public IntBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public IntBuffer(ByteBuffer buffer, int length) {
        super(buffer, length);
    }

    @Override
    protected DataBuffer create(long length) {
        return new IntBuffer(length);
    }

    public IntBuffer(int[] data) {
        super(data);
    }

    public IntBuffer(double[] data) {
        super(data);
    }

    public IntBuffer(float[] data) {
        super(data);
    }

    @Override
    public DataBuffer create(double[] data) {
        return new IntBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new IntBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new IntBuffer(data);
    }

    public IntBuffer(int[] data, boolean copy) {
        super(data, copy);
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 4;
        type = Type.INT;
    }


}
