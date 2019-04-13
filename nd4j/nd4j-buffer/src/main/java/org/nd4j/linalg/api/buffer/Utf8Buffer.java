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


import lombok.NonNull;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;

/**
 * UTF-8 buffer
 *
 * @author Adam Gibson
 */
public class Utf8Buffer extends BaseDataBuffer {

    protected Collection<Pointer> references = new ArrayList<>();

    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public Utf8Buffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public Utf8Buffer(long length) {
        super(length);
    }

    public Utf8Buffer(long length, boolean initialize) {
        super(length, initialize);
    }

    public Utf8Buffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, initialize, workspace);
    }

    public Utf8Buffer(int[] ints, boolean copy, MemoryWorkspace workspace) {
        super(ints, copy, workspace);
    }

    public Utf8Buffer(ByteBuffer buffer, int length, long offset) {
        super(buffer, length, offset);
    }

    public Utf8Buffer(byte[] data, int length) {
        super(data, length);
    }

    public Utf8Buffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public Utf8Buffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public Utf8Buffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public Utf8Buffer(long[] data, boolean copy) {
        super(data, copy);
    }

    public Utf8Buffer(long[] data, boolean copy, MemoryWorkspace workspace) {
        super(data, copy, workspace);
    }

    public Utf8Buffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public Utf8Buffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public Utf8Buffer(int length, int elementSize) {
        super(length, elementSize);
    }

    public Utf8Buffer(int length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    public Utf8Buffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public Utf8Buffer(@NonNull Collection<String> strings) {
        super(strings.size(), false);
    }

    public Utf8Buffer(ByteBuffer buffer, int length) {
        super(buffer, length);
    }

    @Override
    protected DataBuffer create(long length) {
        return new Utf8Buffer(length);
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


    public void put(long index, Pointer pointer) {
        references.add(pointer);
        ((LongIndexer) indexer).put(index, pointer.address());
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 8;
        type = DataType.UTF8;
    }


}
