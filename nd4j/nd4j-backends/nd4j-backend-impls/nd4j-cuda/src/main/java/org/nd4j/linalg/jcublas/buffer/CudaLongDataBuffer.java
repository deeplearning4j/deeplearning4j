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

import lombok.NonNull;
import lombok.val;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Cuda double  buffer
 *
 * @author Adam Gibson
 */
public class CudaLongDataBuffer extends BaseCudaDataBuffer {
    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaLongDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }


    public CudaLongDataBuffer(@NonNull Pointer hostPointer, @NonNull Pointer devicePointer, long numberOfElements) {
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        this.offset = 0;
        this.originalOffset = 0;
        this.underlyingLength = numberOfElements;
        this.length = numberOfElements;
        initTypeAndSize();

        this.pointer = new CudaPointer(hostPointer, numberOfElements).asLongPointer();
        indexer = LongIndexer.create((LongPointer) this.pointer);

        this.allocationPoint = AtomicAllocator.getInstance().pickExternalBuffer(this);

        val pp = new PointersPair(devicePointer, this.pointer);
        allocationPoint.setPointers(pp);
        trackingPoint = allocationPoint.getObjectId();
    }

    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaLongDataBuffer(long length) {
        super(length, 8);
    }

    public CudaLongDataBuffer(long length, boolean initialize) {
        super(length, 8, initialize);
    }

    public CudaLongDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 8, initialize, workspace);
    }

    public CudaLongDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }

    public CudaLongDataBuffer(long length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    public CudaLongDataBuffer(double[] data, boolean copy, MemoryWorkspace workspace) {
        super(data, copy,0, workspace);
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        type = DataType.LONG;
        elementSize = 8;
    }

    public CudaLongDataBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }


    /**
     * Instantiate based on the given data
     *
     * @param data the data to instantiate with
     */
    public CudaLongDataBuffer(double[] data) {
        this(data.length);
        setData(data);
    }

    public CudaLongDataBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public CudaLongDataBuffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaLongDataBuffer(double[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        super(data, copy, offset, workspace);
    }

    public CudaLongDataBuffer(float[] data) {
        super(data);
    }

    public CudaLongDataBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public CudaLongDataBuffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaLongDataBuffer(int[] data) {
        super(data);
    }

    public CudaLongDataBuffer(long[] data) {
        this(data, true);
    }

    public CudaLongDataBuffer(long[] data, boolean copy) {
        super(data.length, 8);

        set(data, data.length, 0, 0);
    }

    public CudaLongDataBuffer(long[] data, MemoryWorkspace workspace) {
        super(data.length, 8, workspace);

        set(data, data.length, 0, 0);
    }

    public CudaLongDataBuffer(int[] data, boolean copy) {
        super(data, copy);
    }

    public CudaLongDataBuffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaLongDataBuffer(byte[] data, long length) {
        super(data, length, DataType.LONG);
    }

    public CudaLongDataBuffer(ByteBuffer buffer, long length) {
        super(buffer, (int) length, DataType.LONG);
    }

    public CudaLongDataBuffer(ByteBuffer buffer, long length, long offset) {
        super(buffer, length, offset, DataType.LONG);
    }


    @Override
    protected DataBuffer create(long length) {
        return new CudaLongDataBuffer(length);
    }


    @Override
    public DataBuffer create(double[] data) {
        return new CudaLongDataBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new CudaLongDataBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new CudaLongDataBuffer(data);
    }

    private void writeObject(java.io.ObjectOutputStream stream) throws java.io.IOException {
        stream.defaultWriteObject();

        if (getHostPointer() == null) {
            stream.writeInt(0);
        } else {
            double[] arr = this.asDouble();

            stream.writeInt(arr.length);
            for (int i = 0; i < arr.length; i++) {
                stream.writeDouble(arr[i]);
            }
        }
    }

    private void readObject(java.io.ObjectInputStream stream) throws java.io.IOException, ClassNotFoundException {
        stream.defaultReadObject();

        int n = stream.readInt();
        double[] arr = new double[n];

        for (int i = 0; i < n; i++) {
            arr[i] = stream.readDouble();
        }

        this.length = n;
        this.elementSize = 8;

        //wrappedBuffer = ByteBuffer.allocateDirect(length() * getElementSize());
        //wrappedBuffer.order(ByteOrder.nativeOrder());

        this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this,
                        new AllocationShape(length, elementSize, DataType.LONG), false);
        this.trackingPoint = allocationPoint.getObjectId();
        //this.wrappedBuffer = allocationPoint.getPointers().getHostPointer().asByteBuffer();
        //this.wrappedBuffer.order(ByteOrder.nativeOrder());

        setData(arr);
    }

}
