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
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.common.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Cuda double buffer
 *
 * @author Adam Gibson
 */
public class CudaDoubleDataBuffer extends BaseCudaDataBuffer {
    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaDoubleDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public CudaDoubleDataBuffer(Pointer pointer, Pointer specialPointer, Indexer indexer, long length){
        super(pointer, specialPointer, indexer, length);
    }

    public CudaDoubleDataBuffer(ByteBuffer buffer, DataType dataType, long length, long offset) {
        super(buffer, dataType, length, offset);
    }

    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaDoubleDataBuffer(long length) {
        super(length, 8);
    }

    public CudaDoubleDataBuffer(long length, boolean initialize) {
        super(length, 8, initialize);
    }

    public CudaDoubleDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 8, initialize, workspace);
    }

    public CudaDoubleDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }

    public CudaDoubleDataBuffer(long length, int elementSize, boolean initialize, @NonNull MemoryWorkspace workspace) {
        super(length, elementSize, initialize, workspace);
    }

    public CudaDoubleDataBuffer() {
    }

    public CudaDoubleDataBuffer(double[] data, boolean copy, MemoryWorkspace workspace) {
        super(data, copy, workspace);
    }

    public CudaDoubleDataBuffer(ByteBuffer underlyingBuffer, DataType dataType, long length) {
        super(underlyingBuffer, dataType, length);
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        type = DataType.DOUBLE;
        elementSize = 8;
    }

    /**
     * Instantiate based on the given data
     *
     * @param data the data to instantiate with
     */
    public CudaDoubleDataBuffer(double[] data) {
        super(data.length, 8, true);
        setData(data);
    }

    public CudaDoubleDataBuffer(float[] data) {
        super(data.length, 8, true);
        setData(data);
    }

    public CudaDoubleDataBuffer(int[] data) {
        super(data.length, 8, true);
        setData(data);
    }

    @Override
    protected DataBuffer create(long length) {
        return new CudaDoubleDataBuffer(length);
    }

    @Override
    public void setData(int[] data) {
        if (data.length == 0)
            return;
        double[] doubleData = ArrayUtil.toDoubles(data);
        copyDataFromSrc(new DoublePointer(doubleData), data.length, 0, 0);
    }

    @Override
    public void setData(float[] data) {
        if (data.length == 0)
            return;
        double[] doubleData = ArrayUtil.toDoubles(data);
        copyDataFromSrc(new DoublePointer(doubleData), data.length, 0, 0);
    }

    @Override
    public void setData(double[] data) {
        if (data.length == 0)
            return;
        copyDataFromSrc(new DoublePointer(data), data.length, 0, 0);
    }

    @Override
    public DataType dataType() {
        return DataType.DOUBLE;
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
        return super.getDouble(i);
    }

    @Override
    public DataBuffer create(double[] data) {
        return new CudaDoubleDataBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new CudaDoubleDataBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new CudaDoubleDataBuffer(data);
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

        this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, elementSize, DataType.DOUBLE), false);

        setData(arr);
    }
}
