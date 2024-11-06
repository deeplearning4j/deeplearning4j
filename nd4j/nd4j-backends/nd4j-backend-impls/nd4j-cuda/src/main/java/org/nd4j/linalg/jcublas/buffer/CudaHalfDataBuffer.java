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

import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.common.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Cuda Half precision buffer
 *
 * @author raver119@gmail.com
 */
public class CudaHalfDataBuffer extends BaseCudaDataBuffer {
    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaHalfDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public CudaHalfDataBuffer(Pointer pointer, Pointer specialPointer, Indexer indexer, long length){
        super(pointer, specialPointer, indexer, length);
    }

    public CudaHalfDataBuffer(ByteBuffer buffer, DataType dataType, long length, long offset) {
        super(buffer, dataType, length, offset);
    }

    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaHalfDataBuffer(long length) {
        super(length, 2);
    }

    public CudaHalfDataBuffer(long length, boolean initialize) {
        super(length, 2, initialize);
    }

    public CudaHalfDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }


    public CudaHalfDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 2, initialize, workspace);
    }

    public CudaHalfDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        super(data.length, 2, true, workspace);
        setData(data);
    }

    public CudaHalfDataBuffer(ByteBuffer underlyingBuffer, DataType dataType, long length) {
        super(underlyingBuffer, dataType, length);
    }

    public CudaHalfDataBuffer(double[] data, boolean copy) {
        super(data.length, 2, true);
        setData(data);
    }

    public CudaHalfDataBuffer(float[] data, boolean copy) {
        super(data.length, 2, true);
        setData(data);
    }

    public CudaHalfDataBuffer(float[] data, boolean b, long offset) {
        super(data.length, 2, true);
        setData(data);
    }

    public CudaHalfDataBuffer(int[] data, boolean copy) {
        super(data.length, 2, true);
        setData(data);
    }


    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 2;
        type = DataType.HALF;
    }


    public CudaHalfDataBuffer(float[] buffer) {
        super(buffer.length, 2, true);
        setData(buffer);
    }




    public CudaHalfDataBuffer(double[] data) {
        super(data.length, 2, true);
        setData(data);
    }



    public CudaHalfDataBuffer(int[] data) {
        super(data.length, 2, true);
        setData(data);
    }



    @Override
    protected DataBuffer create(long length) {
        return new CudaHalfDataBuffer(length);
    }

    @Override
    public void setData(float[] data) {
        if (data.length == 0)
            return;
        val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
        copyDataFromSrc(pointer, data.length, 0, 0);
    }

    @Override
    public void setData(int[] data) {
        if (data.length == 0)
            return;
        val shortData = ArrayUtil.toShorts(data);
        set(shortData, shortData.length, 0, 0);
    }

    @Override
    public void setData(double[] data) {
        if (data.length == 0)
            return;
        val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
        copyDataFromSrc(pointer, data.length, 0, 0);
    }

    @Override
    public void setData(long[] data) {
        if (data.length == 0)
            return;
        val shortData = ArrayUtil.toShorts(data);
        set(shortData, shortData.length, 0, 0);
    }

    @Override
    public void setData(byte[] data) {
        if (data.length == 0)
            return;
        float[] floats = new float[data.length];
        for(int i = 0; i < data.length; i++) {
            floats[i] = data[i];
        }
        setData(floats);
    }

    @Override
    public void setData(short[] data) {
        if (data.length == 0)
            return;
        val pointer = new ShortPointer(data);
        copyDataFromSrc(pointer, data.length, 0, 0);
    }

    @Override
    public void setData(boolean[] data) {
        if (data.length == 0)
            return;
        float[] floats = ArrayUtil.toFloatArray(data);
        setData(floats);
    }

    @Override
    public float[] getFloatsAt(long offset, long inc, int length) {
        return super.getFloatsAt(offset, inc, length);
    }

    @Override
    public double[] getDoublesAt(long offset, long inc, int length) {
        return ArrayUtil.toDoubles(getFloatsAt(offset, inc, length));
    }

    @Override
    public DataType dataType() {
        return DataType.HALF;
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
        return new CudaHalfDataBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new CudaHalfDataBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new CudaHalfDataBuffer(data);
    }

    @Override
    public void flush() {

    }
}
