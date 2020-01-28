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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Cuda Short buffer
 *
 * @author raver119@gmail.com
 */
public class CudaBoolDataBuffer extends BaseCudaDataBuffer {
    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaBoolDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public CudaBoolDataBuffer(Pointer pointer, Pointer specialPointer, Indexer indexer, long length){
        super(pointer, specialPointer, indexer, length);
    }

    public CudaBoolDataBuffer(ByteBuffer buffer, DataType dataType, long length, long offset) {
        super(buffer, dataType, length, offset);
    }

    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaBoolDataBuffer(long length) {
        super(length, 1);
    }

    public CudaBoolDataBuffer(long length, boolean initialize) {
        super(length, 1, initialize);
    }

    public CudaBoolDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }

    public CudaBoolDataBuffer(long length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    public CudaBoolDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 1, initialize, workspace);
    }

    public CudaBoolDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        super(data, copy,0, workspace);
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 1;
        type = DataType.BOOL;
    }

    public CudaBoolDataBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public CudaBoolDataBuffer(float[] buffer) {
        super(buffer);
    }

    public CudaBoolDataBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public CudaBoolDataBuffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaBoolDataBuffer(float[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        super(data, copy, offset, workspace);
    }

    public CudaBoolDataBuffer(double[] data) {
        super(data);
    }

    public CudaBoolDataBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public CudaBoolDataBuffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaBoolDataBuffer(int[] data) {
        super(data);
    }

    public CudaBoolDataBuffer(int[] data, boolean copy) {
        super(data, copy);
    }

    public CudaBoolDataBuffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    @Override
    protected DataBuffer create(long length) {
        return new CudaBoolDataBuffer(length);
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
    public void setData(float[] data) {
        setData(ArrayUtil.toShorts(data));
    }

    @Override
    public void setData(int[] data) {
        setData(ArrayUtil.toShorts(data));
    }



    @Override
    public void setData(double[] data) {
        setData(ArrayUtil.toFloats(data));
    }

    @Override
    public DataType dataType() {
        return DataType.BOOL;
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
        return new CudaBoolDataBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new CudaBoolDataBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new CudaBoolDataBuffer(data);
    }

    @Override
    public void flush() {

    }



}
