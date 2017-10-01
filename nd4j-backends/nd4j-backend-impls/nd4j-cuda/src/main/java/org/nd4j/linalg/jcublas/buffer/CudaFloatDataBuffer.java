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

package org.nd4j.linalg.jcublas.buffer;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Cuda float buffer
 *
 * @author Adam Gibson
 */
public class CudaFloatDataBuffer extends BaseCudaDataBuffer {
    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public CudaFloatDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaFloatDataBuffer(long length) {
        super(length, 4);
    }

    public CudaFloatDataBuffer(long length, boolean initialize) {
        super(length, 4, initialize);
    }

    public CudaFloatDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 4, initialize, workspace);
    }


    public CudaFloatDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }

    public CudaFloatDataBuffer(long length, int elementSize, long offset) {
        super(length, elementSize, offset);
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        elementSize = 4;
        type = Type.FLOAT;
    }

    public CudaFloatDataBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public CudaFloatDataBuffer(float[] buffer) {
        super(buffer);
    }

    public CudaFloatDataBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public CudaFloatDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        super(data, copy, workspace);
    }

    public CudaFloatDataBuffer(float[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaFloatDataBuffer(float[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        super(data, copy, offset, workspace);
    }

    public CudaFloatDataBuffer(double[] data) {
        super(data);
    }

    public CudaFloatDataBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public CudaFloatDataBuffer(double[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaFloatDataBuffer(int[] data) {
        super(data);
    }

    public CudaFloatDataBuffer(int[] data, boolean copy) {
        super(data, copy);
    }

    public CudaFloatDataBuffer(int[] data, boolean copy, long offset) {
        super(data, copy, offset);
    }

    public CudaFloatDataBuffer(byte[] data, long length) {
        super(data, length);
    }

    public CudaFloatDataBuffer(ByteBuffer buffer, long length) {
        super(buffer, (int) length);
    }

    public CudaFloatDataBuffer(ByteBuffer buffer, long length, long offset) {
        super(buffer, length, offset);
    }


    @Override
    public void assign(long[] indices, float[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                            + length() + " where the indices are of length " + data.length);

        if (contiguous) {
            /*   long offset = indices[0];
            Pointer p = Pointer.to(data);
            set(offset, data.length, p, inc);
            */
            throw new UnsupportedOperationException();
        } else
            throw new UnsupportedOperationException("Only contiguous supported");
    }

    @Override
    public void assign(long[] indices, double[] data, boolean contiguous, long inc) {

        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                            + length() + " where the indices are of length " + data.length);

        if (contiguous) {
            /*long offset = indices[0];
            Pointer p = Pointer.to(data);
            set(offset, data.length, p, inc);
            */
            throw new UnsupportedOperationException();
        } else
            throw new UnsupportedOperationException("Only contiguous supported");
    }

    @Override
    protected DataBuffer create(long length) {
        return new CudaFloatDataBuffer(length);
    }


    @Override
    public double[] getDoublesAt(long offset, long inc, int length) {
        return ArrayUtil.toDoubles(getFloatsAt(offset, inc, length));
    }


    @Override
    public void setData(int[] data) {
        setData(ArrayUtil.toFloats(data));
    }



    @Override
    public void setData(double[] data) {
        setData(ArrayUtil.toFloats(data));
    }

    @Override
    public byte[] asBytes() {
        float[] data = asFloat();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        for (int i = 0; i < data.length; i++)
            try {
                dos.writeFloat(data[i]);
            } catch (IOException e) {
                e.printStackTrace();
            }
        return bos.toByteArray();
    }

    @Override
    public Type dataType() {
        return type;
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
        return new CudaFloatDataBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new CudaFloatDataBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new CudaFloatDataBuffer(data);
    }

    @Override
    public void flush() {

    }



}
