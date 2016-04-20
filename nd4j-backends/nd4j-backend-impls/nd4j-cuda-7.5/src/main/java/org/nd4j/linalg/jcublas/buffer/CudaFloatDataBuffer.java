/*
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

import io.netty.buffer.ByteBuf;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
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
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaFloatDataBuffer(long length) {
        super(length, 4);
    }

    public CudaFloatDataBuffer(long length, int elementSize) {
        super(length, elementSize);
    }

    public CudaFloatDataBuffer(long length, int elementSize, long offset) {
        super(length, elementSize, offset);
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

    public CudaFloatDataBuffer(float[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public CudaFloatDataBuffer(double[] data) {
        super(data);
    }

    public CudaFloatDataBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public CudaFloatDataBuffer(double[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public CudaFloatDataBuffer(int[] data) {
        super(data);
    }

    public CudaFloatDataBuffer(int[] data, boolean copy) {
        super(data, copy);
    }

    public CudaFloatDataBuffer(int[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public CudaFloatDataBuffer(ByteBuf buf, long length) {
        super(buf, length);
    }

    public CudaFloatDataBuffer(ByteBuf buf, long length, int offset) {
        super(buf, length, offset);
    }

    public CudaFloatDataBuffer(byte[] data, long length) {
        super(data, length);
    }

    public CudaFloatDataBuffer(ByteBuffer buffer, long length) {
        super(buffer, length);
    }

    public CudaFloatDataBuffer(ByteBuffer buffer, long length, int offset) {
        super(buffer, length, offset);
    }


    @Override
    public void assign(long[] indices, float[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

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
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

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
        for(int i = 0; i < data.length; i++)
            try {
                dos.writeFloat(data[i]);
            } catch (IOException e) {
                e.printStackTrace();
            }
        return bos.toByteArray();
    }

    @Override
    public Type dataType() {
        return Type.FLOAT;
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
    public DataBuffer create(ByteBuf buf, int length) {
        return new CudaFloatDataBuffer(buf,length);
    }

    @Override
    public void flush() {

    }



}