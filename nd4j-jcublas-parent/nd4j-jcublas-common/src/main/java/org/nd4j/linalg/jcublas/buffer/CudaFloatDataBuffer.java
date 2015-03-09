/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jcublas.buffer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.util.ArrayUtil;

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
    public CudaFloatDataBuffer(int length) {
        super(length, Sizeof.FLOAT);
    }

    public CudaFloatDataBuffer(float[] buffer) {
        this(buffer.length);
        setData(buffer);
    }


    @Override
    public void assign(int[] indices, float[] data, boolean contiguous, int inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

        if (contiguous) {
            int offset = indices[0];
            Pointer p = Pointer.to(data);
            set(offset, data.length, p, inc);
        } else
            throw new UnsupportedOperationException("Only contiguous supported");
    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous, int inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

        if (contiguous) {
            int offset = indices[0];
            Pointer p = Pointer.to(data);
            set(offset, data.length, p, inc);
        } else
            throw new UnsupportedOperationException("Only contiguous supported");
    }


    @Override
    public double[] getDoublesAt(int offset, int inc, int length) {
        return ArrayUtil.toDoubles(getFloatsAt(offset, inc, length));
    }

    @Override
    public float[] getFloatsAt(int offset, int inc, int length) {
        if (offset + length > length())
            length -= offset;
        float[] ret = new float[length];
        Pointer p = Pointer.to(ret);
        get(offset, inc, length, p);
        return ret;
    }

    @Override
    public void assign(Number value, int offset) {
        int arrLength = length - offset;
        float[] data = new float[arrLength];
        for (int i = 0; i < data.length; i++)
            data[i] = value.floatValue();
        set(offset, arrLength, Pointer.to(data));

    }

    @Override
    public void setData(int[] data) {
        setData(ArrayUtil.toFloats(data));
    }

    @Override
    public void setData(float[] data) {

        if (data.length != length)
            throw new IllegalArgumentException("Unable to set vector, must be of length " + length() + " but found length " + data.length);

        if (pointer() != null) {
            destroy();
            pointer = null;
        }

        if (pointer() == null)
            alloc();
        try {
            JCublas.cublasSetVector(
                    length(),
                    elementSize()
                    , Pointer.to(data)
                    , 1
                    , pointer()
                    , 1);
        }catch(Exception e) {
            try {
                JCuda.cudaMemcpy(pointer(), Pointer.to(data), elementSize() * length(), cudaMemcpyKind.cudaMemcpyHostToDevice);
            }catch(Exception e1) {
                throw new RuntimeException(e1);
            }
        }

    }

    @Override
    public void setData(double[] data) {
        setData(ArrayUtil.toFloats(data));
    }

    @Override
    public byte[] asBytes() {
        return new byte[0];
    }

    @Override
    public int dataType() {
        return DataBuffer.FLOAT;
    }

    @Override
    public float[] asFloat() {
        float[] ret = new float[length];
        Pointer p = Pointer.to(ret);
        JCublas.cublasGetVector(
                length,
                elementSize(),
                pointer(),
                1,
                p,
                1);
        return ret;
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
    public double getDouble(int i) {
        return getFloat(i);
    }

    @Override
    public float getFloat(int i) {
        float[] data = new float[1];
        Pointer p = Pointer.to(data);
        get(i, p);
        return data[0];
    }

    @Override
    public Number getNumber(int i) {
        return getFloat(i);
    }


    @Override
    public void put(int i, float element) {
        float[] data = new float[]{element};
        Pointer p = Pointer.to(data);
        set(i, p);
    }

    @Override
    public void put(int i, double element) {
        put(i, (float) element);
    }

    @Override
    public void put(int i, int element) {
        put(i, (float) element);
    }


    @Override
    public int getInt(int ix) {
        return (int) getFloat(ix);
    }

    @Override
    public DataBuffer dup() {
        CudaFloatDataBuffer buf = new CudaFloatDataBuffer(length());
        copyTo(buf);
        return buf;
    }

    @Override
    public void flush() {

    }

    @Override
    public void put(int i, IComplexNumber result) {

    }

    private void writeObject(java.io.ObjectOutputStream stream)
            throws java.io.IOException
    {
        stream.defaultWriteObject();

        if (pointer() == null) {
            stream.writeInt(0);
        }
        else {
            float[] arr = this.asFloat();

            stream.writeInt(arr.length);
            for (int i = 0; i < arr.length; i ++) {
                stream.writeFloat(arr[i]);
            }
        }
    }

    private void readObject(java.io.ObjectInputStream stream)
            throws java.io.IOException, ClassNotFoundException
    {
        stream.defaultReadObject();

        int n = stream.readInt();
        float[] arr = new float[n];

        for (int i = 0; i < n; i ++) {
            arr[i] = stream.readFloat();
        }
        setData(arr);
    }
}
