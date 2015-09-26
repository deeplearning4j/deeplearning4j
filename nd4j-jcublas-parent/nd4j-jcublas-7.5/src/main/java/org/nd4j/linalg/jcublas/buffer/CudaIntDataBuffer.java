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
import jcuda.Pointer;
import jcuda.Sizeof;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Cuda int buffer
 *
 * @author Adam Gibson
 */
public class CudaIntDataBuffer extends BaseCudaDataBuffer {
    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaIntDataBuffer(int length) {
        super(length, Sizeof.INT);
    }

    public CudaIntDataBuffer(int[] data) {
        this(data.length);
        setData(data);
    }

    @Override
    public void assign(int[] indices, float[] data, boolean contiguous, int inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

        if (!contiguous)
            throw new UnsupportedOperationException("Non contiguous is not supported");

    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous, int inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

        if (!contiguous)
            throw new UnsupportedOperationException("Non contiguous is not supported");

    }

    @Override
    public double[] getDoublesAt(int offset, int length) {
        return new double[0];
    }

    @Override
    public float[] getFloatsAt(int offset, int length) {
        return new float[0];
    }

    @Override
    public double[] getDoublesAt(int offset, int inc, int length) {
        return new double[0];
    }

    @Override
    public float[] getFloatsAt(int offset, int inc, int length) {
        return new float[0];
    }

    @Override
    public void assign(Number value, int offset) {
        int arrLength = length - offset;
        int[] data = new int[arrLength];
        for (int i = 0; i < data.length; i++)
            data[i] = value.intValue();
        set(offset, arrLength, Pointer.to(data));
    }

    @Override
    public void setData(int[] data) {

    }

    @Override
    public void setData(float[] data) {

    }

    @Override
    public void setData(double[] data) {

    }

    @Override
    public byte[] asBytes() {
        return new byte[0];
    }

    @Override
    public DataBuffer.Type dataType() {
        return DataBuffer.Type.INT;
    }

    @Override
    public float[] asFloat() {
        return new float[0];
    }

    @Override
    public double[] asDouble() {
        return new double[0];
    }

    @Override
    public int[] asInt() {
        return new int[0];
    }


    @Override
    public double getDouble(int i) {
        return 0;
    }

    @Override
    public float getFloat(int i) {
        return 0;
    }

    @Override
    public Number getNumber(int i) {
        return null;
    }

    @Override
    public void put(int i, float element) {

    }

    @Override
    public void put(int i, double element) {

    }

    @Override
    public void put(int i, int element) {

    }


    @Override
    public int getInt(int ix) {
        return 0;
    }

    @Override
    public DataBuffer dup() {
        return null;
    }

    @Override
    protected DataBuffer create(int length) {
        return new CudaIntDataBuffer(length);
    }

    @Override
    public DataBuffer create(double[] data) {
        return null;
    }

    @Override
    public DataBuffer create(float[] data) {
        return null;
    }

    @Override
    public DataBuffer create(int[] data) {
        return null;
    }

    @Override
    public DataBuffer create(ByteBuf buf, int length) {
        return null;
    }

    @Override
    public void flush() {

    }


    private void writeObject(java.io.ObjectOutputStream stream)
            throws java.io.IOException {
        stream.defaultWriteObject();

        if (getHostPointer() == null) {
            stream.writeInt(0);
        } else {
            int[] arr = this.asInt();

            stream.writeInt(arr.length);
            for (int i = 0; i < arr.length; i++) {
                stream.writeInt(arr[i]);
            }
        }
    }

    private void readObject(java.io.ObjectInputStream stream)
            throws java.io.IOException, ClassNotFoundException {
        stream.defaultReadObject();

        int n = stream.readInt();
        int[] arr = new int[n];

        for (int i = 0; i < n; i++) {
            arr[i] = stream.readInt();
        }
        setData(arr);
    }


}
