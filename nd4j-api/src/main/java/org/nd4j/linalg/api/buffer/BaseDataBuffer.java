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

package org.nd4j.linalg.api.buffer;

import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.ElementWiseOp;

import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Base class for a data buffer handling basic byte operations among other things.
 *
 * @author Adam Gibson
 */
public abstract class BaseDataBuffer implements DataBuffer {

    public static final int MAPPING_SIZE = 1 << 30;
    protected final List<ByteBuffer> mappings = new ArrayList<>();
    protected int length;
    //memory mapped file
    protected String path;
    protected RandomAccessFile memoryMappedBuffer;

    /**
     * Instantiate a buffer with the given length
     * @param length the length of the buffer
     */
    protected BaseDataBuffer(int length) {
        this.length = length;
    }

    public static byte[] toByteArray(double value) {
        byte[] bytes = new byte[8];
        ByteBuffer.wrap(bytes).putDouble(value);
        return bytes;
    }

    public static byte[] toByteArray(float value) {
        byte[] bytes = new byte[4];
        ByteBuffer.wrap(bytes).putFloat(value);
        return bytes;
    }

    public static byte[] toByteArray(int value) {
        byte[] bytes = new byte[4];
        ByteBuffer.wrap(bytes).putFloat(value);
        return bytes;
    }

    public static double toDouble(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getDouble();
    }

    public static int toInt(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getInt();
    }

    public static float toFloat(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getFloat();
    }

    @Override
    public void assign(int[] indices, float[] data, boolean contiguous, int inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);
        for (int i = 0; i < indices.length; i++) {
            put(indices[i], data[i]);
        }
    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous, int inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);
        for (int i = 0; i < indices.length; i += inc) {
            put(indices[i], data[i]);
        }
    }

    @Override
    public void assign(DataBuffer data) {
        if (data.length() != length())
            throw new IllegalArgumentException("Unable to assign buffer of length " + data.length() + " to this buffer of length " + length());


        for (int i = 0; i < data.length(); i++) {
            put(i, data);
        }
    }

    @Override
    public void assign(int[] indices, float[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public int length() {
        return length;
    }

    @Override
    public void assign(Number value) {
        assign(value, 0);
    }

    @Override
    public <E> E getElement(int i) {
        throw new UnsupportedOperationException();

    }

    @Override
    public double[] getDoublesAt(int offset, int length) {
        return getDoublesAt(offset, 1, length);
    }

    @Override
    public float[] getFloatsAt(int offset, int inc, int length) {
        if (length + offset > length())
            throw new IllegalArgumentException("Unable to get length " + length + " offset of " + offset + " was too high");

        if (length >= length())
            throw new IllegalArgumentException("Length must not be > " + length);
        if (offset >= length())
            throw new IllegalArgumentException("Length must not be > " + length);
        float[] ret = new float[length];
        for (int i = 0; i < length; i++) {
            ret[i] = getFloat(i + offset);
        }
        return ret;
    }

    @Override
    public double[] getDoublesAt(int offset, int inc, int length) {
        if (length + offset > length()) {
            length -= offset;
        }

        if (length > length())
            throw new IllegalArgumentException("Length must not be > " + length);
        if (offset > length())
            throw new IllegalArgumentException("Length must not be > " + length);
        if (offset + length > length())
            length = length() - offset;
        double[] ret = new double[length];
        for (int i = 0; i < length; i++) {
            ret[i] = getDouble(i + offset);
        }


        return ret;
    }

    @Override
    public float[] getFloatsAt(int offset, int length) {
        return getFloatsAt(offset, 1, length);
    }

    @Override
    public IComplexFloat getComplexFloat(int i) {
        return Nd4j.createFloat(getFloat(i), getFloat(i) + 1);
    }

    @Override
    public IComplexDouble getComplexDouble(int i) {
        return Nd4j.createDouble(getDouble(i), getDouble(i + 1));
    }

    @Override
    public IComplexNumber getComplex(int i) {
        return dataType() == DataBuffer.FLOAT ? getComplexFloat(i) : getComplexDouble(i);
    }

    @Override
    public void apply(ElementWiseOp op, int offset) {
        INDArray from = op.from();
        if (from instanceof IComplexNDArray) {
            for (int i = offset; i < length(); i++) {
                IComplexNumber result = op.apply(from, getComplex(i), i);
                put(i, result);
            }
        } else {
            for (int i = offset; i < length(); i++) {
                double result = op.apply(from, getDouble(i), i);
                put(i, result);
            }
        }

    }

    @Override
    public void apply(ElementWiseOp op) {
        apply(op, 0);
    }


    @Override
    public void addi(Number n) {
        addi(n, 1, 0);
    }

    @Override
    public void subi(Number n) {
        subi(n, 1, 0);
    }

    @Override
    public void muli(Number n) {
        muli(n, 1, 0);
    }

    @Override
    public void divi(Number n) {
        divi(n, 1, 0);
    }

    @Override
    public void addi(Number n, int inc, int offset) {
        for (int i = offset; i < length(); i += inc) {
            put(i, getDouble(i) + n.doubleValue());
        }
    }

    @Override
    public void subi(Number n, int inc, int offset) {
        for (int i = offset; i < length(); i += inc) {
            put(i, getDouble(i) - n.doubleValue());

        }
    }

    @Override
    public void muli(Number n, int inc, int offset) {
        for (int i = offset; i < length(); i += inc) {
            put(i, getDouble(i) * n.doubleValue());

        }
    }

    @Override
    public void divi(Number n, int inc, int offset) {
        for (int i = offset; i < length(); i += inc) {
            put(i, getDouble(i) / n.doubleValue());

        }
    }

    @Override
    public void addi(DataBuffer buffer) {
        addi(buffer, length(), 0, 0, 1, 1);
    }

    @Override
    public void subi(DataBuffer buffer) {
        subi(buffer, length(), 0, 0, 1, 1);
    }

    @Override
    public void muli(DataBuffer buffer) {
        muli(buffer, length(), 0, 0, 1, 1);
    }

    @Override
    public void divi(DataBuffer buffer) {
        divi(buffer, length(), 0, 0, 1, 1);
    }

    @Override
    public void addi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {
        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                put(i, getDouble(i) + buffer.getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                put(yi, getDouble(yi) + buffer.getDouble(xi));
            }


        }

    }

    @Override
    public void subi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {


        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                put(i, getDouble(i) - buffer.getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                put(yi, getDouble(yi) - buffer.getDouble(xi));
            }


        }
    }

    @Override
    public void muli(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {


        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                put(i, getDouble(i) * buffer.getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                put(yi, getDouble(yi) * buffer.getDouble(xi));
            }


        }
    }

    @Override
    public void divi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {


        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                put(i, getDouble(i) / buffer.getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                put(yi, getDouble(yi) / buffer.getDouble(xi));
            }


        }
    }

    @Override
    public void rsubi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {

        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                put(i, buffer.getDouble(i) - getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                put(yi,  buffer.getDouble(xi) - getDouble(yi));
            }


        }
    }

    @Override
    public void rdivi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {

        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                put(i, buffer.getDouble(i) / getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                put(yi, buffer.getDouble(xi) / getDouble(yi));
            }


        }
    }

    @Override
    public void rsubi(DataBuffer buffer) {
        for (int i = 0; i < buffer.length(); i++) {
            put(i, buffer.getDouble(i) - getDouble(i));
        }
    }

    @Override
    public void rdivi(DataBuffer buffer) {
        for (int i = 0; i < buffer.length(); i++) {
            put(i, buffer.getDouble(i) / getDouble(i));
        }
    }

    @Override
    public void rdivi(Number n, int inc, int offset) {
        for (int i = offset; i < length(); i += inc) {
            put(i, n.doubleValue() / getDouble(i));

        }
    }

    @Override
    public void rsubi(Number n, int inc, int offset) {
        for (int i = offset; i < length(); i += inc) {
            put(i, n.doubleValue() / getDouble(i));

        }
    }

    @Override
    public void rdivi(Number n) {
       rdivi(n,0,1);
    }

    @Override
    public void rsubi(Number n) {
      rsubi(n,0,1);
    }


    @Override
    public void rsubi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                result.put(i, buffer.getDouble(i) - getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                result.put(yi, buffer.getDouble(xi) - getDouble(yi));
            }


        }
    }

    @Override
    public void rdivi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                result.put(i, buffer.getDouble(i) / getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                result.put(yi, buffer.getDouble(xi) / getDouble(yi));
            }


        }
    }

    @Override
    public void divi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                result.put(i, getDouble(i) / buffer.getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                result.put(yi, getDouble(yi) / buffer.getDouble(xi));
            }


        }
    }

    @Override
    public void muli(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                result.put(i, getDouble(i) * buffer.getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                result.put(yi, getDouble(yi) * buffer.getDouble(xi));
            }


        }
    }

    @Override
    public void subi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                result.put(i, getDouble(i) - buffer.getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                result.put(yi, getDouble(yi) - buffer.getDouble(xi));
            }


        }
    }

    @Override
    public void addi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        if (incx == 1 && incy == 1 && offset == 0 && yOffset == 0) {
            for (int i = 0; i < n; i++) {
                result.put(i, getDouble(i) + buffer.getDouble(i));
            }

        } else {
            for (int c = 0, xi = offset, yi = yOffset; c < n; c++, xi += incx, yi += incy) {
                result.put(yi, getDouble(yi) + buffer.getDouble(xi));
            }


        }
    }

    @Override
    public void rsubi(DataBuffer buffer, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,buffer.getDouble(i) - getDouble(i));
        }
    }

    @Override
    public void rdivi(DataBuffer buffer, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,buffer.getDouble(i) / getDouble(i));
        }
    }

    @Override
    public void divi(DataBuffer buffer, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,getDouble(i) / buffer.getDouble(i));
        }
    }

    @Override
    public void muli(DataBuffer buffer, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,getDouble(i) * buffer.getDouble(i));
        }
    }

    @Override
    public void subi(DataBuffer buffer, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,getDouble(i) - buffer.getDouble(i));
        }
    }

    @Override
    public void addi(DataBuffer buffer, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,getDouble(i) + buffer.getDouble(i));
        }
    }

    @Override
    public void rdivi(Number n, int inc, int offset, DataBuffer result) {
        for(int i = offset; i < length(); i+= inc) {
            result.put(i,n.doubleValue() / getDouble(i));
        }
    }

    @Override
    public void divi(Number n, int inc, int offset, DataBuffer result) {
        for(int i = offset; i < length(); i+= inc) {
            result.put(i,getDouble(i) / n.doubleValue());
        }
    }

    @Override
    public void muli(Number n, int inc, int offset, DataBuffer result) {
        for(int i = offset; i < length(); i+= inc) {
            result.put(i,getDouble(i) * n.doubleValue());
        }
    }

    @Override
    public void rsubi(Number n, int inc, int offset, DataBuffer result) {
        for(int i = offset; i < length(); i+= inc) {
            result.put(i,n.doubleValue() - getDouble(i));
        }
    }

    @Override
    public void subi(Number n, int inc, int offset, DataBuffer result) {
        for(int i = offset; i < length(); i+= inc) {
            result.put(i,getDouble(i) - n.doubleValue());
        }
    }

    @Override
    public void addi(Number n, int inc, int offset, DataBuffer result) {
        for(int i = offset; i < length(); i+= inc) {
            result.put(i,getDouble(i) + n.doubleValue());
        }
    }

    @Override
    public void rdivi(Number n, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,n.doubleValue() / getDouble(i));
        }
    }

    @Override
    public void divi(Number n, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,getDouble(i) / n.doubleValue());
        }
    }

    @Override
    public void muli(Number n, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,getDouble(i) * n.doubleValue());
        }
    }

    @Override
    public void rsubi(Number n, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,n.doubleValue() - getDouble(i));
        }
    }

    @Override
    public void subi(Number n, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,getDouble(i) - n.doubleValue());
        }
    }

    @Override
    public void addi(Number n, DataBuffer result) {
        for(int i = 0; i < length(); i++) {
            result.put(i,getDouble(i) + n.doubleValue());
        }
    }

    @Override
    public <E> void put(int i, E element) {
        throw new UnsupportedOperationException();
    }

    @Override
    public <E> E[] asType() {
        throw new UnsupportedOperationException();
    }
}
