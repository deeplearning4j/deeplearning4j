package org.nd4j.linalg.api.buffer;


import com.google.common.primitives.Bytes;

import java.util.Arrays;

/**
 * Double buffer implementation of data buffer
 * @author Adam Gibson
 */
public class DoubleBuffer extends BaseDataBuffer {

    private double[] buffer;

    public DoubleBuffer(int length) {
        super(length);
        this.buffer = new double[length];
    }
    public DoubleBuffer(double[] buffer) {
        super(buffer.length);
        this.buffer = Arrays.copyOf(buffer,buffer.length);
    }

    @Override
    public byte[] asBytes() {
        byte[][] ret1 = new byte[length][];
        for(int i = 0; i < length; i++) {
            ret1[i] = toByteArray(buffer[i]);
        }

        return Bytes.concat(ret1);
    }

    @Override
    public String dataType() {
        return DataBuffer.DOUBLE;
    }

    @Override
    public float[] asFloat() {
        float[] ret = new float[length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (float) buffer[i];
        }
        return ret;
    }

    @Override
    public double[] asDouble() {
        return buffer;
    }

    @Override
    public int[] asInt() {
        int[] ret = new int[length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (int) buffer[i];
        }
        return ret;
    }

    @Override
    public <E> E[] asType() {
        return null;
    }

    @Override
    public double getDouble(int i) {
        return buffer[i];
    }

    @Override
    public float getFloat(int i) {
        return (float) buffer[i];
    }

    @Override
    public Number getNumber(int i) {
        return buffer[i];
    }



    @Override
    public void put(int i, float element) {
        buffer[i] = element;
    }

    @Override
    public void put(int i, double element) {
        buffer[i] = element;
    }

    @Override
    public void put(int i, int element) {
        buffer[i] = element;
    }




    @Override
    public int getInt(int ix) {
        return (int) buffer[ix];
    }

    @Override
    public DataBuffer dup() {
        return new DoubleBuffer(buffer);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DoubleBuffer)) return false;

        DoubleBuffer that = (DoubleBuffer) o;

        if (!Arrays.equals(buffer, that.buffer)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return buffer != null ? Arrays.hashCode(buffer) : 0;
    }
}
