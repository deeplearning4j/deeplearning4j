package org.nd4j.linalg.api.buffer;

import com.google.common.primitives.Bytes;

import java.util.Arrays;

/**
 *  Data buffer for floats
 *
 *  @author Adam Gibson
 */
public class FloatBuffer extends BaseDataBuffer {

    private float[] buffer;

    public FloatBuffer(int length) {
        super(length);
        this.buffer = new float[length];
    }
    public FloatBuffer(float[] buffer) {
        super(buffer.length);
        this.buffer = Arrays.copyOf(buffer, buffer.length);
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
        return DataBuffer.FLOAT;
    }

    @Override
    public float[] asFloat() {
       return buffer;
    }

    @Override
    public double[] asDouble() {
        double[] ret = new double[length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] =  buffer[i];
        }
        return ret;
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
        return buffer[i];
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
        buffer[i] = (float) element;
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
        return new FloatBuffer(buffer);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FloatBuffer)) return false;

        FloatBuffer that = (FloatBuffer) o;

        if (!Arrays.equals(buffer, that.buffer)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(buffer);
    }
}
