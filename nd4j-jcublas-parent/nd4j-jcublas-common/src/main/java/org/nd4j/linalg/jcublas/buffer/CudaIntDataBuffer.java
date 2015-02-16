package org.nd4j.linalg.jcublas.buffer;

import jcuda.Pointer;
import jcuda.Sizeof;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.ops.ElementWiseOp;

/**
 * Cuda int buffer
 * @author Adam Gibson
 */
public class CudaIntDataBuffer extends BaseCudaDataBuffer {
    /**
     * Base constructor
     *
     * @param length      the length of the buffer
     */
    public CudaIntDataBuffer(int length) {
        super(length, Sizeof.INT);
    }

    public CudaIntDataBuffer(int[] data) {
        this(data.length);
        setData(data);
    }

    @Override
    public void assign(int[] indices, float[] data, boolean contiguous,int inc) {
        if(indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if(indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

        if(contiguous) {
            int offset = indices[0];
            Pointer p = Pointer.to(data);
        }
        else
            throw new UnsupportedOperationException("Non contiguous is not supported");

    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous,int inc) {
        if(indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if(indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

        if(contiguous) {
            int offset = indices[0];
            Pointer p = Pointer.to(data);
        }
        else
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
    public void assign(Number value, int offset) {
        int arrLength = length - offset;
        int[] data = new int[arrLength];
        for(int i = 0; i < data.length; i++)
            data[i] = value.intValue();
        set(offset,arrLength, Pointer.to(data));
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
    public int dataType() {
        return DataBuffer.INT;
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
    public <E> E[] asType() {
        return null;
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
    public <E> E getElement(int i) {
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
    public <E> void put(int i, E element) {

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
    public void flush() {

    }

    @Override
    public void apply(ElementWiseOp op, int offset) {

    }


}
