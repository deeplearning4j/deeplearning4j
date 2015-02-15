package org.nd4j.linalg.jcublas.buffer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Cuda float buffer
 * @author Adam Gibson
 */
public class CudaFloatDataBuffer extends BaseCudaDataBuffer {
    /**
     * Base constructor
     *
     * @param length      the length of the buffer
     */
    public CudaFloatDataBuffer(int length) {
        super(length, Sizeof.FLOAT);
    }

    public CudaFloatDataBuffer(float[] buffer) {
        this(buffer.length);
        setData(buffer);
    }

    @Override
    public void setData(int[] data) {

    }

    @Override
    public void setData(float[] data) {

        if(data.length != length)
            throw new IllegalArgumentException("Unable to set vector, must be of length " + length() + " but found length " + data.length);

        if(pointer() == null)
            alloc();

        JCublas.cublasSetVector(
                length,
                elementSize,
                Pointer.to(data),
                1,
                pointer(),
                1);
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


}
