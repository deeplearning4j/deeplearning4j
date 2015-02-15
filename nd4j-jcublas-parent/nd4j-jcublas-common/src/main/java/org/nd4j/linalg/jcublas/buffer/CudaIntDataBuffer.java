package org.nd4j.linalg.jcublas.buffer;

import jcuda.Sizeof;
import org.nd4j.linalg.api.buffer.DataBuffer;

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


}
