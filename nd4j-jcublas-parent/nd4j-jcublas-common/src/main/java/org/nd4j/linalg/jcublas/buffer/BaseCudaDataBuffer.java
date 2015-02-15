package org.nd4j.linalg.jcublas.buffer;

import jcuda.Pointer;
import jcuda.jcublas.JCublas;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.SimpleJCublas;

/**
 * Base class for a data buffer
 * @author Adam Gibson
 */
public abstract class BaseCudaDataBuffer implements JCudaBuffer {
    protected Pointer pointer;
    protected int length;
    protected int elementSize;

    /**
     * Base constructor
     * @param length the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(int length,int elementSize) {
        this.length = length;
        this.elementSize = elementSize;
    }

    @Override
    public Pointer pointer() {
        return pointer;
    }

    @Override
    public void alloc() {
        SimpleJCublas.init();
        JCublas.cublasInit();
        pointer = new Pointer();
        //allocate memory for the pointer
        JCublas.cublasAlloc(
                length,
                elementSize
                , pointer());
    }


    @Override
    public void set(Pointer pointer) {
        JCublas.cublasInit();
        JCublas.cublasSetVector(
                length,
                elementSize,
                pointer,
                1,
                pointer(),
                1);
    }

    @Override
    public void destroy() {
        JCublas.cublasFree(pointer);
    }


    @Override
    public int elementSize() {
        return elementSize;
    }

    @Override
    public int length() {
        return length;
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
        throw new UnsupportedOperationException();
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
        throw new UnsupportedOperationException();
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
        throw new UnsupportedOperationException();
    }



    @Override
    public int getInt(int ix) {
        return 0;
    }

    @Override
    public DataBuffer dup() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void flush() {
        throw new UnsupportedOperationException();
    }


}
