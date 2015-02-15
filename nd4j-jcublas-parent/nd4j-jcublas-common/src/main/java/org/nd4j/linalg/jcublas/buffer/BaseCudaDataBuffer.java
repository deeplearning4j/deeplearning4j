package org.nd4j.linalg.jcublas.buffer;

import jcuda.Pointer;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
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


    /**
     * Copy the data of this buffer to another buffer on the gpu
     * @param to the buffer to copy data to
     */
    protected void copyTo(JCudaBuffer to) {
        JCuda.cudaMemcpy(to.pointer(), pointer(), length() * elementSize(), cudaMemcpyKind.cudaMemcpyDeviceToDevice);
    }

    /**
     * Get element with the specified index
     * @param index the index of the element to get
     * @param init the initialized pointer
     */
    protected void get(int index,Pointer init) {

        JCublas.cublasGetVector(
                1,
                elementSize(),
                pointer().withByteOffset(index *  elementSize()),
                1,
                init,
                1);

    }


    @Override
    public IComplexFloat getComplexFloat(int i) {
        return Nd4j.createFloat(getFloat(i), getFloat(i) + 1);
    }

    @Override
    public IComplexDouble getComplexDouble(int i) {
        return Nd4j.createDouble(getDouble(i),getDouble(i + 1));
    }

    @Override
    public IComplexNumber getComplex(int i) {
        return dataType() == DataBuffer.FLOAT ? getComplexFloat(i) : getComplexDouble(i);
    }

    /**
     * Set an individual element
     * @param index the index of the element
     * @param from the element to get data from
     */
    protected void set(int index,Pointer from) {
        JCublas.cublasInit();
        JCublas.cublasSetVector(
                1,
                elementSize,
                from,
                1,
                pointer().withByteOffset(index *  elementSize()),
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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BaseCudaDataBuffer)) return false;

        BaseCudaDataBuffer that = (BaseCudaDataBuffer) o;

        if (elementSize != that.elementSize) return false;
        if (length != that.length) return false;
        for(int i = 0; i < length; i++) {
            double element = getDouble(i);
            double other = that.getDouble(i);
            if(element != other)
                return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = pointer != null ? pointer.hashCode() : 0;
        result = 31 * result + length;
        result = 31 * result + elementSize;
        return result;
    }
}
