package org.nd4j.linalg.jcublas.buffer;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.jcublas.JCublas;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.ops.ElementWiseOp;

/**
 * Base class for a data buffer
 * @author Adam Gibson
 */
public abstract class BaseCudaDataBuffer implements JCudaBuffer {
    protected Pointer pointer;
    protected int length;
    protected int elementSize;

    static {
        SimpleJCublas.init();
    }


    /**
     * Base constructor
     * @param length the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(int length,int elementSize) {
        this.length = length;
        this.elementSize = elementSize;
        if(pointer() == null)
            alloc();
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
        if(to.dataType() != dataType())
            throw new IllegalArgumentException("Unable to copy buffer, mis matching data types.");
        if(dataType() == DataBuffer.FLOAT) {
            JCublas.cublasCcopy(length(),pointer(),1,to.pointer(),1);
        }
        else {
            JCublas.cublasDcopy(length(),pointer(),1,to.pointer(),1);
        }

    }


    @Override
    public void assign(Number value) {
        assign(value,0);
    }

    /**
     * Get element with the specified index
     * @param index the index of the element to get
     * @param init the initialized pointer
     */
    protected void get(int index,int length,Pointer init) {
        if(index == 0 && length == length()) {
            if(dataType() == DataBuffer.FLOAT)
                JCublas.cublasCcopy(length,pointer(),1,init,1);
            else
                JCublas.cublasDcopy(length,pointer(),1,init,1);

        }


        else {
            try {
                JCublas.cublasGetVector(
                        length,
                        elementSize(),
                        pointer().withByteOffset(index * elementSize()),
                        1,
                        init,
                        1);
            }catch(CudaException e) {
                throw new RuntimeException("Cuda exception with offset " + index + " and " + length + " out of total length " + length(),e);
            }
        }

    }

    /**
     * Get element with the specified index
     * @param index the index of the element to get
     * @param init the initialized pointer
     */
    protected void get(int index,Pointer init) {
        get(index, 1, init);
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
    protected void set(int index,int length,Pointer from,int inc) {
        JCublas.cublasSetVector(
                length,
                elementSize(),
                from,
                1,
                pointer().withByteOffset(index *  elementSize()),
                inc);
    }
    /**
     * Set an individual element
     * @param index the index of the element
     * @param from the element to get data from
     */
    protected void set(int index,int length,Pointer from) {
        set(index,length,from,1);
    }

    @Override
    public void assign(DataBuffer data) {
        JCudaBuffer buf = (JCudaBuffer) data;
        set(0,buf.pointer());
    }

    /**
     * Set an individual element
     * @param index the index of the element
     * @param from the element to get data from
     */
    protected void set(int index,Pointer from) {
        set(index,1,from);
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
        throw new UnsupportedOperationException();

    }

    @Override
    public void put(int i, double element) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void put(int i, int element) {
        throw new UnsupportedOperationException();
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
    public void apply(ElementWiseOp op) {
        apply(op,0);
    }

    @Override
    public void assign(int[] indices, float[] data, boolean contiguous) {
        assign(indices,data,contiguous,1);
    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous) {
        assign(indices, data, contiguous,1);
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
