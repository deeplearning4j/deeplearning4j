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

package org.nd4j.linalg.jcublas.buffer;

import jcuda.Pointer;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.complex.CudaComplexConversion;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for a data buffer
 *
 * @author Adam Gibson
 */
public abstract class BaseCudaDataBuffer implements JCudaBuffer {
    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);
    static {
        SimpleJCublas.init();
    }
    protected transient Pointer pointer;
    protected int length;
    protected int elementSize;


    /**
     * Base constructor
     *
     * @param length      the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(int length, int elementSize) {
        this.length = length;
        this.elementSize = elementSize;
        if (pointer() == null)
            alloc();
    }

    @Override
    public void put(int i, IComplexNumber result) {
        if(dataType() == DataBuffer.FLOAT) {
            JCublas.cublasSetVector(
                    length(),
                    new cuComplex[]{CudaComplexConversion.toComplex(result.asFloat())}
                    ,i
                    ,1
                    ,pointer()
                    ,1);
        }
        else {
            JCublas.cublasSetVector(
                    length(),
                    new cuDoubleComplex[]{CudaComplexConversion.toComplexDouble(result.asDouble())}
                    ,i
                    ,1
                    ,pointer()
                    ,1);
        }
    }

    @Override
    public Pointer pointer() {
        return pointer;
    }

    @Override
    public void alloc() {
        pointer = new Pointer();
        //allocate memory for the pointer
        try {
            JCuda.cudaMalloc(pointer(),
                    length() * elementSize()
            );

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public void set(Pointer pointer) {
        if (dataType() == DOUBLE) {
            JCublas.cublasDcopy(
                    length(),
                    pointer,
                    1,
                    pointer(),
                    1
            );
        } else {
            JCublas.cublasScopy(
                    length(),
                    pointer,
                    1,
                    pointer(),
                    1
            );
        }


    }


    /**
     * Copy the data of this buffer to another buffer on the gpu
     *
     * @param to the buffer to copy data to
     */
    protected void copyTo(JCudaBuffer to) {
        if (to.dataType() != dataType())
            throw new IllegalArgumentException("Unable to copy buffer, mis matching data types.");

        JCuda.cudaMemcpy(
                to.pointer()
                , pointer()
                , length() * elementSize()
                , cudaMemcpyKind.cudaMemcpyDeviceToDevice);


    }




    @Override
    public void assign(Number value) {
        assign(value, 0);
    }


    /**
     * Get element with the specified index
     *
     * @param index  the index of the element to get
     * @param inc    the increment step when getting data
     * @param length the length to iterate for
     * @param init   the initialized pointer
     */
    protected void get(int index, int inc, int length, Pointer init) {
        JCublas.cublasGetVector(
                length
                , elementSize(),
                pointer().withByteOffset(index * elementSize())
                ,
                inc,
                init
                , 1);
    }

    /**
     * Get element with the specified index
     *
     * @param index the index of the element to get
     * @param init  the initialized pointer
     */
    protected void get(int index, int length, Pointer init) {
        get(index, 1, length, init);
    }

    /**
     * Get element with the specified index
     *
     * @param index the index of the element to get
     * @param init  the initialized pointer
     */
    protected void get(int index, Pointer init) {
        get(index, 1, init);
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

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    protected void set(int index, int length, Pointer from, int inc) {
        JCublas.cublasSetVector(
                length,
                elementSize(),
                from,
                inc,
                pointer().withByteOffset(elementSize() * index)
                , 1);


    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    protected void set(int index, int length, Pointer from) {
        set(index, length, from, 1);
    }

    @Override
    public void assign(DataBuffer data) {
        JCudaBuffer buf = (JCudaBuffer) data;
        set(0, buf.pointer());
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    protected void set(int index, Pointer from) {
        set(index, 1, from);
    }


    @Override
    public void destroy() {
        JCublas.cublasFree(pointer);

    }

    @Override
    public double[] getDoublesAt(int offset, int length) {
        return getDoublesAt(offset, 1, length);
    }

    @Override
    public float[] getFloatsAt(int offset, int length) {
        return getFloatsAt(offset, 1, length);
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
    public void assign(int[] indices, float[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BaseCudaDataBuffer)) return false;

        BaseCudaDataBuffer that = (BaseCudaDataBuffer) o;

        if (elementSize != that.elementSize) return false;
        if (length != that.length) return false;
        for (int i = 0; i < length; i++) {
            double element = getDouble(i);
            double other = that.getDouble(i);
            if (element != other)
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




    /**
     * Execute an operation on this buffer with a scalar
     * @param dType the data type to operate on (float or double)
     * @param dYIdx the index to
     * @param scalar the scalar to operate with
     * @param n the number of items to iterate on
     * @param incy the increment for this buffer
     * @param op the operation to execute
     */
    protected void execScalar(String dType,int dYIdx,Number scalar,int n,int incy,String op,DataBuffer result) {
        Pointer scalarP = dType.equals("double") ? Pointer.to(new double[]{scalar.doubleValue()}) : Pointer.to(new float[]{scalar.floatValue()});
        Pointer twoP = Pointer.to(pointer());
        JCudaBuffer resultBuffer = (JCudaBuffer) result;
        int blocks = KernelFunctions.numBlocks(n);
        int threads = (int) Math.ceil((double) n / blocks);


        Pointer resultP = Pointer.to(resultBuffer.pointer());


        Pointer kernelParameters = KernelFunctions.constructKernelParameters(
                //number of elements
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{dYIdx})
                , scalarP
                , twoP
                , Pointer.to(new int[]{incy}),resultP);
        //actually call the kernel
        KernelFunctions.invoke(
                blocks,threads,
                KernelFunctionLoader.getInstance().getFunction(op,dType)
                , kernelParameters);
    }


    @Override
    public void assign(int[] offsets, int[] strides, int n, DataBuffer... buffers) {

    }

    @Override
    public void assign(DataBuffer... buffers) {
        int[] offsets = new int[buffers.length];
        int[] strides = new int[buffers.length];
        for(int i = 0; i < strides.length; i++)
            strides[i] = 1;
        assign(offsets,strides,buffers);
    }

    @Override
    public void assign(int[] offsets, int[] strides, DataBuffer... buffers) {
        assign(offsets,strides,length(),buffers);
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        destroy();
    }
}
