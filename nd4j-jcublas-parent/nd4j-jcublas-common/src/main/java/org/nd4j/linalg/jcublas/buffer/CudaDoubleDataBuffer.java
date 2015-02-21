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
import jcuda.Sizeof;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.jcublas.JCublas;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.jcublas.complex.CudaComplexConversion;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Cuda double  buffer
 *
 * @author Adam Gibson
 */
public class CudaDoubleDataBuffer extends BaseCudaDataBuffer {


    /**
     * Base constructor
     *
     * @param length the length of the buffer
     */
    public CudaDoubleDataBuffer(int length) {
        super(length, Sizeof.DOUBLE);
        if (pointer() == null)
            alloc();
    }

    /**
     * Instantiate based on the given data
     * @param data the data to instantiate with
     */
    public CudaDoubleDataBuffer(double[] data) {
        this(data.length);
        setData(data);
    }


    @Override
    public void assign(int[] indices, float[] data, boolean contiguous, int inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

        if (contiguous) {
            int offset = indices[0];
            Pointer p = Pointer.to(data);
            set(offset, data.length, p, inc);

        } else
            throw new UnsupportedOperationException("Non contiguous is not supported");

    }

    @Override
    public void assign(int[] indices, double[] data, boolean contiguous, int inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length " + length() + " where the indices are of length " + data.length);

        if (contiguous) {
            int offset = indices[0];
            Pointer p = Pointer.to(data);
            set(offset, data.length, p, inc);
        } else
            throw new UnsupportedOperationException("Non contiguous is not supported");

    }

    @Override
    public double[] getDoublesAt(int offset, int length) {
        return getDoublesAt(0, 1, length);
    }

    @Override
    public float[] getFloatsAt(int offset, int length) {
        return ArrayUtil.toFloats(getDoublesAt(offset, length));
    }

    @Override
    public double[] getDoublesAt(int offset, int inc, int length) {
        if (offset + length > length())
            length -= offset;

        double[] ret = new double[length];
        Pointer p = Pointer.to(ret);
        get(offset, inc, length, p);
        return ret;
    }

    @Override
    public float[] getFloatsAt(int offset, int inc, int length) {
        return ArrayUtil.toFloats(getDoublesAt(offset, 1, length));
    }

    @Override
    public void assign(Number value, int offset) {
        int arrLength = length - offset;
        double[] data = new double[arrLength];
        for (int i = 0; i < data.length; i++)
            data[i] = value.doubleValue();
        set(offset, arrLength, Pointer.to(data));
    }

    @Override
    public void setData(int[] data) {
        setData(ArrayUtil.toDoubles(data));
    }

    @Override
    public void setData(float[] data) {
        setData(ArrayUtil.toDoubles(data));
    }

    @Override
    public void setData(double[] data) {

        if (data.length != length)
            throw new IllegalArgumentException("Unable to set vector, must be of length " + length() + " but found length " + data.length);

        if (pointer() == null)
            alloc();


        JCublas.cublasSetVector(
                length()
                , elementSize()
                , Pointer.to(data)
                , 1
                , pointer()
                , 1);


    }

    @Override
    public byte[] asBytes() {
        return new byte[0];
    }

    @Override
    public int dataType() {
        return DataBuffer.DOUBLE;
    }

    @Override
    public float[] asFloat() {
        return new float[0];
    }

    @Override
    public double[] asDouble() {
        double[] ret = new double[length];
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
    public int[] asInt() {
        return new int[0];
    }


    @Override
    public double getDouble(int i) {
        double[] d = new double[1];
        Pointer p = Pointer.to(d);
        get(i, p);
        return d[0];
    }

    @Override
    public float getFloat(int i) {
        return (float) getDouble(i);
    }

    @Override
    public Number getNumber(int i) {
        return getDouble(i);
    }


    @Override
    public void put(int i, float element) {
        put(i, (double) element);
    }

    @Override
    public void put(int i, double element) {
        double[] d = new double[]{element};
        Pointer p = Pointer.to(d);
        set(i, p);

    }

    @Override
    public void put(int i, int element) {
        put(i, (double) element);
    }


    @Override
    public int getInt(int ix) {
        return (int) getDouble(ix);
    }

    @Override
    public DataBuffer dup() {
        CudaDoubleDataBuffer buffer = new CudaDoubleDataBuffer(length());
        copyTo(buffer);
        return buffer;
    }




    @Override
    public void apply(ElementWiseOp op, int offset) {
        if (offset >= length)
            throw new IllegalArgumentException("Illegal start " + offset + " greater than length of " + length);
        int arrLength = Math.abs(length - offset);
        double[] data = new double[arrLength];
        Pointer p = Pointer.to(data);
        get(offset, data.length, p);
        DataBuffer floatBuffer = new DoubleBuffer(data, false);
        floatBuffer.apply(op);
        p = Pointer.to(data);
        set(offset, arrLength, p);
    }

    @Override
    public void rsubi(Number n) {
      rsubi(n,1,0);
    }

    @Override
    public void rdivi(Number n) {
      rdivi(n,1,0);
    }

    @Override
    public void addi(Number n, int inc, int offset) {
        execScalar("double",offset,n,length(),inc,"add_scalar");
    }

    @Override
    public void subi(Number n, int inc, int offset) {
        execScalar("double",offset,n,length(),inc,"sub_scalar");
    }

    @Override
    public void rsubi(Number n, int inc, int offset) {
        execScalar("double",offset,n,length(),inc,"rsub_scalar");
    }

    @Override
    public void muli(Number n, int inc, int offset) {
        execScalar("double",offset,n,length(),inc,"mul_scalar");
    }

    @Override
    public void divi(Number n, int inc, int offset) {
        execScalar("double",offset,n,length(),inc,"div_scalar");
    }

    @Override
    public void rdivi(Number n, int inc, int offset) {

    }

    @Override
    public void rdivi(DataBuffer buffer) {
        exec2d(buffer,"double",0,0,length(),1,1,"rdiv_strided");

    }

    @Override
    public void rsubi(DataBuffer buffer) {
        exec2d(buffer,"double",0,0,length(),1,1,"rsub_strided");
    }

    @Override
    public void addi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {
        JCudaBuffer b = (JCudaBuffer) buffer;
        JCublas.cublasDaxpy(n, 1.0, b.pointer(), incx, pointer(), incy);
    }

    @Override
    public void subi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {
        JCudaBuffer b = (JCudaBuffer) buffer;
        JCublas.cublasDaxpy(n, -1.0, b.pointer(), incx, pointer(), incy);

    }

    @Override
    public void muli(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"mul_strided");

    }

    @Override
    public void divi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"mul_strided");

    }

    @Override
    public void rdivi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"rdiv_strided");

    }

    @Override
    public void rsubi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"rsub_strided");

    }

    @Override
    public void addi(Number n, DataBuffer result) {
        execScalar("double",0,n,length(),1,"add_scalar",result);
    }

    @Override
    public void subi(Number n, DataBuffer result) {
        execScalar("double",0,n,length(),1,"sub_scalar",result);
    }

    @Override
    public void rsubi(Number n, DataBuffer result) {
        execScalar("double",0,n,length(),1,"rsub_sclar",result);
    }

    @Override
    public void muli(Number n, DataBuffer result) {
        execScalar("double",0,n,length(),1,"mul_scalar",result);
    }

    @Override
    public void divi(Number n, DataBuffer result) {
        execScalar("double",0,n,length(),1,"div_scalar",result);
    }

    @Override
    public void rdivi(Number n, DataBuffer result) {
        execScalar("double",0,n,length(),1,"rdiv_scalar",result);
    }

    @Override
    public void addi(Number n, int inc, int offset, DataBuffer result) {
      execScalar("double",offset,n,length(),inc,"add_scalar",result);
    }

    @Override
    public void subi(Number n, int inc, int offset, DataBuffer result) {
        execScalar("double",offset,n,length(),inc,"sub_scalar",result);

    }

    @Override
    public void rsubi(Number n, int inc, int offset, DataBuffer result) {
        execScalar("double",offset,n,length(),inc,"rsub_scalar",result);

    }

    @Override
    public void muli(Number n, int inc, int offset, DataBuffer result) {
        execScalar("double",offset,n,length(),inc,"mul_scalar",result);

    }

    @Override
    public void divi(Number n, int inc, int offset, DataBuffer result) {
        execScalar("double",offset,n,length(),inc,"div_scalar",result);

    }

    @Override
    public void rdivi(Number n, int inc, int offset, DataBuffer result) {
        execScalar("double",offset,n,length(),inc,"rdiv_scalar",result);

    }

    @Override
    public void addi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"add_strided",result);
    }

    @Override
    public void subi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"sub_strided",result);

    }

    @Override
    public void muli(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"mul_strided",result);

    }

    @Override
    public void divi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"div_strided",result);

    }

    @Override
    public void rdivi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"rdiv_strided",result);

    }

    @Override
    public void rsubi(DataBuffer buffer, int n, int offset, int yOffset, int incx, int incy, DataBuffer result) {
        exec2d(buffer,"double",offset,yOffset,n,incx,incy,"rsub_strided",result);
    }
}
