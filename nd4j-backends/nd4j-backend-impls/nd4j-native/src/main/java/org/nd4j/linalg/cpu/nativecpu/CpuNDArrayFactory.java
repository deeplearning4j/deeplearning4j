/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.cpu.nativecpu;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.blas.CpuLapack;
import org.nd4j.linalg.factory.BaseNDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.cpu.nativecpu.blas.CpuLevel1;
import org.nd4j.linalg.cpu.nativecpu.blas.CpuLevel2;
import org.nd4j.linalg.cpu.nativecpu.blas.CpuLevel3;
import org.nd4j.linalg.cpu.nativecpu.complex.ComplexDouble;
import org.nd4j.linalg.cpu.nativecpu.complex.ComplexFloat;
import org.nd4j.linalg.cpu.nativecpu.complex.ComplexNDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;

/**
 * Jblas NDArray Factory
 *
 * @author Adam Gibson
 */
public class CpuNDArrayFactory extends BaseNDArrayFactory {
    public CpuNDArrayFactory() {
    }
    static {
        //invoke the override
        Nd4j.getBlasWrapper();
    }


    public CpuNDArrayFactory(DataBuffer.Type dtype, Character order) {
        super(dtype, order);
    }

    public CpuNDArrayFactory(DataBuffer.Type dtype, char order) {
        super(dtype, order);
    }

    @Override
    public void createLevel1() {
        level1 = new CpuLevel1();
    }

    @Override
    public void createLevel2() {
       level2 = new CpuLevel2();
    }

    @Override
    public void createLevel3() {
        level3 = new CpuLevel3();
    }

    @Override
    public void createLapack() {
        lapack = new CpuLapack();
    }

    @Override
    public INDArray create(int[] shape, DataBuffer buffer) {
        return new NDArray(shape, buffer);
    }

    /**
     * Create float
     *
     * @param real real component
     * @param imag imag component
     * @return
     */
    @Override
    public IComplexFloat createFloat(float real, float imag) {
        return new ComplexFloat(real, imag);
    }

    /**
     * Create an instance of a complex double
     *
     * @param real the real component
     * @param imag the imaginary component
     * @return a new imaginary double with the specified real and imaginary components
     */
    @Override
    public IComplexDouble createDouble(double real, double imag) {
        return new ComplexDouble(real, imag);
    }

    /**
     * Create an ndarray with the given data layout
     *
     * @param data the data to create the ndarray with
     * @return the ndarray with the given data layout
     */
    @Override
    public INDArray create(double[][] data) {
        return new NDArray(data);
    }

    @Override
    public INDArray create(double[][] data, char ordering) {
        return new NDArray(data,ordering);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arr the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(INDArray arr) {
        return new ComplexNDArray(arr);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param data  the data to wrap
     * @param shape
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape) {
            return new ComplexNDArray(data, shape);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arrs  the arr to wrap
     * @param shape
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(List<IComplexNDArray> arrs, int[] shape) {
        return new ComplexNDArray(arrs, shape);
    }

    @Override
    public INDArray create(DataBuffer data) {
        return new NDArray(data);
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data) {
        return new ComplexNDArray(data);
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int rows, int columns, int[] stride, int offset) {
        return new ComplexNDArray(data, new int[]{rows, columns}, stride, offset);
    }

    @Override
    public INDArray create(DataBuffer data, int rows, int columns, int[] stride, int offset) {
        return new NDArray(data, new int[]{rows, columns}, stride, offset);
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] stride, int offset) {
        return new ComplexNDArray(data, shape, stride, offset);
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, int offset) {
        return createComplex(data, shape, stride, offset, order());
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new ComplexNDArray(data, shape, stride, offset, ordering);

    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, char ordering) {
        return new ComplexNDArray(data, shape, stride, 0, ordering);
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int offset, char ordering) {
        return createComplex(data, shape, Nd4j.getComplexStrides(shape), offset, ordering);
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, char ordering) {
        return createComplex(data, shape, Nd4j.getComplexStrides(shape), 0, ordering);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset) {
        return new ComplexNDArray(data, shape, stride, offset);
    }

    @Override
    public INDArray create(int[] shape, char ordering) {
        return new NDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering);
    }

    @Override
    public INDArray create(DataBuffer data, int[] newShape, int[] newStride, int offset, char ordering) {
        return new NDArray(data, newShape, newStride, offset, ordering);
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] newDims, int[] newStrides, int offset, char ordering) {
        return new ComplexNDArray(data, newDims, newStrides, offset, ordering);

    }


    @Override
    public IComplexNDArray createComplex(float[] data, Character order) {
        return new ComplexNDArray(data, order);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int offset, Character order) {
        return new NDArray(data, shape, offset, order);
    }

    @Override
    public INDArray create(float[] data, int rows, int columns, int[] stride, int offset, char ordering) {
        return new NDArray(data, new int[]{rows, columns}, stride, offset, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, char ordering) {
        return new NDArray(Nd4j.createBuffer(data), shape, ordering);
    }

    @Override
    public INDArray create(List<INDArray> list, int[] shape, char ordering) {
        return new NDArray(list, shape, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, int offset) {
        return new NDArray(Nd4j.createBuffer(data), shape, offset);
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new NDArray(Nd4j.createBuffer(data), shape, stride, offset, ordering);
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset) {
        return new NDArray(data, shape, stride, offset);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset) {
        return new ComplexNDArray(Nd4j.createBuffer(data), shape, stride, offset);
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, int offset) {
        return new NDArray(data, shape, stride, offset);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape) {
        return new NDArray(data, shape);
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] shape) {
        return new ComplexNDArray(data, shape);
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] stride) {
        return new ComplexNDArray(data, shape, stride);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape, int[] stride, int offset) {
        return new NDArray(data, shape, stride, offset, Nd4j.order());
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param list
     * @param shape the shape of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(List<INDArray> list, int[] shape) {
        return new NDArray(list, shape, Nd4j.getStrides(shape));

    }


    /**
     * Create a complex ndarray with the given data
     *
     * @param data     the data to use with tne ndarray
     * @param shape    the shape of the ndarray
     * @param stride   the stride for the ndarray
     * @param offset   the offset of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created complex ndarray
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new ComplexNDArray(ArrayUtil.floatCopyOf(data), shape, stride, offset, ordering);
    }

    /**
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset, char ordering) {
        return new ComplexNDArray(ArrayUtil.floatCopyOf(data), shape, offset, ordering);
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer buffer, int[] shape, int offset, char ordering) {
        return new ComplexNDArray(buffer, shape, Nd4j.getComplexStrides(shape), offset, ordering);
    }

    /**
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset) {
        return new ComplexNDArray(ArrayUtil.floatCopyOf(data), shape, offset);
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer buffer, int[] shape, int offset) {
        return new ComplexNDArray(buffer, shape, Nd4j.getComplexStrides(shape), offset, Nd4j.order());
    }

    /**
     * Create a complex ndarray with the given data
     *
     * @param data     the data to use with tne ndarray
     * @param shape    the shape of the ndarray
     * @param stride   the stride for the ndarray
     * @param offset   the offset of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created complex ndarray
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new ComplexNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[][] floats) {
        return new NDArray(floats);
    }

    @Override
    public INDArray create(float[][] data, char ordering) {
        return new NDArray(data,ordering);
    }

    @Override
    public IComplexNDArray createComplex(float[] dim) {
        return new ComplexNDArray(dim);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new NDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, int offset) {
        return new NDArray(buffer, shape, Nd4j.getStrides(shape), offset);
    }

    /**
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int offset, char ordering) {
        return new ComplexNDArray(data, shape, Nd4j.getComplexStrides(shape, ordering), offset, ordering);

    }

    /**
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int offset) {
        return new ComplexNDArray(data, shape, offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int offset) {
        return new NDArray(data, shape, offset);
    }
}
