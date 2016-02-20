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

package org.nd4j.linalg.cpu.complex;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.BaseComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;

import java.util.List;


/**
 * ComplexNDArray for complex numbers.
 * <p/>
 * <p/>
 * Note that the indexing scheme for a complex ndarray is 2 * length
 * not length.
 * <p/>
 * The reason for this is the fact that imaginary components have
 * to be stored alongside realComponent components.
 *
 * @author Adam Gibson
 */
public class ComplexNDArray extends BaseComplexNDArray {
	
	public void close() {
	}


    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data     the data to use
     * @param shape    the shape of the ndarray
     * @param ordering
     */
    public ComplexNDArray(float[] data, int[] shape, char ordering) {
        super(data, shape, ordering);
    }

    public ComplexNDArray(int[] shape, int offset, char ordering) {
        super(shape, offset, ordering);
    }

    public ComplexNDArray(int[] shape) {
        super(shape);
    }

    public ComplexNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        super(data, shape, stride, ordering);
    }

    public ComplexNDArray(int[] shape, char ordering) {
        super(shape, ordering);
    }

    /**
     * Initialize the given ndarray as the real component
     *
     * @param m        the real component
     * @param stride   the stride of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public ComplexNDArray(INDArray m, int[] stride, char ordering) {
        super(m, stride, ordering);
    }

    /**
     * Construct a complex matrix from a realComponent matrix.
     *
     * @param m
     * @param ordering
     */
    public ComplexNDArray(INDArray m, char ordering) {
        super(m, ordering);
    }

    /**
     * Construct a complex matrix from a realComponent matrix.
     *
     * @param m
     */
    public ComplexNDArray(INDArray m) {
        super(m);
    }

    /**
     * Create with the specified ndarray as the real component
     * and the given stride
     *
     * @param m      the ndarray to use as the stride
     * @param stride the stride of the ndarray
     */
    public ComplexNDArray(INDArray m, int[] stride) {
        super(m, stride);
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices the slices of the ndarray
     * @param shape  the final shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public ComplexNDArray(List<IComplexNDArray> slices, int[] shape, int[] stride) {
        super(slices, shape, stride);
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices   the slices of the ndarray
     * @param shape    the final shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public ComplexNDArray(List<IComplexNDArray> slices, int[] shape, int[] stride, char ordering) {
        super(slices, shape, stride, ordering);
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices   the slices of the ndarray
     * @param shape    the final shape of the ndarray
     * @param ordering the ordering of the ndarray
     */
    public ComplexNDArray(List<IComplexNDArray> slices, int[] shape, char ordering) {
        super(slices, shape, ordering);
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices the slices of the ndarray
     * @param shape  the final shape of the ndarray
     */
    public ComplexNDArray(List<IComplexNDArray> slices, int[] shape) {
        super(slices, shape);
    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float[]
     *
     * @param newData the new data for this array
     * @param shape   the shape of the ndarray
     */
    public ComplexNDArray(IComplexNumber[] newData, int[] shape) {
        super(newData, shape);
    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float[]
     *
     * @param newData the new data for this array
     * @param shape   the shape of the ndarray
     * @param stride
     */
    public ComplexNDArray(IComplexNumber[] newData, int[] shape, int[] stride) {
        super(newData, shape, stride);
    }


    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new float[]
     *
     * @param newData  the new data for this array
     * @param shape    the shape of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public ComplexNDArray(IComplexNumber[] newData, int[] shape, char ordering) {
        super(newData, shape, ordering);
    }

    /**
     * Initialize with the given data,shape and stride
     *
     * @param data   the data to use
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public ComplexNDArray(float[] data, int[] shape, int[] stride) {
        super(data, shape, stride);
    }

    /**
     * THe ordering of the ndarray
     *
     * @param data     the data to use
     * @param shape    the final shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param offset   the offset
     * @param ordering the ordering
     */
    public ComplexNDArray(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public ComplexNDArray(float[] data, int[] shape, int[] stride, int offset) {
        super(data, shape, stride, offset);
    }

    public ComplexNDArray(float[] data, int[] shape) {
        super(data, shape);
    }

    public ComplexNDArray(float[] data, int[] shape, int offset, char ordering) {
        super(data, shape, offset, ordering);
    }

    public ComplexNDArray(float[] data, int[] shape, int offset) {
        super(data, shape, offset);
    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public ComplexNDArray(int[] shape, int[] stride, int offset) {
        super(shape, stride, offset);
    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering for the ndarray
     */
    public ComplexNDArray(int[] shape, int[] stride, int offset, char ordering) {
        super(shape, stride, offset, ordering);
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param ordering
     */
    public ComplexNDArray(int[] shape, int[] stride, char ordering) {
        super(shape, stride, ordering);
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public ComplexNDArray(int[] shape, int[] stride) {
        super(shape, stride);
    }

    /**
     * @param shape
     * @param offset
     */
    public ComplexNDArray(int[] shape, int offset) {
        super(shape, offset);
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public ComplexNDArray(int newRows, int newColumns) {
        super(newRows, newColumns);
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     * @param ordering   the ordering of the ndarray
     */
    public ComplexNDArray(int newRows, int newColumns, char ordering) {
        super(newRows, newColumns, ordering);
    }


    public ComplexNDArray(float[] doubles) {
        super(doubles);
    }

    public ComplexNDArray(float[][] floats) {
        this(new NDArray(floats));
    }

    public ComplexNDArray(DataBuffer data) {
        super(data);
    }

    public ComplexNDArray(DataBuffer data, int[] shape, int[] stride, int offset) {
        super(data, shape, stride, offset);
    }

    public ComplexNDArray(IComplexNumber[] data, int[] shape, int[] stride, int offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public ComplexNDArray(DataBuffer data, int[] newDims, int[] newStrides, int offset, char ordering) {
        super(data, newDims, newStrides, offset, ordering);
    }

    public ComplexNDArray(DataBuffer data, int[] shape) {
        super(data, shape);
    }

    public ComplexNDArray(DataBuffer data, int[] shape, int[] stride) {
        super(data, shape, stride);
    }

    public ComplexNDArray(float[] data, Character order) {
        super(data, order);
    }
}
