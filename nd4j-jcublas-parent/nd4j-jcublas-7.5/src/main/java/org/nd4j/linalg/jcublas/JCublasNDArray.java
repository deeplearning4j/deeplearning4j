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

package org.nd4j.linalg.jcublas;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by mjk on 8/23/14.
 *
 * @author mjk
 * @author Adam Gibson
 */

public class JCublasNDArray extends BaseNDArray {
	

    public JCublasNDArray(double[][] data) {
        super(data);
    }

    public JCublasNDArray(double[][] data, char ordering) {
        super(data, ordering);
    }

    public JCublasNDArray(int[] shape, DataBuffer buffer) {
        super(shape, buffer);
    }

    /**
     * Create this JCublasNDArray with the given data and shape and 0 offset
     *
     * @param data     the data to use
     * @param shape    the shape of the JCublasNDArray
     * @param ordering
     */
    public JCublasNDArray(float[] data, int[] shape, char ordering) {
        super(data, shape, ordering);


    }

    /**
     * @param data     the data to use
     * @param shape    the shape of the JCublasNDArray
     * @param offset   the desired offset
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(float[] data, int[] shape, int offset, char ordering) {
        super(data, shape, offset, ordering);


    }

    /**
     * Construct an JCublasNDArray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the JCublasNDArray
     * @param stride   the stride of the JCublasNDArray
     * @param offset   the desired offset
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride, int offset, char ordering) {
        super(shape, stride, offset, ordering);

    }

    /**
     * Create the JCublasNDArray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape    the shape of the JCublasNDArray
     * @param stride   the stride of the JCublasNDArray
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride, char ordering) {

        super(shape, stride, ordering);

    }

    public JCublasNDArray(int[] shape, int offset, char ordering) {

        super(shape, offset, ordering);

    }

    public JCublasNDArray(int[] shape) {

        super(shape);

    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     * @param ordering
     */
    public JCublasNDArray(int newRows, int newColumns, char ordering) {
        super(newRows, newColumns, ordering);

    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices   the slices to merge
     * @param shape    the shape of the JCublasNDArray
     * @param ordering
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, char ordering) {

        super(slices, shape, ordering);

    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices   the slices to merge
     * @param shape    the shape of the JCublasNDArray
     * @param stride
     * @param ordering
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, int[] stride, char ordering) {
        super(slices, shape, stride, ordering);

    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        super(data, shape, stride, ordering);

    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        super(data, shape, stride, offset, ordering);

    }

    public JCublasNDArray(DataBuffer data, int[] shape, int[] stride, int offset) {
        super(data, shape, stride, offset);
    }

    public JCublasNDArray(int[] data, int[] shape, int[] strides) {
        super(data, shape, strides);
    }

    public JCublasNDArray(DataBuffer data, int[] shape) {
        super(data, shape);
    }

    public JCublasNDArray(DataBuffer buffer, int[] shape, int offset) {
        super(buffer, shape, offset);
    }

    /**
     * Create this JCublasNDArray with the given data and shape and 0 offset
     *
     * @param data  the data to use
     * @param shape the shape of the JCublasNDArray
     */
    public JCublasNDArray(float[] data, int[] shape) {
        super(data, shape);
    }

    public JCublasNDArray(float[] data, int[] shape, int offset) {

        super(data, shape, offset);

    }

    /**
     * Construct an JCublasNDArray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the JCublasNDArray
     * @param stride the stride of the JCublasNDArray
     * @param offset the desired offset
     */
    public JCublasNDArray(int[] shape, int[] stride, int offset) {

        super(shape, stride, offset);
    }

    /**
     * Create the JCublasNDArray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the JCublasNDArray
     * @param stride the stride of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride) {
        super(shape, stride);
    }

    public JCublasNDArray(int[] shape, int offset) {
        super(shape, offset);
    }

    public JCublasNDArray(int[] shape, char ordering) {
        super(shape, ordering);
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public JCublasNDArray(int newRows, int newColumns) {
        super(newRows, newColumns);
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the JCublasNDArray
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape) {
        super(slices, shape);
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the JCublasNDArray
     * @param stride
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, int[] stride) {
        super(slices, shape, stride);

    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride) {
        super(data, shape, stride);
    }


    public JCublasNDArray(float[] data, int[] shape, int[] stride, int offset) {
        super(data, shape, stride, offset);
    }

    public JCublasNDArray(float[] data) {
        super(data);
    }


    public JCublasNDArray(JCublasNDArray doubleMatrix) {
        this(new int[]{doubleMatrix.rows, doubleMatrix.columns});
        this.data = dup().data();
    }

    public JCublasNDArray(double[] data, int[] shape, int[] stride, int offset) {
        super(data,shape,stride,offset);
    }

    public JCublasNDArray(float[][] floats) {
        super(floats);
    }

    public JCublasNDArray(float[][] data, char ordering) {
        super(data, ordering);
    }

    public JCublasNDArray(DataBuffer buffer, int[] shape, int offset, char ordering) {
        super(buffer, shape, offset, ordering);
    }

    public JCublasNDArray() {
    }

    public JCublasNDArray(DataBuffer buffer) {
        super(buffer);
    }

    public JCublasNDArray(DataBuffer buffer, int[] shape, int[] stride, int offset, char ordering) {
        super(buffer, shape, stride, offset, ordering);
    }

    public JCublasNDArray(float[] data, char order) {
        super(data, order);
    }

    public JCublasNDArray(FloatBuffer floatBuffer, char order) {
        super(floatBuffer, order);
    }

    public JCublasNDArray(DataBuffer buffer, int[] shape, int[] strides) {
        super(buffer, shape, strides);
    }

    public JCublasNDArray(double[] data, int[] shape, char ordering) {
        super(data, shape ,ordering);
    }

    public JCublasNDArray(double[] data, int[] shape, int[] stride, int offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

}