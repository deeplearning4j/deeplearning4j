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


import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;

import java.util.List;

/**
 *
 * Created by mjk on 8/23/14.
 *
 * @author mjk
 * @author Adam Gibson
 * @author raver119@gmail.com
 */

public class JCublasNDArray extends BaseNDArray {
	private transient Allocator allocator = AtomicAllocator.getInstance();

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

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to get
     * @return the array with the specified elements
     */
    @Override
    public double getDouble(int... indices) {
        allocator.synchronizeHostData(this);
        return super.getDouble(indices);
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param index the indices to get
     * @return the array with the specified elements
     */
    @Override
    public double getDouble(int index) {
        allocator.synchronizeHostData(this);
        return super.getDouble(index);
    }

    @Override
    public INDArray putScalar(int i, double value) {
        try {
            allocator.synchronizeHostData(this);
            return super.putScalar(i, value);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    @Override
    public INDArray putScalar(int i, float value) {
        try {
            allocator.synchronizeHostData(this);
            return super.putScalar(i, value);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    @Override
    public INDArray putScalar(int i, int value) {
        try {
            allocator.synchronizeHostData(this);
            return super.putScalar(i, value);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    @Override
    public INDArray putScalar(int[] indexes, double value) {
        try {
            allocator.synchronizeHostData(this);
            return super.putScalar(indexes, value);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    @Override
    public INDArray putScalar(int[] indexes, float value) {
        try {
            allocator.synchronizeHostData(this);
            return super.putScalar(indexes, value);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    @Override
    public INDArray putScalar(int[] indexes, int value) {
        try {
            allocator.synchronizeHostData(this);
            return super.putScalar(indexes, value);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    /**
     * Inserts the element at the specified index
     *
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int[] indices, INDArray element) {
        try {
            allocator.synchronizeHostData(this);
            return super.put(indices, element);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, int j, INDArray element) {
        try {
            allocator.synchronizeHostData(this);
            return super.put(i, j, element);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, int j, Number element) {
        try {
            allocator.synchronizeHostData(this);
            return super.put(i, j, element);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to put
     * @return this for chainability
     */
    @Override
    public INDArray putSlice(int slice, INDArray put) {
        try {
            allocator.synchronizeHostData(this);
            return super.putSlice(slice, put);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to get
     * @return the array with the specified elements
     */
    @Override
    public float getFloat(int... indices) {
        allocator.synchronizeHostData(this);
        return super.getFloat(indices);
    }

    @Override
    public INDArray dup() {
        allocator.synchronizeHostData(this);
        return super.dup();
    }

    @Override
    public INDArray dup(char order) {
        allocator.synchronizeHostData(this);
        return super.dup(order);
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    @Override
    public int getInt(int... indices) {
        allocator.synchronizeHostData(this);
        return super.getInt(indices);
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, Number element) {
        try {
            allocator.synchronizeHostData(this);
            return super.put(indices, element);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, INDArray element) {
        try {
            allocator.synchronizeHostData(this);
            return super.put(indices, element);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    @Override
    public INDArray getScalar(int i) {
        return super.getScalar(i);
    }

    /**
     * Fetch a particular number on a multi dimensional scale.
     *
     * @param indexes the indexes to get a number from
     * @return the number at the specified indices
     */
    @Override
    public INDArray getScalar(int... indexes) {
        allocator.synchronizeHostData(this);
        return super.getScalar(indexes);
    }

    @Override
    public double getDoubleUnsafe(int offset) {
        allocator.synchronizeHostData(this);
        return super.getDoubleUnsafe(offset);
    }

    @Override
    public double getDouble(int i, int j) {
        allocator.synchronizeHostData(this);
        return super.getDouble(i, j);
    }

    @Override
    public float getFloat(int i) {
        allocator.synchronizeHostData(this);
        return super.getFloat(i);
    }

    @Override
    public float getFloat(int i, int j) {
        allocator.synchronizeHostData(this);
        return super.getFloat(i, j);
    }

    /**
     * Returns the element at the specified row/column
     * This will throw an exception if the
     *
     * @param row    the row of the element to return
     * @param column the row of the element to return
     * @return a scalar indarray of the element at this index
     */
    @Override
    public INDArray getScalar(int row, int column) {
        allocator.synchronizeHostData(this);
        return super.getScalar(row, column);
    }

    /**
     * Insert a row in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param row   the row insert into
     * @param toPut the row to insert
     * @return this
     */
    @Override
    public INDArray putRow(int row, INDArray toPut) {
        try {
            allocator.synchronizeHostData(this);
            return super.putRow(row, toPut);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    /**
     * Insert a column in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param column the column to insert
     * @param toPut  the array to put
     * @return this
     */
    @Override
    public INDArray putColumn(int column, INDArray toPut) {
        try {
            allocator.synchronizeHostData(this);
            return super.putColumn(column, toPut);
        } finally {
            allocator.tickHostWrite(this);
        }
    }

    @Override
    public INDArray putScalarUnsafe(int offset, double value) {
        try {
            allocator.synchronizeHostData(this);
            return super.putScalarUnsafe(offset, value);
        } finally {
            allocator.tickHostWrite(this);
        }
    }
}