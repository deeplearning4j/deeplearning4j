/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.factory;


import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.blas.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.io.File;
import java.util.*;

public interface NDArrayFactory {


    char FORTRAN = 'f';
    char C = 'c';

    /**
     * Return extra blas operations
     * @return
     */
    Blas blas();

    Lapack lapack();

    /**
     * Return the level 1 blas operations
     * @return
     */
    Level1 level1();

    /**
     * Return the level 2 blas operations
     * @return
     */
    Level2 level2();

    /**
     * Return the level 3 blas operations
     * @return
     */
    Level3 level3();

    /**
     * Create blas
     */
    void createBlas();

    /**
     * Create level 1
     */
    void createLevel1();

    /**
     * Create level 2
     */
    void createLevel2();

    /**
     * Create level 3
     */
    void createLevel3();

    /**
     * Create lapack
     */
    void createLapack();



    /**
     * Sets the order. Primarily for testing purposes
     *
     * @param order
     */
    void setOrder(char order);

    /**
     * Sets the data opType
     *
     * @param dtype
     */
    void setDType(DataType dtype);

    /**
     * Create an ndarray with the given shape
     * and data
     * @param shape the shape to use
     * @param buffer the buffer to use
     * @return the ndarray
     */
    INDArray create(int[] shape, DataBuffer buffer);

    /**
     * Create an ndarray with the given shape
     * and data
     * @param buffer the buffer to use
     * @param longShapeDescriptor the shape descriptor to use.
     * @return the ndarray
     */
    INDArray create(DataBuffer buffer,LongShapeDescriptor longShapeDescriptor);

    /**
     * Returns the order for this ndarray for internal data storage
     *
     * @return the order (c or f)
     */
    char order();

    /**
     * Returns the data opType for this ndarray
     *
     * @return the data opType for this ndarray
     */
    DataType dtype();

    /**
     * /**
     * Returns a flattened ndarray with all of the elements in each ndarray
     * regardless of dimension
     *
     * @param matrices the ndarrays to use
     * @return a flattened ndarray of the elements in the order of titerating over the ndarray and the linear view of
     * each
     */
    INDArray toFlattened(Collection<INDArray> matrices);


    /**
     * Returns a flattened ndarray with all of the elements in each ndarray
     * regardless of dimension
     *
     * @param matrices the ndarrays to use
     * @return a flattened ndarray of the elements in the order of titerating over the ndarray and the linear view of
     * each
     */
    INDArray toFlattened(int length, Iterator<? extends INDArray>... matrices);

    /**
     * Returns a flattened ndarray with all elements in each ndarray
     * regardless of dimension.
     * Order is specified to ensure flattening order is consistent across
     * @param matrices the ndarrays to flatten
     * @param order the order in which the ndarray values should be flattened
     * @return
     */
    INDArray toFlattened(char order, Collection<INDArray> matrices);

    /**
     * Returns a column vector where each entry is the nth bilinear
     * product of the nth slices of the two tensors.
     */
    INDArray bilinearProducts(INDArray curr, INDArray in);

    /**
     * Flatten all of the ndarrays in to one long vector
     *
     * @param matrices the matrices to flatten
     * @return the flattened vector
     */
    INDArray toFlattened(INDArray... matrices);

    /**
     * Flatten all of the ndarrays in to one long vector
     *
     * @param matrices the matrices to flatten
     * @return the flattened vector
     */
    INDArray toFlattened(char order, INDArray... matrices);

    /**
     * Create the identity ndarray
     *
     * @param n the number for the identity
     * @return
     */
    INDArray eye(long n);

    /**
     * Rotate a matrix 90 degrees
     *
     * @param toRotate the matrix to rotate
     * @return the rotated matrix
     */
    void rot90(INDArray toRotate);

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     *
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    INDArray rot(INDArray reverse);

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     *
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    INDArray reverse(INDArray reverse);


    /**
     * Array of evenly spaced values.
     * Returns a row vector
     *
     * @param begin the begin of the range
     * @param end   the end of the range
     * @return the range vector
     */
    INDArray arange(double begin, double end, double step);


    /**
     * Copy a to b
     *
     * @param a the origin matrix
     * @param b the destination matrix
     */
    void copy(INDArray a, INDArray b);

    /**
     * Generates a random matrix between min and max
     *
     * @param shape the number of rows of the matrix
     * @param min   the minimum number
     * @param max   the maximum number
     * @param rng   the rng to use
     * @return a drandom matrix of the specified shape and range
     */
    INDArray rand(int[] shape, float min, float max, org.nd4j.linalg.api.rng.Random rng);

    INDArray rand(long[] shape, float min, float max, org.nd4j.linalg.api.rng.Random rng);

    /**
     * Generates a random matrix between min and max
     *
     * @param rows    the number of rows of the matrix
     * @param columns the number of columns in the matrix
     * @param min     the minimum number
     * @param max     the maximum number
     * @param rng     the rng to use
     * @return a drandom matrix of the specified shape and range
     */
    INDArray rand(long rows, long columns, float min, float max, org.nd4j.linalg.api.rng.Random rng);

    INDArray appendBias(INDArray... vectors);

    /**
     * Create an ndarray with the given data layout
     *
     * @param data the data to create the ndarray with
     * @return the ndarray with the given data layout
     */
    INDArray create(double[][] data);

    /**
     * Create a matrix from the given
     * data and ordering
     * @param data
     * @param ordering
     * @return
     */
    INDArray create(double[][] data, char ordering);


    /**
     * Concatneate ndarrays along a dimension
     *
     * @param dimension the dimension to concatneate along
     * @param toConcat  the ndarrays to concateneate
     * @return the concatneated ndarrays
     */
    INDArray concat(int dimension, INDArray... toConcat);

    /**
     * Concatenate ndarrays along a dimension
     *
     * PLEASE NOTE: This method is special for GPU backend, it works on HOST side only.
     *
     * @param dimension the dimension to concatneate along
     * @param toConcat  the ndarrays to concateneate
     * @return the concatneated ndarrays
     */
    INDArray specialConcat(int dimension, INDArray... toConcat);

    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes
     *
     * @param source source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes indexes from source array
     * @return the concatneated ndarrays
     */
    INDArray pullRows(INDArray source, int sourceDimension, int[] indexes);

    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes
     * @param source source tensor
     * @param sourceDimension  dimension of source tensor
     * @param indexes indexes from source array
     * @return the concatneated ndarrays
     */
    INDArray pullRows(INDArray source, int sourceDimension, long[] indexes);

    /**
     * This method produces concatenated array, that consist from tensors,
     * fetched from source array, against some dimension and specified indexes
     *
     * @param source source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes indexes from source array
     * @param order order for result array
     * @return
     */
    INDArray pullRows(INDArray source, int sourceDimension, int[] indexes, char order);


    /**
     * * This method produces concatenated array, that consist from tensors,
     * fetched from source array, against some dimension and specified indexes
     * in to the destination array
     * @param source
     * @param destination
     * @param sourceDimension
     * @param indexes
     * @return
     */
    INDArray pullRows(INDArray source, INDArray destination, int sourceDimension, int[] indexes);

    /**
     * In place shuffle of an ndarray
     * along a specified set of dimensions
     *
     * @param array the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    void shuffle(INDArray array, Random rnd, long... dimension);

    /**
     * Symmetric in place shuffle of an ndarray
     * along a specified set of dimensions. All arrays should have equal shapes.
     *
     * @param array the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    void shuffle(Collection<INDArray> array, Random rnd, long... dimension);

    /**
     * Symmetric in place shuffle of an ndarray
     * along a specified set of dimensions. Each array in list should have it's own dimension at the same index of dimensions array
     *
     * @param array the ndarray to shuffle
     * @param dimensions the dimensions to do the shuffle
     * @return
     */
    void shuffle(List<INDArray> array, Random rnd, List<long[]> dimensions);


    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return the random ndarray with the specified shape
     */
    INDArray rand(long rows, long columns, org.nd4j.linalg.api.rng.Random r);

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param seed    the  seed to use
     * @return the random ndarray with the specified shape
     */
    INDArray rand(long rows, long columns, long seed);

    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    INDArray rand(long rows, long columns);

    /**
     * Create a random (uniform 0-1) NDArray with the specified shape and order
     * @param order      Order ('c' or 'f') of the output array
     * @param rows       Number of rows of the output array
     * @param columns    Number of columns of the output array
     */
    INDArray rand(char order, long rows, long columns);

    /**
     * Random normal using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return
     */
    INDArray randn(long rows, long columns, org.nd4j.linalg.api.rng.Random r);

    /**
     * Random normal (N(0,1)) using the current time stamp
     * as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    INDArray randn(long rows, long columns);

    /**
     * Random normal N(0,1), with specified output order
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     */
    INDArray randn(char order, long rows, long columns);

    /**
     * Random normal using the specified seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    INDArray randn(long rows, long columns, long seed);


    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int[] shape, Distribution r);

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int[] shape, org.nd4j.linalg.api.rng.Random r);

    INDArray rand(long[] shape, org.nd4j.linalg.api.rng.Random r);

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param seed  the  seed to use
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int[] shape, long seed);

    INDArray rand(long[] shape, long seed);

    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int[] shape);


    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    INDArray rand(long[] shape);

    /**
     * Create a random ndarray with the given shape, and specified output order
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    INDArray rand(char order, int[] shape);

    /**
     * Create a random ndarray with the given shape
     * and specified output order
     * @param order the order of the array
     * @param shape the shape of the array
     * @return the created ndarray
     */
    INDArray rand(char order, long[] shape);

    /**
     * Random normal using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     */
    INDArray randn(int[] shape, org.nd4j.linalg.api.rng.Random r);

    INDArray randn(long[] shape, org.nd4j.linalg.api.rng.Random r);

    /**
     * Random normal N(0,1) using the current time stamp
     * as the seed
     *
     * @param shape the shape of the ndarray
     */
    INDArray randn(int[] shape);

    /**
     * Random normal N(0,1) using the current time stamp
     * as the seed
     *
     * @param shape the shape of the ndarray
     */
    INDArray randn(long[] shape);

    /**
     * Random normal N(0,1) with the specified shape and order
     *
     * @param order the order ('c' or 'f') of the output array
     * @param shape the shape of the ndarray
     */
    INDArray randn(char order, int[] shape);

    /**
     * Random normal N(0,1) with the specified shape and order
     *
     * @param order the order ('c' or 'f') of the output array
     * @param shape the shape of the ndarray
     */
    INDArray randn(char order, long[] shape);

    /**
     * Random normal using the specified seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    INDArray randn(int[] shape, long seed);


    /**
     * Random normal using the specified seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    INDArray randn(long[] shape, long seed);


    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    INDArray create(double[] data);

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data);


    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    INDArray create(DataBuffer data);


    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray create(long columns);

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray zeros(long rows, long columns);


    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray zeros(long columns);


    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return the created ndarray
     */
    INDArray valueArrayOf(int[] shape, double value);

    INDArray valueArrayOf(long[] shape, double value);


    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param value   the value to assign
     * @return the created ndarray
     */
    INDArray valueArrayOf(long rows, long columns, double value);


    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray ones(long rows, long columns);

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray ones(long columns);


    /**
     * Concatenates two matrices horizontally. Matrices must have identical
     * numbers of rows.
     *
     * @param arrs
     */
    INDArray hstack(INDArray... arrs);

    /**
     * Concatenates two matrices vertically. Matrices must have identical
     * numbers of columns.
     *
     * @param arrs
     */
    INDArray vstack(INDArray... arrs);


    /**
     * Create an ndarray of zeros
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    INDArray zeros(int[] shape);

    /**
     * Create an ndarray of zeros
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    INDArray zeros(long[] shape);

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    INDArray ones(int[] shape);

    INDArray ones(long[] shape);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param data    the data to use with the ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    INDArray create(DataBuffer data, long rows, long columns, int[] stride, long offset);


    /**
     * @param data
     * @param rows
     * @param columns
     * @param stride
     * @param offset
     * @return
     */
    INDArray create(float[] data, long rows, long columns, int[] stride, long offset);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    INDArray create(float[] data, int[] shape, int[] stride, long offset);

    INDArray create(float[] data, long[] shape, long[] stride, long offset);

    INDArray create(float[] data, long[] shape, long[] stride, char order, DataType dataType);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    INDArray create(double[] data, int[] shape);

    INDArray create(double[] data, long[] shape);
    INDArray create(float[] data, long[] shape);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data, int[] shape);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param data    the data to use with tne ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    INDArray create(double[] data, long rows, long columns, int[] stride, long offset);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    INDArray create(double[] data, int[] shape, int[] stride, long offset);

    INDArray create(double[] data, long[] shape, long[] stride, long offset);

    INDArray create(double[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace);
    INDArray create(float[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace);
    INDArray create(long[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace);
    INDArray create(int[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace);
    INDArray create(short[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace);
    INDArray create(byte[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace);
    INDArray create(boolean[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace);


    INDArray create(double[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace);
    INDArray create(float[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace);
    INDArray create(long[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace);
    INDArray create(int[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace);
    INDArray create(short[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace);
    INDArray create(byte[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace);
    INDArray create(boolean[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace);


    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    INDArray create(DataBuffer data, int[] shape);


    INDArray create(DataBuffer data, long[] shape);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    INDArray create(DataBuffer data, int[] shape, int[] stride, long offset);

    /**
     * Creates an ndarray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */

    INDArray create(DataBuffer data, long[] shape, long[] stride, long offset);


    /**
     * Creates an ndarray with the specified shape
     * concatneated from the given input arrays
     * @param shape the shape of the ndarray
     * @return the instance
     */
    INDArray create(List<INDArray> list, int[] shape);

    /**
     * Creates an ndarray with the specified shape
     * concatneated from the given input arrays
     * @param list  the list of ndarrays to concatneate
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    INDArray create(List<INDArray> list, long[] shape);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    INDArray create(long rows, long columns, int[] stride, long offset);

    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(long rows, long columns, long[] stride, long offset);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(int[] shape, int[] stride, long offset);


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(long[] shape, long[] stride, long offset);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    INDArray create(long rows, long columns, int[] stride);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    INDArray create(int[] shape, int[] stride);

    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    INDArray create(long[] shape, long[] stride);

    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    INDArray create(long[] shape);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    INDArray create(long rows, long columns);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    INDArray create(int[] shape);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    INDArray scalar(float value, long offset);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    INDArray scalar(double value, long offset);


    INDArray scalar(int value, long offset);

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    INDArray scalar(Number value);

    INDArray empty(DataType type);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    INDArray scalar(float value);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    INDArray scalar(double value);


    /**
     * Create a scalar nd array with the data type
     * and a default value depending on the data type.
     * Generally this will be an empty string for
     * UTF8 or 0.0 for numerical values, or true for booleans.
     *
     * @param dataType the dataType of the scalar
     *                 * @return the scalar nd array
     */
    INDArray scalar(DataType dataType);

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    INDArray create(float[] data, int[] shape, long offset);

    /**
     * Create an ndarray with the specified data
     * @param data the data to use
     * @param shape the shape of the ndarray
     * @param ordering  the ordering of the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data, int[] shape, char ordering);

    /**
     * Create an ndarray with the specified data
     * @param floats the data to use
     * @return the created ndarray
     */
    INDArray create(float[][] floats);

    /**
     * Create an ndarray with the specified data
     * @param data the data to use
     * @param ordering the ordering to use
     * @return the created ndarray
     */
    INDArray create(float[][] data, char ordering);

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     * @return
     */
    INDArray create(float[] data, int[] shape, int[] stride, long offset, char ordering);

    /**
     * Create an ndarray with the specified shape
     * @param buffer the buffer to use
     * @param shape the shape of the ndarray
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(DataBuffer buffer, int[] shape, long offset);

    /**
     * Create an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param ordering the ordering of the ndarray
     * @return the created ndarray
     */
    INDArray create(int[] shape, char ordering);


    /**
     * Create an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param ordering the ordering of the ndarray
     * @return the created ndarray
     */
    INDArray create(long[] shape, char ordering);

    /**
     * Create an ndarray with the specified shape
     * @param dataType the data type of the ndarray
     * @param shape the shape of the ndarray
     * @param ordering the ordering of the ndarray
     * @param workspace the workspace to allocate the ndarray in
     * @return
     */
    INDArray create(DataType dataType, long[] shape, char ordering, MemoryWorkspace workspace);

    /**
     * Create an ndarray with the specified shape
     * @param dataType  Data type of the new array
     * @param shape     Shape of the new array
     * @param strides   Strides of the new array
     * @param ordering  Fortran 'f' or C/C++ 'c' ordering.
     * @param workspace Workspace to allocate the array in
     * @return
     */
    INDArray create(DataType dataType, long[] shape, long[] strides, char ordering, MemoryWorkspace workspace);

   /**
     * Create an ndArray with padded Buffer
     * @param dataType
     * @param shape
     * @param paddings
     * @param paddingOffsets
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @param workspace
     * @return
     */
    INDArray create(DataType dataType, long[] shape, long[] paddings, long[] paddingOffsets, char ordering, MemoryWorkspace workspace);

    INDArray createUninitialized(int[] shape, char ordering);

    INDArray createUninitialized(long[] shape, char ordering);

    INDArray createUninitialized(DataType dataType, long[] shape, char ordering, MemoryWorkspace workspace);

    default INDArray createUninitialized(DataType dataType, long[] shape, long[] strides, char ordering) {
        return createUninitialized(dataType, shape, strides, ordering, Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread());
    }


    /**
     * Create an uninitialized ndArray. Detached from workspace.
     * @param dataType data type. Exceptions will be thrown for UTF8, COMPRESSED and UNKNOWN data types.
     * @param ordering  Fortran 'f' or C/C++ 'c' ordering.
     * @param shape the shape of the array.
     * @return the created detached array.
     */
    INDArray createUninitializedDetached(DataType dataType, char ordering, long... shape);

    /**
     *
     * @param data
     * @param newShape
     * @param newStride
     * @param offset
     * @param ordering
     * @return
     */
    INDArray create(DataBuffer data, int[] newShape, int[] newStride, long offset, char ordering);

    /**
     * Create an ndarray from the given data buffer
     * @param data the data buffer to use
     * @param newShape the new shape of the ndarray
     * @param newStride the new stride of the ndarray
     * @param offset the offset of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering);

    /**
     * Create an ndarray from the given data buffer
     * @param data the data buffer to use
     * @param newShape the new shape of the ndarray
     * @param newStride  the new stride of the ndarray
     * @param offset the offset of the ndarray
     * @param ews  the element wise stride of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, long ews, char ordering);

    /**
     * Create an ndarray from the given data buffer
     * @param data the data buffer to use
     * @param newShape the new shape of the ndarray
     * @param newStride  the new stride of the ndarray
     * @param offset the offset of the ndarray
     * @param ews  the element wise stride of the ndarray
     * @param ordering the ordering for the ndarray
     * @param isView  whether the ndarray is a view or not
     * @return the created ndarray
     */
    INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, long ews, char ordering,boolean isView);

    /**
     * Create an ndarray from the given data buffer
     * @param data the data buffer to use
     * @param newShape the new shape of the ndarray
     * @param newStride the new stride of the ndarray
     * @param offset the offset of the ndarray
     * @param ordering the ordering for the ndarray
     * @param dataType the data type of the ndarray
     * @return the created ndarray
     */
    INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering, DataType dataType);

    /**
     * Create an ndarray from the given data buffer
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param min the min number for the uniform distribution
     * @param max the max number for the uniform distribution
     * @param rng the rng to use
     * @return the created ndarray
     */
    INDArray rand(long rows, long columns, double min, double max, org.nd4j.linalg.api.rng.Random rng);

    /**
     * Create an ndarray from the given data buffer
     * @param data the data buffer to use
     * @param shape the shape of the ndarray
     * @param offset the offset of the ndarray
     * @param order  the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data, int[] shape, long offset, Character order);

    /**
     * Create an ndarray from the given data buffer
     * @param data the data buffer to use
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the offset of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data, long rows, long columns, int[] stride, long offset, char ordering);

    /**
     * Create an ndarray from the given data buffer
     * @param data the data buffer to use
     * @param shape the shape of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(double[] data, int[] shape, char ordering);

    /**
     * Create an ndarray from the given list of ndarrays.
     * @param list  the list to create the ndarray from
     * @param shape the shape of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(List<INDArray> list, int[] shape, char ordering);

    /**
     * Create an ndarray from the given list of ndarrays.
     * @param list the list to create the ndarray from
     * @param shape the shape of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(List<INDArray> list, long[] shape, char ordering);

    /**
     * Create an ndarray from the double array as the data.
     * @param data the data to create the ndarray from
     * @param shape the shape of the ndarray
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(double[] data, int[] shape, long offset);

    /**
     * Create an ndarray from the double array as the data.
     * @param data the data to create the ndarray from
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the offset of the ndarray
     * @param ordering the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(double[] data, int[] shape, int[] stride, long offset, char ordering);

    /**
     * Create a random ndrray  with the min and max values specified.
     * @param shape  the shape of the ndarray
     * @param min the min number for the uniform distribution
     * @param max the max number for the uniform distribution
     * @param rng the rng to use
     * @return the created ndarray
     */
    INDArray rand(int[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng);

    /**
     * Create a random ndrray  with the min and max values specified.
     * @param shape the shape of the ndarray
     * @param min the min number for the uniform distribution
     * @param max the max number for the uniform distribution
     * @param rng the rng to use
     * @return the created ndarray
     */
    INDArray rand(long[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng);

    /**
     * Create an int ndarray from the givne data.
     * @param ints the shape of the ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(int[] ints, int[] shape, int[] stride, long offset);

    /**
     * Create an int ndarray from the given data.
     * @param data the shape of the ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param order the ordering for the ndarray
     * @param offset  the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(int[] data, int[] shape, int[] stride, char order, long offset);

    /**
     * Create an ndarray of data type
     * {@link Nd4j#dataType()}
     * @param rows  the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param ordering the ordering for the ndarray
     * @return  the created ndarray
     */
    INDArray create(long rows, long columns, char ordering);

    /**
     * Create an ndarray of data type
     * @param shape the shape of the ndarray
     * @param dataType the data type of the ndarray
     * @return the created ndarray
     */
    INDArray create(int[] shape, DataType dataType, MemoryWorkspace workspace);

    /**
     * Create an ndarray of data type
     * @param data the data to use
     * @param order the ordering for the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data, char order);

    /**
     * Create an ndarray of data type
     * @param data the data to use
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param order the ordering for the ndarray
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data, int[] shape, int[] stride, char order, long offset);

    /**
     * Create an ndarray of data type
     * @param data the data to use
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param order the ordering for the ndarray
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data, long[] shape, long[] stride, char order, long offset);

    /**
     *
     * @param buffer
     * @param shape
     * @param stride
     * @param order
     * @param offset
     * @return
     */
    INDArray create(DataBuffer buffer, int[] shape, int[] stride, char order, long offset);

    /**
     *
     * @param data
     * @param order
     * @return
     */
    INDArray create(double[] data, char order);

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param order
     * @param offset
     * @return
     */
    INDArray create(double[] data, int[] shape, int[] stride, char order, long offset);

    /**
     *
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     * @return
     */
    INDArray create(int[] shape, int[] stride, long offset, char ordering);

    INDArray create(long[] shape, long[] stride, long offset, char ordering);


    /**
     *
     * @param typeSrc
     * @param source
     * @param typeDst
     * @return
     */

    INDArray convertDataEx(DataTypeEx typeSrc, INDArray source, DataTypeEx typeDst);

    /**
     *
     * @param typeSrc
     * @param source
     * @param typeDst
     * @return
     */
    DataBuffer convertDataEx(DataTypeEx typeSrc, DataBuffer source, DataTypeEx typeDst);

    /**
     *
     * @param typeSrc
     * @param source
     * @param typeDst
     * @param target
     */
    void convertDataEx(DataTypeEx typeSrc, DataBuffer source, DataTypeEx typeDst, DataBuffer target);

    /**
     *
     * @param typeSrc
     * @param source
     * @param typeDst
     * @param target
     * @param length
     */
    void convertDataEx(DataTypeEx typeSrc, Pointer source, DataTypeEx typeDst, Pointer target, long length);

    /**
     *
     * @param typeSrc
     * @param source
     * @param typeDst
     * @param buffer
     */
    void convertDataEx(DataTypeEx typeSrc, Pointer source, DataTypeEx typeDst, DataBuffer buffer);

    /**
     * Create from an in memory numpy pointer
     * @param pointer the pointer to the
     *                numpy array
     * @return an ndarray created from the in memory
     * numpy pointer
     */
    INDArray createFromNpyPointer(Pointer pointer);

    INDArray createFromDescriptor(DataBuffer shapeInformation);


    /**
     * Create from an in memory numpy header.
     * Use this when not interopping
     * directly from python.
     *
     * @param pointer the pointer to the
     *                numpy header
     * @return an ndarray created from the in memory
     * numpy pointer
     */
    INDArray createFromNpyHeaderPointer(Pointer pointer);

    /**
     * Create from a given numpy file.
     * @param file the file to create the ndarray from
     * @return the created ndarray
     */
    INDArray createFromNpyFile(File file);

    /**
     * Create a Map<String, INDArray> from given npz file.
     * @param file the file to create the map from
     * @return Map<String, INDArray>
     */
    public Map<String, INDArray> createFromNpzFile(File file) throws Exception;

    /**
     * Convert an {@link INDArray}
     * to a numpy array.
     * This will usually be used
     * for writing out to an external source.
     * Note that this will create a zero copy reference
     * to this ndarray's underlying data.
     *
     * @param array the array to convert
     * @returnthe created pointer representing
     * a pointer to a numpy header
     */
    Pointer convertToNumpy(INDArray array);

    /**
     * Convert an {@link INDArray}
     * to a numpy array.
     * This will usually be used
     * for writing out to an external source.
     * Note that this will create a zero copy reference
     * to this ndarray's underlying data.
     *
     *
     * @param array the array to convert
     * @returnthe created pointer representing
     * a pointer to a numpy header
     */
    DataBuffer convertToNumpyBuffer(INDArray array);

    INDArray create(float[] data, long[] shape, long[] stride, long offset, char ordering);

    INDArray create(double[] data, long[] shape, long[] stride, long offset, char ordering);



    /**
     *
     * @param x
     * @param descending
     * @return
     */
    INDArray sort(INDArray x, boolean descending);

    INDArray sort(INDArray x, boolean descending, long... dimensions);

    INDArray sortCooIndices(INDArray x);

    INDArray create(float[] data, long[] shape, long offset, Character order);
    INDArray create(double[] data, long[] shape, long offset, Character order);
    INDArray create(float[] data, long[] shape, char ordering);
    INDArray create(double[] data, long[] shape, char ordering);

    /**
     * Create from a {@link LongShapeDescriptor}
     * a buffer will be allocated if the descriptor is not marked as empty.
     * @param longShapeDescriptor the shape descriptor
     * @return
     */
    INDArray create(LongShapeDescriptor longShapeDescriptor);

    // =========== String methods ============

    INDArray create(Collection<String> strings, long[] shape, char order);

    INDArray createUninitialized(DataType dataType, long[] shape, long[] strides, char ordering, MemoryWorkspace currentWorkspace);

    INDArray create(DataBuffer dataBuffer, DataBuffer descriptor);
}
