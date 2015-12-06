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

package org.nd4j.linalg.factory;


import org.nd4j.linalg.api.blas.Lapack;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.blas.Level2;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Creation of ndarrays via classpath discovery.
 *
 * @author Adam Gibson
 */
public interface NDArrayFactory {


    char FORTRAN = 'f';
    char C = 'c';

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
     * Creates an 1 x num ndarray with the specified value
     *
     * @param num   the number of columns
     * @param value the value to assign
     * @return a complex ndarray of the specified size
     * and value
     */
    IComplexNDArray complexValueOf(int num, IComplexNumber value);

    /**
     * Creates an shape ndarray with the specified value
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return a complex ndarray of the specified size
     * and value
     */
    IComplexNDArray complexValueOf(int[] shape, IComplexNumber value);

    /**
     * Creates an 1 x num ndarray with the specified value
     *
     * @param num   the number of columns
     * @param value the value to assign
     * @return a complex ndarray of the specified size
     * and value
     */
    IComplexNDArray complexValueOf(int num, double value);

    /**
     * Creates an shape ndarray with the specified value
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return a complex ndarray of the specified size
     * and value
     */
    IComplexNDArray complexValueOf(int[] shape, double value);

    /**
     * Sets the order. Primarily for testing purposes
     *
     * @param order
     */
    void setOrder(char order);

    /**
     * Sets the data type
     *
     * @param dtype
     */
    void setDType(DataBuffer.Type dtype);

    /**
     * Create an ndarray with the given shape
     * and data
     * @param shape the shape to use
     * @param buffer the buffer to use
     * @return the ndarray
     */
    INDArray create(int[] shape, DataBuffer buffer);

    /**
     * Returns the order for this ndarray for internal data storage
     *
     * @return the order (c or f)
     */
    char order();

    /**
     * Returns the data type for this ndarray
     *
     * @return the data type for this ndarray
     */
    DataBuffer.Type dtype();

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(int rows, int columns, int[] stride, int offset);

    /**
     * Generate a linearly spaced vector
     *
     * @param lower upper bound
     * @param upper lower bound
     * @param num   the step size
     * @return the linearly spaced vector
     */
    INDArray linspace(int lower, int upper, int num);

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
    INDArray eye(int n);

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
    INDArray arange(double begin, double end);

    /**
     * Create float
     *
     * @param real real component
     * @param imag imag component
     * @return
     */
    IComplexFloat createFloat(float real, float imag);


    /**
     * Create an instance of a complex double
     *
     * @param real the real component
     * @param imag the imaginary component
     * @return a new imaginary double with the specified real and imaginary components
     */
    IComplexDouble createDouble(double real, double imag);


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
    INDArray rand(int rows, int columns, float min, float max, org.nd4j.linalg.api.rng.Random rng);

    INDArray appendBias(INDArray... vectors);

    /**
     * Create an ndarray with the given data layout
     *
     * @param data the data to create the ndarray with
     * @return the ndarray with the given data layout
     */
    INDArray create(double[][] data);

    INDArray create(double[][] data,char ordering);

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arr the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    IComplexNDArray createComplex(INDArray arr);

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param data the data to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    IComplexNDArray createComplex(IComplexNumber[] data, int[] shape);


    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arrs the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    IComplexNDArray createComplex(List<IComplexNDArray> arrs, int[] shape);


    /**
     * Concatneate ndarrays along a dimension
     *
     * @param dimension the dimension to concatneate along
     * @param toConcat  the ndarrays to concateneate
     * @return the concatneated ndarrays
     */
    INDArray concat(int dimension, INDArray... toConcat);

    /**
     * Concatneate ndarrays along a dimension
     *
     * @param dimension the dimension to concatneate along
     * @param toConcat  the ndarrays to concateneate
     * @return the concatneated ndarrays
     */
    IComplexNDArray concat(int dimension, IComplexNDArray... toConcat);


    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int rows, int columns, org.nd4j.linalg.api.rng.Random r);

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param seed    the  seed to use
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int rows, int columns, long seed);

    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int rows, int columns);

    /**
     * Random normal using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return
     */
    INDArray randn(int rows, int columns, org.nd4j.linalg.api.rng.Random r);

    /**
     * Random normal using the current time stamp
     * as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    INDArray randn(int rows, int columns);

    /**
     * Random normal using the specified seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    INDArray randn(int rows, int columns, long seed);


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

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param seed  the  seed to use
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int[] shape, long seed);

    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    INDArray rand(int[] shape);

    /**
     * Random normal using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return
     */
    INDArray randn(int[] shape, org.nd4j.linalg.api.rng.Random r);

    /**
     * Random normal using the current time stamp
     * as the seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    INDArray randn(int[] shape);

    /**
     * Random normal using the specified seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    INDArray randn(int[] shape, long seed);


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
     * Creates an ndarray with the specified data
     *
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    IComplexNDArray createComplex(double[] data);


    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    INDArray create(DataBuffer data);

    /**
     * Creates an ndarray with the specified data
     *
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    IComplexNDArray createComplex(DataBuffer data);


    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray create(int columns);

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    IComplexNDArray createComplex(int columns);

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray zeros(int rows, int columns);

    /**
     * Creates a matrix of zeros
     *
     * @param rows    te number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    IComplexNDArray complexZeros(int rows, int columns);


    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray zeros(int columns);

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    IComplexNDArray complexZeros(int columns);


    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return the created ndarray
     */
    INDArray valueArrayOf(int[] shape, double value);


    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param value   the value to assign
     * @return the created ndarray
     */
    INDArray valueArrayOf(int rows, int columns, double value);


    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray ones(int rows, int columns);

    /**
     * Creates an ndarray
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    IComplexNDArray complexOnes(int rows, int columns);

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    INDArray ones(int columns);

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    IComplexNDArray complexOnes(int columns);


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
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    IComplexNDArray complexZeros(int[] shape);

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    INDArray ones(int[] shape);

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    IComplexNDArray complexOnes(int[] shape);

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data    the data to use with the ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(float[] data, int rows, int columns, int[] stride, int offset);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data    the data to use with the ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(DataBuffer data, int rows, int columns, int[] stride, int offset);

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
    INDArray create(DataBuffer data, int rows, int columns, int[] stride, int offset);

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] stride, int offset);


    /**
     * @param data
     * @param rows
     * @param columns
     * @param stride
     * @param offset
     * @return
     */
    INDArray create(float[] data, int rows, int columns, int[] stride, int offset);

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, int offset);

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, int offset, char ordering);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, char ordering);

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param offset the stride for the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int offset, char ordering);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data  the data to use with the ndarray
     * @param shape the shape of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, char ordering);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    INDArray create(float[] data, int[] shape, int[] stride, int offset);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    INDArray create(double[] data, int[] shape);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    INDArray create(float[] data, int[] shape);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    IComplexNDArray createComplex(double[] data, int[] shape);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    IComplexNDArray createComplex(float[] data, int[] shape);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    IComplexNDArray createComplex(double[] data, int[] shape, int[] stride);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    IComplexNDArray createComplex(float[] data, int[] shape, int[] stride);

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(double[] data, int rows, int columns, int[] stride, int offset);

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
    INDArray create(double[] data, int rows, int columns, int[] stride, int offset);

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    INDArray create(double[] data, int[] shape, int[] stride, int offset);


    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    INDArray create(DataBuffer data, int[] shape);

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    IComplexNDArray createComplex(DataBuffer data, int[] shape);


    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] stride);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    INDArray create(DataBuffer data, int[] shape, int[] stride, int offset);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    INDArray create(List<INDArray> list, int[] shape);


    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    INDArray create(int rows, int columns, int[] stride, int offset);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(int[] shape, int[] stride, int offset);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    INDArray create(int[] shape, int[] stride, int offset);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(int rows, int columns, int[] stride);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    INDArray create(int rows, int columns, int[] stride);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(int[] shape, int[] stride);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    INDArray create(int[] shape, int[] stride);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(int rows, int columns);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    INDArray create(int rows, int columns);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    IComplexNDArray createComplex(int[] shape);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    INDArray create(int[] shape);

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    INDArray scalar(Number value, int offset);

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    IComplexNDArray complexScalar(Number value, int offset);


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    IComplexNDArray complexScalar(Number value);


    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    INDArray scalar(float value, int offset);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    INDArray scalar(double value, int offset);


    INDArray scalar(int value, int offset);

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    INDArray scalar(Number value);

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
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    IComplexNDArray scalar(IComplexNumber value, int offset);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    IComplexNDArray scalar(IComplexFloat value);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    IComplexNDArray scalar(IComplexDouble value);

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    IComplexNDArray scalar(IComplexNumber value);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    IComplexNDArray scalar(IComplexFloat value, int offset);

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    IComplexNDArray scalar(IComplexDouble value, int offset);


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
    IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset, char ordering);


    /**
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    IComplexNDArray createComplex(double[] data, int[] shape, int offset, char ordering);


    IComplexNDArray createComplex(DataBuffer buffer, int[] shape, int offset, char ordering);

    /**
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    IComplexNDArray createComplex(double[] data, int[] shape, int offset);

    /**
     * @param buffer
     * @param shape
     * @param offset
     * @return
     */
    IComplexNDArray createComplex(DataBuffer buffer, int[] shape, int offset);

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    INDArray create(float[] data, int[] shape, int offset);

    /**
     *
     * @param data
     * @param shape
     * @param ordering
     * @return
     */
    INDArray create(float[] data, int[] shape, char ordering);

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    IComplexNDArray createComplex(float[] data, int[] shape, int offset, char ordering);

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    IComplexNDArray createComplex(float[] data, int[] shape, int offset);

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     * @return
     */
    IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset, char ordering);

    /**
     *
     * @param floats
     * @return
     */
    INDArray create(float[][] floats);

    INDArray create(float[][] data,char ordering);

    /**
     *
     * @param dim
     * @return
     */
    IComplexNDArray createComplex(float[] dim);

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     * @return
     */
    INDArray create(float[] data, int[] shape, int[] stride, int offset, char ordering);

    /**
     *
     * @param flatten
     * @return
     */
    IComplexNDArray complexFlatten(List<IComplexNDArray> flatten);

    /**
     *
     * @param flatten
     * @return
     */
    IComplexNDArray complexFlatten(IComplexNDArray[] flatten);

    /**
     *
     * @param buffer
     * @param shape
     * @param offset
     * @return
     */
    INDArray create(DataBuffer buffer, int[] shape, int offset);

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @return
     */
    IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset);

    /**
     *
     * @param shape
     * @param ordering
     * @return
     */
    INDArray create(int[] shape, char ordering);

    /**
     *
     * @param data
     * @param newShape
     * @param newStride
     * @param offset
     * @param ordering
     * @return
     */
    INDArray create(DataBuffer data, int[] newShape, int[] newStride, int offset, char ordering);

    /**
     *
     * @param data
     * @param newDims
     * @param newStrides
     * @param offset
     * @param ordering
     * @return
     */
    IComplexNDArray createComplex(DataBuffer data, int[] newDims, int[] newStrides, int offset, char ordering);

    /**
     *
     * @param rows
     * @param columns
     * @param min
     * @param max
     * @param rng
     * @return
     */
    INDArray rand(int rows, int columns, double min, double max, org.nd4j.linalg.api.rng.Random rng);

    /**
     *
     * @param data
     * @param order
     * @return
     */
    IComplexNDArray createComplex(float[] data, Character order);

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @param order
     * @return
     */
    INDArray create(float[] data, int[] shape, int offset, Character order);

    /**
     *
     * @param data
     * @param rows
     * @param columns
     * @param stride
     * @param offset
     * @param ordering
     * @return
     */
    INDArray create(float[] data, int rows, int columns, int[] stride, int offset, char ordering);

    /**
     *
     * @param data
     * @param shape
     * @param ordering
     * @return
     */
    INDArray create(double[] data, int[] shape, char ordering);

    /**
     *
     * @param list
     * @param shape
     * @param ordering
     * @return
     */
    INDArray create(List<INDArray> list, int[] shape, char ordering);

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    INDArray create(double[] data, int[] shape, int offset);

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     * @return
     */
    INDArray create(double[] data, int[] shape, int[] stride, int offset, char ordering);

    /**
     *
     * @param shape
     * @param min
     * @param max
     * @param rng
     * @return
     */
    INDArray rand(int[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng);

    /**
     *
     * @param ints
     * @param ints1
     * @param stride
     * @param offset
     * @return
     */
    IComplexNDArray createComplex(int[] ints, int[] ints1, int[] stride, int offset);

    /**
     *
     * @param ints
     * @param ints1
     * @param stride
     * @param offset
     * @return
     */
    INDArray create(int[] ints, int[] ints1, int[] stride, int offset);

    /**
     *
     * @param shape
     * @param ints1
     * @param stride
     * @param order
     * @param offset
     * @return
     */
    INDArray create(int[] shape, int[] ints1, int[] stride, char order, int offset);

    /**
     *
     * @param rows
     * @param columns
     * @param ordering
     * @return
     */
    INDArray create(int rows, int columns, char ordering);

    /**
     *
     * @param shape
     * @param dataType
     * @return
     */
    INDArray create(int[] shape, DataBuffer.Type dataType);

    /**
     *
     * @param data
     * @param order
     * @return
     */
    INDArray create(float[] data, char order);

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param order
     * @param offset
     * @return
     */
    INDArray create(float[] data, int[] shape, int[] stride, char order, int offset);

    /**
     *
     * @param buffer
     * @param shape
     * @param stride
     * @param order
     * @param offset
     * @return
     */
    INDArray create(DataBuffer buffer, int[] shape, int[] stride, char order, int offset);

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
    INDArray create(double[] data, int[] shape, int[] stride, char order, int offset);

    /**
     *
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     * @return
     */
    INDArray create(int[] shape, int[] stride, int offset, char ordering);

    /**
     *
     * @param shape
     * @param complexStrides
     * @param offset
     * @param ordering
     * @return
     */
    IComplexNDArray createComplex(int[] shape, int[] complexStrides, int offset, char ordering);
}
