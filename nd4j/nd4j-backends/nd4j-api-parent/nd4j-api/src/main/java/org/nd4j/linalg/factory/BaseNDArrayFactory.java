/*-
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


import lombok.val;
import org.nd4j.linalg.api.blas.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * Base NDArrayFactory class.
 * <p/>
 * Allows specification or data opType and row (c) or column(fortran) major order
 *
 * @author Adam Gibson
 */
public abstract class BaseNDArrayFactory implements NDArrayFactory {

    // We don't really care about dtype field we'll use context instead
    // protected DataBuffer.Type dtype;
    protected char order;
    protected Blas blas;
    protected Level1 level1;
    protected Level2 level2;
    protected Level3 level3;
    protected Lapack lapack;

    public BaseNDArrayFactory() {}

    @Override
    public Lapack lapack() {
        if (lapack == null)
            createLapack();
        return lapack;
    }

    @Override
    public Blas blas() {
        if (blas == null)
            createBlas();
        return blas;
    }

    @Override
    public Level1 level1() {
        if (level1 == null)
            createLevel1();
        return level1;
    }

    @Override
    public Level2 level2() {
        if (level2 == null)
            createLevel2();
        return level2;
    }

    @Override
    public Level3 level3() {
        if (level3 == null)
            createLevel3();
        return level3;
    }

    /**
     *
     * Initialize with the given data opType and ordering
     * The ndarray factory will use this for
     * @param dtype the data opType
     * @param order the ordering in mem
     */
    protected BaseNDArrayFactory(DataBuffer.Type dtype, Character order) {
        // this.dtype = dtype;
        if (Character.toLowerCase(order) != 'c' && Character.toLowerCase(order) != 'f')
            throw new IllegalArgumentException("Order must either be c or f");

        this.order = order;
    }

    /**
     * @param dtype the data opType
     * @param order the ordering
     */
    protected BaseNDArrayFactory(DataBuffer.Type dtype, char order) {
        // this.dtype = dtype;
        if (Character.toLowerCase(order) != 'c' && Character.toLowerCase(order) != 'f')
            throw new IllegalArgumentException("Order must either be c or f");

        this.order = order;
    }

    //input arrays must have same number of dimensions
    protected static void validateConcat(int dimension, INDArray... arrs) {
        if (arrs[0].isScalar()) {
            for (int i = 1; i < arrs.length; i++)
                if (!arrs[i].isScalar())
                    throw new IllegalArgumentException("All arrays must have same dimensions");
        } else {
            int dims = arrs[0].shape().length;
            long[] shape = ArrayUtil.removeIndex(arrs[0].shape(), dimension);
            for (int i = 1; i < arrs.length; i++) {
                assert Arrays.equals(shape, ArrayUtil.removeIndex(arrs[i].shape(), dimension));
                assert arrs[i].shape().length == dims;
            }
        }


    }

    /**
     * Sets the order. Primarily for testing purposes
     *
     * @param order
     */
    @Override
    public void setOrder(char order) {
        assert order == 'c' || order == 'f' : "Order specified must be either c or f";
        this.order = order;

    }

    @Override
    public INDArray rand(long[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        Nd4j.getRandom().setSeed(rng.getSeed());
        return Nd4j.getDistributions().createUniform(min, max).sample(shape);
    }

    @Override
    public INDArray rand(int[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        Nd4j.getRandom().setSeed(rng.getSeed());
        return Nd4j.getDistributions().createUniform(min, max).sample(shape);
    }

    @Override
    public INDArray rand(long rows, long columns, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        Nd4j.getRandom().setSeed(rng.getSeed());
        return rand(new long[] {rows, columns}, min, max, rng);
    }

    /**
     * Sets the data opType
     *
     * @param dtype
     */
    @Override
    public void setDType(DataBuffer.Type dtype) {
        assert dtype == DataBuffer.Type.DOUBLE || dtype == DataBuffer.Type.FLOAT
                        || dtype == DataBuffer.Type.INT : "Invalid opType passed, must be float or double";
        // this.dtype = dtype;
    }

    @Override
    public INDArray create(int[] shape, DataBuffer.Type dataType) {
        return create(shape, Nd4j.createBuffer(shape, dataType));
    }

    /**
     * Returns the order for this ndarray for internal data storage
     *
     * @return the order (c or f)
     */
    @Override
    public char order() {
        return order;
    }

    /**
     * Returns the data opType for this ndarray
     *
     * @return the data opType for this ndarray
     */
    @Override
    public DataBuffer.Type dtype() {
        return Nd4j.dataType();
    }

    /**
     * Generate a linearly spaced vector
     *
     * @param lower upper bound
     * @param upper lower bound
     * @param num   the step size
     * @return the linearly spaced vector
     */
    @Override
    public INDArray linspace(int lower, int upper, int num) {
        double[] data = new double[num];
        for (int i = 0; i < num; i++) {
            double t = (double) i / (num - 1);
            data[i] = lower * (1 - t) + t * upper;

        }

        //edge case for scalars
        INDArray ret = Nd4j.create(data.length);
        if (ret.isScalar())
            return ret;

        for (int i = 0; i < ret.length(); i++)
            ret.putScalar(i, data[i]);
        return ret;
    }

    @Override
    public IComplexNDArray createComplex(int[] ints, int[] ints1, int[] stride, long offset) {
        return createComplex(Nd4j.createBuffer(ints), ints1, stride, offset);
    }

    @Override
    public INDArray create(int[] ints, int[] ints1, int[] stride, long offset) {
        return create(Nd4j.createBuffer(ints), ints1, stride, offset);
    }

    @Override
    public INDArray create(long rows, long columns, char ordering) {
        return create(new long[] {rows, columns}, ordering);
    }


    /**
     * Returns a vector with all of the elements in every nd array
     * equal to the sum of the lengths of the ndarrays
     *
     * @param matrices the ndarrays to getFloat a flattened representation of
     * @return the flattened ndarray
     */
    @Override
    public INDArray toFlattened(Collection<INDArray> matrices) {
        int length = 0;
        for (INDArray m : matrices)
            length += m.length();
        INDArray ret = Nd4j.create(1, length);
        int linearIndex = 0;
        for (INDArray d : matrices) {
            ret.put(new INDArrayIndex[] {NDArrayIndex.interval(linearIndex, linearIndex + d.length())}, d);
            linearIndex += d.length();
        }

        return ret;

    }

    @Override
    public INDArray toFlattened(int length, Iterator<? extends INDArray>... matrices) {
        List<INDArray> arr = new ArrayList<>();
        for (Iterator<? extends INDArray> arrs : matrices) {
            while (arrs.hasNext())
                arr.add(arrs.next());
        }
        return toFlattened(arr);
    }

    /**
     * Returns a column vector where each entry is the nth bilinear
     * product of the nth slices of the two tensors.
     */
    @Override
    public INDArray bilinearProducts(INDArray curr, INDArray in) {
        assert curr.shape().length == 3;
        if (in.columns() != 1) {
            throw new AssertionError("Expected a column vector");
        }
        if (in.rows() != curr.size(curr.shape().length - 1)) {
            throw new AssertionError("Number of rows in the input does not match number of columns in tensor");
        }


        if (curr.size(curr.shape().length - 2) != curr.size(curr.shape().length - 1)) {
            throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
        }

        INDArray ret = Nd4j.create(curr.slices(), 1);
        INDArray inT = in.transpose();
        for (int i = 0; i < curr.slices(); i++) {
            INDArray slice = curr.slice(i);
            INDArray inTTimesSlice = inT.mmul(slice);
            ret.putScalar(i, Nd4j.getBlasWrapper().dot(inTTimesSlice, in));
        }
        return ret;
    }

    @Override
    public INDArray toFlattened(INDArray... matrices) {
        int length = 0;
        for (INDArray m : matrices)
            length += m.length();
        INDArray ret = Nd4j.create(1, length);
        int linearIndex = 0;
        for (INDArray d : matrices) {
            ret.put(new INDArrayIndex[] {NDArrayIndex.interval(linearIndex, linearIndex + d.length())}, d);
            linearIndex += d.length();
        }

        return ret;
    }


    @Override
    public INDArray toFlattened(char order, INDArray... matrices) {
        return toFlattened(order, Arrays.asList(matrices));
    }

    /**
     * Create the identity ndarray
     *
     * @param n the number for the identity
     * @return
     */
    @Override
    public INDArray eye(long n) {
        INDArray ret = Nd4j.create(n, n);
        for (int i = 0; i < n; i++) {
            ret.put(i, i, 1.0);
        }

        return ret.reshape(n, n);

    }

    /**
     * Rotate a matrix 90 degrees
     *
     * @param toRotate the matrix to rotate
     * @return the rotated matrix
     */
    @Override
    public void rot90(INDArray toRotate) {
        if (!toRotate.isMatrix())
            throw new IllegalArgumentException("Only rotating matrices");

        INDArray start = toRotate.transpose();
        for (int i = 0; i < start.rows(); i++)
            start.putRow(i, reverse(start.getRow(i)));

    }

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     *
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    @Override
    public INDArray rot(INDArray reverse) {
        INDArray ret = Nd4j.create(reverse.shape());
        if (reverse.isVector())
            return reverse(reverse);
        else {
            for (int i = 0; i < reverse.slices(); i++) {
                ret.putSlice(i, reverse(reverse.slice(i)));
            }
        }
        return ret.reshape(reverse.shape());
    }

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     *
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    
    @Override
    public INDArray reverse(INDArray reverse) {
        // FIXME: native method should be used instead
        INDArray rev = reverse.linearView();
        INDArray ret = Nd4j.create(rev.shape());
        int count = 0;
        for (long i = rev.length() - 1; i >= 0; i--) {
            ret.putScalar(count++, rev.getFloat(i));

        }

        return ret.reshape(reverse.shape());
    }

    /**
     * Array of evenly spaced values.
     *
     * @param begin the begin of the range
     * @param end   the end of the range
     * @return the range vector
     */
    @Override
    public INDArray arange(double begin, double end) {
        return Nd4j.create(ArrayUtil.toDoubles(ArrayUtil.range((int) begin, (int) end)));
    }

    /**
     * Create float
     *
     * @param real real component
     * @param imag imag component
     * @return
     */
    public abstract IComplexFloat createFloat(float real, float imag);

    /**
     * Create an instance of a complex double
     *
     * @param real the real component
     * @param imag the imaginary component
     * @return a new imaginary double with the specified real and imaginary components
     */
    public abstract IComplexDouble createDouble(double real, double imag);

    /**
     * Copy a to b
     *
     * @param a the origin matrix
     * @param b the destination matrix
     */
    @Override
    public void copy(INDArray a, INDArray b) {
        b.assign(a);
    }

    /**
     * Generates a random matrix between min and max
     *
     * @param shape the number of rows of the matrix
     * @param min   the minimum number
     * @param max   the maximum number
     * @param rng   the rng to use
     * @return a random matrix of the specified shape and range
     */
    @Override
    public INDArray rand(int[] shape, float min, float max, org.nd4j.linalg.api.rng.Random rng) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return Nd4j.getDistributions().createUniform(min, max).sample(shape);
    }

    @Override
    public INDArray rand(long[] shape, float min, float max, org.nd4j.linalg.api.rng.Random rng) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new long[] {1, 1};
        }
        return Nd4j.getDistributions().createUniform(min, max).sample(shape);
    }

    /**
     * Generates a random matrix between min and max
     *
     * @param rows    the number of rows of the matrix
     * @param columns the number of columns in the matrix
     * @param min     the minimum number
     * @param max     the maximum number
     * @param rng     the rng to use
     * @return a random matrix of the specified shape and range
     */
    @Override
    public INDArray rand(long rows, long columns, float min, float max, org.nd4j.linalg.api.rng.Random rng) {
        return rand(new long[] {rows, columns}, min, max, rng);
    }

    /**
     * Merge the vectors and append a bias.
     * Each vector must be either row or column vectors.
     * An exception is thrown for inconsistency (mixed row and column vectors)
     *
     * @param vectors the vectors to merge
     * @return the merged ndarray appended with the bias
     */
    @Override
    public INDArray appendBias(INDArray... vectors) {
        int size = 0;
        for (INDArray vector : vectors) {
            size += vector.rows();
        }


        INDArray result = Nd4j.create(size + 1, vectors[0].columns());
        int index = 0;
        for (INDArray vector : vectors) {
            INDArray put = toFlattened(vector, Nd4j.ones(1));
            result.put(new INDArrayIndex[] {NDArrayIndex.interval(index, index + vector.rows() + 1),
                            NDArrayIndex.interval(0, vectors[0].columns())}, put);
            index += vector.rows();
        }

        return result;

    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arr the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public abstract IComplexNDArray createComplex(INDArray arr);

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param data the data to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public abstract IComplexNDArray createComplex(IComplexNumber[] data, int[] shape);

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arrs the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public abstract IComplexNDArray createComplex(List<IComplexNDArray> arrs, int[] shape);

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return the random ndarray with the specified shape
     */
    @Override
    public INDArray rand(long rows, long columns, org.nd4j.linalg.api.rng.Random r) {
        return rand(new long[] {rows, columns}, r);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param seed    the  seed to use
     * @return the random ndarray with the specified shape
     */
    @Override
    public INDArray rand(long rows, long columns, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return rand(new long[] {rows, columns}, Nd4j.getRandom());
    }

    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    @Override
    public INDArray rand(long rows, long columns) {
        return rand(new long[] {rows, columns}, System.currentTimeMillis());
    }

    /**
     * Create a random (uniform 0-1) NDArray with the specified shape and order
     * @param order      Order ('c' or 'f') of the output array
     * @param rows       Number of rows of the output array
     * @param columns    Number of columns of the output array
     */
    @Override
    public INDArray rand(char order, long rows, long columns) {
        return Nd4j.getRandom().nextDouble(order, new long[] {rows, columns});
    }

    /**
     * Random normal using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return
     */
    @Override
    public INDArray randn(long rows, long columns, org.nd4j.linalg.api.rng.Random r) {
        return randn(new long[] {rows, columns}, r);
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    @Override
    public INDArray randn(long rows, long columns) {
        return randn(new long[] {rows, columns}, System.currentTimeMillis());
    }

    /**
     * Generate a random normal N(0,1) with the specified order and shape
     * @param order   Order of the output array
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    @Override
    public INDArray randn(char order, long rows, long columns) {
        return Nd4j.getRandom().nextGaussian(order, new long[] {rows, columns});
    }

    /**
     * Random normal using the specified seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    @Override
    public INDArray randn(long rows, long columns, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return randn(new long[] {rows, columns}, Nd4j.getRandom());
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return the random ndarray with the specified shape
     */
    @Override
    public INDArray rand(int[] shape, Distribution r) {
        INDArray ret = r.sample(shape);
        return ret;
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return the random ndarray with the specified shape
     */
    @Override
    public INDArray rand(int[] shape, org.nd4j.linalg.api.rng.Random r) {
        INDArray ret = r.nextDouble(shape);
        return ret;
    }

    @Override
    public INDArray rand(long[] shape, org.nd4j.linalg.api.rng.Random r) {
        INDArray ret = r.nextDouble(shape);
        return ret;
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param seed  the  seed to use
     * @return the random ndarray with the specified shape
     */
    @Override
    public INDArray rand(int[] shape, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return rand(shape, Nd4j.getRandom());
    }

    @Override
    public INDArray rand(long[] shape, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return rand(shape, Nd4j.getRandom());
    }

    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    @Override
    public INDArray rand(int[] shape) {
        return rand(shape, System.currentTimeMillis());
    }

    @Override
    public INDArray rand(long[] shape) {
        return rand(shape, System.currentTimeMillis());
    }

    /**
     * Create a random ndarray with the given shape and order
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    @Override
    public INDArray rand(char order, int[] shape) {
        return Nd4j.getRandom().nextDouble(order, shape);
    }

    @Override
    public INDArray rand(char order, long[] shape) {
        return Nd4j.getRandom().nextDouble(order, shape);
    }

    /**
     * Random normal using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return
     */
    @Override
    public INDArray randn(int[] shape, org.nd4j.linalg.api.rng.Random r) {
        return r.nextGaussian(shape);
    }

    @Override
    public INDArray randn(long[] shape, org.nd4j.linalg.api.rng.Random r) {
        return r.nextGaussian(shape);
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    @Override
    public INDArray randn(char order, int[] shape) {
        return Nd4j.getRandom().nextGaussian(order, shape);
    }

    @Override
    public INDArray randn(char order, long[] shape) {
        return Nd4j.getRandom().nextGaussian(order, shape);
    }

    /**
     * Random normal N(0,1) with the specified shape and
     *
     * @param shape the shape of the ndarray
     * @return
     */
    @Override
    public INDArray randn(int[] shape) {
        return randn(shape, System.currentTimeMillis());
    }

    @Override
    public INDArray randn(long[] shape) {
        return randn(shape, System.currentTimeMillis());
    }

    /**
     * Random normal using the specified seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    @Override
    public INDArray randn(int[] shape, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return randn(shape, Nd4j.getRandom());
    }

    @Override
    public INDArray randn(long[] shape, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return randn(shape, Nd4j.getRandom());
    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(double[] data) {
        return create(data, new int[] {1, data.length});
    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(float[] data) {
        return create(data, new int[] {1, data.length});
    }

    /**
     * Creates an ndarray with the specified data
     *
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray createComplex(double[] data) {
        assert data.length
                        % 2 == 0 : "Length of data must be even. A complex ndarray is made up of pairs of real and imaginary components";
        return createComplex(data, new int[] {1, data.length / 2});
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(long columns) {
        return create(new long[] {1, columns});
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray createComplex(long columns) {
        // return createComplex(new long[] {1, columns});
        throw new UnsupportedOperationException();
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray zeros(long rows, long columns) {
        return zeros(new long[] {rows, columns});
    }

    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes
     *
     * @param source source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes indexes from source array
     * @return
     */
    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes, char order) {
        long vectorLength = source.shape()[sourceDimension];
        INDArray ret = Nd4j.createUninitialized(new long[] {indexes.length, vectorLength}, order);

        for (int cnt = 0; cnt < indexes.length; cnt++) {
            ret.putRow(cnt, source.tensorAlongDimension((int) indexes[cnt], sourceDimension));
        }

        return ret;
    }

    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes
     *
     * @param source          source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes         indexes from source array
     * @return
     */
    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes) {
        return pullRows(source, sourceDimension, indexes, Nd4j.order());
    }

    /**
     * Creates a matrix of zeros
     *
     * @param rows    te number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray complexZeros(long rows, long columns) {
        //return createComplex(new long[] {rows, columns});
        throw new UnsupportedOperationException();
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray zeros(long columns) {
        return zeros(new long[] {1, columns});
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray complexZeros(long columns) {
        //return createComplex(new int[] {1, columns});
        throw new UnsupportedOperationException();
    }

    /**
     * Creates an shape ndarray with the specified value
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return a complex ndarray of the specified size
     * and value
     */
    @Override
    public IComplexNDArray complexValueOf(int[] shape, IComplexNumber value) {
        IComplexNDArray ones = complexOnes(shape);
        ones.assign(Nd4j.scalar(value));
        return ones;
    }

    /**
     * Creates an 1 x num ndarray with the specified value
     *
     * @param num   the number of columns
     * @param value the value to assign
     * @return a complex ndarray of the specified size
     * and value
     */
    @Override
    public IComplexNDArray complexValueOf(int num, double value) {
        IComplexNDArray ones = complexOnes(num);
        ones.assign(Nd4j.createDouble(value, 0.0));
        return ones;
    }

    /**
     * Creates an shape ndarray with the specified value
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return a complex ndarray of the specified size
     * and value
     */
    @Override
    public IComplexNDArray complexValueOf(int[] shape, double value) {
        IComplexNDArray ones = complexOnes(shape);
        ones.assign(Nd4j.scalar(value));
        return ones;
    }

    /**
     * Creates an 1 x num ndarray with the specified value
     *
     * @param num   the number of columns
     * @param value the value to assign
     * @return a complex ndarray of the specified size
     * and value
     */
    @Override
    public IComplexNDArray complexValueOf(int num, IComplexNumber value) {
        IComplexNDArray ones = complexOnes(num);
        ones.assign(Nd4j.scalar(value));
        return ones;
    }

    @Override
    public IComplexNDArray createComplex(int[] shape, int[] complexStrides, long offset, char ordering) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return createComplex(Nd4j.createBuffer(ArrayUtil.prodLong(shape) * 2), shape, complexStrides, offset, ordering);
    }

    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return the created ndarray
     */
    @Override
    public INDArray valueArrayOf(int[] shape, double value) {
        INDArray ret = Nd4j.createUninitialized(shape, Nd4j.order());
        ret.assign(value);
        return ret;
    }

    @Override
    public INDArray valueArrayOf(long[] shape, double value) {
        INDArray ret = Nd4j.createUninitialized(shape, Nd4j.order());
        ret.assign(value);
        return ret;
    }

    @Override
    public INDArray create(int[] shape, int[] stride, long offset, char ordering) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return create(Nd4j.createBuffer(ArrayUtil.prodLong(shape)), shape, stride, offset, ordering);
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param value   the value to assign
     * @return the created ndarray
     */
    @Override
    public INDArray valueArrayOf(long rows, long columns, double value) {
        INDArray create = createUninitialized(new long[] {rows, columns}, Nd4j.order());
        create.assign(value);
        return create;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray ones(long rows, long columns) {
        return ones(new long[] {rows, columns});
    }

    /**
     * Creates an ndarray
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray complexOnes(long rows, long columns) {
        //return complexOnes(new long[] {rows, columns});
        throw new UnsupportedOperationException();
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray ones(long columns) {
        return ones(new long[] {1, columns});
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray complexOnes(long columns) {
        /*
        IComplexNDArray base = createComplex(new long[] {1, columns});
        base.assign(1);
        return base;
        */
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray create(float[] data, int[] shape, char ordering) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return create(Nd4j.createBuffer(data), shape, Nd4j.getStrides(shape, ordering), 0, ordering);
    }

    /**
     * concatenate ndarrays along a dimension
     *
     * @param dimension the dimension to concatenate along
     * @param toConcat  the ndarrays to concatenate
     * @return the concatenate ndarrays
     */
    @Override
    public INDArray concat(int dimension, INDArray... toConcat) {
        if (toConcat.length == 1)
            return toConcat[0];
        int sumAlongDim = 0;
        boolean allC = toConcat[0].ordering() == 'c';

        long[] outputShape = ArrayUtil.copy(toConcat[0].shape());
        outputShape[dimension] = sumAlongDim;

        for (int i = 0; i < toConcat.length; i++) {
            sumAlongDim += toConcat[i].size(dimension);
            allC = allC && toConcat[i].ordering() == 'c';
            for (int j = 0; j < toConcat[i].rank(); j++) {
                if (j != dimension && toConcat[i].size(j) != outputShape[j] && !toConcat[i].isVector()) {
                    throw new IllegalArgumentException(
                                    "Illegal concatenation at array " + i + " and shape element " + j);
                }
            }
        }



        long[] sortedStrides = Nd4j.getStrides(outputShape);

        INDArray ret = Nd4j.create(outputShape, sortedStrides);
        allC &= (ret.ordering() == 'c');

        if (toConcat[0].isScalar()) {
            INDArray retLinear = ret.linearView();
            for (int i = 0; i < retLinear.length(); i++)
                retLinear.putScalar(i, toConcat[i].getDouble(0));
            return ret;
        }



        if (dimension == 0 && allC) {
            int currBuffer = 0;
            int currBufferOffset = 0;
            for (int i = 0; i < ret.length(); i++) {
                ret.data().put(i, toConcat[currBuffer].data()
                                .getDouble(toConcat[currBuffer].offset() + currBufferOffset++));
                if (currBufferOffset >= toConcat[currBuffer].length()) {
                    currBuffer++;
                    currBufferOffset = 0;
                }
            }

            return ret;
        }

        int arrOffset = 0;

        // FIXME: int cast

        INDArray[] retAlongDimensionArrays = new INDArray[(int) ret.tensorssAlongDimension(dimension)];
        for (int i = 0; i < retAlongDimensionArrays.length; i++)
            retAlongDimensionArrays[i] = ret.tensorAlongDimension(i, dimension);

        for (INDArray arr : toConcat) {
            long arrTensorLength = -1;

            if (arr.tensorssAlongDimension(dimension) != ret.tensorssAlongDimension(dimension))
                throw new IllegalStateException("Illegal concatenate. Tensors along dimension must be same length.");


            for (int i = 0; i < arr.tensorssAlongDimension(dimension); i++) {
                INDArray retLinear = retAlongDimensionArrays[i];
                INDArray arrTensor = arr.tensorAlongDimension(i, dimension);

                arrTensorLength = arrTensor.length();
                for (int j = 0; j < arrTensor.length(); j++) {
                    int idx = j + arrOffset;
                    retLinear.putScalar(idx, arrTensor.getDouble(j));
                }
            }

            //bump the sliding window
            arrOffset += arrTensorLength;

        }

        return ret;

    }

    /**
     * concatenate ndarrays along a dimension
     *
     * @param dimension the dimension to concatenate along
     * @param toConcat  the ndarrays to concatenate
     * @return the concatenate ndarrays
     */
    @Override
    public IComplexNDArray concat(int dimension, IComplexNDArray... toConcat) {
        /*
        if (toConcat.length == 1)
            return toConcat[0];
        validateConcat(dimension, toConcat);
        int sumAlongDim = 0;
        for (int i = 0; i < toConcat.length; i++)
            sumAlongDim += toConcat[i].shape()[dimension];


        long[] outputShape = ArrayUtil.copy(toConcat[0].shape());

        outputShape[dimension] = sumAlongDim;


        IComplexNDArray ret = Nd4j.createComplex(outputShape);
        IComplexNDArray linear = ret.linearView();
        int count = 0;
        for (int i = 0; i < toConcat.length; i++) {
            IComplexNDArray flattened = toConcat[i].linearView();

            for (int j = 0; j < flattened.length(); j++) {
                linear.putScalar(count++, flattened.getComplex(j));
            }
        }


        return ret;

        */
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNDArray complexFlatten(IComplexNDArray[] flatten) {
        int length = 0;
        for (IComplexNDArray m : flatten)
            length += m.length();
        IComplexNDArray ret = Nd4j.createComplex(length);
        int linearIndex = 0;
        for (IComplexNDArray d : flatten) {
            IComplexNDArray flattened = d.linearView();
            for (int i = 0; i < d.length(); i++) {
                ret.putScalar(linearIndex++, flattened.getComplex(i));
            }
        }

        return ret;

    }

    @Override
    public IComplexNDArray complexFlatten(List<IComplexNDArray> flatten) {
        int length = 0;
        for (IComplexNDArray m : flatten)
            length += m.length();
        IComplexNDArray ret = Nd4j.createComplex(length);
        int linearIndex = 0;
        for (IComplexNDArray d : flatten) {
            IComplexNDArray flattened = d.linearView();
            for (int i = 0; i < d.length(); i++) {
                ret.putScalar(linearIndex++, flattened.getComplex(i));
            }
        }

        return ret;

    }

    /**
     * Concatenates two matrices horizontally.
     * Matrices must have identical
     * numbers of rows.
     *
     * @param arrs
     */
    public INDArray hstack(INDArray... arrs) {
        return Nd4j.concat(1, arrs);
    }

    /**
     * Concatenates two matrices vertically. Matrices must have identical
     * numbers of columns.
     *
     * @param arrs
     */
    @Override
    public INDArray vstack(final INDArray... arrs) {
        return Nd4j.concat(0, arrs);

    }


    /**
     * Create an ndarray of zeros
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    @Override
    public INDArray zeros(int[] shape) {
        INDArray ret = create(shape);
        return ret;
    }

    @Override
    public INDArray zeros(long[] shape) {
        INDArray ret = create(shape);
        return ret;
    }

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    @Override
    public IComplexNDArray complexZeros(int[] shape) {
        IComplexNDArray ret = createComplex(shape);
        return ret;

    }


    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    @Override
    public INDArray ones(int[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }

        INDArray ret = create(shape);
        ret.assign(1);
        return ret;
    }

    @Override
    public INDArray ones(long[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape

        INDArray ret = create(shape);
        ret.assign(1);
        return ret;
    }

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    @Override
    public IComplexNDArray complexOnes(int[] shape) {
        IComplexNDArray ret = createComplex(shape);
        ret.assign(1);
        return ret;

    }


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
    @Override
    public IComplexNDArray createComplex(float[] data, long rows, long columns, int[] stride, long offset) {
        //return createComplex(data, new int[] {rows, columns}, stride, offset);
        throw new UnsupportedOperationException();
    }


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
    @Override
    public INDArray create(float[] data, long rows, long columns, int[] stride, long offset) {
        return create(data, new long[] {rows, columns}, ArrayUtil.toLongArray(stride), offset);
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
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, long offset) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return createComplex(Nd4j.createBuffer(data), shape, stride, offset, Nd4j.order());
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public abstract INDArray create(float[] data, int[] shape, int[] stride, long offset);


    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(double[] data, int[] shape) {
        return create(data, shape, Nd4j.getStrides(shape), 0);
    }


    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(float[] data, int[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return create(data, shape, Nd4j.getStrides(shape), 0);
    }

    @Override
    public INDArray create(float[] data, long[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new long[] {1, 1};
        }
        return create(data, shape, Nd4j.getStrides(shape), 0);
    }

    @Override
    public INDArray create(double[] data, long[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new long[] {1, 1};
        }
        return create(data, shape, Nd4j.getStrides(shape), 0);
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return createComplex(data, shape, Nd4j.getComplexStrides(shape), 0);
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return createComplex(data, shape, Nd4j.getComplexStrides(shape), 0);
    }


    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return createComplex(data, shape, stride, 0);
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride) {
        return createComplex(data, shape, stride, 0);
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(double[] data, long rows, long columns, int[] stride, long offset) {
        //return createComplex(data, new int[] {rows, columns}, stride, offset);
        throw new UnsupportedOperationException();
    }


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
    @Override
    public INDArray create(double[] data, long rows, long columns, int[] stride, long offset) {
        return create(data, new long[] {rows, columns}, ArrayUtil.toLongArray(stride), offset);
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public abstract IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, long offset);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public abstract INDArray create(double[] data, int[] shape, int[] stride, long offset);

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public abstract INDArray create(List<INDArray> list, int[] shape);


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(long rows, long columns, int[] stride, long offset) {
        /*
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return createComplex(new double[rows * columns * 2], new int[] {rows, columns}, stride, offset);
        else if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.HALF)
            return createComplex(new float[rows * columns * 2], new int[] {rows, columns}, stride, offset);
        else if (Nd4j.dataType() == DataBuffer.Type.INT)
            return createComplex(new int[rows * columns * 2], new int[] {rows, columns}, stride, offset);

        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());
        */

        throw new UnsupportedOperationException();
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(long rows, long columns, int[] stride, long offset) {
        /*
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return create(new double[rows * columns], new int[] {rows, columns}, stride, offset);
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.HALF)
            return create(new float[rows * columns], new int[] {rows, columns}, stride, offset);
        if (Nd4j.dataType() == DataBuffer.Type.INT)
            return create(new int[rows * columns], new int[] {rows, columns}, stride, offset);
        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());
        */

        throw new UnsupportedOperationException();
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public IComplexNDArray createComplex(int[] shape, int[] stride, long offset) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return createComplex(new double[ArrayUtil.prod(shape) * 2], shape, stride, offset);
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.HALF)
            return createComplex(new float[ArrayUtil.prod(shape) * 2], shape, stride, offset);
        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());

    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(int[] shape, int[] stride, long offset) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        DataBuffer buffer = Nd4j.createBuffer(ArrayUtil.prodLong(shape));
        return create(buffer, shape, stride, offset);
    }

    @Override
    public INDArray create(long[] shape, long[] stride, long offset) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new long[] {1, 1};
        }

        DataBuffer buffer = Nd4j.createBuffer(ArrayUtil.prodLong(shape));
        return create(buffer, shape, stride, offset);
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(long rows, long columns, int[] stride) {
        //return createComplex(new int[] {rows, columns}, stride);
        throw new UnsupportedOperationException();
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(long rows, long columns, int[] stride) {
        return create(new long[] {rows, columns}, ArrayUtil.toLongArray(stride));
    }

    @Override
    public INDArray create(long[] shape, long[] stride) {
        return create(shape, stride, 0, Nd4j.order());
    }

    @Override
    public INDArray create(long[] shape, long[] stride, long offset, char ordering) {
        if (shape.length == 1 && shape[0] == 0) {
            shape = new long[] {1, 1};
        }
        return create(Nd4j.createBuffer(ArrayUtil.prodLong(shape)), shape, stride, offset, ordering);
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(int[] shape, int[] stride) {
        return createComplex(shape, stride, 0);
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(int[] shape, int[] stride) {
        return create(shape, stride, 0);
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(long rows, long columns) {
        /*
        return createComplex(new int[] {rows, columns});
         */
        throw new UnsupportedOperationException();
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(long rows, long columns) {
        return create(new long[] {rows, columns});
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(int[] shape) {
        return createComplex(shape, Nd4j.getComplexStrides(shape), 0);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(long[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape

        return create(shape, Nd4j.getStrides(shape), 0L);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(int[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return create(shape, Nd4j.getStrides(shape), 0);
    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray scalar(Number value, long offset) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return scalar(value.doubleValue(), offset);
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.HALF)
            return scalar(value.floatValue(), offset);
        if (Nd4j.dataType() == DataBuffer.Type.INT)
            return scalar(value.intValue(), offset);
        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());
    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray complexScalar(Number value, long offset) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return scalar(createDouble(value.doubleValue(), 0), offset);
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.INT
                        || Nd4j.dataType() == DataBuffer.Type.HALF)
            return scalar(createFloat(value.floatValue(), 0), offset);

        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());
    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray complexScalar(Number value) {
        return complexScalar(value, 0);
    }


    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(float value, long offset) {
        return create(new float[] {value}, new int[] {1, 1}, new int[] {1, 1}, offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(double value, long offset) {
        return create(new double[] {value}, new int[] {1, 1}, new int[] {1, 1}, offset);
    }

    @Override
    public INDArray trueScalar(Number value) {
        val dtype = Nd4j.dataType();
        switch (dtype) {
            case DOUBLE:
                return create(new double[] {value.doubleValue()}, new int[] {}, new int[] {}, 0);
            case FLOAT:
                return create(new float[] {value.floatValue()}, new int[] {}, new int[] {}, 0);
            case HALF:
                return create(new float[] {value.floatValue()}, new int[] {}, new int[] {}, 0);
            default:
                throw new UnsupportedOperationException("Unsupported data type: [" + dtype + "]");

        }
    }

    public INDArray trueVector(float[] data) {
        return create(data, new int[] {data.length}, new int[]{1}, 0);
    }

    public INDArray trueVector(double[] data) {
        return create(data, new int[] {data.length}, new int[]{1}, 0);
    }



    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(int value, long offset) {
        return create(new int[] {value}, new int[] {1, 1}, new int[] {1, 1}, offset);
    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    @Override
    public INDArray scalar(Number value) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return scalar(value.doubleValue(), 0);
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.HALF)
            return scalar(value.floatValue(), 0);
        if (Nd4j.dataType() == DataBuffer.Type.INT)
            return scalar(value.intValue(), 0);
        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(float value) {
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.HALF)
            return create(new float[] {value}, new int[] {1, 1}, new int[] {1, 1}, 0);
        else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return scalar((double) value);
        else
            return scalar((int) value);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(double value) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return create(new double[] {value}, new int[] {1, 1}, new int[] {1, 1}, 0);
        else
            return scalar((float) value);
    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray scalar(IComplexNumber value, long offset) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return scalar(value.asDouble(), offset);
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.HALF)
            return scalar(value.asFloat(), offset);
        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    @Override
    public IComplexNDArray scalar(IComplexFloat value) {
        return createComplex(new float[] {value.realComponent(), value.imaginaryComponent()}, new int[] {1},
                        new int[] {1}, 0);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    @Override
    public IComplexNDArray scalar(IComplexDouble value) {
        return createComplex(new double[] {value.realComponent(), value.imaginaryComponent()}, new int[] {1},
                        new int[] {1}, 0);

    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray scalar(IComplexNumber value) {
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE)
            return scalar(value.asDouble(), 0);
        if (Nd4j.dataType() == DataBuffer.Type.FLOAT || Nd4j.dataType() == DataBuffer.Type.HALF)
            return scalar(value.asFloat(), 0);
        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public IComplexNDArray scalar(IComplexFloat value, long offset) {
        return createComplex(new float[] {value.realComponent(), value.imaginaryComponent()}, new int[] {1},
                        new int[] {1}, offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public IComplexNDArray scalar(IComplexDouble value, long offset) {
        return createComplex(new double[] {value.realComponent(), value.imaginaryComponent()}, new int[] {1},
                        new int[] {1}, offset);

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
    public abstract IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, long offset, char ordering);

    /**
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, long offset, char ordering) {
        return createComplex(Nd4j.createBuffer(data), shape, offset, ordering);
    }


    /**
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, long offset) {
        return createComplex(Nd4j.createBuffer(data), shape, offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset) {
        return create(Nd4j.createBuffer(data), shape, offset);
    }

    @Override
    public INDArray create(float[] data, char order) {
        int[] shape = new int[] {1, data.length};
        return create(Nd4j.createBuffer(data), shape, Nd4j.getStrides(shape, order), order, 0);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, char order, long offset) {
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);
    }


    @Override
    public INDArray create(double[] data, char order) {
        return create(data, new int[] {1, data.length}, Nd4j.getStrides(new int[] {1, data.length}, order), order, 0);
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, char order, long offset) {
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);

    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, int[] stride, char order, long offset) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return create(buffer, shape, stride, offset, order);
    }

    @Override
    public INDArray create(int[] data, int[] shape, int[] stride, char order, long offset) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        }
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);
    }
}
