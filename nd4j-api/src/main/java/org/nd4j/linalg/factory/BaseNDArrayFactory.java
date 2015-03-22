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

package org.nd4j.linalg.factory;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Base NDArrayFactory class.
 * <p/>
 * Allows specification or data type and row (c) or column(fortran) major order
 *
 * @author Adam Gibson
 */
public abstract class BaseNDArrayFactory implements NDArrayFactory {


    protected int dtype;
    protected char order;

    public BaseNDArrayFactory() {
    }

    /**
     * @param dtype the data type
     * @param order the ordering
     */
    protected BaseNDArrayFactory(int dtype, Character order) {
        this.dtype = dtype;
        if (Character.toLowerCase(order) != 'c' && Character.toLowerCase(order) != 'f')
            throw new IllegalArgumentException("Order must either be c or f");

        this.order = order;
    }

    /**
     * @param dtype the data type
     * @param order the ordering
     */
    protected BaseNDArrayFactory(Integer dtype, Character order) {
        this.dtype = dtype;
        if (Character.toLowerCase(order) != 'c' && Character.toLowerCase(order) != 'f')
            throw new IllegalArgumentException("Order must either be c or f");

        this.order = order;
    }

    /**
     * @param dtype the data type
     * @param order the ordering
     */
    protected BaseNDArrayFactory(int dtype, char order) {
        this.dtype = dtype;
        if (Character.toLowerCase(order) != 'c' && Character.toLowerCase(order) != 'f')
            throw new IllegalArgumentException("Order must either be c or f");

        this.order = order;
    }

    //input arrays must have same number of dimensions
    protected static void validateConcat(int dimension, INDArray... arrs) {
        if(arrs[0].isScalar()) {
            for(int i = 1; i < arrs.length; i++)
                if(!arrs[i].isScalar())
                    throw new IllegalArgumentException("All arrays must have same dimensions");
        }
        else {
            int dims = arrs[0].shape().length;
            int[] shape = ArrayUtil.removeIndex(arrs[0].shape(), dimension);
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
    public INDArray rand(int[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        return Nd4j.getDistributions().createUniform(min, max).sample(shape);
    }

    @Override
    public INDArray rand(int rows, int columns, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        return rand(new int[]{rows, columns}, min, max, rng);
    }

    /**
     * Sets the data type
     *
     * @param dtype
     */
    @Override
    public void setDType(int dtype) {
        assert dtype == DataBuffer.DOUBLE || dtype == DataBuffer.FLOAT || dtype == DataBuffer.INT : "Invalid type passed, must be float or double";
        this.dtype = dtype;
    }

    @Override
    public INDArray create(int[] shape, int dataType) {
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
     * Returns the data type for this ndarray
     *
     * @return the data type for this ndarray
     */
    @Override
    public int dtype() {
        return dtype;
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

        INDArray ret = create(data);
        return ret;
    }

    @Override
    public IComplexNDArray createComplex(int[] ints, int[] ints1, int[] stride, int offset) {
        return createComplex(Nd4j.createBuffer(ints), ints1, stride, offset);
    }

    @Override
    public INDArray create(int[] ints, int[] ints1, int[] stride, int offset) {
        return create(Nd4j.createBuffer(ints), ints1, stride, offset);
    }

    @Override
    public INDArray create(int rows, int columns, char ordering) {
        return create(new int[]{rows, columns}, ordering);
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

        INDArray ret = Nd4j.create(length);
        DataBuffer[] buffers = new DataBuffer[matrices.size()];
        int[] strides = new int[matrices.size()];
        int[] offsets = new int[matrices.size()];
        int count = 0;
        for (INDArray matrix : matrices) {
            buffers[count] = matrix.data();
            strides[count] = matrix.majorStride();
            offsets[count] = matrix.offset();
            count++;
        }


        ret.data().assign(offsets, strides, buffers);
        return ret;

    }

    @Override
    public INDArray toFlattened(int length, Iterator<? extends INDArray>... matrices) {
        INDArray ret = Nd4j.create(length);
        int linearIndex = 0;

        for (Iterator<? extends INDArray> iter1 : matrices) {
            while (iter1.hasNext()) {
                INDArray d = iter1.next();
                INDArray flattened = d.linearView();
                for (int i = 0; i < d.length(); i++) {
                    ret.putScalar(linearIndex++, flattened.getDouble(i));
                }
            }

        }

        return ret;
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
            if (!d.isVector())
                d = Nd4j.create(d.data(), new int[]{1, d.length()}, d.offset());
            for (int i = 0; i < d.length(); i++) {
                ret.putScalar(linearIndex++, d.getFloat(i));
            }
        }

        return ret;
    }

    /**
     * Create the identity ndarray
     *
     * @param n the number for the identity
     * @return
     */
    @Override
    public INDArray eye(int n) {
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
        INDArray rev = reverse.linearView();
        INDArray ret = Nd4j.create(rev.shape());
        int count = 0;
        for (int i = rev.length() - 1; i >= 0; i--) {
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
        a = a.linearView();
        b = b.linearView();
        for (int i = 0; i < a.length(); i++) {
            b.put(i, a.getScalar(i));
        }
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
    public INDArray rand(int rows, int columns, float min, float max, org.nd4j.linalg.api.rng.Random rng) {
        return rand(new int[]{rows, columns}, min, max, rng);
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
            result.put(new NDArrayIndex[]{NDArrayIndex.interval(index, index + vector.rows() + 1), NDArrayIndex.interval(0, vectors[0].columns())}, put);
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
    public INDArray rand(int rows, int columns, org.nd4j.linalg.api.rng.Random r) {
        return rand(new int[]{rows, columns}, r);
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
    public INDArray rand(int rows, int columns, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return rand(new int[]{rows, columns}, Nd4j.getRandom());
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
    public INDArray rand(int rows, int columns) {
        return randn(new int[]{rows, columns}, System.currentTimeMillis());
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
    public INDArray randn(int rows, int columns, org.nd4j.linalg.api.rng.Random r) {
        return randn(new int[]{rows, columns}, r);
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
    public INDArray randn(int rows, int columns) {
        return randn(new int[]{rows, columns}, System.currentTimeMillis());
    }

    /**
     * Random normal using the specified seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    @Override
    public INDArray randn(int rows, int columns, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return randn(new int[]{rows, columns}, Nd4j.getRandom());
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

    /**
     * Random normal using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return
     */
    @Override
    public INDArray randn(int[] shape, org.nd4j.linalg.api.rng.Random r) {
        return Nd4j.getDistributions().createNormal(0, 1).sample(shape);
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    @Override
    public INDArray randn(int[] shape) {
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

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(double[] data) {
        return create(data, new int[]{data.length});
    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(float[] data) {
        return create(data, new int[]{data.length});
    }

    /**
     * Creates an ndarray with the specified data
     *
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray createComplex(double[] data) {
        assert data.length % 2 == 0 : "Length of data must be even. A complex ndarray is made up of pairs of real and imaginary components";
        return createComplex(data, new int[]{data.length / 2});
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(int columns) {
        return create(new int[]{columns});
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray createComplex(int columns) {
        return createComplex(new int[]{columns});
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray zeros(int rows, int columns) {
        return zeros(new int[]{rows, columns});
    }

    /**
     * Creates a matrix of zeros
     *
     * @param rows    te number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray complexZeros(int rows, int columns) {
        return createComplex(new int[]{rows, columns});
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray zeros(int columns) {
        return zeros(new int[]{columns});
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray complexZeros(int columns) {
        return createComplex(new int[]{columns});
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
        INDArray create = create(shape);
        create.assign(value);
        return create;
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
    public INDArray valueArrayOf(int rows, int columns, double value) {
        INDArray create = create(rows, columns);
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
    public INDArray ones(int rows, int columns) {
        return ones(new int[]{rows, columns});
    }

    /**
     * Creates an ndarray
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public INDArray complexOnes(int rows, int columns) {
        return createComplex(new int[]{rows, columns});
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray ones(int columns) {
        return ones(new int[]{columns});
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    @Override
    public IComplexNDArray complexOnes(int columns) {
        IComplexNDArray base = createComplex(new int[]{columns});
        base.assign(1);
        return base;
    }

    @Override
    public INDArray create(float[] data, int[] shape, char ordering) {
        return create(Nd4j.createBuffer(data),shape,Nd4j.getStrides(shape),0,ordering);
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

        validateConcat(dimension, toConcat);

        if(toConcat[0].isScalar()) {
            int[] outputShape = dimension == 0 ? new int[]{toConcat.length,1} : new int[]{toConcat.length};
            INDArray ret = Nd4j.create(outputShape);
            for(int i = 0; i < ret.length(); i++) {
                ret.putScalar(i,toConcat[i].getDouble(0));
            }
            return ret;
        }

        else {
            int sumAlongDim = 0;
            for (int i = 0; i < toConcat.length; i++)
                sumAlongDim += toConcat[i].shape()[dimension];


            int[] outputShape = ArrayUtil.copy(toConcat[0].shape());

            outputShape[dimension] = sumAlongDim;


            INDArray ret = Nd4j.create(outputShape);
            INDArray linear = ret.linearView();
            int count = 0;
            for (int i = 0; i < toConcat.length; i++) {
                INDArray flattened = toConcat[i].linearView();

                for (int j = 0; j < flattened.length(); j++) {
                    linear.putScalar(count++, flattened.getFloat(j));
                }
            }


            return ret;
        }
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
        if (toConcat.length == 1)
            return toConcat[0];
        validateConcat(dimension, toConcat);
        int sumAlongDim = 0;
        for (int i = 0; i < toConcat.length; i++)
            sumAlongDim += toConcat[i].shape()[dimension];


        int[] outputShape = ArrayUtil.copy(toConcat[0].shape());

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

    }

    @Override
    public IComplexNDArray complexFlatten(IComplexNDArray[] flatten) {
        int length = 0;
        for (IComplexNDArray m : flatten) length += m.length();
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
        for (IComplexNDArray m : flatten) length += m.length();
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
     * Concatenates two matrices horizontally. Matrices must have identical
     * numbers of rows.
     *
     * @param arrs
     */
    public INDArray hstack(INDArray... arrs) {
        int cols = arrs[0].columns();
        int rows = arrs[0].rows();

        for (int i = 1; i < arrs.length; i++) {
            cols += arrs[i].columns();
            if (arrs[i].rows() != rows)
                throw new IllegalStateException("Illegal number of rows for array " + i);

        }


        final INDArray ret = Nd4j.create(rows, cols);
        final AtomicInteger i = new AtomicInteger(0);
        for (INDArray a : arrs) {
            a.iterateOverAllColumns(new SliceOp() {


                @Override
                public void operate(INDArray nd) {
                    for (int j = 0; j < nd.length(); j++) {
                        ret.putScalar(i.get(), nd.getDouble(j));

                    }
                    i.incrementAndGet();
                }
            });

        }


        return ret;
    }

    /**
     * Concatenates two matrices vertically. Matrices must have identical
     * numbers of columns.
     *
     * @param arrs
     */
    @Override
    public INDArray vstack(final INDArray... arrs) {
        int cols = arrs[0].columns();
        int rows = arrs[0].rows();

        for (int i = 1; i < arrs.length; i++) {
            rows += arrs[i].rows();
            if (arrs[i].columns() != cols)
                throw new IllegalStateException("Illegal number of rows for array " + i);

        }


        final INDArray ret = Nd4j.create(rows, cols);
        final AtomicInteger i = new AtomicInteger(0);


        for (INDArray arr : arrs) {
            arr.iterateOverAllRows(new SliceOp() {
                @Override
                public void operate(INDArray nd) {
                    ret.putRow(i.get(), nd);
                    i.incrementAndGet();
                }
            });
        }


        return ret;
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
    public IComplexNDArray createComplex(float[] data, int rows, int columns, int[] stride, int offset) {
        return createComplex(data, new int[]{rows, columns}, stride, offset);
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
    public INDArray create(float[] data, int rows, int columns, int[] stride, int offset) {
        return create(data, new int[]{rows, columns}, stride, offset);
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
    public abstract IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public abstract INDArray create(float[] data, int[] shape, int[] stride, int offset);


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
    public IComplexNDArray createComplex(double[] data, int rows, int columns, int[] stride, int offset) {
        return createComplex(data, new int[]{rows, columns}, stride, offset);
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
    public INDArray create(double[] data, int rows, int columns, int[] stride, int offset) {
        return create(data, new int[]{rows, columns}, stride, offset);
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public abstract IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset);


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public abstract INDArray create(double[] data, int[] shape, int[] stride, int offset);

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
    public IComplexNDArray createComplex(int rows, int columns, int[] stride, int offset) {
        if (dtype == DataBuffer.DOUBLE)
            return createComplex(new double[rows * columns * 2], new int[]{rows, columns}, stride, offset);
        if (dtype == DataBuffer.FLOAT)
            return createComplex(new float[rows * columns * 2], new int[]{rows, columns}, stride, offset);
        if (dtype == DataBuffer.INT)
            return createComplex(new int[rows * columns * 2], new int[]{rows, columns}, stride, offset);

        throw new IllegalStateException("Illegal data type " + dtype);
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
    public INDArray create(int rows, int columns, int[] stride, int offset) {
        if (dtype == DataBuffer.DOUBLE)
            return create(new double[rows * columns], new int[]{rows, columns}, stride, offset);
        if (dtype == DataBuffer.FLOAT)
            return create(new float[rows * columns], new int[]{rows, columns}, stride, offset);
        if (dtype == DataBuffer.INT)
            return create(new int[rows * columns], new int[]{rows, columns}, stride, offset);
        throw new IllegalStateException("Illegal data type " + dtype);
    }


    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public IComplexNDArray createComplex(int[] shape, int[] stride, int offset) {
        if (dtype == DataBuffer.DOUBLE)
            return createComplex(new double[ArrayUtil.prod(shape) * 2], shape, stride, offset);
        if (dtype == DataBuffer.FLOAT)
            return createComplex(new float[ArrayUtil.prod(shape) * 2], shape, stride, offset);
        throw new IllegalStateException("Illegal data type " + dtype);

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
    public INDArray create(int[] shape, int[] stride, int offset) {
        return create(Nd4j.createBuffer(shape), shape, stride, offset);

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
    public IComplexNDArray createComplex(int rows, int columns, int[] stride) {
        return createComplex(new int[]{rows, columns}, stride);
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
    public INDArray create(int rows, int columns, int[] stride) {
        return create(new int[]{rows, columns}, stride);
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
    public IComplexNDArray createComplex(int rows, int columns) {
        return createComplex(new int[]{rows, columns});
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(int rows, int columns) {
        return create(new int[]{rows, columns});
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
    public INDArray create(int[] shape) {
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
    public INDArray scalar(Number value, int offset) {
        if (dtype == DataBuffer.DOUBLE)
            return scalar(value.doubleValue(), offset);
        if (dtype == DataBuffer.FLOAT)
            return scalar(value.floatValue(), offset);
        if (dtype == DataBuffer.INT)
            return scalar(value.intValue(), offset);
        throw new IllegalStateException("Illegal data type " + dtype);
    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray complexScalar(Number value, int offset) {
        if (dtype == DataBuffer.DOUBLE)
            return scalar(createDouble(value.doubleValue(), 0), offset);
        if (dtype == DataBuffer.FLOAT || dtype == DataBuffer.INT)
            return scalar(createFloat(value.floatValue(), 0), offset);

        throw new IllegalStateException("Illegal data type " + dtype);
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
    public INDArray scalar(float value, int offset) {
        return create(new float[]{value}, new int[]{1}, new int[]{1}, offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(double value, int offset) {
        return create(new double[]{value}, new int[]{1}, new int[]{1}, offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(int value, int offset) {
        return create(new int[]{value}, new int[]{1}, new int[]{1}, offset);
    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    @Override
    public INDArray scalar(Number value) {
        if (dtype == DataBuffer.DOUBLE)
            return scalar(value.doubleValue(), 0);
        if (dtype == DataBuffer.FLOAT)
            return scalar(value.floatValue(), 0);
        if (dtype == DataBuffer.INT)
            return scalar(value.intValue(), 0);
        throw new IllegalStateException("Illegal data type " + dtype);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(float value) {
        if (dtype == DataBuffer.FLOAT)
            return create(new float[]{value}, new int[]{1}, new int[]{1}, 0);
        else if (dtype == DataBuffer.DOUBLE)
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
        if (dtype == DataBuffer.DOUBLE)
            return create(new double[]{value}, new int[]{1}, new int[]{1}, 0);
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
    public IComplexNDArray scalar(IComplexNumber value, int offset) {
        if (dtype == DataBuffer.DOUBLE)
            return scalar(value.asDouble(), offset);
        if (dtype == DataBuffer.FLOAT)
            return scalar(value.asFloat(), offset);
        throw new IllegalStateException("Illegal data type " + dtype);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    @Override
    public IComplexNDArray scalar(IComplexFloat value) {
        return createComplex(new float[]{value.realComponent(), value.imaginaryComponent()}, new int[]{1}, new int[]{1}, 0);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    @Override
    public IComplexNDArray scalar(IComplexDouble value) {
        return createComplex(new double[]{value.realComponent(), value.imaginaryComponent()}, new int[]{1}, new int[]{1}, 0);

    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    @Override
    public IComplexNDArray scalar(IComplexNumber value) {
        if (dtype == DataBuffer.DOUBLE)
            return scalar(value.asDouble(), 0);
        if (dtype == DataBuffer.FLOAT)
            return scalar(value.asFloat(), 0);
        throw new IllegalStateException("Illegal data type " + dtype);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public IComplexNDArray scalar(IComplexFloat value, int offset) {
        return createComplex(new float[]{value.realComponent(), value.imaginaryComponent()}, new int[]{1}, new int[]{1}, offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    @Override
    public IComplexNDArray scalar(IComplexDouble value, int offset) {
        return createComplex(new double[]{value.realComponent(), value.imaginaryComponent()}, new int[]{1}, new int[]{1}, offset);

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
    public abstract IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset, char ordering);

    /**
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset, char ordering) {
        return createComplex(Nd4j.createBuffer(data), shape, offset, ordering);
    }


    /**
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset) {
        return createComplex(Nd4j.createBuffer(data), shape, offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int offset) {
        return create(Nd4j.createBuffer(data), shape, offset);
    }

    @Override
    public INDArray create(float[] data, char order) {
        return create(Nd4j.createBuffer(data), new int[]{data.length}, ArrayUtil.calcStrides(new int[]{data.length}), order, 0);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, char order, int offset) {
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);
    }


    @Override
    public INDArray create(double[] data, char order) {
        return create(data, new int[]{data.length}, ArrayUtil.calcStrides(new int[]{data.length}), order, 0);
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, char order, int offset) {
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);

    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, int[] stride, char order, int offset) {
        return create(buffer, shape, stride, offset, order);
    }

    @Override
    public INDArray create(int[] data, int[] shape, int[] stride, char order, int offset) {
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);
    }
}
