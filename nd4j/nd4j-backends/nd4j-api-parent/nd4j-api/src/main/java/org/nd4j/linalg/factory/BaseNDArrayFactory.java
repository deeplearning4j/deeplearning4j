/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.factory;


import com.google.common.util.concurrent.AtomicDouble;
import lombok.val;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.blas.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.random.impl.Range;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base NDArrayFactory class.
 * <p/>
 * Allows specification or data opType and row (c) or column(fortran) major order
 *
 * @author Adam Gibson
 */
public abstract class BaseNDArrayFactory implements NDArrayFactory {

    // We don't really care about dtype field we'll use context instead
    // protected DataType dtype;
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
    protected BaseNDArrayFactory(DataType dtype, Character order) {
        // this.dtype = dtype;
        if (Character.toLowerCase(order) != 'c' && Character.toLowerCase(order) != 'f')
            throw new IllegalArgumentException("Order must either be c or f");

        this.order = Character.toLowerCase(order);
    }

    /**
     * @param dtype the data opType
     * @param order the ordering
     */
    protected BaseNDArrayFactory(DataType dtype, char order) {
        // this.dtype = dtype;
        if (Character.toLowerCase(order) != 'c' && Character.toLowerCase(order) != 'f')
            throw new IllegalArgumentException("Order must either be c or f");

        this.order = Character.toLowerCase(order);
    }

    /**
     * Sets the order. Primarily for testing purposes
     *
     * @param order
     */
    @Override
    public void setOrder(char order) {
        Preconditions.checkArgument(order == 'c' || order == 'f', "Order specified must be either c or f: got %s", String.valueOf(order));
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
    public void setDType(DataType dtype) {
        assert dtype == DataType.DOUBLE || dtype == DataType.FLOAT
                        || dtype == DataType.INT : "Invalid opType passed, must be float or double";
        // this.dtype = dtype;
    }

    @Override
    public INDArray create(int[] shape, DataType dataType, MemoryWorkspace workspace) {
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
    public DataType dtype() {
        return Nd4j.dataType();
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
        return toFlattened('c', matrices.toArray(new INDArray[matrices.size()]));
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
        Preconditions.checkArgument(curr.rank() == 3, "Argument 'curr' must be rank 3. Got input with rank: %s", curr.rank());
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
        return toFlattened(Nd4j.order(), Arrays.asList(matrices));
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
        INDArray rev = reverse.reshape(-1);
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
    public INDArray arange(double begin, double end, double step) {
        long length = (long)Math.floor((end-begin)/step);
        DynamicCustomOp op = new Range(begin, end, step, DataType.FLOAT);
        INDArray out = Nd4j.create(op.calculateOutputShape().get(0));
        op.setOutputArgument(0, out);
        Nd4j.exec(op);
        return out;
    }

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
        Preconditions.checkArgument(vectors != null && vectors.length > 0, "vectros must be not null and have at least one element");
        int size = 0;
        for (INDArray vector : vectors) {
            size += vector.rows();
            Preconditions.checkArgument(vectors[0].dataType() == vector.dataType(), "appendBias: all arrays must have same type");
        }


        INDArray result = Nd4j.create(vectors[0].dataType(), size + 1, vectors[0].columns());
        int index = 0;
        for (INDArray vector : vectors) {
            INDArray put = toFlattened(vector, Nd4j.ones(vector.dataType(), 1));
            result.put(new INDArrayIndex[] {NDArrayIndex.interval(index, index + vector.rows() + 1),
                            NDArrayIndex.interval(0, vectors[0].columns())}, put);
            index += vector.rows();
        }

        return result;
    }

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
        Shape.assertValidOrder(order);
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
        Shape.assertValidOrder(order);
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
        Shape.assertValidOrder(order);
        return Nd4j.getRandom().nextDouble(order, shape);
    }

    @Override
    public INDArray rand(char order, long[] shape) {
        Shape.assertValidOrder(order);
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
        Shape.assertValidOrder(order);
        return Nd4j.getRandom().nextGaussian(order, shape);
    }

    @Override
    public INDArray randn(char order, long[] shape) {
        Shape.assertValidOrder(order);
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
        return create(data, new long[] {data.length});
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray create(long columns) {
        return create(new long[] {columns});
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
        Shape.assertValidOrder(order);
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
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray zeros(long columns) {
        return zeros(new long[] {columns});
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
        Shape.assertValidOrder(ordering);
        //ensure shapes that wind up being scalar end up with the write shape
        long length = ArrayUtil.prodLong(shape);
        if(length == 0)
            return scalar(0.0);
        return create(Nd4j.createBuffer(length), shape, stride, offset, ordering);
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
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    @Override
    public INDArray ones(long columns) {
        return ones(new long[] {columns});
    }

    @Override
    public INDArray create(float[] data, int[] shape, char ordering) {
        Shape.assertValidOrder(ordering);
        long length = ArrayUtil.prodLong(shape);
        if(length == 0)
            return scalar(0.0);
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
            INDArray retLinear = ret.reshape(-1);
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

        INDArray[] retAlongDimensionArrays = new INDArray[(int) ret.tensorsAlongDimension(dimension)];
        for (int i = 0; i < retAlongDimensionArrays.length; i++)
            retAlongDimensionArrays[i] = ret.tensorAlongDimension(i, dimension);

        for (INDArray arr : toConcat) {
            long arrTensorLength = -1;

            if (arr.tensorsAlongDimension(dimension) != ret.tensorsAlongDimension(dimension))
                throw new IllegalStateException("Illegal concatenate. Tensors along dimension must be same length.");


            for (int i = 0; i < arr.tensorsAlongDimension(dimension); i++) {
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
    public INDArray ones(int[] shape) {
        INDArray ret = createUninitialized(shape, Nd4j.order());
        ret.assign(1);
        return ret;
    }

    @Override
    public INDArray ones(long[] shape) {
        //ensure shapes that wind up being scalar end up with the write shape
        INDArray ret = createUninitialized(shape, Nd4j.order());
        ret.assign(1);
        return ret;
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
        return create(data, shape, Nd4j.getStrides(shape), 0);
    }

    @Override
    public INDArray create(float[] data, long[] shape) {
        return create(data, shape, Nd4j.getStrides(shape), 0);
    }

    @Override
    public INDArray create(double[] data, long[] shape) {
        return create(data, shape, Nd4j.getStrides(shape), 0);
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
        if (Nd4j.dataType() == DataType.DOUBLE)
            return create(new double[rows * columns], new int[] {rows, columns}, stride, offset);
        if (Nd4j.dataType() == DataType.FLOAT || Nd4j.dataType() == DataType.HALF)
            return create(new float[rows * columns], new int[] {rows, columns}, stride, offset);
        if (Nd4j.dataType() == DataType.INT)
            return create(new int[rows * columns], new int[] {rows, columns}, stride, offset);
        throw new IllegalStateException("Illegal data opType " + Nd4j.dataType());
        */

        throw new UnsupportedOperationException();
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
        DataBuffer buffer = Nd4j.createBuffer(ArrayUtil.prodLong(shape));
        return create(buffer, shape, stride, offset);
    }

    @Override
    public INDArray create(long[] shape, long[] stride, long offset) {
        DataBuffer buffer = Nd4j.createBuffer(ArrayUtil.prodLong(shape));
        return create(buffer, shape, stride, offset);
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
        Shape.assertValidOrder(ordering);
        return create(Nd4j.createBuffer(ArrayUtil.prodLong(shape)), shape, stride, offset, ordering);
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
        if (Nd4j.dataType() == DataType.DOUBLE)
            return scalar(value.doubleValue(), offset);
        if (Nd4j.dataType() == DataType.FLOAT || Nd4j.dataType() == DataType.HALF)
            return scalar(value.floatValue(), offset);
        if (Nd4j.dataType() == DataType.INT)
            return scalar(value.intValue(), offset);
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
    public INDArray scalar(float value, long offset) {
        return create(new float[] {value}, new int[0], new int[0], offset);
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
        return create(new double[] {value}, new int[0], new int[0], offset);
    }

    public INDArray trueVector(boolean[] data) {
        return create(data, new long[] {data.length}, new long[]{1}, DataType.BOOL, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    public INDArray trueVector(byte[] data) {
        return create(data, new long[] {data.length}, new long[]{1}, DataType.BYTE, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    public INDArray trueVector(short[] data) {
        return create(data, new long[] {data.length}, new long[]{1}, DataType.SHORT, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    public INDArray trueVector(int[] data) {
        return create(data, new long[] {data.length}, new long[]{1}, DataType.INT, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    public INDArray trueVector(long[] data) {
        return create(data, new long[] {data.length}, new long[]{1}, DataType.LONG, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    public INDArray trueVector(float[] data) {
        return create(data, new long[] {data.length}, new long[]{1}, DataType.FLOAT, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    public INDArray trueVector(double[] data) {
        return create(data, new long[] {data.length}, new long[]{1}, DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
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
        return create(new int[] {value}, new long[0], new long[0], DataType.INT, Nd4j.getMemoryManager().getCurrentWorkspace());
    }


    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    @Override
    public INDArray scalar(Number value) {
        if (value instanceof Double)
            return create(new double[]{value.doubleValue()}, new long[0], new long[0], DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (value instanceof Float)
            return create(new float[]{value.floatValue()}, new long[0], new long[0], DataType.FLOAT, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (value instanceof Long)
            return create(new long[]{value.longValue()}, new long[0], new long[0], DataType.LONG, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (value instanceof Integer)
            return create(new int[]{value.intValue()}, new long[0], new long[0], DataType.INT, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (value instanceof Short)
            return create(new short[]{value.shortValue()}, new long[0], new long[0], DataType.SHORT, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (value instanceof Byte)
            return create(new byte[]{value.byteValue()}, new long[0], new long[0], DataType.BYTE, Nd4j.getMemoryManager().getCurrentWorkspace());
        throw new IllegalStateException("Unknown instance of Number: [" + value.getClass().getCanonicalName() + "]");
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(float value) {
            return create(new float[] {value}, new long[0], new long[0], DataType.FLOAT, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    @Override
    public INDArray scalar(double value) {
        return create(new double[] {value}, new long[0], new long[0], DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset) {
        return create(Nd4j.createBuffer(data), shape, offset);
    }

    public abstract INDArray create(float[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace);

    @Override
    public INDArray create(float[] data, char order) {
        val shape = new long[] {data.length};
        val stride = Nd4j.getStrides(shape, order);
        return create(data, shape, stride, order, DataType.FLOAT);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, char order, long offset) {
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);
    }


    @Override
    public INDArray create(double[] data, char order) {
        Shape.assertValidOrder(order);
        return create(data, new long[] {data.length}, new long[]{1}, DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, char order, long offset) {
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);

    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, int[] stride, char order, long offset) {
        Shape.assertValidOrder(order);
        return create(buffer, shape, stride, offset, order);
    }

    @Override
    public INDArray create(int[] data, int[] shape, int[] stride, char order, long offset) {
        Shape.assertValidOrder(order);
        return create(Nd4j.createBuffer(data), shape, stride, order, offset);
    }
}
