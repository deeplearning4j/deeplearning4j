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

package org.nd4j.linalg.api.ndarray;


import com.google.common.base.Function;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.dimensionfunctions.DimensionFunctions;
import org.nd4j.linalg.api.instrumentation.Instrumentation;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarNotEquals;
import org.nd4j.linalg.api.ops.impl.transforms.Negative;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.nd4j.linalg.util.Shape;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.nd4j.linalg.util.ArrayUtil.calcStrides;
import static org.nd4j.linalg.util.ArrayUtil.calcStridesFortran;


/**
 * NDArray: (think numpy)
 * <p/>
 * A few things of note.
 * <p/>
 * An NDArray can have any number of dimensions.
 * <p/>
 * An NDArray is accessed via strides.
 * <p/>
 * Strides are how to index over
 * a contiguous block of data.
 * <p/>
 * This block of data has 2 orders(as of right now):
 * fortran and c
 *
 * @author Adam Gibson
 */
public abstract class BaseNDArray implements INDArray {


    protected int[] shape;
    protected int[] stride;
    protected int offset = 0;
    protected char ordering;
    protected DataBuffer data;
    protected int rows, columns;
    protected int length;
    protected INDArray linearView;
    protected String id = UUID.randomUUID().toString();
    protected boolean cleanedUp = false;
    protected transient WeakReference<INDArray> ref;

    public BaseNDArray() {
    }


    public BaseNDArray(DataBuffer buffer) {
        this.data = buffer;
        init(new int[]{1, buffer.length()});
    }

    public BaseNDArray(DataBuffer buffer, int[] shape, int[] stride, int offset, char ordering) {
        this.data = buffer;
        if (ArrayUtil.prod(shape) > buffer.length())
            throw new IllegalArgumentException("Shape must be <= buffer length");
        this.stride = stride;
        this.offset = offset;
        this.ordering = ordering;
        init(shape);

    }

    public BaseNDArray(double[][] data) {
        this(Nd4j.createBuffer(ArrayUtil.flatten(data)), new int[]{data.length, data[0].length});

        for (int r = 0; r < rows; r++) {
            assert (data[r].length == columns);
        }

        this.data = Nd4j.createBuffer(length);


        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                putScalar(new int[]{r, c}, data[r][c]);
            }
        }
    }


    /**
     * Create with the specified shape and buffer
     *
     * @param shape  the shape
     * @param buffer the buffer
     */
    public BaseNDArray(int[] shape, DataBuffer buffer) {
        this.data = buffer;
        init(shape);
    }

    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data  the data to use
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape, char ordering) {
        this(data, shape, 0, ordering);
    }

    /**
     * @param data     the data to use
     * @param shape    the shape of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape, int offset, char ordering) {
        this(data, shape, ordering == NDArrayFactory.C ? calcStrides(shape) : calcStridesFortran(shape), offset);

    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride, int offset, char ordering) {
        this(Nd4j.createBuffer(ArrayUtil.prod(shape)), shape, stride, offset, ordering);
    }


    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride, char ordering) {
        this(shape, stride, 0, ordering);
    }

    public BaseNDArray(int[] shape, int offset, char ordering) {
        this(shape, Nd4j.getStrides(shape, ordering), offset, ordering);
    }


    public BaseNDArray(int[] shape) {
        this(shape, 0, Nd4j.order());
    }


    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public BaseNDArray(int newRows, int newColumns, char ordering) {
        this.ordering = ordering;
        this.data = Nd4j.createBuffer(newRows * newColumns);
        init(new int[]{newRows, newColumns});
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, char ordering) {
        this(slices, shape, Nd4j.getStrides(shape), ordering);
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, int[] stride, char ordering) {

        DataBuffer ret = slices.get(0).data().dataType() == (DataBuffer.FLOAT) ?
                Nd4j.createBuffer(new float[ArrayUtil.prod(shape)]) :
                Nd4j.createBuffer(new double[ArrayUtil.prod(shape)]);

        this.stride = stride;
        this.ordering = ordering;
        this.data = ret;

        init(shape);

        for (int i = 0; i < slices(); i++) {
            putSlice(i, slices.get(i));
        }
    }


    public BaseNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        this(data, shape, stride, 0, ordering);
    }


    public BaseNDArray(float[] data, int[] shape, int[] stride, int offset, char ordering) {


        this.offset = offset;
        this.stride = stride;
        this.ordering = ordering;

        if (data != null && data.length > 0) {
            this.data = Nd4j.createBuffer(data);
            if (offset >= data.length)
                throw new IllegalArgumentException("invalid offset: must be < data.length");


        }

        init(shape);


    }

    public BaseNDArray(DataBuffer data, int[] shape, int[] stride, int offset) {
        this.data = data;
        this.stride = stride;
        this.offset = offset;
        this.ordering = Nd4j.order();
        init(shape);

    }


    public BaseNDArray(int[] data, int[] shape, int[] strides) {
        this(Nd4j.createBuffer(data), shape, strides);
    }

    public BaseNDArray(DataBuffer data, int[] shape) {
        this(data, shape, Nd4j.getStrides(shape), 0, Nd4j.order());
    }

    public BaseNDArray(DataBuffer buffer, int[] shape, int offset) {
        this(buffer, shape, Nd4j.getStrides(shape), offset);
    }

    public BaseNDArray(double[] data, int[] shape, char ordering) {
        this(new DoubleBuffer(data), shape, ordering);
    }

    public BaseNDArray(double[] data, int[] shape, int[] stride, int offset, char ordering) {
        this(new DoubleBuffer(data), shape, stride, offset, ordering);
    }

    public BaseNDArray(float[] data, char order) {
        this(new FloatBuffer(data), order);
    }

    public BaseNDArray(FloatBuffer floatBuffer, char order) {
        this(floatBuffer, new int[]{floatBuffer.length()}, Nd4j.getStrides(new int[]{floatBuffer.length()}), 0, order);
    }

    public BaseNDArray(DataBuffer buffer, int[] shape, int[] strides) {
        this(buffer, shape, strides, 0, Nd4j.order());
    }


    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data  the data to use
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape) {
        this(data, shape, 0);
    }

    public BaseNDArray(float[] data, int[] shape, int offset) {
        this(data, shape, offset, Nd4j.order());

    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public BaseNDArray(int[] shape, int[] stride, int offset) {
        this(new float[ArrayUtil.prod(shape)], shape, stride, offset, Nd4j.order());
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride) {
        this(shape, stride, 0);
    }

    public BaseNDArray(int[] shape, int offset) {
        this(shape, calcStrides(shape), offset);
    }


    public BaseNDArray(int[] shape, char ordering) {
        this(shape, 0, ordering);
    }


    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public BaseNDArray(int newRows, int newColumns) {
        this(newRows, newColumns, Nd4j.order());
    }

    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape) {
        this(slices, shape, Nd4j.order());
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, int[] stride) {
        this(slices, shape, stride, Nd4j.order());

    }


    public BaseNDArray(float[] data, int[] shape, int[] stride) {
        this(data, shape, stride, Nd4j.order());
    }


    public BaseNDArray(float[] data, int[] shape, int[] stride, int offset) {
        this(data, shape, stride, offset, Nd4j.order());
    }


    public BaseNDArray(float[] data) {
        this.data = Nd4j.createBuffer(data);
    }


    public BaseNDArray(float[][] data) {
        this(data.length, data[0].length);

        for (int r = 0; r < rows; r++) {
            assert (data[r].length == columns);
        }

        this.data = Nd4j.createBuffer(length);


        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                putScalar(new int[]{r, c}, data[r][c]);
            }
        }
    }

    /**
     * Constructor for stride and offset
     * @param buffer
     * @param shape
     * @param offset
     * @param ordering
     */
    public BaseNDArray(DataBuffer buffer, int[] shape, int offset, char ordering) {
        this(buffer,shape,Nd4j.getStrides(shape,ordering),offset,ordering);
    }

    /**
     * Mainly an internal method (public  for testing)
     * for given an offset and stride, and index,
     * calculating the beginning index of a query given indices
     *
     * @param offset  the desired offset
     * @param stride  the desired stride
     * @param indexes the desired indexes to test on
     * @return the index for a query given stride and offset
     */
    public static int getIndex(int offset, int[] stride, int... indexes) {
        if (stride.length > indexes.length)
            throw new IllegalArgumentException("Invalid number of items in stride array: should be <= number of indexes");

        int ix = offset;


        for (int i = 0; i < indexes.length; i++) {
            ix += indexes[i] * stride[i];
        }
        return ix;
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public INDArray linearViewColumnOrder() {
        ensureNotCleanedUp();
        return Nd4j.create(data, new int[]{length, 1}, offset());
    }

    protected void ensureNotCleanedUp() {
        if(cleanedUp)
            throw new IllegalStateException("Invalid operation: already collected");
    }

    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     */
    @Override
    public INDArray linearView() {
        ensureNotCleanedUp();
        if (isVector())
            return this;
        if (linearView == null)
            resetLinearView();

        return linearView;
    }

    @Override
    public void resetLinearView() {
        ensureNotCleanedUp();
        linearView = Nd4j.create(data, new int[]{length}, new int[]{1}, offset());
    }

    @Override
    public void setData(float[] data) {
        ensureNotCleanedUp();
        this.data = Nd4j.createBuffer(data);
    }

    @Override
    public int majorStride() {
        ensureNotCleanedUp();
        //return ordering == NDArrayFactory.C ? stride[0] : stride[stride.length - 1];r
        return stride[0];
    }

    @Override
    public int secondaryStride() {
        ensureNotCleanedUp();
        if (stride.length >= 2) {
            if (ordering == NDArrayFactory.C) {
                if (isColumnVector())
                    return majorStride();
                return stride[1];
            } else
                return majorStride();
        }
        return majorStride();
    }

    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    @Override
    public int vectorsAlongDimension(int dimension) {
        ensureNotCleanedUp();
        if (dimension >= shape.length)
            return length / size(shape.length - 1);
        return length / size(dimension);
    }

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    @Override
    public INDArray vectorAlongDimension(int index, int dimension) {
        ensureNotCleanedUp();
        assert dimension <= shape.length : "Invalid dimension " + dimension;
        if (dimension > shape().length - 1)
            dimension = shape.length - 1;
        if (ordering == NDArrayFactory.C) {

            if (dimension == shape.length - 1 && dimension != 0)
                return Nd4j.create(data,
                        new int[]{shape[dimension]}
                        , new int[]{stride[shape.length - 1]},
                        offset + index * stride[dimension - 1]);

            else if (dimension == 0)
                return Nd4j.create(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index);

            if (size(dimension) == 1) {

                return Nd4j.create(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index);
            } else
                return Nd4j.create(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * stride[0]);
        } else if (ordering == NDArrayFactory.FORTRAN) {

            if (dimension == shape.length - 1 && dimension != 0) {
                if (size(dimension) == 1)
                    return Nd4j.create(data,
                            new int[]{1, shape[dimension]}
                            , ArrayUtil.removeIndex(stride, 0),
                            offset + index);
                return Nd4j.create(data,
                        new int[]{1, shape[dimension]}
                        , ArrayUtil.removeIndex(stride, 0),
                        offset + index * stride[dimension - 1]);

            } else {
                if (size(dimension) == 1)
                    return Nd4j.create(data,
                            new int[]{shape[dimension], 1}
                            , new int[]{stride[dimension], 1},
                            offset + index);

                return Nd4j.create(data,
                        new int[]{shape[dimension], 1}
                        , new int[]{stride[dimension], 1},
                        offset + index * stride[stride.length - 1]);
            }

        }

        throw new IllegalStateException("Illegal ordering..none declared");


    }

    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public INDArray cumsumi(int dimension) {
        ensureNotCleanedUp();
        if (isVector()) {
            double s = 0.0;
            for (int i = 0; i < length; i++) {
                s += getDouble(i);
                putScalar(i, s);
            }
        } else if (dimension == Integer.MAX_VALUE || dimension == shape.length - 1) {
            INDArray flattened = ravel();
            double prevVal = flattened.getDouble(0);
            for (int i = 1; i < flattened.length(); i++) {
                double d = prevVal + flattened.getDouble(i);
                flattened.putScalar(i, d);
                prevVal = d;
            }

            return flattened;
        } else {
            for (int i = 0; i < vectorsAlongDimension(dimension); i++) {
                INDArray vec = vectorAlongDimension(i, dimension);
                vec.cumsumi(0);

            }
        }


        return this;
    }

    /**
     * Cumulative sum along a dimension (in place)
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public INDArray cumsum(int dimension) {
        ensureNotCleanedUp();
        return dup().cumsumi(dimension);
    }

    /**
     * Assign all of the elements in the given
     * ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    @Override
    public INDArray assign(INDArray arr) {
        ensureNotCleanedUp();
        if (!arr.isVector() && !isVector())
            LinAlgExceptions.assertSameShape(this, arr);
        else if (isVector() && arr.isVector() && length() != arr.length())
            throw new IllegalArgumentException("Illegal assignment, must be of same length");
        Nd4j.getBlasWrapper().copy(arr, this);
        return this;
    }

    @Override
    public INDArray putScalar(int i, double value) {
        ensureNotCleanedUp();
        int idx = linearIndex(i);
        data.put(idx, value);
        return this;
    }

    @Override
    public INDArray putScalar(int i, float value) {
        ensureNotCleanedUp();
        int idx = linearIndex(i);
        data.put(idx, value);
        return this;
    }

    @Override
    public INDArray putScalar(int i, int value) {
        ensureNotCleanedUp();
        int idx = linearIndex(i);
        data.put(idx, value);
        return this;
    }

    @Override
    public INDArray putScalar(int[] indexes, double value) {
        ensureNotCleanedUp();
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        if (ix >= data.length())
            throw new IllegalArgumentException("Illegal indices " + Arrays.toString(indexes));
        data.put(ix, value);
        return this;
    }

    @Override
    public INDArray putScalar(int[] indexes, float value) {
        ensureNotCleanedUp();
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        if (ix >= data.length())
            throw new IllegalArgumentException("Illegal indices " + Arrays.toString(indexes));
        data.put(ix, value);
        return this;
    }

    @Override
    public INDArray putScalar(int[] indexes, int value) {
        ensureNotCleanedUp();
        int ix = offset;
        for (int i = 0; i < shape.length; i++) {
            ix += indexes[i] * stride[i];
        }
        if (ix >= data.length())
            throw new IllegalArgumentException("Illegal indices " + Arrays.toString(indexes));
        data.put(ix, value);
        return this;
    }

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    @Override
    public INDArray eps(Number other) {
        return dup().epsi(other);
    }

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    @Override
    public INDArray epsi(Number other) {
        ensureNotCleanedUp();
        INDArray linearView = linearView();
        double otherVal = other.doubleValue();

        for (int i = 0; i < linearView.length(); i++) {
            double val = linearView.getDouble(i);
            double diff = Math.abs(val - otherVal);
            if (diff <= Nd4j.EPS_THRESHOLD)
                linearView.putScalar(i, 1);
            else
                linearView.putScalar(i, 0);

        }
        return this;
    }

    /**
     * epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    @Override
    public INDArray eps(INDArray other) {
        return dup().epsi(other);
    }

    /**
     * In place epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    @Override
    public INDArray epsi(INDArray other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new Eps(linearView(), other.linearView(), this, length()));
        return this;
    }

    @Override
    public INDArray lt(Number other) {
        return dup().lti(other);
    }

    @Override
    public INDArray lti(Number other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new ScalarLessThan(linearView(), other));
        return this;
    }

    @Override
    public INDArray eq(Number other) {
        return dup().eqi(other);
    }

    @Override
    public INDArray eqi(Number other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new ScalarEquals(linearView(), other));
        return this;
    }

    @Override
    public INDArray gt(Number other) {
        return dup().gti(other);
    }

    @Override
    public INDArray gti(Number other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new ScalarGreaterThan(linearView(), other));
        return this;
    }

    @Override
    public INDArray lt(INDArray other) {
        return dup().lti(other);
    }

    @Override
    public INDArray lti(INDArray other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new LessThan(linearView(), other, linearView(), length()));
        return this;
    }

    @Override
    public INDArray neq(Number other) {
        return dup().neqi(other);
    }

    @Override
    public INDArray neqi(Number other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new ScalarNotEquals(linearView(), other));
        return this;
    }

    @Override
    public INDArray neq(INDArray other) {
        return dup().neqi(other);
    }

    @Override
    public INDArray neqi(INDArray other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new NotEqualTo(linearView(), other.linearView(), linearView(), length()));
        return this;
    }

    @Override
    public INDArray eq(INDArray other) {
        return dup().eqi(other);
    }

    @Override
    public INDArray eqi(INDArray other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new EqualTo(linearView(), other.linearView(), linearView(), length()));
        return this;
    }

    @Override
    public INDArray gt(INDArray other) {
        return dup().gti(other);
    }

    @Override
    public INDArray gti(INDArray other) {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new GreaterThan(linearView(), other.linearView(), linearView(), length()));
        return this;
    }

    /**
     * Negate each element.
     */
    @Override
    public INDArray neg() {
        return dup().negi();
    }

    /**
     * Negate each element (in-place).
     */
    @Override
    public INDArray negi() {
        ensureNotCleanedUp();
        Nd4j.getExecutioner().exec(new Negative(linearView()));
        return this;
    }

    @Override
    public INDArray rdiv(Number n, INDArray result) {
        return dup().rdivi(n, result);
    }

    @Override
    public INDArray rdivi(Number n, INDArray result) {
        ensureNotCleanedUp();
        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;
        Nd4j.getExecutioner().exec(new ScalarReverseDivision(linearView(), null, result.linearView(), result.length(), n));
        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);
        return result;
    }

    @Override
    public INDArray rsub(Number n, INDArray result) {
        return dup().rsubi(n, result);
    }

    @Override
    public INDArray rsubi(Number n, INDArray result) {
        ensureNotCleanedUp();
        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;

        Nd4j.getExecutioner().exec(new ScalarReverseSubtraction(linearView(), null, result.linearView(), result.length(), n));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);
        return result;
    }

    @Override
    public INDArray div(Number n, INDArray result) {
        return dup().divi(n, result);
    }

    @Override
    public INDArray divi(Number n, INDArray result) {
        ensureNotCleanedUp();
        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;
        Nd4j.getExecutioner().exec(new ScalarDivision(linearView(), null, result.linearView(), result.length(), n));


        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    @Override
    public INDArray mul(Number n, INDArray result) {
        return dup().muli(n, result);
    }

    @Override
    public INDArray muli(Number n, INDArray result) {
        ensureNotCleanedUp();
        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;
        Nd4j.getExecutioner().exec(new ScalarMultiplication(linearView(), null, result.linearView(), result.length(), n));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    @Override
    public INDArray sub(Number n, INDArray result) {
        return dup().subi(n, result);
    }

    @Override
    public INDArray subi(Number n, INDArray result) {
        ensureNotCleanedUp();
        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;

        Nd4j.getExecutioner().exec(new ScalarSubtraction(linearView(), null, result.linearView(), result.length(), n));
        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    @Override
    public INDArray add(Number n, INDArray result) {
        return dup().addi(n, result);
    }

    @Override
    public INDArray addi(Number n, INDArray result) {
        ensureNotCleanedUp();
        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;
        Nd4j.getExecutioner().exec(new ScalarAdd(linearView(), null, result.linearView(), result.length(), n));
        return this;
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
        return getScalar(new int[]{row, column});
    }

    @Override
    public INDArray dup() {
        ensureNotCleanedUp();
        DataBuffer dup = data().dup();
        INDArray ret = Nd4j.create(dup, shape(), stride(), offset());
        return ret;
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    @Override
    public int getInt(int... indices) {
        ensureNotCleanedUp();
        int ix = offset;
        for (int i = 0; i < indices.length; i++)
            ix += indices[i] * stride[i];

        return data.getInt(ix);

    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    @Override
    public double getDouble(int... indices) {
        ensureNotCleanedUp();
        int ix = offset;
        for (int i = 0; i < indices.length; i++)
            ix += indices[i] * stride[i];

        return data.getDouble(ix);

    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    @Override
    public float getFloat(int... indices) {
        ensureNotCleanedUp();
        int ix = offset;
        for (int i = 0; i < indices.length; i++)
            ix += indices[i] * stride[i];

        return data.getFloat(ix);

    }

    /**
     * Test whether a matrix is scalar.
     */
    @Override
    public boolean isScalar() {
        ensureNotCleanedUp();
        if (shape.length == 0)
            return true;
        else if (shape.length == 1 && shape[0] == 1)
            return true;
        else if (shape.length >= 2) {
            for (int i = 0; i < shape.length; i++)
                if (shape[i] != 1)
                    return false;
        }

        return length == 1;
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
        ensureNotCleanedUp();
        if (!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
        int ix = offset;
        for (int i = 0; i < indices.length; i++)
            ix += indices[i] * stride[i];
        if (ix >= data.length())
            throw new IllegalArgumentException("Illegal indices " + Arrays.toString(indices));
        data.put(ix, element.getDouble(0));
        return this;

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
        return put(new int[]{i, j}, element);
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
        ensureNotCleanedUp();
        return putScalar(new int[]{i, j}, element.doubleValue());
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
        ensureNotCleanedUp();
        if (isScalar()) {
            assert put.isScalar() : "Invalid dimension. Can only insert a scalar in to another scalar";
            put(0, put.getScalar(0));
            return this;
        } else if (isVector()) {
            assert put.isScalar() || put.isVector() &&
                    put.length() == length() : "Invalid dimension on insertion. Can only insert scalars input vectors";
            if (put.isScalar())
                putScalar(slice, put.getDouble(0));
            else
                for (int i = 0; i < length(); i++)
                    putScalar(i, put.getDouble(i));

            return this;
        }


        assertSlice(put, slice);


        INDArray view = slice(slice);

        if (put.isScalar())
            putScalar(slice, put.getDouble(0));
        else if (put.isVector())
            for (int i = 0; i < put.length(); i++)
                view.putScalar(i, put.getDouble(i));
        else if (put.shape().length == 2)
            for (int i = 0; i < put.rows(); i++)
                view.putRow(i, put.getRow(i));

        else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for (int i = 0; i < put.slices(); i++)
                view.slice(i).putSlice(i, view.slice(i));

        }

        return this;

    }

    protected void assertSlice(INDArray put, int slice) {
        ensureNotCleanedUp();
        assert slice <= slices() : "Invalid slice specified " + slice;
        int[] sliceShape = put.shape();
        int[] requiredShape = ArrayUtil.removeIndex(shape(), 0);

        //no need to compare for scalar; primarily due to shapes either being [1] or length 0
        if (put.isScalar())
            return;

        //edge case for column vectors
        if (Shape.isColumnVectorShape(sliceShape))
            return;
        assert Shape.shapeEquals(sliceShape, requiredShape) : String.format("Invalid shape size of %s . Should have been %s ", Arrays.toString(sliceShape), Arrays.toString(requiredShape));

    }

    /**
     * Returns true if this ndarray is 2d
     * or 3d with a singleton element
     *
     * @return true if the element is a matrix, false otherwise
     */
    public boolean isMatrix() {
        ensureNotCleanedUp();
        return (shape().length == 2
                && (shape[0] != 1 && shape[1] != 1));
    }


    @Override
    public int index(int row, int column) {
        ensureNotCleanedUp();
        if (!isMatrix()) {
            if (isColumnVector()) {
                int idx = linearIndex(row);
                return idx;
            } else if (isRowVector()) {
                int idx = linearIndex(column);
                return idx;
            } else
                throw new IllegalStateException("Unable to getFromOrigin row/column from a non matrix");
        }


        return offset + (row * stride[0] + column * stride[1]);
    }

    protected INDArray newShape(int[] newShape, char ordering) {
        ensureNotCleanedUp();
        if (Arrays.equals(newShape, this.shape()))
            return this;

        else if (Shape.isVector(newShape) && isVector()) {
            if (isRowVector() && Shape.isColumnVectorShape(newShape)) {
                return Nd4j.create(data, newShape, new int[]{stride[0], 1}, offset);
            } else if (isColumnVector() && Shape.isRowVectorShape(newShape)) {
                return Nd4j.create(data, newShape, new int[]{stride[1]}, offset);

            }
        }

        INDArray newCopy = this;
        int[] newStrides = null;
        //create a new copy of the ndarray
        if (shape().length > 1 && ((ordering == NDArrayFactory.C && this.ordering != NDArrayFactory.C) ||
                (ordering == NDArrayFactory.FORTRAN && this.ordering != NDArrayFactory.FORTRAN))) {
            newStrides = noCopyReshape(newShape, ordering);
            if (newStrides == null) {
                newCopy = Nd4j.create(shape(), ordering);
                for (int i = 0; i < vectorsAlongDimension(0); i++) {
                    INDArray copyFrom = vectorAlongDimension(i, 0);
                    INDArray copyTo = newCopy.vectorAlongDimension(i, 0);
                    for (int j = 0; j < copyFrom.length(); j++) {
                        copyTo.putScalar(j, copyFrom.getDouble(i));
                    }
                }
            }


        }

        //needed to copy data
        if (newStrides == null)
            newStrides = Nd4j.getStrides(newShape);
        if (this instanceof IComplexNDArray)
            return Nd4j.createComplex(newCopy.data(), newShape, newStrides, offset);
        return Nd4j.create(newCopy.data(), newShape, newStrides, offset);


    }

    /**
     * Return the new strides based on the shape and ordering or null
     * if we can't do a reshape
     *
     * @param newShape the new shape
     * @param ordering the ordering of the new shape
     * @return the new strides or null if we can't reshape
     */
    protected int[] noCopyReshape(int[] newShape, char ordering) {
        ensureNotCleanedUp();
        List<Integer> oldDims = new ArrayList<>();
        List<Integer> oldStrides = new ArrayList<>();
        for (int i = 0; i < shape.length; i++) {
            if (size(i) != 1) {
                oldDims.add(size(i));
                oldStrides.add(stride[i]);
            }
        }

        int np = 1;
        for (int ni = 0; ni < newShape.length; ni++) {
            np *= newShape[ni];
        }

        int op = 1;
        for (int oi = 0; oi < oldDims.size(); oi++) {
            op *= oldDims.get(oi);
        }
        if (np != op) {
        /* different total sizes; no hope */
            return null;
        }

        if (np == 0) {
        /* the current code does not handle 0-sized arrays, so give up */
            return null;
        }


          /* oi to oj and ni to nj give the axis ranges currently worked with */
        int oi = 0;
        int oj = 1;
        int ni = 0;
        int nj = 1;

        List<Integer> newStrides = new ArrayList<>();
        while (ni < newShape.length && oi < oldDims.size()) {
            np = newShape[ni];
            op = oldDims.get(oi);

            while (np != op) {
                if (np < op)
                    np *= newShape[nj++];
                else
                    op *= oldDims.get(oj++);
            }

                    /* Check whether the original axes can be combined */
            for (int ok = oi; ok < oj - 1; ok++) {
                if (ordering == NDArrayFactory.FORTRAN) {
                    if (oldStrides.get(ok + 1) != oldDims.get(ok) * oldStrides.get(ok)) {
                     /* not contiguous enough */
                        return null;
                    }
                } else {
                /* C order */
                    if (oldStrides.get(ok) != oldDims.get(ok + 1) * oldStrides.get(ok + 1)) {
                    /* not contiguous enough */
                        return null;
                    }
                }
            }


             /* Calculate new strides for all axes currently worked with */
            if (ordering == NDArrayFactory.FORTRAN) {
                newStrides.set(ni, oldStrides.get(oi));
                for (int nk = ni + 1; nk < nj; nk++) {
                    newStrides.set(nk, newStrides.get(nk - 1) * newShape[nk - 1]);
                }
            } else {
            /* C order */
                newStrides.set(nj - 1, oldStrides.get(oj - 1));

                for (int nk = nj - 1; nk > ni; nk--) {
                    newStrides.set(nk - 1, newStrides.get(nk) * newShape[nk]);
                }
            }
            ni = nj++;
            oi = oj++;
        }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
        int lastStride;
        if (ni >= 1) {
            lastStride = newStrides.get(ni - 1);
        } else {
            lastStride = length;
        }
        if (ordering == NDArrayFactory.FORTRAN) {
            lastStride *= newShape[ni - 1];
        }
        for (int nk = ni; nk < newShape.length; nk++) {
            newStrides.set(nk, lastStride);
        }


        return ArrayUtil.toArray(newStrides);


    }


    /**
     * Returns the squared (Euclidean) distance.
     */
    @Override
    public double squaredDistance(INDArray other) {
        ensureNotCleanedUp();
        double sd = 0.0;
        for (int i = 0; i < length; i++) {
            double d = getDouble(i) - other.getDouble(i);
            sd += d * d;
        }
        return sd;
    }

    /**
     * Returns the (euclidean) distance.
     */
    @Override
    public double distance2(INDArray other) {
        ensureNotCleanedUp();
        return Math.sqrt(squaredDistance(other));
    }

    /**
     * Returns the (1-norm) distance.
     */
    @Override
    public double distance1(INDArray other) {
        ensureNotCleanedUp();
        return other.sub(this).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, INDArray element) {
        ensureNotCleanedUp();
        if (Indices.isContiguous(indices)) {
            INDArray get = get(indices);
            INDArray linear = get.linearView();
            if (element.isScalar()) {
                for (int i = 0; i < linear.length(); i++) {
                    linear.putScalar(i, element.getDouble(0));
                }
            }

            if (Shape.shapeEquals(element.shape(), get.shape()) || element.length() <= get.length()) {
                INDArray elementLinear = element.linearView();

                for (int i = 0; i < elementLinear.length(); i++) {
                    linear.putScalar(i, elementLinear.getDouble(i));
                }
            }


        } else {
            if (isVector()) {
                assert indices.length == 1 : "Indices must only be of length 1.";
                assert element.isScalar() || element.isVector() : "Unable to assign elements. Element is not a vector.";
                assert indices[0].length() == element.length() : "Number of specified elements in index does not match length of element.";
                int[] assign = indices[0].indices();
                for (int i = 0; i < element.length(); i++) {
                    putScalar(assign[i], element.getDouble(i));
                }

                return this;

            }

            if (element.isVector())
                slice(indices[0].indices()[0]).put(Arrays.copyOfRange(indices, 1, indices.length), element);


            else {
                for (int i = 0; i < element.slices(); i++) {
                    INDArray slice = slice(indices[0].indices()[i]);
                    slice.put(Arrays.copyOfRange(indices, 1, indices.length), element.slice(i));
                }
            }

        }

        return this;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, Number element) {
        return put(indices, Nd4j.scalar(element));
    }

    /**
     * Iterate over every column of every slice
     *
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllColumns(SliceOp op) {
        ensureNotCleanedUp();
        if (isVector())
            op.operate(this);
        else if (isMatrix()) {
            for (int i = 0; i < columns(); i++) {
                op.operate(getColumn(i));
            }
        } else {
            for (int i = 0; i < slices(); i++) {
                slice(i).iterateOverAllRows(op);
            }
        }
    }

    /**
     * Iterate over every row of every slice
     *
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllRows(SliceOp op) {
        if (isVector())
            op.operate(this);
        else if (isMatrix()) {
            for (int i = 0; i < rows(); i++) {
                op.operate(getRow(i));
            }
        } else {
            for (int i = 0; i < slices(); i++) {
                slice(i).iterateOverAllRows(op);
            }
        }
    }

    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with      the one to swap it with
     * @return the swapped axes view
     */
    @Override
    public INDArray swapAxes(int dimension, int with) {
        ensureNotCleanedUp();
        int[] shape = ArrayUtil.range(0, shape().length);
        shape[dimension] = with;
        shape[with] = dimension;
        return permute(shape);
    }

    /**
     * Gives the indices for the ending of each slice
     *
     * @return the off sets for the beginning of each slice
     */
    @Override
    public int[] endsForSlices() {
        ensureNotCleanedUp();
        int[] ret = new int[slices()];
        int currOffset = offset + stride[0] - 1;
        for (int i = 0; i < slices(); i++) {
            ret[i] = currOffset;
            currOffset += stride[0];
        }
        return ret;
    }

    @Override
    public DataBuffer data() {
        ensureNotCleanedUp();
        return data;
    }

    @Override
    public void setData(DataBuffer data) {
        ensureNotCleanedUp();
        this.data = data;
    }

    /**
     * Number of slices: aka shape[0]
     *
     * @return the number of slices
     * for this nd array
     */
    @Override
    public int slices() {
        ensureNotCleanedUp();
        if (shape.length < 1)
            return 0;
        return shape[0];
    }

    @Override
    public INDArray subArray(int[] offsets, int[] shape, int[] stride) {
        ensureNotCleanedUp();
        int n = shape.length;
        if (shape.length < 1)
            return Nd4j.create(Nd4j.createBuffer(shape));
        if (offsets.length != n)
            throw new IllegalArgumentException("Invalid offset " + Arrays.toString(offsets));
        if (shape.length != n)
            throw new IllegalArgumentException("Invalid shape " + Arrays.toString(shape));

        if (Arrays.equals(shape, this.shape)) {
            if (ArrayUtil.isZero(offsets)) {
                return this;
            } else {
                throw new IllegalArgumentException("Invalid subArray offsets");
            }
        }


        int offset = this.offset + ArrayUtil.dotProduct(offsets, this.stride);

        return Nd4j.create(
                data
                , Arrays.copyOf(shape, shape.length)
                , stride
                , offset, ordering
        );
    }

    @Override
    public INDArray cond(Condition condition) {
        return dup().condi(condition);
    }

    @Override
    public INDArray condi(Condition condition) {
        ensureNotCleanedUp();
        INDArray linear = linearView();
        for (int i = 0; i < length(); i++) {
            boolean met = condition.apply(linear.getDouble(i));
            linear.putScalar(i, met ? 1 : 0);
        }
        return this;
    }

    /**
     * Iterate along a dimension.
     * This encapsulates the process of sum, mean, and other processes
     * take when iterating over a dimension.
     *
     * @param dimension the dimension to iterate over
     * @param op        the operation to apply
     * @param modify    whether to modify this array while iterating
     */
    @Override
    public void iterateOverDimension(int dimension, SliceOp op, boolean modify) {
        ensureNotCleanedUp();
        if (dimension >= shape.length)
            throw new IllegalArgumentException("Unable to remove dimension  " + dimension + " was >= shape length");
        int vectors = vectorsAlongDimension(dimension);
        for (int i = 0; i < vectors; i++) {
            INDArray vector = vectorAlongDimension(i, dimension);
            op.operate(vector);
        }

    }

    @Override
    public void setStride(int[] stride) {
        ensureNotCleanedUp();
        this.stride = stride;
    }

    protected void init(int[] shape) {
        ensureNotCleanedUp();
        this.shape = shape;

        if (this.shape.length == 1) {
            rows = 1;
            columns = this.shape[0];
        } else if (this.shape().length == 2) {
            if (shape[0] == 1) {
                this.shape = new int[1];
                this.shape[0] = shape[1];
                rows = 1;
                columns = shape[1];
                //weird case here: when its fortran contiguous, we have scencrios
                //where we need to retain the non zero stride of the ndarray.
                //this is in response to a strange scenario that happens in subArray
                //hopefully we can work out something better eventually
                if (stride != null && ordering == NDArrayFactory.FORTRAN)
                    this.stride = new int[]{ArrayUtil.nonOneStride(this.stride)};

            } else {
                rows = shape[0];
                columns = shape[1];
            }


        }

        //default row vector
        else if (this.shape.length == 1) {
            columns = this.shape[0];
            rows = 1;
        }

        //null character
        if (this.ordering == '\u0000')
            this.ordering = Nd4j.order();

        this.length = ArrayUtil.prod(this.shape);
        if (this.stride == null) {
            if (ordering == NDArrayFactory.FORTRAN)
                this.stride = ArrayUtil.calcStridesFortran(shape);
            else
                this.stride = ArrayUtil.calcStrides(this.shape);
        }

        //recalculate stride: this should only happen with row vectors
        if (this.stride.length != this.shape.length) {
            if (ordering == NDArrayFactory.FORTRAN)
                this.stride = ArrayUtil.calcStridesFortran(this.shape);
            else
                this.stride = ArrayUtil.calcStrides(this.shape);
        }

        //add the reference for clean up later (clean up the buffer when this becomes a weak reference)
        data().addReferencing(id());
        ref = new WeakReference<>((INDArray) this, Nd4j.refQueue());


    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        cleanup();
    }

    @Override
    public INDArray getScalar(int i) {
        ensureNotCleanedUp();
        if (!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        if (idx >= data.length())
            throw new IllegalArgumentException("Illegal indices " + i);
        return Nd4j.scalar(data.getDouble(idx));
    }

    protected void assertColumnVector(INDArray column) {
        assert column.isColumnVector() || column.columns() == columns() && column.rows() == 1 : "Must only add a column vector";
        assert column.length() == rows() || column.columns() == columns() && column.rows() == 1 : "Illegal column vector must have the same length as the number of rows in this ndarray";

    }

    /**
     * Do a row wise op (a,s,m,d)
     * a : add
     * s : subtract
     * m : multiply
     * d : divide
     * h : reverse subtraction
     * t : reverse division
     *
     * @param columnVector the column  vector
     * @param operation    the operation
     * @return
     */
    protected INDArray doColumnWise(INDArray columnVector, char operation) {
        ensureNotCleanedUp();
        if (rows() == 1 && columnVector.isScalar()) {
            switch (operation) {
                case 'a':
                    addi(columnVector.getDouble(0));
                    break;
                case 's':
                    subi(columnVector.getDouble(0));
                    break;
                case 'm':
                    muli(columnVector.getDouble(0));
                    break;
                case 'd':
                    divi(columnVector.getDouble(0));
                    break;
                case 'h':
                    rsubi(columnVector.getDouble(0));
                    break;
                case 't':
                    rdivi(columnVector.getDouble(0));
                    break;
            }

            return this;
        }
        assertColumnVector(columnVector);
        for (int i = 0; i < columns(); i++) {
            INDArray slice = slice(i, 0);
            switch (operation) {

                case 'a':
                    slice.addi(columnVector);
                    break;
                case 's':
                    slice.subi(columnVector);
                    break;
                case 'm':
                    slice.muli(columnVector);
                    break;
                case 'd':
                    slice.divi(columnVector);
                    break;
                case 'h':
                    slice.rsubi(columnVector);
                    break;
                case 't':
                    slice.rdivi(columnVector);
                    break;
            }
        }

        return this;


    }

    @Override
    public boolean isCleanedUp() {
        return cleanedUp;
    }

    @Override
    public  void cleanup() {
        cleanedUp = true;
        if (Nd4j.shouldInstrument)
            Nd4j.getInstrumentation().log(this, Instrumentation.DESTROYED);
        Nd4j.getResourceManager().remove(id());
        data().removeReferencing(id());


    }

    protected void assertRowVector(INDArray rowVector) {
        ensureNotCleanedUp();
        assert rowVector.isRowVector() || rowVector.rows() == rows() && rowVector.columns() == 1 : "Must only add a row vector";
        assert rowVector.length() == columns() || rowVector.rows() == rows() && rowVector.columns() == 1 : "Illegal row vector must have the same length as the number of rows in this ndarray";

    }

    /**
     * Do a row wise op (a,s,m,d)
     * a : add
     * s : subtract
     * m : multiply
     * d : divide
     * h : reverse subtraction
     * t : reverse division
     *
     * @param rowVector the row vector
     * @param operation the operation
     * @return
     */
    protected INDArray doRowWise(INDArray rowVector, char operation) {
        ensureNotCleanedUp();
        if (columns() == 1 && rowVector.isScalar()) {
            switch (operation) {
                case 'a':
                    addi(rowVector.getDouble(0));
                    break;
                case 's':
                    subi(rowVector.getDouble(0));
                    break;
                case 'm':
                    muli(rowVector.getDouble(0));
                    break;
                case 'd':
                    divi(rowVector.getDouble(0));
                    break;
                case 'h':
                    rsubi(rowVector.getDouble(0));
                    break;
                case 't':
                    rdivi(rowVector.getDouble(0));
                    break;
            }

            return this;
        }

        assertRowVector(rowVector);
        for (int i = 0; i < rows(); i++) {
            switch (operation) {

                case 'a':
                    getRow(i).addi(rowVector);
                    break;
                case 's':
                    getRow(i).subi(rowVector);
                    break;
                case 'm':
                    getRow(i).muli(rowVector);
                    break;
                case 'd':
                    getRow(i).divi(rowVector);
                    break;
                case 'h':
                    getRow(i).rsubi(rowVector);
                    break;
                case 't':
                    getRow(i).rdivi(rowVector);
                    break;
            }
        }


        return this;

    }

    @Override
    public INDArray rdiviColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 't');
    }

    @Override
    public INDArray rdivColumnVector(INDArray columnVector) {
        return dup().rdiviColumnVector(columnVector);
    }

    @Override
    public INDArray rdiviRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 't');
    }

    @Override
    public INDArray rdivRowVector(INDArray rowVector) {
        return dup().rdiviRowVector(rowVector);
    }

    @Override
    public INDArray rsubiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'h');
    }

    @Override
    public INDArray rsubColumnVector(INDArray columnVector) {
        return dup().rsubiColumnVector(columnVector);
    }

    @Override
    public INDArray rsubiRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'h');
    }

    @Override
    public INDArray rsubRowVector(INDArray rowVector) {
        return dup().rsubiRowVector(rowVector);
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, INDArray element) {
        ensureNotCleanedUp();
        if (element == null)
            throw new IllegalArgumentException("Unable to insert null element");
        assert element.isScalar() : "Unable to insert non scalar element";
        int idx = linearIndex(i);
        if (idx >= data.length())
            throw new IllegalArgumentException("Illegal indices " + i);
        data.put(idx, element.getDouble(0));
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray diviColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'd');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray divColumnVector(INDArray columnVector) {
        return dup().diviColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray diviRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'd');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray divRowVector(INDArray rowVector) {
        return dup().diviRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray muliColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'm');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray mulColumnVector(INDArray columnVector) {
        return dup().muliColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray muliRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'm');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray mulRowVector(INDArray rowVector) {
        return dup().muliRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 's');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subColumnVector(INDArray columnVector) {
        return dup().subiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subiRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 's');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subRowVector(INDArray rowVector) {
        return dup().subiRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'a');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addColumnVector(INDArray columnVector) {
        return dup().addiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addiRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'a');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addRowVector(INDArray rowVector) {
        return dup().addiRowVector(rowVector);
    }

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other) {
        ensureNotCleanedUp();
        int[] shape = {rows(), other.columns()};
        char order = Nd4j.factory().order();
        boolean switchedOrder = false;
        synchronized (Nd4j.factory()) {
            if (order != NDArrayFactory.FORTRAN) {
                Nd4j.factory().setOrder(NDArrayFactory.FORTRAN);
                switchedOrder = true;
            }
        }
        INDArray result = Nd4j.create(shape, other.data().dataType());
        synchronized (Nd4j.factory()) {
            if (switchedOrder && order != NDArrayFactory.FORTRAN)
                Nd4j.factory().setOrder(NDArrayFactory.C);
        }
        return mmuli(other, result);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other, INDArray result) {
        return dup().mmuli(other, result);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public INDArray div(INDArray other) {
        return dup().divi(other);
    }

    /**
     * copy (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    public INDArray div(INDArray other, INDArray result) {
        return dup().divi(other, result);
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public INDArray mul(INDArray other) {
        return dup().muli(other);
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    public INDArray mul(INDArray other, INDArray result) {
        return dup().muli(other, result);
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public INDArray sub(INDArray other) {
        return dup().subi(other);
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    public INDArray sub(INDArray other, INDArray result) {
        return dup().subi(other, result);
    }

    /**
     * copy addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public INDArray add(INDArray other) {
        return dup().addi(other);
    }

    /**
     * copy addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public INDArray add(INDArray other, INDArray result) {
        return dup().addi(other, result);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other) {
        return dup().mmuli(other, this);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other, INDArray result) {
        ensureNotCleanedUp();
        INDArray otherArray = other;
        INDArray resultArray = result;

        if (other.shape().length > 2) {
            for (int i = 0; i < other.slices(); i++) {
                result.putSlice(i, slice(i).mmul(other.slice(i)));
            }

            return result;

        }
        LinAlgExceptions.assertMultiplies(this, other);


        if (other.isScalar()) {
            return muli(otherArray.getDouble(0), resultArray);
        }
        if (isScalar()) {
            return otherArray.muli(getDouble(0), resultArray);
        }

        /* check sizes and resize if necessary */
        //assertMultipliesWith(other);


        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            INDArray temp = Nd4j.create(resultArray.shape(), ArrayUtil.calcStridesFortran(resultArray.shape()));

            if (otherArray.columns() == 1) {
                if (data.dataType() == (DataBuffer.DOUBLE))
                    Nd4j.getBlasWrapper().gemv(1.0, this, otherArray, 0.0, temp);
                else
                    Nd4j.getBlasWrapper().gemv(1.0f, this, otherArray, 0.0f, temp);
            } else {
                if (data.dataType() == (DataBuffer.DOUBLE))
                    Nd4j.getBlasWrapper().gemm(1.0, this, otherArray, 0.0, temp);
                else
                    Nd4j.getBlasWrapper().gemm(1.0f, this, otherArray, 0.0f, temp);
            }

            Nd4j.getBlasWrapper().copy(temp, resultArray);


        } else {
            if (otherArray.columns() == 1)
                if (data.dataType() == (DataBuffer.DOUBLE))
                    Nd4j.getBlasWrapper().gemv(1.0, this, otherArray, 0.0, resultArray);
                else
                    Nd4j.getBlasWrapper().gemv(1.0f, this, otherArray, 0.0f, resultArray);
            else {
                if (data.dataType() == (DataBuffer.DOUBLE))
                    Nd4j.getBlasWrapper().gemm(1.0, this, otherArray, 0.0, resultArray);
                else
                    Nd4j.getBlasWrapper().gemm(1.0f, this, otherArray, 0.0f, resultArray);
            }
        }

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(resultArray);
        return resultArray;
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public INDArray divi(INDArray other) {
        return divi(other, this);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    public INDArray divi(INDArray other, INDArray result) {
        ensureNotCleanedUp();
        if (other.isScalar()) {
            return divi(other.getDouble(0), result);
        }

        if (isScalar()) {
            return other.divi(getDouble(0), result);
        }

        Nd4j.getExecutioner().exec(new DivOp(this.linearView(), other.linearView(), result.linearView(), length()));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);
        return result;
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public INDArray muli(INDArray other) {
        return muli(other, this);
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    public INDArray muli(INDArray other, INDArray result) {
        ensureNotCleanedUp();
        if (other.isScalar()) {
            return muli(other.getDouble(0), result);
        }
        if (isScalar()) {
            return other.muli(getDouble(0), result);
        }

        Nd4j.getExecutioner().exec(new MulOp(this.linearView(), other.linearView(), result.linearView(), length()));
        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public INDArray subi(INDArray other) {
        return subi(other, this);
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    public INDArray subi(INDArray other, INDArray result) {
        ensureNotCleanedUp();
        if (other.isScalar()) {
            return subi(other.getDouble(0), result);
        }
        if (isScalar()) {
            return other.rsubi(getDouble(0), result);
        }


        if (result == this) {
            if (data.dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(-1.0, other, result);
            else
                Nd4j.getBlasWrapper().axpy(-1.0f, other, result);
        } else if (result == other) {
            if (data.dataType() == DataBuffer.DOUBLE) {
                Nd4j.getBlasWrapper().scal(-1.0, result);
                Nd4j.getBlasWrapper().axpy(1.0, this, result);
            } else {
                Nd4j.getBlasWrapper().scal(-1.0f, result);
                Nd4j.getBlasWrapper().axpy(1.0f, this, result);
            }
        } else {
            if (data.dataType() == DataBuffer.FLOAT) {
                Nd4j.getBlasWrapper().copy(this, result);
                Nd4j.getBlasWrapper().axpy(-1.0f, other, result);
            } else {
                Nd4j.getBlasWrapper().copy(this, result);
                Nd4j.getBlasWrapper().axpy(-1.0, other, result);
            }

        }

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public INDArray addi(INDArray other) {
        return addi(other, this);
    }

    /**
     * in place addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public INDArray addi(INDArray other, INDArray result) {
        ensureNotCleanedUp();
        if (other.isScalar()) {
            return result.addi(other.getDouble(0), result);
        }
        if (isScalar()) {
            return other.addi(getDouble(0), result);
        }


        if (result == this) {
            if (data.dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(1.0, other, result);


            else
                Nd4j.getBlasWrapper().axpy(1.0f, other, result);

        } else if (result == other) {
            if (data.dataType() == (DataBuffer.DOUBLE))
                Nd4j.getBlasWrapper().axpy(1.0, this, result);
            else
                Nd4j.getBlasWrapper().axpy(1.0f, this, result);
        } else {
            Nd4j.getExecutioner().exec(new AddOp(linearView(), other.linearView(), result.linearView()));
        }

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray normmax(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.normmax(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.normmax(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }

    /**
     * Reverse division
     *
     * @param other the matrix to divide from
     * @return
     */
    @Override
    public INDArray rdiv(INDArray other) {
        return dup().rdivi(other);
    }

    /**
     * Reverse divsion (in place)
     *
     * @param other
     * @return
     */
    @Override
    public INDArray rdivi(INDArray other) {
        return rdivi(other, this);
    }

    /**
     * Reverse division
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public INDArray rdiv(INDArray other, INDArray result) {
        return dup().rdivi(other, result);
    }

    /**
     * Reverse division (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    public INDArray rdivi(INDArray other, INDArray result) {
        return other.divi(this, result);
    }

    /**
     * Reverse subtraction
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public INDArray rsub(INDArray other, INDArray result) {
        return dup().rsubi(other, result);
    }

    /**
     * @param other
     * @return
     */
    @Override
    public INDArray rsub(INDArray other) {
        return dup().rsubi(other);
    }

    /**
     * @param other
     * @return
     */
    @Override
    public INDArray rsubi(INDArray other) {
        return rsubi(other, this);
    }

    /**
     * Reverse subtraction (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    public INDArray rsubi(INDArray other, INDArray result) {
        return other.subi(this, result);
    }

    /**
     * Set the value of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    @Override
    public INDArray assign(Number value) {
        data().assign(value);
        return this;
    }

    @Override
    public int linearIndex(int i) {
        int realStride = secondaryStride();
        int idx = offset + i * realStride;

        if (data != null && idx >= data.length())
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }

    /**
     * Returns the specified slice of this matrix.
     * In matlab, this would be equivalent to (given a 2 x 2 x 2):
     * A(:,:,x) where x is the slice you want to return.
     * <p/>
     * The slice is always relative to the final dimension of the matrix.
     *
     * @param slice the slice to return
     * @return the specified slice of this matrix
     */
    @Override
    public INDArray slice(int slice) {

        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d NDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1) {
            if (size(0) == 1)
                return Nd4j.create(data, ArrayUtil.empty(), ArrayUtil.empty(), offset + slice);
            else
                return Nd4j.create(data, ArrayUtil.empty(), ArrayUtil.empty(), offset + slice * stride[0]);


        }


        //slice of a matrix is a vector
        else if (shape.length == 2) {
            if (size(0) == 1) {
                INDArray slice2 = Nd4j.create(
                        data,
                        ArrayUtil.of(shape[1]),
                        Arrays.copyOfRange(stride, 1, stride.length),
                        offset + slice, ordering
                );
                return slice2;
            } else {
                INDArray slice2 = Nd4j.create(
                        data,
                        ArrayUtil.of(shape[1]),
                        Arrays.copyOfRange(stride, 1, stride.length),
                        offset + slice * stride[0], ordering
                );
                return slice2;
            }


        } else {
            int offset = this.offset + (slice * stride[0]);
            if (size(0) == 1) {

                INDArray slice2 = Nd4j.create(data,
                        Arrays.copyOfRange(shape, 1, shape.length),
                        Arrays.copyOfRange(stride, 1, stride.length),
                        offset, ordering);
                return slice2;
            } else {
                INDArray slice2 = Nd4j.create(data,
                        Arrays.copyOfRange(shape, 1, shape.length),
                        Arrays.copyOfRange(stride, 1, stride.length),
                        offset, ordering);
                return slice2;
            }


        }
    }

    /**
     * Returns the slice of this from the specified dimension
     *
     * @param slice     the dimension to return from
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    @Override
    public INDArray slice(int slice, int dimension) {
        if (shape.length == 2) {
            //rows
            if (dimension == 1)
                return getRow(slice);


            else if (dimension == 0)
                return getColumn(slice);

            else throw new IllegalAccessError("Illegal dimension for matrix");

        }

        if (slice == shape.length - 1)
            return slice(dimension);

        INDArray slice2 = Nd4j.create(data,
                ArrayUtil.removeIndex(shape, dimension),
                ArrayUtil.removeIndex(stride, dimension),
                offset + slice * stride[dimension], ordering);
        return slice2;
    }

    /**
     * Fetch a particular number on a multi dimensional scale.
     *
     * @param indexes the indexes to getFromOrigin a number from
     * @return the number at the specified indices
     */
    @Override
    public INDArray getScalar(int... indexes) {
        int ix = offset;


        for (int i = 0; i < indexes.length; i++) {
            ix += indexes[i] * stride[i];
        }

        if (ix >= data.length())
            throw new IllegalArgumentException("Illegal index " + Arrays.toString(indexes));

        return Nd4j.scalar(data.getDouble(ix));
    }

    @Override
    public INDArray rdiv(Number n) {
        return dup().rdivi(n);
    }

    @Override
    public INDArray rdivi(Number n) {
        return rdivi(n, this);
    }

    @Override
    public INDArray rsub(Number n) {
        return dup().rsubi(n);
    }

    @Override
    public INDArray rsubi(Number n) {
        return rsubi(n, this);
    }

    @Override
    public INDArray div(Number n) {
        return dup().divi(n);
    }

    @Override
    public INDArray divi(Number n) {
        return divi(n, this);
    }

    @Override
    public INDArray mul(Number n) {
        return dup().muli(n);
    }

    @Override
    public INDArray muli(Number n) {
        return muli(n, this);
    }

    @Override
    public INDArray sub(Number n) {
        return dup().subi(n);
    }

    @Override
    public INDArray subi(Number n) {
        return subi(n, this);
    }

    @Override
    public INDArray add(Number n) {
        return dup().addi(n);
    }

    @Override
    public INDArray addi(Number n) {
        return addi(n, this);
    }

    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    @Override
    public INDArray repmat(int[] shape) {
        int[] newShape = new int[shape.length];
        assert shape.length <= newShape.length : "Illegal shape: The passed in shape must be <= the current shape length";
        int[] oldShape = isRowVector() ? new int[]{1, this.shape[0]} : Arrays.copyOf(this.shape, 2);
        for (int i = 0; i < newShape.length; i++) {
            if (i < this.shape.length)
                newShape[i] = oldShape[i] * shape[i];
            else
                newShape[i] = oldShape[i];
        }

        INDArray result = Nd4j.create(newShape);
        //nd copy
        if (isScalar()) {
            for (int i = 0; i < result.length(); i++) {
                result.put(i, getScalar(0));

            }
        } else if (isRowVector()) {
            if (Shape.isColumnVectorShape(newShape))
                return transpose();
            else if (Shape.isMatrix(newShape)) {
                INDArray ret = Nd4j.create(newShape);
                for (int i = 0; i < ret.rows(); i++) {
                    ret.putRow(i, this);
                }
                return ret;
            }
        } else if (isMatrix()) {

            for (int c = 0; c < shape()[1]; c++) {
                for (int r = 0; r < shape()[0]; r++) {
                    for (int i = 0; i < rows(); i++) {
                        for (int j = 0; j < columns(); j++) {
                            result.put(r * rows() + i, c * columns() + j, getScalar(i, j));
                        }
                    }
                }
            }

        } else {
            int[] sliceRepmat = ArrayUtil.removeIndex(shape, 0);
            for (int i = 0; i < result.slices(); i++) {
                result.putSlice(i, repmat(sliceRepmat));
            }
        }

        INDArray ret = result;
        return ret;
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
        assert toPut.isVector() && toPut.length() == columns : "Illegal length for row " + toPut.length() + " should have been " + columns;
        INDArray r = getRow(row);
        r.assign(toPut);
        return this;
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
        assert toPut.isVector() && toPut.length() == rows() : "Illegal length for row " + toPut.length() + " should have been " + rows();
        INDArray r = getColumn(column);
        r.assign(toPut);
        return this;
    }


    @Override
    public double getDouble(int i) {
        int idx = linearIndex(i);
        if (idx < 0)
            throw new IllegalStateException("Illegal index " + i);
        return data.getDouble(idx);
    }

    @Override
    public double getDouble(int i, int j) {
        int idx = index(i, j);
        if (idx >= data.length())
            throw new IllegalArgumentException("Invalid index " + i + " , " + j);
        return data.getDouble(idx);
    }

    @Override
    public float getFloat(int i) {
        int idx = linearIndex(i);
        if (idx < 0)
            throw new IllegalStateException("Illegal index " + i);
        if (idx >= data.length())
            throw new IllegalArgumentException("Invalid index " + i);
        return data.getFloat(idx);
    }

    @Override
    public float getFloat(int i, int j) {
        int idx = index(i, j);
        if (idx >= data.length())
            throw new IllegalArgumentException("Invalid index " + i + " , " + j);
        return data.getFloat(idx);
    }

    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public INDArray transpose() {
        return transposei();
    }


    /**
     * Return transposed version of this matrix.
     */
    @Override
    public INDArray transposei() {
        if (isRowVector())
            return Nd4j.create(data, new int[]{shape[0], 1}, offset);
        else if (isColumnVector())
            return Nd4j.create(data, new int[]{shape[0]}, offset);
        if (isMatrix()) {
            INDArray ret = Nd4j.create(columns, rows);
            for (int i = 0; i < ret.rows(); i++) {
                ret.putRow(i, getColumn(i));
            }

            return ret;
        }
        INDArray ret = permute(ArrayUtil.range(shape.length - 1, -1));
        return ret;
    }


    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     *
     * @param shape
     * @return
     */
    @Override
    public INDArray reshape(int[] shape) {
        assert ArrayUtil.prod(shape) == ArrayUtil.prod(this.shape()) : "Illegal reshape must be of same length as data";
        return newShape(shape, ordering);
    }


    @Override
    public void checkDimensions(INDArray other) {
        assert Arrays.equals(shape, other.shape()) : " Other array should have been shape: " + Arrays.toString(shape) + " but was " + Arrays.toString(other.shape());
        assert Arrays.equals(stride, other.stride()) : " Other array should have been stride: " + Arrays.toString(stride) + " but was " + Arrays.toString(other.stride());
        assert offset == other.offset() : "Offset of this array is " + offset + " but other was " + other.offset();

    }


    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    @Override
    public INDArray prod(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.prod(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.prod(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray mean(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.mean(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.mean(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }

    /**
     * Returns the overall variance of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray var(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.var(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.var(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }

    /**
     * Returns the overall max of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray max(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.max(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.max(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray min(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.min(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.min(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    @Override
    public INDArray sum(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.sum(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.sum(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }


    /**
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray norm1(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.norm1(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.norm1(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }


    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    @Override
    public INDArray std(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.std(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.std(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }


    /**
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    @Override
    public INDArray norm2(int dimension) {
        Triple<SliceOp, INDArray, int[]> pair = getOp(new AtomicInteger(0), DimensionFunctions.norm2(dimension), dimension);
        return doDimensionWise(
                DimensionFunctions.norm2(),
                pair.getLeft(), pair.getMiddle(), pair.getRight(), dimension, false);
    }

    /**
     * Return a slice op
     *
     * @param i         the shape of the solution
     * @param func      the function to apply in the
     *                  slice op
     * @param dimension the dimension to perform on
     * @return a tuple representing a slice operation along
     * a dimension
     */
    private Triple<SliceOp, INDArray, int[]> getOp(final AtomicInteger i, final Function<INDArray, INDArray> func, int dimension) {
        ensureNotCleanedUp();
        int[] shape = shape().length == 1 || dimension == Integer.MAX_VALUE ? new int[]{1} : ArrayUtil.removeIndex(shape(), dimension);
        final INDArray put = Nd4j.create(new int[]{ArrayUtil.prod(shape)});
        SliceOp op = new SliceOp() {


            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                put.put(i.get(), func.apply(nd));
                i.incrementAndGet();

            }
        };

        return Triple.of(op, put, shape);

    }


    /**
     * Number of columns (shape[1]), throws an exception when
     * called when not 2d
     *
     * @return the number of columns in the array (only 2d)
     */
    @Override
    public int columns() {
        ensureNotCleanedUp();
        if (isMatrix()) {
            if (shape().length > 2)
                return Shape.squeeze(shape)[1];
            else if (shape().length == 2)
                return shape[1];
        }
        if (isVector()) {
            if (isColumnVector())
                return 1;
            else
                return shape[0];
        }
        throw new IllegalStateException("Unable to getFromOrigin number of of rows for a non 2d matrix");
    }

    /**
     * Returns the number of rows
     * in the array (only 2d) throws an exception when
     * called when not 2d
     *
     * @return the number of rows in the matrix
     */
    @Override
    public int rows() {
        ensureNotCleanedUp();
        if (isMatrix()) {
            if (shape().length > 2)
                return Shape.squeeze(shape)[0];
            else if (shape().length == 2)
                return shape[0];
        } else if (isVector()) {
            if (isRowVector())
                return 1;
            else
                return shape[0];
        }
        throw new IllegalStateException("Unable to get number of of rows for a non 2d matrix");
    }


    /**
     * Do an operation over a dimension
     *
     * @param baseCase  the base case for a vector
     * @param sliceOp   the slice operation to perform
     * @param arr       the array to return
     * @param newShape  the new shape of the solution
     * @param dimension the dimension to operate on
     * @param modify    whether to modify the array
     * @return the new result
     */
    protected INDArray doDimensionWise(
            Function<INDArray, INDArray> baseCase
            , SliceOp sliceOp
            , INDArray arr
            , int[] newShape
            , int dimension
            , boolean modify) {
        ensureNotCleanedUp();
        if (dimension == Integer.MAX_VALUE || dimension >= shape().length)
            return baseCase.apply(this.linearView());
        else if(isRowVector()) {
            //same shape
            if(dimension == 0) {
                //no reduction
                return this;
            }
            else if(dimension == 1) {
                return baseCase.apply(this.linearView());
            }
        }
        else if(isColumnVector()) {
            if(dimension == 0) {
                return baseCase.apply(this.linearView());

            }
            //row vector
            else if(dimension == 1) {
                //make a row vector
                return baseCase.apply(this).transpose();

            }
        }

        iterateOverDimension(dimension, sliceOp, modify);
        return arr.reshape(newShape);
    }


    /**
     * Flattens the array for linear indexing
     *
     * @return the flattened version of this array
     */
    @Override
    public INDArray ravel() {

        ensureNotCleanedUp();
        INDArray ret = Nd4j.create(length, ordering);

        int dimension = shape.length == 2 ? 1 : shape.length;
        int count = 0;
        for (int i = 0; i < vectorsAlongDimension(dimension); i++) {
            INDArray vec = vectorAlongDimension(i, dimension);
            for (int j = 0; j < vec.length(); j++) {
                ret.putScalar(count++, vec.getDouble(j));
            }
        }

        return ret;

    }

    /**
     * Flattens the array for linear indexing
     *
     * @return the flattened version of this array
     */
    @Override
    public void sliceVectors(List<INDArray> list) {
        ensureNotCleanedUp();
        if (isVector())
            list.add(this);
        else {
            for (int i = 0; i < slices(); i++) {
                slice(i).sliceVectors(list);
            }
        }
    }

    /**
     * Reshape the matrix. Number of elements must not change.
     *
     * @param newRows
     * @param newColumns
     */
    @Override
    public INDArray reshape(int newRows, int newColumns) {
        ensureNotCleanedUp();
        assert newRows * newColumns == length : "Illegal new shape " + newRows + " x " + newColumns;
        return reshape(new int[]{newRows, newColumns});
    }

    /**
     * Get the specified column
     *
     * @param c
     */
    @Override
    public INDArray getColumn(int c) {
        ensureNotCleanedUp();
        if (shape.length == 2) {
            if (ordering == NDArrayFactory.C) {
                INDArray ret = Nd4j.create(
                        data,
                        new int[]{shape[0], 1},
                        new int[]{stride[0], 1},
                        offset + c, ordering
                );

                return ret;
            } else {
                INDArray ret = Nd4j.create(
                        data,
                        new int[]{shape[0], 1},
                        new int[]{stride[0], 1},
                        offset + c * rows(), ordering
                );

                return ret;
            }

        } else if (isRowVector()) {
            return Nd4j.scalar(getDouble(c));
        } else if (isColumnVector() && c == 0)
            return this;


        else
            throw new IllegalArgumentException("Unable to getFloat scalar column of non 2d matrix");
    }


    /**
     * Get whole rows from the passed indices.
     *
     * @param rindices
     */
    @Override
    public INDArray getRows(int[] rindices) {
        ensureNotCleanedUp();
        INDArray rows = Nd4j.create(rindices.length, columns());
        for (int i = 0; i < rindices.length; i++) {
            rows.putRow(i, getRow(rindices[i]));
        }
        return rows;
    }

    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    @Override
    public INDArray get(NDArrayIndex... indexes) {
        ensureNotCleanedUp();
        //fill in to match the rest of the dimensions: aka grab all the content
        //in the dimensions not filled in
        //also prune indices greater than the shape to be the shape instead

        indexes = Indices.adjustIndices(shape(), indexes);


        int[] offsets = Indices.offsets(indexes);
        int[] shape = Indices.shape(shape(), indexes);
        //no stride will help here, need to do manually
        if (!Indices.isContiguous(indexes)) {
            INDArray ret = Nd4j.create(shape);
            if (ret.isVector() && isVector()) {
                int[] indices = indexes[0].indices();
                for (int i = 0; i < ret.length(); i++) {
                    ret.putScalar(i, getDouble(indices[i]));
                }

                return ret;
            }
            if (!ret.isVector()) {
                //overrides when shouldn't
                for (int i = 0; i < ret.slices(); i++) {
                    INDArray putSlice = slice(i).get(Arrays.copyOfRange(indexes, 1, indexes.length));
                    ret.putSlice(i, putSlice);

                }
            } else {
                INDArray putSlice = slice(0).get(Arrays.copyOfRange(indexes, 1, indexes.length));
                ret.putSlice(0, putSlice);

            }


            return ret;
        }

        if (ArrayUtil.prod(shape) > length())
            return this;


        int[] strides = null;

        strides = ArrayUtil.copy(stride());

        if (offsets.length != shape.length)
            offsets = Arrays.copyOfRange(offsets, 0, shape.length);

        if (strides.length != shape.length)
            strides = Arrays.copyOfRange(strides, 0, shape.length);

        return subArray(offsets, shape, strides);
    }


    /**
     * Get whole columns from the passed indices.
     *
     * @param cindices
     */
    @Override
    public INDArray getColumns(int[] cindices) {
        ensureNotCleanedUp();
        INDArray rows = Nd4j.create(rows(), cindices.length);
        for (int i = 0; i < cindices.length; i++) {
            rows.putColumn(i, getColumn(cindices[i]));
        }
        return rows;
    }

    /**
     * Get a copy of a row.
     *
     * @param r
     */
    @Override
    public INDArray getRow(int r) {
        ensureNotCleanedUp();
        if (shape.length == 2) {
            if (ordering == NDArrayFactory.C) {
                INDArray ret = Nd4j.create(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset + r * columns(),
                        ordering
                );

                return ret;
            } else if (isColumnVector()) {
                return Nd4j.scalar(getDouble(r));
            } else {
                INDArray ret = Nd4j.create(
                        data,
                        new int[]{shape[1]},
                        new int[]{stride[1]},
                        offset + r,
                        ordering
                );
                return ret;
            }


        } else if (isRowVector() && r == 0)
            return this;


        else
            throw new IllegalArgumentException("Unable to getFloat row of non 2d matrix");
    }

    /**
     * Compare two matrices. Returns true if and only if other is also a
     * DoubleMatrix which has the same size and the maximal absolute
     * difference in matrix elements is smaller than 1e-6.
     *
     * @param o
     */
    @Override
    public boolean equals(Object o) {
        ensureNotCleanedUp();
        INDArray n = null;

        if (!(o instanceof INDArray))
            return false;

        if (n == null)
            n = (INDArray) o;

        //epsilon equals
        if (isScalar() && n.isScalar()) {
            if (data.dataType() == DataBuffer.FLOAT) {
                double val = getDouble(0);
                double val2 = n.getDouble(0);
                return Math.abs(val - val2) < 1e-6;
            } else {
                double val = getDouble(0);
                double val2 = n.getDouble(0);
                return Math.abs(val - val2) < 1e-6;
            }

        } else if (isVector() && n.isVector()) {
            for (int i = 0; i < length; i++) {
                if (data.dataType() == DataBuffer.FLOAT) {
                    double curr = getDouble(i);
                    double comp = n.getDouble(i);
                    if (Math.abs(curr - comp) > 1e-3)
                        return false;
                } else {
                    double curr = getDouble(i);
                    double comp = n.getDouble(i);
                    if (Math.abs(curr - comp) > 1e-3)
                        return false;
                }
            }

            return true;

        }


        if (!Shape.shapeEquals(shape(), n.shape()))
            return false;


        if (slices() != n.slices())
            return false;

        for (int i = 0; i < slices(); i++) {
            INDArray slice = slice(i);
            INDArray nSlice = n.slice(i);

            if (!slice.equals(nSlice))
                return false;
        }

        return true;

    }


    /**
     * Returns the shape(dimensions) of this array
     *
     * @return the shape of this matrix
     */
    public int[] shape() {
        ensureNotCleanedUp();
        return shape;
    }

    /**
     * Returns the stride(indices along the linear index for which each slice is accessed) of this array
     *
     * @return the stride of this array
     */
    @Override
    public int[] stride() {
        ensureNotCleanedUp();
        return stride;
    }


    @Override
    public int offset() {
        ensureNotCleanedUp();
        return offset;
    }

    @Override
    public char ordering() {
        ensureNotCleanedUp();
        return ordering;
    }

    /**
     * Returns the size of this array
     * along a particular dimension
     *
     * @param dimension the dimension to return from
     * @return the shape of the specified dimension
     */
    @Override
    public int size(int dimension) {
        ensureNotCleanedUp();
        if (isScalar()) {
            if (dimension == 0)
                return length;
            else
                throw new IllegalArgumentException("Illegal dimension for scalar " + dimension);
        } else if (isVector()) {
            if (dimension == 0)
                return length;
            else if (dimension == 1)
                return 1;
        }

        return shape[dimension];
    }

    /**
     * Returns the total number of elements in the ndarray
     *
     * @return the number of elements in the ndarray
     */
    @Override
    public int length() {
        return length;
    }

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public INDArray broadcast(int[] shape) {
        ensureNotCleanedUp();
        if (Shape.shapeEquals(shape, shape()))
            return this;
        boolean compatible = true;
        int count = shape.length - 1;
        int thisCount = this.shape.length - 1;
        for (int i = shape.length - 1; i > 0; i--) {
            if (count < 0 || thisCount < 0)
                break;
            if (shape[count] != shape()[thisCount] && shape[count] != 1 && shape()[thisCount] != 1) {
                compatible = false;
                break;
            }

            count--;
            thisCount--;
        }

        if (!compatible)
            throw new IllegalArgumentException("Incompatible broadcast from " + Arrays.toString(shape()) + " to " + Arrays.toString(shape));
        //broadcast to shape
        if (isScalar()) {
            INDArray ret = Nd4j.valueArrayOf(shape, getDouble(0));
            return ret;
        } else if (isColumnVector() && Shape.isMatrix(shape)) {
            INDArray ret = Nd4j.create(shape);
            for (int i = 0; i < ret.columns(); i++)
                ret.putColumn(i, this.dup());


            return ret;
        }

        else if(isRowVector()) {
            INDArray ret = Nd4j.create(shape);
            for(int i = 0; i < ret.rows(); i++)
                ret.putRow(i,this.dup());
            return ret;
        }



        int[] retShape = new int[shape.length];

        for (int i = 0; i < retShape.length; i++) {
            if (i < shape().length)
                retShape[i] = Math.max(shape[i], shape()[i]);
            else
                retShape[i] = shape[i];
        }

        INDArray ret = Nd4j.create(retShape);
        INDArray linear = ret.linearView();
        INDArray thisLinear = linearView();
        int bufferIdx = 0;
        for (int i = 0; i < ret.length(); i++) {
            linear.putScalar(i, thisLinear.getDouble(bufferIdx));
            bufferIdx++;
            if (bufferIdx >= length())
                bufferIdx = 0;
        }

        return ret;


    }


    /**
     * Dimshuffle: an extension of permute that adds the ability
     * to broadcast various dimensions.
     * <p/>
     * See theano for more examples.
     * This will only accept integers and xs.
     * <p/>
     * An x indicates a dimension should be broadcasted rather than permuted.
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable) {
        ensureNotCleanedUp();
        assert broadCastable.length == shape.length : "The broadcastable dimensions must be the same length as the current shape";

        boolean broadcast = false;
        Set<Object> set = new HashSet<>();
        for (int i = 0; i < rearrange.length; i++) {
            set.add(rearrange[i]);
            if (rearrange[i] instanceof Integer) {
                Integer j = (Integer) rearrange[i];
                if (j >= broadCastable.length)
                    throw new IllegalArgumentException("Illegal dimension, dimension must be < broadcastable.length (aka the real dimensions");
            } else if (rearrange[i] instanceof Character) {
                Character c = (Character) rearrange[i];
                if (c != 'x')
                    throw new IllegalArgumentException("Illegal input: Must be x");
                broadcast = true;

            } else
                throw new IllegalArgumentException("Only characters and integers allowed");
        }

        //just do permute
        if (!broadcast) {
            int[] ret = new int[rearrange.length];
            for (int i = 0; i < ret.length; i++)
                ret[i] = (Integer) rearrange[i];
            return permute(ret);
        } else {
            List<Integer> drop = new ArrayList<>();
            for (int i = 0; i < broadCastable.length; i++) {
                if (!set.contains(i)) {
                    if (broadCastable[i])
                        drop.add(i);
                    else
                        throw new IllegalArgumentException("We can't drop the given dimension because its not broadcastable");
                }

            }


            //list of dimensions to keep
            int[] shuffle = new int[broadCastable.length];
            int count = 0;
            for (int i = 0; i < rearrange.length; i++) {
                if (rearrange[i] instanceof Integer) {
                    shuffle[count++] = (Integer) rearrange[i];
                }
            }


            List<Integer> augment = new ArrayList<>();
            for (int i = 0; i < rearrange.length; i++) {
                if (rearrange[i] instanceof Character)
                    augment.add(i);
            }

            Integer[] augmentDims = augment.toArray(new Integer[1]);

            count = 0;

            int[] newShape = new int[shuffle.length + drop.size()];
            for (int i = 0; i < newShape.length; i++) {
                if (i < shuffle.length) {
                    newShape[count++] = shuffle[i];
                } else
                    newShape[count++] = drop.get(i);
            }


            INDArray ret = permute(newShape);
            List<Integer> newDims = new ArrayList<>();
            int[] shape = Arrays.copyOfRange(ret.shape(), 0, shuffle.length);
            for (int i = 0; i < shape.length; i++) {
                newDims.add(shape[i]);
            }

            for (int i = 0; i < augmentDims.length; i++) {
                newDims.add(augmentDims[i], 1);
            }

            int[] toReshape = ArrayUtil.toArray(newDims);


            ret = ret.reshape(toReshape);
            return ret;

        }


    }

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.htsliceml
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public INDArray permute(int[] rearrange) {
        ensureNotCleanedUp();
        if (rearrange.length != shape.length)
            return dup();

        checkArrangeArray(rearrange);

        int[] newShape = doPermuteSwap(shape, rearrange);
        int[] newStride = doPermuteSwap(stride, rearrange);

        INDArray value = Nd4j.create(
                data(),
                newShape,
                newStride,
                offset,
                ordering);
        return value;

    }


    protected void copyRealTo(INDArray arr) {
        ensureNotCleanedUp();
        INDArray flattened = linearView();
        INDArray arrLinear = arr.linearView();
        for (int i = 0; i < flattened.length(); i++) {
            arrLinear.putScalar(i, flattened.getDouble(i));
        }

    }

    protected int[] doPermuteSwap(int[] shape, int[] rearrange) {
        ensureNotCleanedUp();
        int[] ret = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            ret[i] = shape[rearrange[i]];
        }
        return ret;
    }


    protected void checkArrangeArray(int[] arr) {
        ensureNotCleanedUp();
        assert arr.length == shape.length : "Invalid rearrangement: number of arrangement != shape";
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] >= arr.length)
                throw new IllegalArgumentException("The specified dimensions can't be swapped. Given element " + i + " was >= number of dimensions");
            if (arr[i] < 0)
                throw new IllegalArgumentException("Invalid dimension: " + i + " : negative value");


        }

        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length; j++) {
                if (i != j && arr[i] == arr[j])
                    throw new IllegalArgumentException("Permute array must have unique elements");
            }
        }

    }

    /**
     * Checks whether the matrix is a vector.
     */
    @Override
    public boolean isVector() {
        ensureNotCleanedUp();
        return shape.length == 1
                ||
                shape.length == 2 && (shape[0] == 1 || shape[1] == 1) && !isScalar();
    }

    @Override
    public boolean isSquare() {
        ensureNotCleanedUp();
        return isMatrix() && rows() == columns();
    }

    /**
     * Checks whether the matrix is a row vector.
     */
    @Override
    public boolean isRowVector() {
        ensureNotCleanedUp();
        if (shape().length == 1)
            return true;

        if (isVector())
            return shape()[0] == 1;

        return false;
    }

    /**
     * Checks whether the matrix is a column vector.
     */
    @Override
    public boolean isColumnVector() {
        ensureNotCleanedUp();
        if (shape().length == 1)
            return false;

        if (isVector())
            return shape()[1] == 1;

        return false;

    }

    /**
     * Generate string representation of the matrix.
     */
    @Override
    public String toString() {
        ensureNotCleanedUp();
        if (isScalar()) {
            return element().toString();
        }


        if (isVector()) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            if (rows > 1)
                sb.append("[");
            int numElementsToPrint = Nd4j.MAX_ELEMENTS_PER_SLICE < 0 ? length : Nd4j.MAX_ELEMENTS_PER_SLICE;
            for (int i = 0; i < length; i++) {
                sb.append(getDouble(i));
                if (i < length - 1) {
                    if (rows > 1)
                        sb.append(",\n  ");
                    else
                        sb.append(", ");
                }
                if (i >= numElementsToPrint) {
                    int numElementsLeft = length - i;
                    //set towards the end of the buffer
                    if (numElementsLeft > numElementsToPrint) {
                        i += numElementsLeft - numElementsToPrint - 1;
                        sb.append(", ..., ");
                    }
                }

            }
            if (rows > 1)
                sb.append("]");
            sb.append("]\n ");
            return sb.toString();
        }


        StringBuilder sb = new StringBuilder();
        int length = shape[0];
        sb.append("[");
        if (length > 0) {
            sb.append(slice(0).toString());
            int slices = Nd4j.MAX_SLICES_TO_PRINT > 0 ? Nd4j.MAX_SLICES_TO_PRINT : slices();
            if (slices > slices())
                slices = slices();
            for (int i = 1; i < slices; i++) {
                sb.append(slice(i).toString());
                if (i < length - 1)
                    sb.append(" ,");

            }


        }
        sb.setLength(sb.length() - 2);
        sb.append("]\n");
        return sb.toString();
    }


    /**
     * Returns a scalar (individual element)
     * of a scalar ndarray
     *
     * @return the individual item in this ndarray
     */
    @Override
    public Object element() {
        ensureNotCleanedUp();
        if (!isScalar())
            throw new IllegalStateException("Unable to retrieve element from non scalar matrix");
        if (data.dataType() == DataBuffer.FLOAT)
            return data.getFloat(offset);
        return data.getDouble(offset);
    }


    @Override
    public IComplexNDArray rdiv(IComplexNumber n) {
        return dup().rdivi(n);
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n) {
        return rdivi(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n) {
        return dup().rsubi(n);
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n) {
        return rsubi(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray div(IComplexNumber n) {
        return dup().divi(n);
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n) {
        ensureNotCleanedUp();
        return divi(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray mul(IComplexNumber n) {
        return dup().muli(n);
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n) {
        ensureNotCleanedUp();
        return muli(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray sub(IComplexNumber n) {
        return dup().subi(n);
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n) {
        return subi(n, Nd4j.createComplex(shape()));
    }

    @Override
    public IComplexNDArray add(IComplexNumber n) {
        return dup().addi(n);
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n) {
        ensureNotCleanedUp();
        return addi(n, Nd4j.createComplex(shape()));

    }

    @Override
    public IComplexNDArray rdiv(IComplexNumber n, IComplexNDArray result) {
        return dup().rdivi(n, result);
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result) {
        return Nd4j.createComplex(this).rdivi(n, result);

    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n, IComplexNDArray result) {
        return dup().rsubi(n, result);
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result) {
        ensureNotCleanedUp();
        return Nd4j.createComplex(this).rsubi(n, result);
    }

    @Override
    public IComplexNDArray div(IComplexNumber n, IComplexNDArray result) {
        return dup().divi(n, result);
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n, IComplexNDArray result) {
        ensureNotCleanedUp();
        return Nd4j.createComplex(this).divi(n, result);

    }

    @Override
    public IComplexNDArray mul(IComplexNumber n, IComplexNDArray result) {
        return dup().muli(n, result);
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n, IComplexNDArray result) {
        ensureNotCleanedUp();
        return Nd4j.createComplex(this).muli(n, result);

    }

    @Override
    public IComplexNDArray sub(IComplexNumber n, IComplexNDArray result) {
        return dup().subi(n, result);
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n, IComplexNDArray result) {
        ensureNotCleanedUp();
        return Nd4j.createComplex(this).subi(n, result);

    }

    @Override
    public IComplexNDArray add(IComplexNumber n, IComplexNDArray result) {
        return dup().addi(n, result);
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n, IComplexNDArray result) {
        ensureNotCleanedUp();
        return Nd4j.createComplex(this).addi(n, result);

    }
}