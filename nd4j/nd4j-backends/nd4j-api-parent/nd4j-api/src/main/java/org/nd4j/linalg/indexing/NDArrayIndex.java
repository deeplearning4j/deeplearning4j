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

package org.nd4j.linalg.indexing;

import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * NDArray indexing
 *
 * @author Adam Gibson
 */
@Slf4j
public class NDArrayIndex implements INDArrayIndex {

    private long[] indices = new long[1];
    private boolean isInterval = false;
    private static NDArrayIndexEmpty EMPTY = new NDArrayIndexEmpty();
    private static NewAxis NEW_AXIS = new NewAxis();


    /**
     * Returns a point index
     * @param point the point index
     * @return the point index based
     * on the specified point
     */
    public static INDArrayIndex point(long point) {
        return new PointIndex(point);
    }

    /**
     * Add indexes for the given shape
     * @param shape the shape ot convert to indexes
     * @return the indexes for the given shape
     */
    public static INDArrayIndex[] indexesFor(long... shape) {
        INDArrayIndex[] ret = new INDArrayIndex[shape.length];
        for (int i = 0; i < shape.length; i++) {
            ret[i] = NDArrayIndex.point(shape[i]);
        }

        return ret;
    }

    /**
     * Compute the offset given an array of offsets.
     * The offset is computed(for both fortran an d c ordering) as:
     * sum from i to n - 1 o[i] * s[i]
     * where i is the index o is the offset and s is the stride
     * Notice the -1 at the end.
     * @param arr the array to compute the offset for
     * @param offsets the offsets for each dimension
     * @return the offset that should be used for indexing
     */
    public static long offset(INDArray arr, long... offsets) {
        return offset(arr.stride(), offsets);
    }

    /**
     * Compute the offset given an array of offsets.
     * The offset is computed(for both fortran an d c ordering) as:
     * sum from i to n - 1 o[i] * s[i]
     * where i is the index o is the offset and s is the stride
     * Notice the -1 at the end.
     * @param arr the array to compute the offset for
     * @param indices the offsets for each dimension
     * @return the offset that should be used for indexing
     */
    public static long offset(INDArray arr, INDArrayIndex... indices) {
        return offset(arr.stride(), Indices.offsets(arr.shape(), indices));
    }

    /**
     * Set the shape and stride for
     * new axes based dimensions
     * @param arr the array to update
     *            the shape/strides for
     * @param indexes the indexes to update based on
     */
    public static void updateForNewAxes(INDArray arr, INDArrayIndex... indexes) {
        int numNewAxes = NDArrayIndex.numNewAxis(indexes);
        if (numNewAxes >= 1 && (indexes[0].length() > 1 || indexes[0] instanceof NDArrayIndexAll)) {
            List<Long> newShape = new ArrayList<>();
            List<Long> newStrides = new ArrayList<>();
            int currDimension = 0;
            for (int i = 0; i < indexes.length; i++) {
                if (indexes[i] instanceof NewAxis) {
                    newShape.add(1L);
                    newStrides.add(0L);
                } else {
                    newShape.add(arr.size(currDimension));
                    newStrides.add(arr.size(currDimension));
                    currDimension++;
                }
            }

            while (currDimension < arr.rank()) {
                newShape.add((long) currDimension);
                newStrides.add((long) currDimension);
                currDimension++;
            }

            long[] newShapeArr = Longs.toArray(newShape);
            long[] newStrideArr = Longs.toArray(newStrides);

            // FIXME: this is wrong, it breaks shapeInfo immutability
            arr.setShape(newShapeArr);
            arr.setStride(newStrideArr);


        } else {
            if (numNewAxes > 0) {
                long[] newShape = Longs.concat(ArrayUtil.toLongArray(ArrayUtil.nTimes(numNewAxes, 1)), arr.shape());
                long[] newStrides = Longs.concat(new long[numNewAxes], arr.stride());
                arr.setShape(newShape);
                arr.setStride(newStrides);
            }
        }

    }



    /**
     * Compute the offset given an array of offsets.
     * The offset is computed(for both fortran an d c ordering) as:
     * sum from i to n - 1 o[i] * s[i]
     * where i is the index o is the offset and s is the stride
     * Notice the -1 at the end.
     * @param strides the strides to compute the offset for
     * @param offsets the offsets for each dimension
     * @return the offset that should be used for indexing
     */
    public static long offset(long[] strides, long[] offsets) {
        int ret = 0;

        if (ArrayUtil.prod(offsets) == 1) {
            for (int i = 0; i < offsets.length; i++) {
                ret += offsets[i] * strides[i];
            }
        } else {
            for (int i = 0; i < offsets.length; i++) {
                ret += offsets[i] * strides[i];
            }

        }

        return ret;
    }

    public static long offset(int[] strides, long[] offsets) {
        int ret = 0;

        if (ArrayUtil.prodLong(offsets) == 1) {
            for (int i = 0; i < offsets.length; i++) {
                ret += offsets[i] * strides[i];
            }
        } else {
            for (int i = 0; i < offsets.length; i++) {
                ret += offsets[i] * strides[i];
            }

        }

        return ret;
    }


    /**
     * Repeat a copy of copy n times
     * @param copy the ndarray index to copy
     * @param n the number of times to copy
     * @return an array of length n containing copies of
     * the given ndarray index
     */
    public static INDArrayIndex[] nTimes(INDArrayIndex copy, int n) {
        INDArrayIndex[] ret = new INDArrayIndex[n];
        for (int i = 0; i < n; i++) {
            ret[i] = copy;
        }

        return ret;
    }

    /**
     * NDArrayIndexing based on the given
     * indexes
     * @param indices
     */
    public NDArrayIndex(long... indices) {
        this.indices = indices;
    }

    /**
     * Represents collecting no elements
     *
     * @return an ndarray index
     * meaning collect
     * no elements
     */
    public static INDArrayIndex empty() {
        return EMPTY;
    }

    /**
     * Represents collecting all elements
     *
     * @return an ndarray index
     * meaning collect
     * all elements
     */
    public static INDArrayIndex all() {
        return new NDArrayIndexAll(true);
    }

    /**
     * Returns an instance of {@link SpecifiedIndex}.
     * Note that SpecifiedIndex works differently than the other indexing options, in that it always returns a copy
     * of the (subset of) the underlying array, for get operations. This means that INDArray.get(..., indices(x,y,z), ...)
     * will be a copy of the relevant subset of the array.
     * @param indices Indices to get
     */
    public static INDArrayIndex indices(long... indices){
        return new SpecifiedIndex(indices);
    }


    /**
     * Represents adding a new dimension
     * @return the indexing for
     * adding a new dimension
     */
    public static INDArrayIndex newAxis() {
        return NEW_AXIS;
    }

    /**
     * Given an all index and
     * the intended indexes, return an
     * index array containing a combination of all elements
     * for slicing and overriding particular indexes where necessary
     * @param arr the array to resolve indexes for
     * @param intendedIndexes the indexes specified by the user
     * @return the resolved indexes (containing all where nothing is specified, and the intended index
     * for a particular dimension otherwise)
     */
    public static INDArrayIndex[] resolve(INDArray arr, INDArrayIndex... intendedIndexes) {
        return resolve(NDArrayIndex.allFor(arr), intendedIndexes);
    }

    /**
     * Number of point indexes
     * @param indexes the indexes
     *                to count for points
     * @return the number of point indexes
     * in the array
     */
    public static int numPoints(INDArrayIndex... indexes) {
        int ret = 0;
        for (int i = 0; i < indexes.length; i++)
            if (indexes[i] instanceof PointIndex)
                ret++;
        return ret;
    }

    /**
     * Given an all index and
     * the intended indexes, return an
     * index array containing a combination of all elements
     * for slicing and overriding particular indexes where necessary
     * @param shapeInfo the index containing all elements
     * @param intendedIndexes the indexes specified by the user
     * @return the resolved indexes (containing all where nothing is specified, and the intended index
     * for a particular dimension otherwise)
     */
    public static INDArrayIndex[] resolve(DataBuffer shapeInfo, INDArrayIndex... intendedIndexes) {
        int numSpecified = 0;
        for (int i = 0; i < intendedIndexes.length; i++) {
            if (intendedIndexes[i] instanceof SpecifiedIndex)
                numSpecified++;
        }

        if (numSpecified > 0) {
            DataBuffer shape = Shape.shapeOf(shapeInfo);
            INDArrayIndex[] ret = new INDArrayIndex[intendedIndexes.length];
            for (int i = 0; i < intendedIndexes.length; i++) {
                if (intendedIndexes[i] instanceof SpecifiedIndex)
                    ret[i] = intendedIndexes[i];
                else {
                    if (intendedIndexes[i] instanceof NDArrayIndexAll) {
                        // FIXME: LONG
                        SpecifiedIndex specifiedIndex = new SpecifiedIndex(ArrayUtil.range(0L, (long) shape.getInt(i)));
                        ret[i] = specifiedIndex;
                    } else if (intendedIndexes[i] instanceof NDArrayIndexEmpty) {
                        ret[i] = new SpecifiedIndex(new long[0]);
                    } else if (intendedIndexes[i] instanceof IntervalIndex) {
                        IntervalIndex intervalIndex = (IntervalIndex) intendedIndexes[i];
                        ret[i] = new SpecifiedIndex(ArrayUtil.range(intervalIndex.begin, intervalIndex.end(),
                                        intervalIndex.stride()));
                    } else if(intendedIndexes[i] instanceof PointIndex){
                        ret[i] = intendedIndexes[i];
                    }
                }
            }

            return ret;
        }


        /**
         * If it's a vector and index asking
         * for a scalar just return the array
         */
        int rank = Shape.rank(shapeInfo);
        DataBuffer shape = Shape.shapeOf(shapeInfo);
        if (intendedIndexes.length >= rank || Shape.isVector(shapeInfo) && intendedIndexes.length == 1) {
            if (Shape.isRowVectorShape(shapeInfo) && intendedIndexes.length == 1) {
                INDArrayIndex[] ret = new INDArrayIndex[2];
                ret[0] = NDArrayIndex.point(0);
                int size;
                if (1 == shape.getInt(0) && rank == 2)
                    size = shape.getInt(1);
                else
                    size = shape.getInt(0);
                ret[1] = validate(size, intendedIndexes[0]);
                return ret;
            }
            List<INDArrayIndex> retList = new ArrayList<>(intendedIndexes.length);
            for (int i = 0; i < intendedIndexes.length; i++) {
                if (i < rank)
                    retList.add(validate(shape.getInt(i), intendedIndexes[i]));
                else
                    retList.add(intendedIndexes[i]);
            }
            return retList.toArray(new INDArrayIndex[retList.size()]);
        }

        List<INDArrayIndex> retList = new ArrayList<>(intendedIndexes.length + 1);
        int numNewAxes = 0;

        if (Shape.isMatrix(shape) && intendedIndexes.length == 1) {
            retList.add(validate(shape.getInt(0), intendedIndexes[0]));
            retList.add(NDArrayIndex.all());
        } else {
            for (int i = 0; i < intendedIndexes.length; i++) {
                retList.add(validate(shape.getInt(i), intendedIndexes[i]));
                if (intendedIndexes[i] instanceof NewAxis)
                    numNewAxes++;
            }
        }

        int length = rank + numNewAxes;
        //fill the rest with all
        while (retList.size() < length)
            retList.add(NDArrayIndex.all());

        return retList.toArray(new INDArrayIndex[retList.size()]);
    }

    /**
     * Given an all index and
     * the intended indexes, return an
     * index array containing a combination of all elements
     * for slicing and overriding particular indexes where necessary
     * @param shape the index containing all elements
     * @param intendedIndexes the indexes specified by the user
     * @return the resolved indexes (containing all where nothing is specified, and the intended index
     * for a particular dimension otherwise)
     */
    public static INDArrayIndex[] resolve(int[] shape, INDArrayIndex... intendedIndexes) {
        return resolve(ArrayUtil.toLongArray(shape), intendedIndexes);
    }

    public static INDArrayIndex[] resolve(long[] shape, INDArrayIndex... intendedIndexes) {
        /**
         * If it's a vector and index asking for a scalar just return the array
         */
        if (intendedIndexes.length >= shape.length || Shape.isVector(shape) && intendedIndexes.length == 1) {
            if (Shape.isRowVectorShape(shape) && intendedIndexes.length == 1) {
                INDArrayIndex[] ret = new INDArrayIndex[2];
                ret[0] = NDArrayIndex.point(0);
                long size;
                if (1 == shape[0] && shape.length == 2)
                    size = shape[1];
                else
                    size = shape[0];
                ret[1] = validate(size, intendedIndexes[0]);
                return ret;
            }
            List<INDArrayIndex> retList = new ArrayList<>(intendedIndexes.length);
            for (int i = 0; i < intendedIndexes.length; i++) {
                if (i < shape.length)
                    retList.add(validate(shape[i], intendedIndexes[i]));
                else
                    retList.add(intendedIndexes[i]);
            }
            return retList.toArray(new INDArrayIndex[retList.size()]);
        }

        List<INDArrayIndex> retList = new ArrayList<>(intendedIndexes.length + 1);
        int numNewAxes = 0;

        if (Shape.isMatrix(shape) && intendedIndexes.length == 1) {
            retList.add(validate(shape[0], intendedIndexes[0]));
            retList.add(NDArrayIndex.all());
        } else {
            for (int i = 0; i < intendedIndexes.length; i++) {
                retList.add(validate(shape[i], intendedIndexes[i]));
                if (intendedIndexes[i] instanceof NewAxis)
                    numNewAxes++;
            }
        }

        int length = shape.length + numNewAxes;
        //fill the rest with all
        while (retList.size() < length)
            retList.add(NDArrayIndex.all());



        return retList.toArray(new INDArrayIndex[retList.size()]);
    }

    protected static INDArrayIndex validate(long size, INDArrayIndex index) {
        if ((index instanceof IntervalIndex || index instanceof PointIndex) && size <= index.current() && size > 1)
            throw new IllegalArgumentException("NDArrayIndex is out of range. Beginning index: " + index.current()
                            + " must be less than its size: " + size);
        if (index instanceof IntervalIndex && size < index.end()) {
            long begin = ((IntervalIndex) index).begin;
            index = NDArrayIndex.interval(begin, index.stride(), size);
        }
        return index;
    }


    /**
     * Given an all index and
     * the intended indexes, return an
     * index array containing a combination of all elements
     * for slicing and overriding particular indexes where necessary
     * @param allIndex the index containing all elements
     * @param intendedIndexes the indexes specified by the user
     * @return the resolved indexes (containing all where nothing is specified, and the intended index
     * for a particular dimension otherwise)
     */
    public static INDArrayIndex[] resolve(INDArrayIndex[] allIndex, INDArrayIndex... intendedIndexes) {

        int numNewAxes = numNewAxis(intendedIndexes);
        INDArrayIndex[] all = new INDArrayIndex[allIndex.length + numNewAxes];
        Arrays.fill(all, NDArrayIndex.all());
        for (int i = 0; i < allIndex.length; i++) {
            //collapse single length indexes in to point indexes
            if (i >= intendedIndexes.length)
                break;

            if (intendedIndexes[i] instanceof NDArrayIndex) {
                NDArrayIndex idx = (NDArrayIndex) intendedIndexes[i];
                if (idx.indices.length == 1)
                    intendedIndexes[i] = new PointIndex(idx.indices[0]);
            }
            all[i] = intendedIndexes[i];
        }

        return all;
    }

    /**
     * Given an array of indexes
     * return the number of new axis elements
     * in teh array
     * @param axes the indexes to get the number
     *             of new axes for
     * @return the number of new axis elements in the given array
     */
    public static int numNewAxis(INDArrayIndex... axes) {
        int ret = 0;
        for (INDArrayIndex index : axes)
            if (index instanceof NewAxis)
                ret++;
        return ret;
    }


    /**
     * Generate an all index
     * equal to the rank of the given array
     * @param arr the array to generate the all index for
     * @return an ndarray index array containing of length
     * arr.rank() containing all elements
     */
    public static INDArrayIndex[] allFor(INDArray arr) {
        INDArrayIndex[] ret = new INDArrayIndex[arr.rank()];
        for (int i = 0; i < ret.length; i++)
            ret[i] = NDArrayIndex.all();

        return ret;
    }

    /**
     * Creates an index covering the given shape
     * (for each dimension 0,shape[i])
     * @param shape the shape to cover
     * @return the ndarray indexes to cover
     */
    public static INDArrayIndex[] createCoveringShape(int[] shape) {
        INDArrayIndex[] ret = new INDArrayIndex[shape.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = NDArrayIndex.interval(0, shape[i]);
        }
        return ret;
    }

    public static INDArrayIndex[] createCoveringShape(long[] shape) {
        INDArrayIndex[] ret = new INDArrayIndex[shape.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = NDArrayIndex.interval(0, shape[i]);
        }
        return ret;
    }


    /**
     * Create a range based on the given indexes.
     * This is similar to create covering shape in that it approximates
     * the length of each dimension (ignoring elements) and
     * reproduces an index of the same dimension and length.
     *
     * @param indexes the indexes to create the range for
     * @return the index ranges.
     */
    public static INDArrayIndex[] rangeOfLength(INDArrayIndex[] indexes) {
        INDArrayIndex[] indexesRet = new INDArrayIndex[indexes.length];
        for (int i = 0; i < indexes.length; i++)
            indexesRet[i] = NDArrayIndex.interval(0, indexes[i].length());
        return indexesRet;
    }



    /**
     * Create from a matrix. The rows are the indices
     * The columns are the individual element in each ndarrayindex
     *
     * @param index the matrix to getFloat indices from
     * @return the indices to getFloat
     */
    public static INDArrayIndex[] create(INDArray index) {

        if (index.isMatrix()) {

            if (index.rows() > Integer.MAX_VALUE)
                throw new ND4JArraySizeException();

            NDArrayIndex[] ret = new NDArrayIndex[(int) index.rows()];
            for (int i = 0; i < index.rows(); i++) {
                INDArray row = index.getRow(i);
                val nums = new long[(int) index.getRow(i).columns()];
                for (int j = 0; j < row.columns(); j++) {
                    nums[j] = (int) row.getFloat(j);
                }

                NDArrayIndex idx = new NDArrayIndex(nums);
                ret[i] = idx;

            }


            return ret;

        } else if (index.isVector()) {
            long[] indices = NDArrayUtil.toLongs(index);
            return new NDArrayIndex[] {new NDArrayIndex(indices)};
        }


        throw new IllegalArgumentException("Passed in ndarray must be a matrix or a vector");

    }

    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     *
     * @param begin the begin
     * @param stride  the stride at which to increment
     * @param end   the end index
     * @param max the max length for this domain
     * @return the interval
     */
    public static INDArrayIndex interval(long begin, long stride, long end,long max) {
        if(begin < 0) {
            begin += max;
        }

        if(end < 0) {
            end += max;
        }

        if (Math.abs(begin - end) < 1)
            end++;
        if (stride > 1 && Math.abs(begin - end) == 1) {
            end *= stride;
        }
        return interval(begin, stride, end, false);
    }

    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     *
     * @param begin the begin
     * @param stride  the stride at which to increment
     * @param end   the end index
     * @return the interval
     */
    public static INDArrayIndex interval(long begin, long stride, long end) {
        if (Math.abs(begin - end) < 1)
            end++;
        if (stride > 1 && Math.abs(begin - end) == 1) {
            end *= stride;
        }
        return interval(begin, stride, end, false);
    }

    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     *
     * @param begin     the begin
     * @param stride the stride at which to increment
     * @param end       the end index
     * @param inclusive whether the end should be inclusive or not
     * @return the interval
     */
    public static INDArrayIndex interval(int begin, int stride, int end, boolean inclusive) {
        Preconditions.checkArgument(begin <= end, "Beginning index (%s) in range must be less than or equal to end (%s)", begin, end);
        INDArrayIndex index = new IntervalIndex(inclusive, stride);
        index.init(begin, end);
        return index;
    }



    public static INDArrayIndex interval(long begin, long stride, long end,long max, boolean inclusive) {
        Preconditions.checkArgument(begin <= end, "Beginning index (%s) in range must be less than or equal to end (%s)", begin, end);
        INDArrayIndex index = new IntervalIndex(inclusive, stride);
        index.init(begin, end);
        return index;
    }


    public static INDArrayIndex interval(long begin, long stride, long end, boolean inclusive) {
        Preconditions.checkArgument(begin <= end, "Beginning index (%s) in range must be less than or equal to end (%s)", begin, end);
        INDArrayIndex index = new IntervalIndex(inclusive, stride);
        index.init(begin, end);
        return index;
    }


    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     *
     * @param begin the begin
     * @param end   the end index
     * @return the interval
     */
    public static INDArrayIndex interval(int begin, int end) {
        return interval(begin, 1, end, false);
    }

    public static INDArrayIndex interval(long begin, long end) {
        return interval(begin, 1, end, false);
    }

    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     *
     * @param begin     the begin
     * @param end       the end index
     * @param inclusive whether the end should be inclusive or not
     * @return the interval
     */
    public static INDArrayIndex interval(int begin, int end, boolean inclusive) {
        return interval(begin, 1, end, inclusive);
    }

    @Override
    public long end() {
        if (indices != null && indices.length > 0)
            return indices[indices.length - 1];
        return 0;
    }

    @Override
    public long offset() {
        if (indices.length < 1)
            return 0;
        return indices[0];
    }

    /**
     * Returns the length of the indices
     *
     * @return the length of the range
     */
    @Override
    public long length() {
        return indices.length;
    }

    @Override
    public long stride() {
        return 1;
    }

    @Override
    public long current() {
        return 0;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public long next() {
        return 0;
    }

    @Override
    public void reverse() {
        ArrayUtil.reverse(indices);
    }

    @Override
    public String toString() {
        return "NDArrayIndex{" + "indices=" + Arrays.toString(indices) + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof INDArrayIndex))
            return false;

        NDArrayIndex that = (NDArrayIndex) o;

        if (!Arrays.equals(indices, that.indices))
            return false;
        return true;
    }


    @Override
    public int hashCode() {
        return Arrays.hashCode(indices);
    }

    @Override
    public boolean isInterval() {
        return isInterval;
    }

    @Override
    public void setInterval(boolean isInterval) {
        this.isInterval = isInterval;
    }

    @Override
    public void init(INDArray arr, long begin, int dimension) {

    }

    @Override
    public void init(INDArray arr, int dimension) {

    }

    @Override
    public void init(long begin, long end, long max) {

    }

    @Override
    public void init(long begin, long end) {

    }

    @Override
    public void reset() {

    }


}
