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

package org.nd4j.linalg.indexing;

import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * NDArray indexing
 *
 * @author Adam Gibson
 */
public class NDArrayIndex implements INDArrayIndex {

    private int[] indices = new int[1];
    private boolean isInterval = false;
    private static NDArrayIndexEmpty EMPTY = new NDArrayIndexEmpty();
    private static NewAxis NEW_AXIS = new NewAxis();


    /**
     * Returns a point index
     * @param point the point index
     * @return the point index based
     * on the specified point
     */
    public static INDArrayIndex point(int point) {
        return new PointIndex(point);
    }

    /**
     * Add indexes for the given shape
     * @param shape the shape ot convert to indexes
     * @return the indexes for the given shape
     */
    public static INDArrayIndex[] indexesFor(int...shape) {
        INDArrayIndex[] ret = new INDArrayIndex[shape.length];
        for(int i = 0; i < shape.length; i++) {
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
    public static int offset(INDArray arr,int...offsets) {
        return offset(arr.stride(),offsets);
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
    public static int offset(INDArray arr,INDArrayIndex...indices) {
        return offset(arr.stride(),Indices.offsets(arr.shape(), indices));
    }

    /**
     * Set the shape and stride for
     * new axes based dimensions
     * @param arr the array to update
     *            the shape/strides for
     * @param indexes the indexes to update based on
     */
    public static void updateForNewAxes(INDArray arr,INDArrayIndex... indexes) {
        int numNewAxes = NDArrayIndex.numNewAxis(indexes);
        if( numNewAxes >= 1 && (indexes[0].length() > 1 || indexes[0] instanceof NDArrayIndexAll)) {
            List<Integer> newShape = new ArrayList<>();
            List<Integer> newStrides = new ArrayList<>();
            int currDimension = 0;
            for(int i = 0; i < indexes.length; i++) {
                if(indexes[i] instanceof NewAxis) {
                    newShape.add(1);
                    newStrides.add(0);
                }
                else {
                    newShape.add(arr.size(currDimension));
                    newStrides.add(arr.size(currDimension));
                    currDimension++;
                }
            }

            while(currDimension < arr.rank()) {
                newShape.add(currDimension);
                newStrides.add(currDimension);
                currDimension++;
            }

            int[] newShapeArr = Ints.toArray(newShape);
            int[] newStrideArr = Ints.toArray(newStrides);
            arr.setShape(newShapeArr);
            arr.setStride(newStrideArr);


        }
        else {
            if(numNewAxes > 0) {
                int[] newShape = Ints.concat(ArrayUtil.nTimes(numNewAxes, 1),arr.shape());
                int[] newStrides = Ints.concat(new int[numNewAxes],arr.stride());
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
    public static int offset(int[] strides,int[] offsets) {
        int ret = 0;

        if(ArrayUtil.prod(offsets) == 1) {
            for(int i = 0; i < offsets.length ; i++) {
                ret += offsets[i] * strides[i];
            }
        }
        else {
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
    public static INDArrayIndex[] nTimes(INDArrayIndex copy,int n) {
        INDArrayIndex[] ret = new INDArrayIndex[n];
        for(int i = 0; i < n; i++) {
            ret[i] = copy;
        }

        return ret;
    }

    /**
     * NDArrayIndexing based on the given
     * indexes
     * @param indices
     */
    public NDArrayIndex(int... indices) {
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
        return resolve(NDArrayIndex.allFor(arr),intendedIndexes);
    }

    /**
     * Number of point indexes
     * @param indexes the indexes
     *                to count for points
     * @return the number of point indexes
     * in the array
     */
    public static int numPoints(INDArrayIndex...indexes) {
        int ret = 0;
        for(int i = 0; i < indexes.length; i++)
            if(indexes[i] instanceof PointIndex)
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
    public static INDArrayIndex[] resolve(IntBuffer shapeInfo, INDArrayIndex... intendedIndexes) {
        /**
         * If it's a vector and index asking for a scalar just return the array
         */
        int rank = Shape.rank(shapeInfo);
        IntBuffer shape = Shape.shapeOf(shapeInfo);
        if(intendedIndexes.length >= rank || Shape.isVector(shapeInfo) && intendedIndexes.length == 1) {
            if (Shape.isRowVectorShape(shapeInfo) && intendedIndexes.length == 1) {
                INDArrayIndex[] ret = new INDArrayIndex[2];
                ret[0] = NDArrayIndex.point(0);
                int size;
                if(1 == shape.get(0) && rank == 2)
                    size = shape.get(1);
                else
                    size = shape.get(0);
                ret[1] = validate(size , intendedIndexes[0]);
                return ret;
            }
            List<INDArrayIndex> retList = new ArrayList<>(intendedIndexes.length);
            for (int i = 0; i < intendedIndexes.length; i++) {
                if(i < rank)
                    retList.add(validate(shape.get(i), intendedIndexes[i]));
                else
                    retList.add(intendedIndexes[i]);
            }
            return retList.toArray(new INDArrayIndex[retList.size()]);
        }

        List<INDArrayIndex> retList = new ArrayList<>(intendedIndexes.length + 1);
        int numNewAxes = 0;

        if (Shape.isMatrix(shape) && intendedIndexes.length == 1) {
            retList.add(validate(shape.get(0), intendedIndexes[0]));
            retList.add(NDArrayIndex.all());
        }
        else {
            for (int i = 0; i < intendedIndexes.length; i++) {
                retList.add(validate(shape.get(i), intendedIndexes[i]));
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
        /**
         * If it's a vector and index asking for a scalar just return the array
         */
        if(intendedIndexes.length >= shape.length || Shape.isVector(shape) && intendedIndexes.length == 1) {
            if (Shape.isRowVectorShape(shape) && intendedIndexes.length == 1) {
                INDArrayIndex[] ret = new INDArrayIndex[2];
                ret[0] = NDArrayIndex.point(0);
                int size;
                if(1 == shape[0] && shape.length == 2)
                    size = shape[1];
                else
                    size = shape[0];
                ret[1] = validate(size , intendedIndexes[0]);
                return ret;
            }
            List<INDArrayIndex> retList = new ArrayList<>(intendedIndexes.length);
            for (int i = 0; i < intendedIndexes.length; i++) {
                if(i < shape.length)
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
        }
        else {
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

    protected static INDArrayIndex validate(int size, INDArrayIndex index) {
        if ((index instanceof IntervalIndex || index instanceof PointIndex) && size <= index.current() && size > 1)
            throw new IllegalArgumentException("NDArrayIndex is out of range. Beginning index: " + index.current() + " must be less than its size: " + size);
        if (index instanceof IntervalIndex && size < index.end()) {
            int begin = ((IntervalIndex) index).begin;
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
    public static INDArrayIndex[] resolve(INDArrayIndex[] allIndex, INDArrayIndex...intendedIndexes) {

        int numNewAxes = numNewAxis(intendedIndexes);
        INDArrayIndex[] all = new INDArrayIndex[allIndex.length + numNewAxes];
        Arrays.fill(all,NDArrayIndex.all());
        for(int i = 0; i < allIndex.length; i++) {
            //collapse single length indexes in to point indexes
            if (i >= intendedIndexes.length) break;

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
    public static int numNewAxis(INDArrayIndex...axes) {
        int ret = 0;
        for(INDArrayIndex index : axes)
            if(index instanceof NewAxis)
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
        for(int i = 0; i < ret.length; i++)
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
        for(int i = 0; i < ret.length; i++) {
            ret[i] = NDArrayIndex.interval(0,shape[i]);
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
        for(int i = 0; i < indexes.length; i++)
            indexesRet[i] = NDArrayIndex.interval(0,indexes[i].length());
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

            NDArrayIndex[] ret = new NDArrayIndex[index.rows()];
            for (int i = 0; i < index.rows(); i++) {
                INDArray row = index.getRow(i);
                int[] nums = new int[index.getRow(i).columns()];
                for (int j = 0; j < row.columns(); j++) {
                    nums[j] = (int) row.getFloat(j);
                }

                NDArrayIndex idx = new NDArrayIndex(nums);
                ret[i] = idx;

            }


            return ret;

        } else if (index.isVector()) {
            int[] indices = NDArrayUtil.toInts(index);
            return new NDArrayIndex[]{new NDArrayIndex(indices)};
        }


        throw new IllegalArgumentException("Passed in ndarray must be a matrix or a vector");

    }

    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     *
     * @param begin the begin
     * @param stride  the stride at which to increment
     * @param end   the end index
     * @return the interval
     */
    public static INDArrayIndex interval(int begin, int stride,int end) {
        if(Math.abs(begin - end) < 1)
            end++;
        if(stride > 1 && Math.abs(begin - end) == 1) {
            end *= stride;
        }
        return interval(begin,stride, end, false);
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
    public static INDArrayIndex interval(int begin,int stride, int end, boolean inclusive) {
        assert begin <= end : "Beginning index in range must be less than end";
        INDArrayIndex index = new IntervalIndex(inclusive,stride);
        index.init(begin,end);
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
        return interval(begin,1, end, false);
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
        return interval(begin,1,end,inclusive);
    }

    @Override
    public int end() {
        if (indices != null && indices.length > 0)
            return indices[indices.length - 1];
        return 0;
    }

    @Override
    public int offset() {
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
    public int length() {
        return indices.length;
    }

    @Override
    public int stride() {
        return 1;
    }

    @Override
    public int current() {
        return 0;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public int next() {
        return 0;
    }

    @Override
    public void reverse() {
        ArrayUtil.reverse(indices);
    }

    @Override
    public String toString() {
        return "NDArrayIndex{" +
                "indices=" + Arrays.toString(indices) +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof INDArrayIndex)) return false;

        NDArrayIndex that = (NDArrayIndex) o;

        if (!Arrays.equals(indices, that.indices)) return false;
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
    public void init(INDArray arr, int begin, int dimension) {

    }

    @Override
    public void init(INDArray arr, int dimension) {

    }

    @Override
    public void init(int begin, int end) {

    }

    @Override
    public void reset() {

    }


}
