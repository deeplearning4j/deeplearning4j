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

package org.nd4j.linalg.indexing;

import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * NDArray indexing
 *
 * @author Adam Gibson
 */
public class NDArrayIndex {

    private int[] indices = new int[1];
    private boolean isInterval = false;
    private static NDArrayIndexAll ALL = new NDArrayIndexAll();

    /**
     * NDArrayIndexing based on the given
     * indexes
     * @param indices
     */
    public NDArrayIndex(int... indices) {
        this.indices = indices;

    }


    /**
     * Represents collecting all elements
     *
     * @return an ndarray index
     * meaning collect
     * all elements
     */
    public static NDArrayIndex all() {
        return ALL;
    }

    /**
     * Creates an index covering the given shape
     * (for each dimension 0,shape[i])
     * @param shape the shape to cover
     * @return the ndarray indexes to cover
     */
    public static NDArrayIndex[] createCoveringShape(int[] shape) {
        NDArrayIndex[] ret = new NDArrayIndex[shape.length];
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
    public static NDArrayIndex[] rangeOfLength(NDArrayIndex...indexes) {
        NDArrayIndex[] indexesRet = new NDArrayIndex[indexes.length];
        for(int i = 0; i < indexes.length; i++)
            indexesRet[i] = NDArrayIndex.interval(0,indexes[i].indices.length);
        return indexesRet;
    }



    /**
     * Create from a matrix. The rows are the indices
     * The columns are the individual element in each ndarrayindex
     *
     * @param index the matrix to getFloat indices from
     * @return the indices to getFloat
     */
    public static NDArrayIndex[] create(INDArray index) {

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
            int[] indices = ArrayUtil.toInts(index);
            return new NDArrayIndex[]{new NDArrayIndex(indices)};
        }


        throw new IllegalArgumentException("Passed in ndarray must be a matrix or a vector");

    }

    /**
     * Concatneate all of the given indices in to one
     *
     * @param indexes the indexes to concatneate
     * @return the merged indices
     */
    public static NDArrayIndex concat(NDArrayIndex... indexes) {
        int[][] indices = new int[indexes.length][];
        for (int i = 0; i < indexes.length; i++)
            indices[i] = indexes[i].indices();
        return new NDArrayIndex(Ints.concat(indices));
    }


    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     *
     * @param begin the begin
     * @param stride  the stride at which to increment
     * @param end   the end index
     * @return the interval
     */
    public static NDArrayIndex interval(int begin, int stride,int end) {
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
    public static NDArrayIndex interval(int begin,int stride, int end, boolean inclusive) {
        assert begin <= end : "Beginning index in range must be less than end";
        NDArrayIndex ret = new NDArrayIndex(ArrayUtil.range(begin, inclusive ? end + 1 : end,stride));
        ret.isInterval = stride == 1;
        return ret;
    }


    /**
     * Generates an interval from begin (inclusive) to end (exclusive)
     *
     * @param begin the begin
     * @param end   the end index
     * @return the interval
     */
    public static NDArrayIndex interval(int begin, int end) {
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
    public static NDArrayIndex interval(int begin, int end, boolean inclusive) {
       return interval(begin,1,end,inclusive);
    }

    public int end() {
        if (indices != null && indices.length > 0)
            return indices[indices.length - 1];
        return 0;
    }

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
    public int length() {
        if (indices.length < 1)
            return 0;
        return indices[indices.length - 1] - indices[0];
    }

    public int[] indices() {
        return indices;
    }

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
        if (!(o instanceof NDArrayIndex)) return false;

        NDArrayIndex that = (NDArrayIndex) o;

        if (!Arrays.equals(indices, that.indices)) return false;

        return true;
    }




    //static type checking used for checking if an index should be represented as all
    public static class NDArrayIndexAll  extends NDArrayIndex {
        public NDArrayIndexAll(int... indices) {
            super(indices);
        }
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(indices);
    }

    public boolean isInterval() {
        return isInterval;
    }

    public void setInterval(boolean isInterval) {
        this.isInterval = isInterval;
    }


}
