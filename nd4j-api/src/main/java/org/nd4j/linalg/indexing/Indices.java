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

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Indexing util.
 *
 * @author Adam Gibson
 */
public class Indices {

    /**
     * The offsets (begin index) for each index
     *
     * @param indices the indices
     * @return the offsets for the given set of indices
     */
    public static int[] offsets(NDArrayIndex... indices) {
        int[] ret = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            int offset = indices[i].offset();
            if (offset == 0 && i > 0 && i < indices.length - 1)
                ret[i] = 1;
            else
                ret[i] = indices[i].offset();
        }
        return ret;
    }


    /**
     * Fill in the missing indices to be the
     * same length as the original shape.
     * <p/>
     * Think of this as what fills in the indices for numpy or matlab:
     * Given a which is (4,3,2) in numpy:
     * <p/>
     * a[1:3] is filled in by the rest
     * to give back the full slice
     * <p/>
     * This algorithm fills in that delta
     *
     * @param shape   the original shape
     * @param indexes the indexes to start from
     * @return the filled in indices
     */
    public static NDArrayIndex[] fillIn(int[] shape, NDArrayIndex... indexes) {
        if (shape.length == indexes.length)
            return indexes;

        NDArrayIndex[] newIndexes = new NDArrayIndex[shape.length];
        System.arraycopy(indexes, 0, newIndexes, 0, indexes.length);

        for (int i = indexes.length; i < shape.length; i++) {
            newIndexes[i] = NDArrayIndex.interval(0, shape[i]);
        }
        return newIndexes;

    }

    /**
     * Prunes indices of greater length than the shape
     * and fills in missing indices if there are any
     *
     * @param originalShape the original shape to adjust to
     * @param indexes       the indexes to adjust
     * @return the  adjusted indices
     */
    public static NDArrayIndex[] adjustIndices(int[] originalShape, NDArrayIndex... indexes) {
        if (indexes.length < originalShape.length)
            indexes = fillIn(originalShape, indexes);
        if (indexes.length > originalShape.length) {
            NDArrayIndex[] ret = new NDArrayIndex[originalShape.length];
            System.arraycopy(indexes, 0, ret, 0, originalShape.length);
            return ret;
        }

        if (indexes.length == originalShape.length)
            return indexes;
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i].end() >= originalShape[i])
                indexes[i] = NDArrayIndex.interval(0, originalShape[i] - 1);
        }

        return indexes;
    }


    /**
     * Calculate the strides based on the given indices
     *
     * @param ordering the ordering to calculate strides for
     * @param indexes  the indices to calculate stride for
     * @return the strides for the given indices
     */
    public static int[] strides(char ordering, NDArrayIndex... indexes) {
        return Nd4j.getStrides(shape(indexes), ordering);
    }

    /**
     * Calculate the shape for the given set of indices.
     * <p/>
     * The shape is defined as (for each dimension)
     * the difference between the end index + 1 and
     * the begin index
     *
     * @param indices the indices to calculate the shape for
     * @return the shape for the given indices
     */
    public static int[] shape(NDArrayIndex... indices) {
        int[] ret = new int[indices.length];
        for (int i = 0; i < ret.length; i++) {
            int[] currIndices = indices[i].indices();

            int end = currIndices[currIndices.length - 1] + 1;
            int begin = currIndices[0];
            ret[i] = Math.abs(end - begin);
        }

        List<Integer> nonZeros = new ArrayList<>();
        for (int i = 0; i < ret.length; i++) {
            if (ret[i] > 0)
                nonZeros.add(ret[i]);
        }

        return ArrayUtil.toArray(nonZeros);
    }

    /**
     * Returns whether the indices are contiguous by one or not
     *
     * @param indexes the indices to test
     * @return whether the indices are contiguous by one or not
     */
    public static boolean isContiguous(NDArrayIndex... indexes) {
        return isContiguous(1, indexes);
    }

    /**
     * Returns whether indices are contiguous
     * by a certain amount or not
     *
     * @param indexes the indices to test
     * @param diff    the difference considered to be contiguous
     * @return whether the given indices are contiguous or not
     */
    public static boolean isContiguous(int diff, NDArrayIndex... indexes) {
        if (indexes.length < 1)
            return true;
        boolean contiguous = isContiguous(indexes[0].indices(), diff);
        for (int i = 1; i < indexes.length; i++)
            contiguous = contiguous && isContiguous(indexes[i].indices(), diff);

        return contiguous;
    }

    /**
     * Returns whether the indices are contiguous by one or not
     *
     * @param indices the indices to test
     * @return whether the indices are contiguous by one or not
     */
    public static boolean isContiguous(int[] indices) {
        return isContiguous(indices, 1);
    }

    /**
     * Returns whether indices are contiguous
     * by a certain amount or not
     *
     * @param indices the indices to test
     * @param diff    the difference considered to be contiguous
     * @return whether the given indices are contiguous or not
     */
    public static boolean isContiguous(int[] indices, int diff) {
        if (indices.length < 1)
            return true;
        for (int i = 1; i < indices.length; i++) {
            if (Math.abs(indices[i] - indices[i - 1]) > diff)
                return false;
        }

        return true;
    }

    /**
     * Calculate the shape for the given set of indices.
     * <p/>
     * The shape is defined as (for each dimension)
     * the difference between the end index + 1 and
     * the begin index
     * <p/>
     * If specified, this will check for whether any of the indices are >= to end - 1
     * and if so, prune it down
     *
     * @param shape   the original shape
     * @param indices the indices to calculate the shape for
     * @return the shape for the given indices
     */
    public static int[] shape(int[] shape, NDArrayIndex... indices) {
        if (indices.length > shape.length)
            return shape;

        int[] ret = new int[indices.length];
        for (int i = 0; i < ret.length; i++) {
            int[] currIndices = indices[i].indices();
            if (currIndices.length < 1)
                continue;
            int end = currIndices[currIndices.length - 1];
            if (end > shape[i])
                end = shape[i] - 1;
            int begin = currIndices[0];

            ret[i] = indices[i].isInterval() ? Math.abs(end - begin) + 1 :
                    indices[i].indices().length;
        }

        List<Integer> nonZeros = new ArrayList<>();
        for (int i = 0; i < ret.length; i++) {
            if (ret[i] > 0)
                nonZeros.add(ret[i]);
        }


        return ArrayUtil.toArray(nonZeros);

    }


}
