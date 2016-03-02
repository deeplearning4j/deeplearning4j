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
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Indexing util.
 *
 * @author Adam Gibson
 */
public class Indices {
    /**
     * Compute the linear offset
     * for an index in an ndarray.
     *
     * For c ordering this is just the index itself.
     * For fortran ordering, the following algorithm is used.
     *
     * Assuming an ndarray is a list of vectors.
     * The index of the vector relative to the given index is calculated.
     *
     * vectorAlongDimension is then used along the last dimension
     * using the computed index.
     *
     * The offset + the computed column wrt the index: (index % the size of the last dimension)
     * will render the given index in fortran ordering
     * @param index the index
     * @param arr the array
     * @return the linear offset
     */
    public static int rowNumber(int index,INDArray arr) {
        double otherTest = ((double) index) / arr.size(-1);
        int test = (int) Math.floor(otherTest);
        int vectors = arr.vectorsAlongDimension(-1);
        if(test >= vectors)
            return vectors - 1;
        return test;
    }

    /**
     * Compute the linear offset
     * for an index in an ndarray.
     *
     * For c ordering this is just the index itself.
     * For fortran ordering, the following algorithm is used.
     *
     * Assuming an ndarray is a list of vectors.
     * The index of the vector relative to the given index is calculated.
     *
     * vectorAlongDimension is then used along the last dimension
     * using the computed index.
     *
     * The offset + the computed column wrt the index: (index % the size of the last dimension)
     * will render the given index in fortran ordering
     * @param index the index
     * @param arr the array
     * @return the linear offset
     */
    public static int linearOffset(int index,INDArray arr) {
        if(arr.ordering() == NDArrayFactory.C) {
            double otherTest = ((double) index) % arr.size(-1);
            int test = (int) Math.floor(otherTest);
            INDArray vec = arr.vectorAlongDimension(test,-1);
            int otherDim = arr.vectorAlongDimension(test,-1).offset() + index;
            return otherDim;
        }
        else {
            int majorStride = arr.stride(-2);
            int vectorsAlongDimension = arr.vectorsAlongDimension(-1);
            double rowCalc = (double) (index * majorStride) / (double) arr.length();
            int floor = (int) Math.floor(rowCalc);

            INDArray arrVector = arr.vectorAlongDimension(floor, -1);

            int columnIndex = index % arr.size(-1);
            int retOffset = arrVector.linearIndex(columnIndex);
            return retOffset;



        }
    }



    /**
     * The offsets (begin index) for each index
     *
     * @param indices the indices
     * @return the offsets for the given set of indices
     */
    public static int[] offsets(int[] shape,INDArrayIndex...indices) {
        //offset of zero for every new axes
        int[] ret = new int[shape.length];

        if(indices.length == shape.length) {
            for (int i = 0; i < indices.length; i++) {
                if(indices[i] instanceof NDArrayIndexEmpty)
                    ret[i] = 0;
                else {
                    ret[i] = indices[i].offset();
                }

            }

            if(ret.length == 1) {
                ret = new int[] {ret[0],0};
            }
        }

        else {
            int numPoints = NDArrayIndex.numPoints(indices);
            if(numPoints > 0) {
                List<Integer> nonZeros = new ArrayList<>();
                for(int i = 0; i < indices.length; i++)
                    if(indices[i].offset() > 0)
                        nonZeros.add(indices[i].offset());
                if(nonZeros.size() > shape.length)
                    throw new IllegalStateException("Non zeros greater than shape unable to continue");
                else {
                    //push all zeros to the back
                    for(int i = 0; i < nonZeros.size(); i++)
                        ret[i] = nonZeros.get(i);
                }
            }
            else {
                int shapeIndex = 0;
                for (int i = 0; i < indices.length; i++) {
                    if(indices[i] instanceof NDArrayIndexEmpty)
                        ret[i] = 0;
                    else {
                        ret[i] = indices[shapeIndex++].offset();
                    }

                }
            }


            if(ret.length == 1) {
                ret = new int[] {ret[0],0};
            }
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
    public static INDArrayIndex[] fillIn(int[] shape, INDArrayIndex... indexes) {
        if (shape.length == indexes.length)
            return indexes;

        INDArrayIndex[] newIndexes = new INDArrayIndex[shape.length];
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
    public static INDArrayIndex[] adjustIndices(int[] originalShape, INDArrayIndex...indexes) {
        if(Shape.isVector(originalShape) && indexes.length == 1)
            return indexes;

        if (indexes.length < originalShape.length)
            indexes = fillIn(originalShape, indexes);
        if (indexes.length > originalShape.length) {
            INDArrayIndex[] ret = new INDArrayIndex[originalShape.length];
            System.arraycopy(indexes, 0, ret, 0, originalShape.length);
            return ret;
        }

        if (indexes.length == originalShape.length)
            return indexes;
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i].end() >= originalShape[i] || indexes[i] instanceof NDArrayIndexAll)
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
    public static int[] shape(INDArrayIndex... indices) {
        int[] ret = new int[indices.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = indices[i].length();
        }

        List<Integer> nonZeros = new ArrayList<>();
        for (int i = 0; i < ret.length; i++) {
            if (ret[i] > 0)
                nonZeros.add(ret[i]);
        }

        return ArrayUtil.toArray(nonZeros);
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
     * Create an n dimensional index
     * based on the given interval indices.
     * Start and end represent the begin and
     * end of each interval
     * @param start the start indexes
     * @param end the end indexes
     * @return the interval index relative to the given
     * start and end indices
     */
    public static INDArrayIndex[] createFromStartAndEnd(INDArray start, INDArray end) {
        if(start.length() != end.length())
            throw new IllegalArgumentException("Start length must be equal to end length");
        else {
            INDArrayIndex[] indexes = new INDArrayIndex[start.length()];
            for(int i = 0; i < indexes.length; i++) {
                indexes[i] = NDArrayIndex.interval(start.getInt(i),end.getInt(i));
            }
            return indexes;
        }
    }


    /**
     * Create indices representing intervals
     * along each dimension
     * @param start the start index
     * @param end the end index
     * @param inclusive whether the last
     *                  index should be included
     * @return the ndarray indexes covering
     * each dimension
     */
    public static INDArrayIndex[] createFromStartAndEnd(INDArray start, INDArray end, boolean inclusive) {
        if(start.length() != end.length())
            throw new IllegalArgumentException("Start length must be equal to end length");
        else {
            INDArrayIndex[] indexes = new INDArrayIndex[start.length()];
            for(int i = 0; i < indexes.length; i++) {
                indexes[i] = NDArrayIndex.interval(start.getInt(i),end.getInt(i),inclusive);
            }
            return indexes;
        }
    }


    /**
     * Calculate the shape for the given set of indices and offsets.
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
    public static int[] shape(int[] shape,INDArrayIndex...indices) {
        int newAxesPrepend = 0;
        boolean encounteredAll = false;
        List<Integer> accumShape = new ArrayList<>();
        //bump number to read from the shape
        int shapeIndex = 0;
        //list of indexes to prepend to for new axes
        //if all is encountered
        List<Integer> prependNewAxes = new ArrayList<>();
        for(int i = 0; i < indices.length; i++) {
            INDArrayIndex idx = indices[i];
            if (idx instanceof NDArrayIndexAll)
                encounteredAll = true;
            //point: do nothing but move the shape counter
            if(idx instanceof PointIndex) {
                shapeIndex++;
                continue;
            }
            //new axes encountered, need to track whether to prepend or
            //to set the new axis in the middle
            else if(idx instanceof NewAxis) {
                //prepend the new axes at different indexes
                if(encounteredAll) {
                    prependNewAxes.add(i);
                }
                //prepend to the beginning
                //rather than a set index
                else
                    newAxesPrepend++;
                continue;

            }

            //points and intervals both have a direct desired length

            else if(idx instanceof IntervalIndex && !(idx instanceof NDArrayIndexAll) || idx instanceof SpecifiedIndex) {
                accumShape.add(idx.length());
                shapeIndex++;
                continue;
            }

            accumShape.add(shape[shapeIndex]);
            shapeIndex++;

        }

        while(shapeIndex < shape.length) {
            accumShape.add(shape[shapeIndex++]);
        }


        while(accumShape.size() < 2) {
            accumShape.add(1);
        }

        //only one index and matrix, remove the first index rather than the last
        //equivalent to this is reversing the list with the prepended one
        if(indices.length == 1 && indices[0] instanceof PointIndex && shape.length == 2) {
            Collections.reverse(accumShape);
        }

        //prepend for new axes; do this first before
        //doing the indexes to prepend to
        if(newAxesPrepend > 0) {
            for(int i = 0; i < newAxesPrepend; i++)
                accumShape.add(0,1);
        }

        /**
         * For each dimension
         * where we want to prepend a dimension
         * we need to add it at the index such that
         * we account for the offset of the number of indexes
         * added up to that point.
         *
         * We do this by doing an offset
         * for each item added "so far"
         *
         * Note that we also have an offset of - 1
         * because we want to prepend to the given index.
         *
         * When prepend new axes for in the middle is triggered
         * i is already > 0
         */
        for(int i = 0; i < prependNewAxes.size(); i++) {
            accumShape.add(prependNewAxes.get(i) - i,1);
        }



        return Ints.toArray(accumShape);
    }



    /**
     * Return the stride to be used for indexing
     * @param arr the array to get the strides for
     * @param indexes the indexes to use for computing stride
     * @param shape the shape of the output
     * @return the strides used for indexing
     */
    public static int[] stride(INDArray arr,INDArrayIndex[] indexes, int... shape) {
        List<Integer> strides = new ArrayList<>();
        int strideIndex = 0;
        //list of indexes to prepend to for new axes
        //if all is encountered
        List<Integer> prependNewAxes = new ArrayList<>();

        for(int i = 0; i < indexes.length; i++) {
            //just like the shape, drops the stride
            if(indexes[i] instanceof PointIndex) {
                strideIndex++;
                continue;
            }
            else if(indexes[i] instanceof NewAxis) {

            }
        }

        /**
         * For each dimension
         * where we want to prepend a dimension
         * we need to add it at the index such that
         * we account for the offset of the number of indexes
         * added up to that point.
         *
         * We do this by doing an offset
         * for each item added "so far"
         *
         * Note that we also have an offset of - 1
         * because we want to prepend to the given index.
         *
         * When prepend new axes for in the middle is triggered
         * i is already > 0
         */
        for(int i = 0; i < prependNewAxes.size(); i++) {
            strides.add(prependNewAxes.get(i) - i,1);
        }

        return Ints.toArray(strides);

    }


    /**
     * Check if the given indexes
     * over the specified array
     * are searching for a scalar
     * @param indexOver the array to index over
     * @param indexes the index query
     * @return true if the given indexes are searching
     * for a scalar false otherwise
     */
    public static boolean isScalar(INDArray indexOver,INDArrayIndex...indexes) {
        boolean allOneLength = true;
        for(int i = 0; i < indexes.length; i++) {
            allOneLength = allOneLength && indexes[i].length() == 1;
        }

        int numNewAxes = NDArrayIndex.numNewAxis(indexes);
        if(allOneLength && numNewAxes == 0 && indexes.length == indexOver.rank())
            return true;
        else if(allOneLength && indexes.length == indexOver.rank() - numNewAxes) {
            return allOneLength;
        }

        return allOneLength;
    }


}
