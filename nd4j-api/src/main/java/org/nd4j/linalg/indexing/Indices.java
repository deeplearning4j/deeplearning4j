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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LinearIndex;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.api.shape.Shape;

import java.util.ArrayList;
import java.util.Arrays;
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
     * Calculate the linear indices for an
     * ndarray
     * @param arr the array to calculate for
     * @return the array for the linear indices
     */
    public static int[] linearIndices(INDArray arr) {
        LinearIndex index = new LinearIndex(arr,arr.dup(),true);
        Nd4j.getExecutioner().iterateOverAllRows(index);
        return index.getIndices();
    }

    /**
     * The offsets (begin index) for each index
     *
     * @param indices the indices
     * @return the offsets for the given set of indices
     */
    public static int[] offsets(int[] shape,INDArrayIndex...indices) {
        int numNewAxes = NDArrayIndex.numNewAxis(indices);
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
            for (int i = 0; i < shape.length; i++) {
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
     * @param offsets the offsets for the indexing
     * @param indices the indices to calculate the shape for
     * @return the shape for the given indices
     */
    public static int[] shape(int[] shape, int[] offsets,INDArrayIndex...indices) {
        int numNewAxes = NDArrayIndex.numNewAxis(indices);
        if (indices.length > shape.length && numNewAxes < 1)
            return shape;

        if(Shape.isRowVectorShape(shape) && numNewAxes < 1 && indices.length <= 2) {
            if(indices.length == 2)
                return new int[] {1,Math.max(indices[0].length(),indices[1].length())};
            else
                return new int[]{1,indices[0].length()};
        }

        int[] ret = new int[offsets.length];
        if(indices[0].length() == 1 && numNewAxes >= 1)
            ret = new int[indices.length - 1];
        if(offsets.length < shape.length) {
            int[] dup = new int[shape.length];
            System.arraycopy(offsets,0,dup,0,offsets.length);
            offsets = dup;
        }

        int shapeIndex = 0;
        for (int i = 0; i < indices.length; i++) {
            if(indices[i] instanceof NDArrayIndexAll) {
                if(shapeIndex < ret.length) {
                    ret[shapeIndex] = shape[shapeIndex];
                    shapeIndex++;
                }

            }
            else if(indices[i] instanceof NDArrayIndexEmpty) {
                ret[i] = 0;
            }

            else if(indices[i] instanceof NewAxis) {
                continue;
            }

            else {
                ret[i] = indices[i].length();
                shapeIndex++;
                ret[i] -= offsets[i];
            }

        }


        List<Integer> nonZeros = new ArrayList<>();
        for (int i = 0; i < ret.length; i++) {
            if (ret[i] > 0)
                nonZeros.add(ret[i]);
        }


        int[] ret2 =  ArrayUtil.toArray(nonZeros);
        if(Shape.isRowVectorShape(ret2) && ret2.length == 1) {
            return new int[]{1,ret2[0]};
        }

        if(ret2.length <= 1) {
            ret2 = new int[] {1,1};
        }

        return ret2;
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
    public static int[] shape(int[] shape, INDArrayIndex...indices) {
        return shape(shape, new int[shape.length], indices);
    }


    /**
     * Return the stride to be used for indexing
     * @param arr the array to get the strides for
     * @param indexes the indexes to use for computing stride
     * @param shape the shape of the output
     * @return the strides used for indexing
     */
    public static int[] stride(INDArray arr,INDArrayIndex[] indexes, int... shape) {
        int[] retStride = null;
        if(indexes.length >= arr.stride().length) {
            //prepend zeros for new axis
            retStride = new int[arr.stride().length];
            for(int i = 0; i < retStride.length; i++) {
                retStride[i] = arr.stride(i) * indexes[i].stride();
            }
        }
        else {
            retStride = Arrays.copyOfRange(arr.stride(), 1, shape.length);
        }



        return retStride;

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
