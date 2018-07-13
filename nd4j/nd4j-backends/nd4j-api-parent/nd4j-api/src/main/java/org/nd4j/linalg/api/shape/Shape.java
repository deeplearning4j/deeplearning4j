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

package org.nd4j.linalg.api.shape;


import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.*;
import java.util.*;

/**
 * Encapsulates all shape related logic (vector of 0 dimension is a scalar is equivalent to
 * a vector of length 1...)
 *
 * @author Adam Gibson
 */
public class Shape {


    private Shape() {}


    /**
     * Return the shape of the largest length array
     * based on the input
     * @param inputs the inputs to get the max shape for
     * @return the largest shape based on the inputs
     */
    public static long[] getMaxShape(INDArray...inputs) {
        if(inputs == null)
            return null;
        else if(inputs.length < 2)
            return inputs[0].shape();
        else {
            long[] currMax = inputs[0].shape();
            for(int i = 1; i <  inputs.length; i++) {
                if(inputs[i] == null) {
                    continue;
                }
                if(ArrayUtil.prod(currMax) < inputs[i].length()) {
                    currMax = inputs[i].shape();
                }
            }

            return currMax;
        }
    }

    /**
     * Returns true if this shape is scalar
     * @param shape the shape that is scalar
     * @return
     */
    public static boolean shapeIsScalar(int[] shape) {
        return shape.length == 0 || ArrayUtil.prodLong(shape) == 1;
    }

    public static boolean shapeIsScalar(long[] shape) {
        return shape.length == 0 || ArrayUtil.prodLong(shape) == 1;
    }

    /**
     * Returns true if any shape has a -1
     * or a null or empty array is passed in
     * @param shape the input shape to validate
     * @return true if the shape is null,empty, or contains a -1 element
     */
    public static boolean isPlaceholderShape(int[] shape) {
        if(shape == null)
            return true;
        else {
            for(int i = 0; i < shape.length; i++) {
                if(shape[i] < 0)
                    return true;
            }
        }

        return false;
    }

    public static boolean isPlaceholderShape(long[] shape) {
        if(shape == null)
            return true;
        else {
            for(int i = 0; i < shape.length; i++) {
                if(shape[i] < 0)
                    return true;
            }
        }

        return false;
    }

    /**
     * Compute the broadcast rules according to:
     * https://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html
     *
     * Note that the array can be null if the arrays are already equal
     * in shape.
     *
     * This function should be used in conjunction with
     * the shape ops.
     *
     * @param left the left array
     * @param right the right array (the array to be broadcasted
     * @return the broadcast dimensions if any
     */
    public static int[] getBroadcastDimensions(int[] left,int[] right) {
        if(Arrays.equals(left,right))
            return null;

        int n = Math.min(left.length,right.length);
        List<Integer> dims = new ArrayList<>();
        int leftIdx = left.length - 1;
        int rightIdx = right.length - 1;
        for(int i = n - 1; i >= 0; i--) {
            if(left[leftIdx] != right[rightIdx] && right[rightIdx] == 1 || left[leftIdx] == 1) {
                dims.add(i);
            }
            else if(left[leftIdx] != right[rightIdx]) {
                throw new IllegalArgumentException("Unable to broadcast dimension " + i + " due to shape mismatch. Right shape must be 1. "
                        + "Left array shape: " + Arrays.toString(left) + ", right array shape: " + Arrays.toString(right));
            }

            leftIdx--;
            rightIdx--;
        }

        Collections.reverse(dims);
        return Ints.toArray(dims);
    }

    public static int[] getBroadcastDimensions(long[] left, long[] right) {
        if(Arrays.equals(left,right))
            return null;

        int n = Math.min(left.length,right.length);
        List<Integer> dims = new ArrayList<>();
        int leftIdx = left.length - 1;
        int rightIdx = right.length - 1;
        for(int i = n - 1; i >= 0; i--) {
            if(left[leftIdx] != right[rightIdx] && right[rightIdx] == 1 || left[leftIdx] == 1) {
                dims.add(i);
            }
            else if(left[leftIdx] != right[rightIdx]) {
                throw new IllegalArgumentException("Unable to broadcast dimension " + i + " due to shape mismatch. Right shape must be 1. "
                        + "Left array shape: " + Arrays.toString(left) + ", right array shape: " + Arrays.toString(right));
            }

            leftIdx--;
            rightIdx--;
        }

        Collections.reverse(dims);
        return Ints.toArray(dims);
    }


    /**
     * Get the broadcast output shape
     * based on the 2 input shapes
     * Result output shape based on:
     * https://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html
     *
     *
     * @param left the left shape
     * @param right the right second
     * @return
     */
    public static int[] broadcastOutputShape(int[] left,int[] right) {
        assertBroadcastable(left, right);
        if(Arrays.equals(left,right))
            return left;
        int n = Math.max(left.length,right.length);
        List<Integer> dims = new ArrayList<>();
        int leftIdx = left.length - 1;
        int rightIdx = right.length - 1;
        for(int i = n - 1; i >= 0; i--) {
            if(leftIdx < 0) {
                dims.add(right[rightIdx]);
            }
            else if(rightIdx < 0) {
                dims.add(left[leftIdx]);
            }
            else if(left[leftIdx] != right[rightIdx] && right[rightIdx] == 1 || left[leftIdx] == 1) {
                dims.add(Math.max(left[leftIdx],right[rightIdx]));
            }
            else if(left[leftIdx] == right[rightIdx]) {
                dims.add(left[leftIdx]);
            }
            else {
                throw new IllegalArgumentException("Unable to broadcast dimension " + i + " due to shape mismatch. Right shape must be 1.");
            }

            leftIdx--;
            rightIdx--;
        }

        Collections.reverse(dims);
        return Ints.toArray(dims);
    }


    public static long[] broadcastOutputShape(long[] left,long[] right) {
        assertBroadcastable(left, right);
        if(Arrays.equals(left,right))
            return left;
        int n = Math.max(left.length,right.length);
        List<Long> dims = new ArrayList<>();
        int leftIdx = left.length - 1;
        int rightIdx = right.length - 1;
        for(int i = n - 1; i >= 0; i--) {
            if(leftIdx < 0) {
                dims.add(right[rightIdx]);
            }
            else if(rightIdx < 0) {
                dims.add(left[leftIdx]);
            }
            else if(left[leftIdx] != right[rightIdx] && right[rightIdx] == 1 || left[leftIdx] == 1) {
                dims.add(Math.max(left[leftIdx],right[rightIdx]));
            }
            else if(left[leftIdx] == right[rightIdx]) {
                dims.add(left[leftIdx]);
            }
            else {
                throw new IllegalArgumentException("Unable to broadcast dimension " + i + " due to shape mismatch. Right shape must be 1.");
            }

            leftIdx--;
            rightIdx--;
        }

        Collections.reverse(dims);
        return Longs.toArray(dims);
    }


    /**
     *
     * @param newShape the new shape possibly
     *                 containing a negative number
     * @param shape the shape to calculate from
     * @return
     */
    public static int[] resolveNegativeShapeIfNeccessary(int[] newShape,int[] shape) {
        int numberNegativesOnes = 0;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0) {
                if (numberNegativesOnes >= 1)
                    throw new IllegalArgumentException("Only one dimension can be negative ones");

                numberNegativesOnes++;

                int shapeLength = 1;
                for (int j = 0; j < shape.length; j++)
                    if (shape[j] >= 1)
                        shapeLength *= shape[j];
                int realShape = Math.abs(ArrayUtil.prod(newShape) / shapeLength);
                int[] thisNewShape = new int[shape.length];
                for (int j = 0; j < shape.length; j++) {
                    if (i != j) {
                        thisNewShape[j] = shape[j];
                    } else
                        thisNewShape[j] = realShape;
                }

                shape = thisNewShape;
                break;

            }

        }

        for(int i = 0; i < shape.length; i++) {
            if(shape[i] == 0) {
                shape[i] = 1;
            }
        }

        return shape;

    }

    /**
     * Returns true if the dimension is null
     * or the dimension length is 1 and the first entry
     * is {@link Integer#MAX_VALUE}
     * @param shape the shape of the input array
     * @param dimension the dimensions specified
     *
     * @return true if the dimension length is equal to the shape length
     * the dimension is null or the dimension length is 1 and the first entry is
     * {@link Integer#MAX_VALUE}
     */
    public static boolean isWholeArray(int[] shape, int... dimension) {
        return isWholeArray(shape.length, dimension);
    }

    public static boolean isWholeArray(long[] shape, int... dimension) {
        return isWholeArray(shape.length, dimension);
    }

    /**
     * Returns true if the dimension is null
     * or the dimension length is 1 and the first entry
     * is {@link Integer#MAX_VALUE}
     * @param rank the rank of the input array
     * @param dimension the dimensions specified
     *
     * @return true if the dimension length is equal to the rank,
     * the dimension is null or the dimension length is 1 and the first entry is
     * {@link Integer#MAX_VALUE}
     */
    public static boolean isWholeArray(int rank, int... dimension){
        return rank == 0 || dimension == null || dimension.length == 0 ||
                (dimension.length == 1 && dimension[0] == Integer.MAX_VALUE) || dimension.length == rank;
    }

    /**
     * Get the shape of the reduced array
     * @param wholeShape the shape of the array
     *                   with the reduce op being performed
     * @param dimensions the dimensions the reduce op is being performed on
     * @return the shape of the result array as the result of the reduce
     */
    public static long[] getReducedShape(int[] wholeShape, int[] dimensions) {
        if (isWholeArray(wholeShape, dimensions))
            return new long[] {};
        else if (dimensions.length == 1 && wholeShape.length == 2) {
            val ret = new long[2];
            if (dimensions[0] == 1) {
                ret[0] = wholeShape[0];
                ret[1] = 1;
            } else if (dimensions[0] == 0) {
                ret[0] = 1;
                ret[1] = wholeShape[1];
            }
            return ret;
        }

        return ArrayUtil.toLongArray(ArrayUtil.removeIndex(wholeShape, dimensions));
    }

    public static long[] getReducedShape(long[] wholeShape, int[] dimensions) {
        if (isWholeArray(wholeShape, dimensions))
            return new long[] {};
        else if (dimensions.length == 1 && wholeShape.length == 2) {
            val ret = new long[2];
            if (dimensions[0] == 1) {
                ret[0] = wholeShape[0];
                ret[1] = 1;
            } else if (dimensions[0] == 0) {
                ret[0] = 1;
                ret[1] = wholeShape[1];
            }
            return ret;
        }

        return ArrayUtil.removeIndex(wholeShape, dimensions);
    }

    /**
     * Get the shape of the reduced array
     *
     * @param wholeShape the shape of the array
     *                   with the reduce op being performed
     * @param dimensions the dimensions the reduce op is being performed on
     * @param keepDims if set to true, corresponding dimensions will be set to 1
     * @return the shape of the result array as the result of the reduce
     */
    public static long[] getReducedShape(int[] wholeShape, int[] dimensions, boolean keepDims, boolean newFormat) {
        // we need to normalize dimensions, in case they have negative values or unsorted, or whatever
        dimensions = Shape.normalizeAxis(wholeShape.length, dimensions);

        // strip leading keepDims argument
        //if (newFormat)
        //    dimensions = Arrays.copyOfRange(dimensions, 1, dimensions.length);

        if (!keepDims)
            if (!newFormat)
                return getReducedShape(wholeShape, dimensions);
            else {
                if (isWholeArray(wholeShape, dimensions))
                    return new long[] {};
                else if (dimensions.length == 1 && wholeShape.length == 2) {
                    val ret = new long[1];
                    if (dimensions[0] == 1) {
                        ret[0] = wholeShape[0];
                    } else if (dimensions[0] == 0) {
                        ret[0] = wholeShape[1];
                    }
                    return ret;
                }

                return ArrayUtil.toLongArray(ArrayUtil.removeIndex(wholeShape, dimensions));
            }


        // we'll return full array of 1 as shape
        if (isWholeArray(wholeShape, dimensions)) {
            val result = new long[wholeShape.length];

            Arrays.fill(result, 1);
            return result;
        }

        val result = ArrayUtil.toLongArray(Arrays.copyOf(wholeShape, wholeShape.length));
        for (val dim: dimensions)
            result[dim] = 1;

        return result;
    }

    public static long[] getReducedShape(long[] wholeShape, int[] dimensions, boolean keepDims, boolean newFormat) {
        // we need to normalize dimensions, in case they have negative values or unsorted, or whatever
        dimensions = Shape.normalizeAxis(wholeShape.length, dimensions);

        // strip leading keepDims argument
        //if (newFormat)
        //    dimensions = Arrays.copyOfRange(dimensions, 1, dimensions.length);

        if (!keepDims)
            if (!newFormat)
                return getReducedShape(wholeShape, dimensions);
            else {
                if (isWholeArray(wholeShape, dimensions))
                    return new long[] {};
                else if (dimensions.length == 1 && wholeShape.length == 2) {
                    val ret = new long[1];
                    if (dimensions[0] == 1) {
                        ret[0] = wholeShape[0];
                    } else if (dimensions[0] == 0) {
                        ret[0] = wholeShape[1];
                    }
                    return ret;
                }

                return ArrayUtil.removeIndex(wholeShape, dimensions);
            }


        // we'll return full array of 1 as shape
        if (isWholeArray(wholeShape, dimensions)) {
            val result = new long[wholeShape.length];

            Arrays.fill(result, 1);
            return result;
        }

        val result = Arrays.copyOf(wholeShape, wholeShape.length);
        for (val dim: dimensions)
            result[dim] = 1;

        return result;
    }


    /**
     * Get the output shape of a matrix multiply
     *
     * @param left the first matrix shape to multiply
     * @param right the second matrix shape to multiply
     * @return the shape of the output array (the left's rows and right's columns)
     */
    public static int[] getMatrixMultiplyShape(int[] left, int[] right) {
        if(Shape.shapeIsScalar(left)) {
            return right;
        }

        if(Shape.shapeIsScalar(right)) {
            return left;
        }

        if (left.length != 2 && right.length != 2) {
            throw new IllegalArgumentException("Illegal shapes for matrix multiply. Must be of length 2. Left shape: "
                    + Arrays.toString(left) + ", right shape: " + Arrays.toString(right));
        }

        for(int i = 0; i < left.length; i++) {
            if(left[i] < 1)
                throw new ND4JIllegalStateException("Left shape contained value < 0 at index " + i + " - left shape " + Arrays.toString(left));
        }



        for(int i = 0; i < right.length; i++) {
            if(right[i] < 1)
                throw new ND4JIllegalStateException("Right shape contained value < 0 at index " + i + " - right shape " + Arrays.toString(right));
        }


        if (left.length > 1 && left[1] != right[0])
            throw new IllegalArgumentException("Columns of left not equal to rows of right: left shape " + Arrays.toString(left)
                    + ", right shape " + Arrays.toString(right));

        if(left.length < right.length) {
            if(left[0] == right[0]) {
                return new int[] {1, right[1]};
            }
        }

        int[] shape = {left[0], right[1]};
        return shape;
    }

    public static long[] getMatrixMultiplyShape(long[] left, long[] right) {
        if(Shape.shapeIsScalar(left)) {
            return right;
        }

        if(Shape.shapeIsScalar(right)) {
            return left;
        }

        if (left.length != 2 && right.length !=2) {
            if (left.length != 3 && right.length != 3) {
                throw new IllegalArgumentException("Illegal shapes for matrix multiply. Must be both of length 2 or both" +
                        "of length 3 (batch-wise matrix multiply). Left shape: "
                        + Arrays.toString(left) + ", right shape: " + Arrays.toString(right));
            }
        }

        for(int i = 0; i < left.length; i++) {
            if(left[i] < 1)
                throw new ND4JIllegalStateException("Left shape contained value < 0 at index " + i + " - left shape " + Arrays.toString(left));
        }

        for(int i = 0; i < right.length; i++) {
            if(right[i] < 1)
                throw new ND4JIllegalStateException("Right shape contained value < 0 at index " + i + " - right shape " + Arrays.toString(right));
        }


        if (left.length == 2 && left[1] != right[0] || left.length == 3 && left[2] != right[1])
            throw new IllegalArgumentException("Columns of left not equal to rows of right: left shape " + Arrays.toString(left)
                    + ", right shape " + Arrays.toString(right));

        if(left.length < right.length) {
            if(left[0] == right[0]) {
                return new long[] {1, right[1]};
            }
        }

        if (left.length == 3 && left[0] != right[0]) {
            throw new IllegalArgumentException("For batch matrix multiplication the leading dimension of both arguments" +
                    "has to match, got left leading dimension" + left[0] + "and right " + right[0]);
        }

        long[] shape;
        if (left.length == 2) {
            shape = new long[]{left[0], right[1]};
        } else {
            shape = new long[]{left[0], left[1], right[2]};
        }
        return shape;
    }

    /**
     * Create a copy of the matrix
     * where the new offset is zero
     *
     * @param arr the array to copy to offset 0
     * @return the same array if offset is zero
     * otherwise a copy of the array with
     * elements set to zero
     */
    public static INDArray toOffsetZero(INDArray arr) {
        if (arr.offset() < 1 && arr.data().length() == arr.length()
                || arr instanceof IComplexNDArray && arr.length() * 2 == arr.data().length())
            if (arr.ordering() == 'f' && arr.stride(-1) != arr.elementStride()
                    || arr.ordering() == 'c' && arr.stride(0) != arr.elementStride())
                return arr;

        if (arr.isRowVector()) {
            if (arr instanceof IComplexNDArray) {
                /*IComplexNDArray ret = Nd4j.createComplex(arr.shape());
                for (int i = 0; i < ret.length(); i++)
                    ret.putScalar(i, ((IComplexNDArray) arr).getComplex(i));
                return ret;*/
                throw new UnsupportedOperationException("Complex arrays aren't supported yet");
            } else {
                INDArray ret = Nd4j.create(arr.shape());
                for (int i = 0; i < ret.length(); i++)
                    ret.putScalar(i, arr.getDouble(i));
                return ret;
            }
        }


        if (arr instanceof IComplexNDArray) {
            /*IComplexNDArray ret = Nd4j.createComplex(arr.shape());
            for (int i = 0; i < ret.slices(); i++)
                ret.putSlice(i, arr.slice(i));
            return ret;*/
            throw new UnsupportedOperationException("Complex arrays aren't supported yet");
        } else {
            INDArray ret = Nd4j.create(arr.shape(), arr.ordering());
            ret.assign(arr);
            return ret;
        }
    }



    /**
     * Create a copy of the ndarray where the new offset is zero
     *
     * @param arr the array to copy to offset 0
     * @return a copy of the array with elements set to zero offset
     */
    public static INDArray toOffsetZeroCopy(INDArray arr) {
        return toOffsetZeroCopyHelper(arr, Nd4j.order(), false);
    }

    /**Create a copy of the ndarray where the new offset is zero, and has specified order
     * @param arr the array to copy to offset 0
     * @param order the order of the returned array
     * @return a copy of the array with elements set to zero offset, and with specified order
     */
    public static INDArray toOffsetZeroCopy(INDArray arr, char order) {
        return toOffsetZeroCopyHelper(arr, order, false);
    }

    /** Create a copy of the ndarray where the new offset is zero.
     * Unlike toOffsetZeroCopy(INDArray) (which always returns arrays of order Nd4j.order()),
     * and toOffsetZeroCopy(INDArray,char) (which always returns arrays of a specified order)
     * this method returns NDArrays of any order (sometimes c, sometimes f).<br>
     * This method may be faster than the other two toOffsetZeroCopyAnyOrder methods as a result,
     * however no performance benefit (or cost) relative to them will be observed in many cases.
     * If a copy is necessary, the output will have order Nd4j.order()
     * @param arr NDArray to duplicate
     * @return Copy with offset 0, but order might be c, or might be f
     */
    public static INDArray toOffsetZeroCopyAnyOrder(INDArray arr) {
        return toOffsetZeroCopyHelper(arr, Nd4j.order(), true);
    }

    private static INDArray toOffsetZeroCopyHelper(final INDArray arr, char order, boolean anyOrder) {
        if (arr instanceof IComplexNDArray) {
            /*
            if (arr.isRowVector()) {
                IComplexNDArray ret = Nd4j.createComplex(arr.shape(), order);
                for (int i = 0; i < ret.length(); i++)
                    ret.putScalar(i, ((IComplexNDArray) arr).getComplex(i));
                return ret;
            }
            IComplexNDArray ret = Nd4j.createComplex(arr.shape(), order);
            for (int i = 0; i < ret.slices(); i++)
                ret.putSlice(i, arr.slice(i));
            return ret;
            */
            throw new UnsupportedOperationException();
        } else {
            //Use CopyOp:
            char outOrder = (anyOrder ? arr.ordering() : order);
            if (outOrder == 'a')
                outOrder = Nd4j.order();
            INDArray z = Nd4j.createUninitialized(arr.shape(), outOrder);
            z.assign(arr);
            return z;
        }
    }


    /**
     * Get a double based on the array and given indices
     *
     * @param arr     the array to retrieve the double from
     * @param indices the indices to iterate over
     * @return the double at the specified index
     */
    public static double getDouble(INDArray arr, int[] indices) {
        long offset = getOffset(arr.shapeInfo(), ArrayUtil.toLongArray(indices));
        return arr.data().getDouble(offset);
    }

    public static double getDouble(INDArray arr, long... indices) {
        long offset = getOffset(arr.shapeInfo(), indices);
        return arr.data().getDouble(offset);
    }

    public static long getLong(INDArray arr, long... indices) {
        long offset = getOffset(arr.shapeInfo(), indices);
        return arr.data().getLong(offset);
    }

    public static int getInt(INDArray arr, long... indices) {
        long offset = getOffset(arr.shapeInfo(), indices);
        return arr.data().getInt(offset);
    }

    public static int getInt(INDArray arr, int[]indices) {
        long offset = getOffset(arr.shapeInfo(), indices);
        return arr.data().getInt(offset);
    }


    /**
     * Iterate over 2
     * coordinate spaces given 2 arrays
     * @param arr the first array
     * @param coordinateFunction the coordinate function to use
     *
     */
    public static void iterate(INDArray arr, CoordinateFunction coordinateFunction) {
        Shape.iterate(0, arr.rank(), arr.shape(), new long[arr.rank()], coordinateFunction);
    }

    /**
     * Iterate over 2
     * coordinate spaces given 2 arrays
     * @param arr the first array
     * @param arr2 the second array
     * @param coordinateFunction the coordinate function to use
     *
     */
    public static void iterate(INDArray arr, INDArray arr2, CoordinateFunction coordinateFunction) {
        Shape.iterate(0, arr.rank(), arr.shape(), new long[arr.rank()], 0, arr2.rank(), arr2.shape(),
                new long[arr2.rank()], coordinateFunction);
    }

    /**
     * Iterate over a pair of coordinates
     * @param dimension
     * @param n
     * @param size
     * @param res
     * @param dimension2
     * @param n2
     * @param size2
     * @param res2
     * @param func
     */
    public static void iterate(int dimension, int n, int[] size, int[] res, int dimension2, int n2, int[] size2,
                               int[] res2, CoordinateFunction func) {
        if (dimension >= n || dimension2 >= n2) {
            // stop clause
            func.process(ArrayUtil.toLongArray(res), ArrayUtil.toLongArray(res2));
            return;
        }

        if (size2.length != size.length) {
            if (dimension >= size.length)
                return;
            for (int i = 0; i < size[dimension]; i++) {
                if (dimension2 >= size2.length)
                    break;
                for (int j = 0; j < size2[dimension2]; j++) {
                    res[dimension] = i;
                    res2[dimension2] = j;
                    iterate(dimension + 1, n, size, res, dimension2 + 1, n2, size2, res2, func);
                }

            }
        } else {
            if (dimension >= size.length)
                return;

            for (int i = 0; i < size[dimension]; i++) {
                for (int j = 0; j < size2[dimension2]; j++) {
                    if (dimension2 >= size2.length)
                        break;
                    res[dimension] = i;
                    res2[dimension2] = j;
                    iterate(dimension + 1, n, size, res, dimension2 + 1, n2, size2, res2, func);
                }

            }
        }
    }

    public static void iterate(int dimension, int n, long[] size, long[] res, int dimension2, int n2, long[] size2,
                               long[] res2, CoordinateFunction func) {
        if (dimension >= n || dimension2 >= n2) {
            // stop clause
            func.process(res, res2);
            return;
        }

        if (size2.length != size.length) {
            if (dimension >= size.length)
                return;
            for (int i = 0; i < size[dimension]; i++) {
                if (dimension2 >= size2.length)
                    break;
                for (int j = 0; j < size2[dimension2]; j++) {
                    res[dimension] = i;
                    res2[dimension2] = j;
                    iterate(dimension + 1, n, size, res, dimension2 + 1, n2, size2, res2, func);
                }

            }
        } else {
            if (dimension >= size.length)
                return;

            for (int i = 0; i < size[dimension]; i++) {
                for (int j = 0; j < size2[dimension2]; j++) {
                    if (dimension2 >= size2.length)
                        break;
                    res[dimension] = i;
                    res2[dimension2] = j;
                    iterate(dimension + 1, n, size, res, dimension2 + 1, n2, size2, res2, func);
                }

            }
        }
    }


    /**
     * Iterate over a pair of coordinates
     * @param dimension
     * @param n
     * @param size
     */
    public static void iterate(int dimension, int n, int[] size, int[] res, CoordinateFunction func) {
        if (dimension >= n) { //stop clause
            func.process(ArrayUtil.toLongArray(res));
            return;
        }
        for (int i = 0; i < size[dimension]; i++) {
            res[dimension] = i;
            iterate(dimension + 1, n, ArrayUtil.toLongArray(size), ArrayUtil.toLongArray(res), func);
        }
    }

    public static void iterate(int dimension, int n, long[] size, long[] res, CoordinateFunction func) {
        if (dimension >= n) { //stop clause
            func.process(res);
            return;
        }
        for (int i = 0; i < size[dimension]; i++) {
            res[dimension] = i;
            iterate(dimension + 1, n, size, res, func);
        }
    }



    /**
     * Get an offset for retrieval
     * from a data buffer
     * based on the given
     * shape stride and given indices
     * @param baseOffset the offset to start from
     * @param shape the shape of the array
     * @param stride the stride of the array
     * @param indices the indices to iterate over
     * @return the double at the specified index
     */
    public static long getOffset(long baseOffset, int[] shape, int[] stride, int... indices) {
        //int ret =  mappers[shape.length].getOffset(baseOffset, shape, stride, indices);
        if (shape.length != stride.length || indices.length != shape.length)
            throw new IllegalArgumentException("Indexes, shape, and stride must be the same length");
        long offset = baseOffset;
        for (int i = 0; i < shape.length; i++) {
            if (indices[i] >= shape[i])
                throw new IllegalArgumentException(
                        String.format("J: Index [%d] must not be >= shape[%d]=%d.", i, i, shape[i]));
            if (shape[i] != 1) {
                offset += indices[i] * stride[i];
            }
        }

        return offset;
    }

    /**
     * Get the offset of the specified indices from the shape info buffer
     *
     * @param shapeInformation    Shape information to get the offset for
     * @param indices             Indices array to get the offset for (must be same length as array rank)
     * @return                    Buffer offset fo the specified indices
     */
    public static long getOffset(IntBuffer shapeInformation, int[] indices) {
        // FIXME: int cast
        return getOffset(shapeInformation, ArrayUtil.toLongArray(indices));
    }

    public static long getOffset(LongBuffer shapeInformation, int[] indices) {
        // FIXME: int cast
        return getOffset(shapeInformation, ArrayUtil.toLongArray(indices));
    }

    public static long getOffset(LongBuffer shapeInformation, long... indices) {
        int rank = rank(shapeInformation);
        if (indices.length != rank)
            throw new IllegalArgumentException("Indexes must be same length as array rank");
        long offset = 0;
        for (int i = 0; i < rank; i++) {
            int size_dimi = (int) size(shapeInformation, i);
            if (size_dimi != 1) {
                offset += indices[i] * stride(shapeInformation, i);
            }
        }
        return offset;
    }

    public static long getOffset(IntBuffer shapeInformation, long... indices) {
        int rank = rank(shapeInformation);
        if (indices.length != rank)
            throw new IllegalArgumentException("Indexes must be same length as array rank");
        long offset = 0;
        for (int i = 0; i < rank; i++) {
            int size_dimi = size(shapeInformation, i);
            if (size_dimi != 1) {
                offset += indices[i] * stride(shapeInformation, i);
            }
        }
        return offset;
    }

    /**
     * Get the offset of the specified indices from the shape info buffer
     *
     * @param shapeInformation    Shape information to get the offset for
     * @param indices             Indices array to get the offset for (must be same length as array rank)
     * @return                    Buffer offset fo the specified indices
     */
    public static long getOffset(DataBuffer shapeInformation, int[] indices) {
        // FIXME: int cast
        return getOffset(shapeInformation, ArrayUtil.toLongArray(indices));
    }
    public static long getOffset(DataBuffer shapeInformation, long... indices) {
        int rank = rank(shapeInformation);
        if (indices.length != rank)
            throw new IllegalArgumentException("Indexes must be same length as array rank");
        long offset = 0;
        for (int i = 0; i < rank; i++) {
            int size_dimi = size(shapeInformation, i);
            if (indices[i] > size_dimi)
                throw new IllegalArgumentException(
                        String.format("J: Index [%d] must not be >= shape[%d]=%d.", i, i, size_dimi));
            if (size_dimi != 1) {
                offset += indices[i] * stride(shapeInformation, i);
            }
        }
        return offset;
    }


    public static long getOffset(int[] shapeInformation, int... indices) {
        int rank = rank(shapeInformation);
         long offset = 0;
        for (int i = 0; i < Math.min(rank,indices.length); i++) {
            int size_dimi = size(shapeInformation, i);
            if (indices[i] > size_dimi)
                throw new IllegalArgumentException(
                        String.format("J: Index [%d] must not be >= shape[%d]=%d.", i, i, size_dimi));
            if (size_dimi != 1) {
                offset += indices[i] * stride(shapeInformation, i);
            }
        }
        return offset;
    }

    public static long getOffset(long[] shapeInformation, int... indices) {
        int rank = rank(shapeInformation);
        long offset = 0;
        for (int i = 0; i < Math.min(rank,indices.length); i++) {
            long size_dimi = size(shapeInformation, i);
            if (indices[i] > size_dimi)
                throw new IllegalArgumentException(
                        String.format("J: Index [%d] must not be >= shape[%d]=%d.", i, i, size_dimi));
            if (size_dimi != 1) {
                offset += indices[i] * stride(shapeInformation, i);
            }
        }
        return offset;
    }

    public static long getOffset(long[] shapeInformation, long... indices) {
        int rank = rank(shapeInformation);
        long offset = 0;
        for (int i = 0; i < Math.min(rank,indices.length); i++) {
            long size_dimi = size(shapeInformation, i);
            if (indices[i] > size_dimi)
                throw new IllegalArgumentException(
                        String.format("J: Index [%d] must not be >= shape[%d]=%d.", i, i, size_dimi));
            if (size_dimi != 1) {
                offset += indices[i] * stride(shapeInformation, i);
            }
        }
        return offset;
    }

    /** Get the offset of the specified [row,col] for the 2d array
     *
     * @param shapeInformation    Shape information
     * @param row                 Row index to get the offset for
     * @param col                 Column index to get the offset for
     * @return                    Buffer offset
     */
    public static long getOffset(DataBuffer shapeInformation, int row, int col) {
        int rank = rank(shapeInformation);
        if (rank != 2)
            throw new IllegalArgumentException(
                    "Cannot use this getOffset method on arrays of rank != 2 (rank is: " + rank + ")");
        return getOffsetUnsafe(shapeInformation, row, col);
    }

    /**
     * Identical to {@link Shape#getOffset(DataBuffer, int, int)} but without input validation on array rank
     */
    public static long getOffsetUnsafe(DataBuffer shapeInformation, int row, int col) {
        long offset = 0;
        int size_0 = sizeUnsafe(shapeInformation, 0);
        int size_1 = sizeUnsafe(shapeInformation, 1);
        if (row >= size_0 || col >= size_1)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + row + "," + col + "] from a "
                    + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += row * strideUnsafe(shapeInformation, 0, 2);
        if (size_1 != 1)
            offset += col * strideUnsafe(shapeInformation, 1, 2);

        return offset;
    }


    public static long getOffsetUnsafe(int[] shapeInformation, int row, int col) {
        long offset = 0;
        int size_0 = sizeUnsafe(shapeInformation, 0);
        int size_1 = sizeUnsafe(shapeInformation, 1);
        if (row >= size_0 || col >= size_1 && !Shape.isVector(Shape.shape(shapeInformation)) && !Shape.shapeIsScalar(Shape.shape(shapeInformation)))
            throw new IllegalArgumentException("Invalid indices: cannot get [" + row + "," + col + "] from a "
                    + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += row * strideUnsafe(shapeInformation, 0, 2);
        if (size_1 != 1)
            offset += col * strideUnsafe(shapeInformation, 1, 2);

        return offset;
    }

    public static long getOffsetUnsafe(long[] shapeInformation, long row, long col) {
        long offset = 0;
        long size_0 = sizeUnsafe(shapeInformation, 0);
        long size_1 = sizeUnsafe(shapeInformation, 1);
        if (row >= size_0 || col >= size_1 && !Shape.isVector(Shape.shape(shapeInformation)) && !Shape.shapeIsScalar(Shape.shape(shapeInformation)))
            throw new IllegalArgumentException("Invalid indices: cannot get [" + row + "," + col + "] from a "
                    + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += row * strideUnsafe(shapeInformation, 0, 2);
        if (size_1 != 1)
            offset += col * strideUnsafe(shapeInformation, 1, 2);

        return offset;
    }

    /** Get the offset of the specified [row,col] for the 2d array
     *
     * @param shapeInformation    Shape information
     * @param row                 Row index to get the offset for
     * @param col                 Column index to get the offset for
     * @return                    Buffer offset
     */
    public static long getOffset(IntBuffer shapeInformation, int row, int col) {
        int rank = rank(shapeInformation);
        if (rank != 2)
            throw new IllegalArgumentException(
                    "Cannot use this getOffset method on arrays of rank != 2 (rank is: " + rank + ")");
        long offset = 0;
        int size_0 = size(shapeInformation, 0);
        int size_1 = size(shapeInformation, 1);
        if (row >= size_0 || col >= size_1)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + row + "," + col + "] from a "
                    + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += row * stride(shapeInformation, 0);
        if (size_1 != 1)
            offset += col * stride(shapeInformation, 1);

        return offset;
    }

    /** Get the offset of the specified [dim0,dim1,dim2] for the 3d array
     *
     * @param shapeInformation    Shape information
     * @param dim0                Row index to get the offset for
     * @param dim1                Column index to get the offset for
     * @param dim2                dimension 2 index to get the offset for
     * @return                    Buffer offset
     */
    public static long getOffset(IntBuffer shapeInformation, int dim0, int dim1, int dim2) {
        int rank = rank(shapeInformation);
        if (rank != 3)
            throw new IllegalArgumentException(
                    "Cannot use this getOffset method on arrays of rank != 3 (rank is: " + rank + ")");
        long offset = 0;
        int size_0 = size(shapeInformation, 0);
        int size_1 = size(shapeInformation, 1);
        int size_2 = size(shapeInformation, 2);
        if (dim0 >= size_0 || dim1 >= size_1 || dim2 >= size_2)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + dim0 + "," + dim1 + "," + dim2
                    + "] from a " + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += dim0 * stride(shapeInformation, 0);
        if (size_1 != 1)
            offset += dim1 * stride(shapeInformation, 1);
        if (size_2 != 1)
            offset += dim2 * stride(shapeInformation, 2);

        return offset;
    }

    /** Get the offset of the specified [dim0,dim1,dim2] for the 3d array
     *
     * @param shapeInformation    Shape information
     * @param dim0                Row index to get the offset for
     * @param dim1                Column index to get the offset for
     * @param dim2                dimension 2 index to get the offset for
     * @return                    Buffer offset
     */
    public static long getOffset(DataBuffer shapeInformation, int dim0, int dim1, int dim2) {
        int rank = rank(shapeInformation);
        if (rank != 3)
            throw new IllegalArgumentException(
                    "Cannot use this getOffset method on arrays of rank != 3 (rank is: " + rank + ")");
        return getOffsetUnsafe(shapeInformation, dim0, dim1, dim2);
    }

    /**
     * Identical to {@link Shape#getOffset(DataBuffer, int, int, int)} but without input validation on array rank
     */
    public static long getOffsetUnsafe(DataBuffer shapeInformation, int dim0, int dim1, int dim2) {
        long offset = 0;
        int size_0 = sizeUnsafe(shapeInformation, 0);
        int size_1 = sizeUnsafe(shapeInformation, 1);
        int size_2 = sizeUnsafe(shapeInformation, 2);
        if (dim0 >= size_0 || dim1 >= size_1 || dim2 >= size_2)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + dim0 + "," + dim1 + "," + dim2
                    + "] from a " + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += dim0 * strideUnsafe(shapeInformation, 0, 3);
        if (size_1 != 1)
            offset += dim1 * strideUnsafe(shapeInformation, 1, 3);
        if (size_2 != 1)
            offset += dim2 * strideUnsafe(shapeInformation, 2, 3);

        return offset;
    }

    public static long getOffsetUnsafe(int[] shapeInformation, int dim0, int dim1, int dim2) {
        int offset = 0;
        int size_0 = sizeUnsafe(shapeInformation, 0);
        int size_1 = sizeUnsafe(shapeInformation, 1);
        int size_2 = sizeUnsafe(shapeInformation, 2);
        if (dim0 >= size_0 || dim1 >= size_1 || dim2 >= size_2)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + dim0 + "," + dim1 + "," + dim2
                    + "] from a " + Arrays.toString(shapeInformation) + " NDArray");

        if (size_0 != 1)
            offset += dim0 * strideUnsafe(shapeInformation, 0, 3);
        if (size_1 != 1)
            offset += dim1 * strideUnsafe(shapeInformation, 1, 3);
        if (size_2 != 1)
            offset += dim2 * strideUnsafe(shapeInformation, 2, 3);

        return offset;
    }

    /** Get the offset of the specified [dim0,dim1,dim2,dim3] for the 4d array
     *
     * @param shapeInformation    Shape information
     * @param dim0                Row index to get the offset for
     * @param dim1                Column index to get the offset for
     * @param dim2                dimension 2 index to get the offset for
     * @param dim3                dimension 3 index to get the offset for
     * @return                    Buffer offset
     */
    public static long getOffset(IntBuffer shapeInformation, int dim0, int dim1, int dim2, int dim3) {
        int rank = rank(shapeInformation);
        if (rank != 4)
            throw new IllegalArgumentException(
                    "Cannot use this getOffset method on arrays of rank != 4 (rank is: " + rank + ")");
        long offset = 0;
        int size_0 = size(shapeInformation, 0);
        int size_1 = size(shapeInformation, 1);
        int size_2 = size(shapeInformation, 2);
        int size_3 = size(shapeInformation, 3);
        if (dim0 >= size_0 || dim1 >= size_1 || dim2 >= size_2 || dim3 >= size_3)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + dim0 + "," + dim1 + "," + dim2 + ","
                    + dim3 + "] from a " + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += dim0 * stride(shapeInformation, 0);
        if (size_1 != 1)
            offset += dim1 * stride(shapeInformation, 1);
        if (size_2 != 1)
            offset += dim2 * stride(shapeInformation, 2);
        if (size_3 != 1)
            offset += dim3 * stride(shapeInformation, 3);

        return offset;
    }

    /** Get the offset of the specified [dim0,dim1,dim2,dim3] for the 4d array
     *
     * @param shapeInformation    Shape information
     * @param dim0                Row index to get the offset for
     * @param dim1                Column index to get the offset for
     * @param dim2                dimension 2 index to get the offset for
     * @param dim3                dimension 3 index to get the offset for
     * @return                    Buffer offset
     */
    public static long getOffset(DataBuffer shapeInformation, int dim0, int dim1, int dim2, int dim3) {
        int rank = rank(shapeInformation);
        if (rank != 4)
            throw new IllegalArgumentException(
                    "Cannot use this getOffset method on arrays of rank != 4 (rank is: " + rank + ")");
        return getOffsetUnsafe(shapeInformation, dim0, dim1, dim2, dim3);
    }

    public static long getOffsetUnsafe(DataBuffer shapeInformation, int dim0, int dim1, int dim2, int dim3) {
        long offset = 0;
        int size_0 = sizeUnsafe(shapeInformation, 0);
        int size_1 = sizeUnsafe(shapeInformation, 1);
        int size_2 = sizeUnsafe(shapeInformation, 2);
        int size_3 = sizeUnsafe(shapeInformation, 3);
        if (dim0 >= size_0 || dim1 >= size_1 || dim2 >= size_2 || dim3 >= size_3)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + dim0 + "," + dim1 + "," + dim2 + ","
                    + dim3 + "] from a " + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += dim0 * strideUnsafe(shapeInformation, 0, 4);
        if (size_1 != 1)
            offset += dim1 * strideUnsafe(shapeInformation, 1, 4);
        if (size_2 != 1)
            offset += dim2 * strideUnsafe(shapeInformation, 2, 4);
        if (size_3 != 1)
            offset += dim3 * strideUnsafe(shapeInformation, 3, 4);

        return offset;
    }


    public static long getOffsetUnsafe(int[] shapeInformation, int dim0, int dim1, int dim2, int dim3) {
        long offset = 0;
        int size_0 = sizeUnsafe(shapeInformation, 0);
        int size_1 = sizeUnsafe(shapeInformation, 1);
        int size_2 = sizeUnsafe(shapeInformation, 2);
        int size_3 = sizeUnsafe(shapeInformation, 3);
        if (dim0 >= size_0 || dim1 >= size_1 || dim2 >= size_2 || dim3 >= size_3)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + dim0 + "," + dim1 + "," + dim2 + ","
                    + dim3 + "] from a " + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += dim0 * strideUnsafe(shapeInformation, 0, 4);
        if (size_1 != 1)
            offset += dim1 * strideUnsafe(shapeInformation, 1, 4);
        if (size_2 != 1)
            offset += dim2 * strideUnsafe(shapeInformation, 2, 4);
        if (size_3 != 1)
            offset += dim3 * strideUnsafe(shapeInformation, 3, 4);

        return offset;
    }

    public static long getOffsetUnsafe(long[] shapeInformation, long dim0, long dim1, long dim2, long dim3) {
        long offset = 0;
        long size_0 = sizeUnsafe(shapeInformation, 0);
        long size_1 = sizeUnsafe(shapeInformation, 1);
        long size_2 = sizeUnsafe(shapeInformation, 2);
        long size_3 = sizeUnsafe(shapeInformation, 3);
        if (dim0 >= size_0 || dim1 >= size_1 || dim2 >= size_2 || dim3 >= size_3)
            throw new IllegalArgumentException("Invalid indices: cannot get [" + dim0 + "," + dim1 + "," + dim2 + ","
                    + dim3 + "] from a " + Arrays.toString(shape(shapeInformation)) + " NDArray");

        if (size_0 != 1)
            offset += dim0 * strideUnsafe(shapeInformation, 0, 4);
        if (size_1 != 1)
            offset += dim1 * strideUnsafe(shapeInformation, 1, 4);
        if (size_2 != 1)
            offset += dim2 * strideUnsafe(shapeInformation, 2, 4);
        if (size_3 != 1)
            offset += dim3 * strideUnsafe(shapeInformation, 3, 4);

        return offset;
    }

    /**
     * Output an int array for a particular dimension
     * @param axes the axes
     * @param shape the current shape
     * @return
     */
    public static int[] sizeForAxes(int[] axes, int[] shape) {
        int[] ret = new int[shape.length];
        for (int i = 0; i < axes.length; i++) {
            ret[i] = shape[axes[i]];
        }
        return ret;
    }

    /**
     * Returns whether the given shape is a vector
     *
     * @param shapeInfo the shapeinfo to test
     * @return whether the given shape is a vector
     */
    public static boolean isVector(IntBuffer shapeInfo) {
        int rank = Shape.rank(shapeInfo);
        if (rank > 2 || rank < 1)
            return false;
        else {
            int len = Shape.length(shapeInfo);
            IntBuffer shape = Shape.shapeOf(shapeInfo);
            return shape.get(0) == len || shape.get(1) == len;
        }
    }

    /**
     * Returns whether the given shape is a vector
     *
     * @param shapeInfo the shapeinfo to test
     * @return whether the given shape is a vector
     */
    public static boolean isVector(DataBuffer shapeInfo) {
        int rank = Shape.rank(shapeInfo);
        if (rank > 2 || rank < 1)
            return false;
        else {
            long len = Shape.length(shapeInfo);
            DataBuffer shape = Shape.shapeOf(shapeInfo);
            return shape.getInt(0) == len || shape.getInt(1) == len;
        }
    }

    /**
     * Returns whether the given shape is a vector
     *
     * @param shape the shape to test
     * @return whether the given shape is a vector
     */
    public static boolean isVector(int[] shape) {
        if (shape.length > 2 || shape.length < 1)
            return false;
        else {
            long len = ArrayUtil.prodLong(shape);
            return shape[0] == len || shape[1] == len;
        }
    }

    public static boolean isVector(long[] shape) {
        if (shape.length > 2 || shape.length < 1)
            return false;
        else {
            long len = ArrayUtil.prodLong(shape);
            return shape[0] == len || shape[1] == len;
        }
    }


    /**
     * Returns whether the passed in shape is a matrix
     *
     * @param shapeInfo whether the passed in shape is a matrix
     * @return true if the shape is a matrix false otherwise
     */
    public static boolean isMatrix(IntBuffer shapeInfo) {
        int rank = Shape.rank(shapeInfo);
        if (rank != 2)
            return false;
        return !isVector(shapeInfo);
    }


    /**
     * Returns whether the passed in shape is a matrix
     *
     * @param shapeInfo whether the passed in shape is a matrix
     * @return true if the shape is a matrix false otherwise
     */
    public static boolean isMatrix(DataBuffer shapeInfo) {
        int rank = Shape.rank(shapeInfo);
        if (rank != 2)
            return false;
        return !isVector(shapeInfo);
    }

    /**
     * Returns whether the passed in shape is a matrix
     *
     * @param shape whether the passed in shape is a matrix
     * @return true if the shape is a matrix false otherwise
     */
    public static boolean isMatrix(int[] shape) {
        if (shape.length != 2)
            return false;
        return !isVector(shape);
    }

    public static boolean isMatrix(long[] shape) {
        if (shape.length != 2)
            return false;
        return !isVector(shape);
    }


    /**
     * Gets rid of any singleton dimensions of the given array
     *
     * @param shape the shape to squeeze
     * @return the array with all of the singleton dimensions removed
     */
    public static int[] squeeze(int[] shape) {
        if (isColumnVectorShape(shape))
            return shape;

        List<Integer> ret = new ArrayList<>();

        //strip all but last dimension
        for (int i = 0; i < shape.length; i++)
            if (shape[i] != 1)
                ret.add(shape[i]);
        return ArrayUtil.toArray(ret);
    }

    /**
     * Gets rid of any singleton dimensions of the given array
     *
     * @param shape the shape to squeeze
     * @return the array with all of the singleton dimensions removed
     */
    public static long[] squeeze(long[] shape) {
        if (isColumnVectorShape(shape))
            return shape;

        List<Long> ret = new ArrayList<>();

        //strip all but last dimension
        for (int i = 0; i < shape.length; i++)
            if (shape[i] != 1)
                ret.add(shape[i]);
        return ArrayUtil.toArrayLong(ret);
    }



    /**
     * Returns whether 2 shapes are equals by checking for dimension semantics
     * as well as array equality
     *
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the shapes are equivalent
     */
    public static boolean shapeEquals(int[] shape1, int[] shape2) {
        if (isColumnVectorShape(shape1) && isColumnVectorShape(shape2)) {
            return Arrays.equals(shape1, shape2);
        }

        if (isRowVectorShape(shape1) && isRowVectorShape(shape2)) {
            int[] shape1Comp = squeeze(shape1);
            int[] shape2Comp = squeeze(shape2);
            return Arrays.equals(shape1Comp, shape2Comp);
        }

        //scalars
        if(shape1.length == 0 || shape2.length == 0) {
            if(shape1.length == 0 && shapeIsScalar(shape2)) {
                return true;
            }

            if(shape2.length == 0 && shapeIsScalar(shape1)) {
                return true;
            }
        }


        shape1 = squeeze(shape1);
        shape2 = squeeze(shape2);

        return scalarEquals(shape1, shape2) || Arrays.equals(shape1, shape2);
    }


    /**
     * Returns whether 2 shapes are equals by checking for dimension semantics
     * as well as array equality
     *
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the shapes are equivalent
     */
    public static boolean shapeEquals(long[] shape1, long[] shape2) {
        if (isColumnVectorShape(shape1) && isColumnVectorShape(shape2)) {
            return Arrays.equals(shape1, shape2);
        }

        if (isRowVectorShape(shape1) && isRowVectorShape(shape2)) {
            long[] shape1Comp = squeeze(shape1);
            long[] shape2Comp = squeeze(shape2);
            return Arrays.equals(shape1Comp, shape2Comp);
        }

        //scalars
        if(shape1.length == 0 || shape2.length == 0) {
            if(shape1.length == 0 && shapeIsScalar(shape2)) {
                return true;
            }

            if(shape2.length == 0 && shapeIsScalar(shape1)) {
                return true;
            }
        }


        shape1 = squeeze(shape1);
        shape2 = squeeze(shape2);

        return scalarEquals(shape1, shape2) || Arrays.equals(shape1, shape2);
    }


    /**
     * Returns true if the given shapes are both scalars (0 dimension or shape[0] == 1)
     *
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the 2 shapes are equal based on scalar rules
     */
    public static boolean scalarEquals(int[] shape1, int[] shape2) {
        if (shape1.length == 0 && shape2.length == 1 && shape2[0] == 1) {
            return true;
        } else if (shape2.length == 0 && shape1.length == 1 && shape1[0] == 1) {
            return true;
        }

        return false;
    }

    public static boolean scalarEquals(long[] shape1, long[] shape2) {
        if (shape1.length == 0 && shape2.length == 1 && shape2[0] == 1) {
            return true;
        } else if (shape2.length == 0 && shape1.length == 1 && shape1[0] == 1) {
            return true;
        }

        return false;
    }

    /**
     * Returns true if the given shape is of length 1
     * or provided the shape length is 2:
     * element 0 is 1
     * @param shapeInfo the shape info to check
     * @return true if the above conditions hold,false otherwise
     */
    public static boolean isRowVectorShape(DataBuffer shapeInfo) {
        int rank = Shape.rank(shapeInfo);
        DataBuffer shape = Shape.shapeOf(shapeInfo);
        return (rank == 2 && shape.getInt(0) == 1) || rank == 1;

    }

    /**
     * Returns true if the given shape is of length 1
     * or provided the shape length is 2:
     * element 0 is 1
     * @param shapeInfo the shape info to check
     * @return true if the above conditions hold,false otherwise
     */
    public static boolean isRowVectorShape(IntBuffer shapeInfo) {
        int rank = Shape.rank(shapeInfo);
        IntBuffer shape = Shape.shapeOf(shapeInfo);
        return (rank == 2 && shape.get(0) == 1) || rank == 1;

    }

    /**
     * Returns true if the given shape is of length 1
     * or provided the shape length is 2:
     * element 0 is 1
     * @param shape the shape to check
     * @return true if the above conditions hold,false otherwise
     */
    public static boolean isRowVectorShape(int[] shape) {
        return (shape.length == 2 && shape[0] == 1) || shape.length == 1;
    }

    public static boolean isRowVectorShape(long[] shape) {
        return (shape.length == 2 && shape[0] == 1) || shape.length == 1;
    }

    /**
     * Returns true if the given shape is length 2 and
     * the size at element 1 is 1
     * @param shape the shape to check
     * @return true if the above listed conditions
     * hold false otherwise
     */
    public static boolean isColumnVectorShape(int[] shape) {
        return (shape.length == 2 && shape[1] == 1);
    }

    /**
     * Returns true if the given shape length is 2
     * and the size at element 1 is 1
     * @param shape
     * @return
     */
    public static boolean isColumnVectorShape(long[] shape) {
        return (shape.length == 2 && shape[1] == 1);
    }



    /**
     * If a shape array is ony 1 in length
     * it returns a row vector
     * @param shape the shape of the array
     * @return the shape as is if its already >= 2 in length
     * otherwise a row vector shape
     */
    public static int[] ensureAtMinRowVector(int... shape) {
        if (shape.length >= 2)
            return shape;
        return new int[] {1, shape[0]};
    }


    public static long getTADLength(int[] shape, int... dimensions) {
        int tadLength = 1;
        for (int i = 0; i < dimensions.length; i++) {
            tadLength *= shape[dimensions[i]];
        }

        return tadLength;
    }


    public static long getTADLength(long[] shape, int... dimensions) {
        int tadLength = 1;
        for (int i = 0; i < dimensions.length; i++) {
            tadLength *= shape[dimensions[i]];
        }

        return tadLength;
    }




    /**
     *
     * @param shape
     * @param stride
     * @param isFOrder
     * @return
     */
    public static int elementWiseStride(int[] shape, int[] stride, boolean isFOrder) {
        // 0D edge case
        if (shape.length == 0 && stride.length == 0)
            return 1;

        if (shape.length == 1 && stride.length == 1)
            return 1;

        int oldnd;
        int[] olddims = ArrayUtil.copy(shape);
        int[] oldstrides = ArrayUtil.copy(stride);
        long np, op, last_stride;
        int oi, oj, ok, ni, nj, nk;
        long[] newStrides = new long[stride.length];
        oldnd = 0;
        //set the shape to be 1 x length
        int newShapeRank = 2;
        long[] newShape = new long[shape.length];
        newShape[0] = 1;
        newShape[1] = ArrayUtil.prodLong(shape);

        /*
         * Remove axes with dimension 1 from the old array. They have no effect
         * but would need special cases since their strides do not matter.
         */
        for (oi = 0; oi < shape.length; oi++) {
            if (shape[oi] != 1) {
                olddims[oldnd] = shape[oi];
                oldstrides[oldnd] = stride[oi];
                oldnd++;
            }
        }

        np = 1;
        for (ni = 0; ni < newShapeRank; ni++) {
            np *= newShape[ni];
        }
        op = 1;
        for (oi = 0; oi < oldnd; oi++) {
            op *= olddims[oi];
        }
        if (np != op) {
            /* different total sizes; no hope */
            return -1;
        }

        if (np == 0) {
            /* the current code does not handle 0-sized arrays, so give up */
            return -1;
        }

        /* oi to oj and ni to nj give the axis ranges currently worked with */
        oi = 0;
        oj = 1;
        ni = 0;
        nj = 1;
        while (ni < newShapeRank && oi < oldnd) {
            np = newShape[ni];
            op = olddims[oi];

            while (np != op) {
                if (np < op) {
                    /* Misses trailing 1s, these are handled later */
                    np *= newShape[nj++];
                } else {
                    op *= olddims[oj++];
                }
            }

            /* Check whether the original axes can be combined */
            for (ok = oi; ok < oj - 1; ok++) {
                if (isFOrder) {
                    if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
                        /* not contiguous enough */
                        return -1;
                    }
                } else {
                    /* C order */
                    if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
                        /* not contiguous enough */
                        return -1;
                    }
                }
            }

            /* Calculate new strides for all axes currently worked with */
            if (isFOrder) {
                newStrides[ni] = oldstrides[oi];
                for (nk = ni + 1; nk < nj; nk++) {
                    newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
                }
            } else {
                /* C order */
                newStrides[nj - 1] = oldstrides[oj - 1];
                for (nk = nj - 1; nk > ni; nk--) {
                    newStrides[nk - 1] = newStrides[nk] * newShape[nk];
                }
            }
            ni = nj++;
            oi = oj++;
        }

        /*
         * Set strides corresponding to trailing 1s of the new shape.
         */
        if (ni >= 1) {
            last_stride = newStrides[ni - 1];
        } else {
            last_stride = stride[shape.length - 1];
        }
        if (isFOrder && ni >= 1) {
            last_stride *= newShape[ni - 1];
        }
        for (nk = ni; nk < newShapeRank; nk++) {
            newStrides[nk] = last_stride;
        }
        if (newStrides[newShapeRank - 1] >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Element size can not be >= Integer.MAX_VALUE");
        //returns the last element of the new stride array
        return (int) newStrides[newShapeRank - 1];
    }

    public static long elementWiseStride(long[] shape, long[] stride, boolean isFOrder) {
        // 0D edge case
        if (shape.length == 0 && stride.length == 0)
            return 1;

        if (shape.length == 1 && stride.length == 1)
            return 1;

        int oldnd;
        long[] olddims = ArrayUtil.copy(shape);
        long[] oldstrides = ArrayUtil.copy(stride);
        long np, op, last_stride;
        int oi, oj, ok, ni, nj, nk;
        long[] newStrides = new long[stride.length];
        oldnd = 0;
        //set the shape to be 1 x length
        int newShapeRank = 2;
        long[] newShape = new long[shape.length];
        newShape[0] = 1;
        newShape[1] = ArrayUtil.prodLong(shape);

        /*
         * Remove axes with dimension 1 from the old array. They have no effect
         * but would need special cases since their strides do not matter.
         */
        for (oi = 0; oi < shape.length; oi++) {
            if (shape[oi] != 1) {
                olddims[oldnd] = shape[oi];
                oldstrides[oldnd] = stride[oi];
                oldnd++;
            }
        }

        np = 1;
        for (ni = 0; ni < newShapeRank; ni++) {
            np *= newShape[ni];
        }
        op = 1;
        for (oi = 0; oi < oldnd; oi++) {
            op *= olddims[oi];
        }
        if (np != op) {
            /* different total sizes; no hope */
            return -1;
        }

        if (np == 0) {
            /* the current code does not handle 0-sized arrays, so give up */
            return -1;
        }

        /* oi to oj and ni to nj give the axis ranges currently worked with */
        oi = 0;
        oj = 1;
        ni = 0;
        nj = 1;
        while (ni < newShapeRank && oi < oldnd) {
            np = newShape[ni];
            op = olddims[oi];

            while (np != op) {
                if (np < op) {
                    /* Misses trailing 1s, these are handled later */
                    np *= newShape[nj++];
                } else {
                    op *= olddims[oj++];
                }
            }

            /* Check whether the original axes can be combined */
            for (ok = oi; ok < oj - 1; ok++) {
                if (isFOrder) {
                    if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
                        /* not contiguous enough */
                        return -1;
                    }
                } else {
                    /* C order */
                    if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
                        /* not contiguous enough */
                        return -1;
                    }
                }
            }

            /* Calculate new strides for all axes currently worked with */
            if (isFOrder) {
                newStrides[ni] = oldstrides[oi];
                for (nk = ni + 1; nk < nj; nk++) {
                    newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
                }
            } else {
                /* C order */
                newStrides[nj - 1] = oldstrides[oj - 1];
                for (nk = nj - 1; nk > ni; nk--) {
                    newStrides[nk - 1] = newStrides[nk] * newShape[nk];
                }
            }
            ni = nj++;
            oi = oj++;
        }

        /*
         * Set strides corresponding to trailing 1s of the new shape.
         */
        if (ni >= 1) {
            last_stride = newStrides[ni - 1];
        } else {
            last_stride = stride[shape.length - 1];
        }
        if (isFOrder && ni >= 1) {
            last_stride *= newShape[ni - 1];
        }
        for (nk = ni; nk < newShapeRank; nk++) {
            newStrides[nk] = last_stride;
        }
        if (newStrides[newShapeRank - 1] >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Element size can not be >= Integer.MAX_VALUE");
        //returns the last element of the new stride array
        return newStrides[newShapeRank - 1];
    }

    public static INDArray newShapeNoCopy(INDArray arr, int[] newShape, boolean isFOrder) {
        return newShapeNoCopy(arr, ArrayUtil.toLongArray(newShape), isFOrder);
    }
    /**
     * A port of numpy's reshaping algorithm that leverages
     * no copy where possible and returns
     * null if the reshape
     * couldn't happen without copying
     * @param arr  the array to reshape
     * @param newShape the new shape
     * @param isFOrder whether the array will be fortran ordered or not
     * @return null if a reshape isn't possible, or a new ndarray
     */
    public static INDArray newShapeNoCopy(INDArray arr, long[] newShape, boolean isFOrder) {
        int oldnd;
        long[] olddims = ArrayUtil.copy(arr.shape());
        long[] oldstrides = ArrayUtil.copy(arr.stride());
        long np, op, last_stride;
        int oi, oj, ok, ni, nj, nk;
        long[] newStrides = new long[newShape.length];
        oldnd = 0;
        /*
         * Remove axes with dimension 1 from the old array. They have no effect
         * but would need special cases since their strides do not matter.
         */
        for (oi = 0; oi < arr.rank(); oi++) {
            if (arr.size(oi) != 1) {
                olddims[oldnd] = arr.size(oi);
                oldstrides[oldnd] = arr.stride(oi);
                oldnd++;
            }
        }

        np = 1;
        for (ni = 0; ni < newShape.length; ni++) {
            np *= newShape[ni];
        }
        op = 1;
        for (oi = 0; oi < oldnd; oi++) {
            op *= olddims[oi];
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
        oi = 0;
        oj = 1;
        ni = 0;
        nj = 1;
        while (ni < newShape.length && oi < oldnd) {
            np = newShape[ni];
            op = olddims[oi];

            while (np != op) {
                if (np < op) {
                    /* Misses trailing 1s, these are handled later */
                    np *= newShape[nj++];
                } else {
                    op *= olddims[oj++];
                }
            }

            /* Check whether the original axes can be combined */
            for (ok = oi; ok < oj - 1; ok++) {
                if (isFOrder) {
                    if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
                        /* not contiguous enough */
                        return null;
                    }
                } else {
                    /* C order */
                    if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
                        /* not contiguous enough */
                        return null;
                    }
                }
            }

            /* Calculate new strides for all axes currently worked with */
            if (isFOrder) {
                newStrides[ni] = oldstrides[oi];
                for (nk = ni + 1; nk < nj; nk++) {
                    newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
                }
            } else {
                /* C order */
                newStrides[nj - 1] = oldstrides[oj - 1];
                for (nk = nj - 1; nk > ni; nk--) {
                    newStrides[nk - 1] = newStrides[nk] * newShape[nk];
                }
            }
            ni = nj++;
            oi = oj++;
        }

        /*
         * Set strides corresponding to trailing 1s of the new shape.
         */
        if (ni >= 1) {
            last_stride = newStrides[ni - 1];
        } else {
            last_stride = arr.elementStride();
        }
        if (isFOrder && ni >= 1) {
            last_stride *= newShape[ni - 1];
        }
        for (nk = ni; nk < newShape.length; nk++) {
            newStrides[nk] = last_stride;
        }

        if (arr instanceof IComplexNDArray)
            //return Nd4j.createComplex(arr.data(), newShape, newStrides, arr.offset());
            throw new UnsupportedOperationException();


        INDArray ret = Nd4j.create(arr.data(), newShape, newStrides, arr.offset(), isFOrder ? 'f' : 'c');


        return ret;
    }

    /**
     * Infer order from
     * @param shape the shape to infer by
     * @param stride the stride to infer by
     * @param elementStride the element stride to start at
     * @return the storage order given shape and element stride
     */
    public static boolean cOrFortranOrder(long[] shape, long[] stride, long elementStride) {
        long sd;
        long dim;
        int i;
        boolean cContiguous = true;
        boolean isFortran = true;

        sd = 1;
        for (i = shape.length - 1; i >= 0; --i) {
            dim = shape[i];

            if (stride[i] != sd) {
                cContiguous = false;
                break;
            }
            /* contiguous, if it got this far */
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }


        /* check if fortran contiguous */
        sd = elementStride;
        for (i = 0; i < shape.length; ++i) {
            dim = shape[i];
            if (stride[i] != sd) {
                isFortran = false;
            }
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }
        return cContiguous || isFortran;
    }

    @Deprecated
    public static boolean cOrFortranOrder(int[] shape, int[] stride, int elementStride) {
        return cOrFortranOrder(ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride), elementStride);
    }

    /**
     * Infer order from
     * @param shape the shape to infer by
     * @param stride the stride to infer by
     * @param elementStride the element stride to start at
     * @return the storage order given shape and element stride
     */
    public static char getOrder(int[] shape, int[] stride, int elementStride) {
        int sd;
        int dim;
        int i;
        boolean cContiguous = true;
        boolean isFortran = true;

        sd = 1;
        for (i = shape.length - 1; i >= 0; --i) {
            dim = shape[i];

            if (stride[i] != sd) {
                cContiguous = false;
                break;
            }
            /* contiguous, if it got this far */
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }


        /* check if fortran contiguous */
        sd = elementStride;
        for (i = 0; i < shape.length; ++i) {
            dim = shape[i];
            if (stride[i] != sd) {
                isFortran = false;
            }
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }

        if (isFortran && cContiguous)
            return 'a';
        else if (isFortran && !cContiguous)
            return 'f';
        else if (!isFortran && !cContiguous)
            return 'c';
        else
            return 'c';

    }


    public static char getOrder(long[] shape, long[] stride, long elementStride) {
        long sd;
        long dim;
        int i;
        boolean cContiguous = true;
        boolean isFortran = true;

        sd = 1;
        for (i = shape.length - 1; i >= 0; --i) {
            dim = shape[i];

            if (stride[i] != sd) {
                cContiguous = false;
                break;
            }
            /* contiguous, if it got this far */
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }


        /* check if fortran contiguous */
        sd = elementStride;
        for (i = 0; i < shape.length; ++i) {
            dim = shape[i];
            if (stride[i] != sd) {
                isFortran = false;
            }
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }

        if (isFortran && cContiguous)
            return 'a';
        else if (isFortran && !cContiguous)
            return 'f';
        else if (!isFortran && !cContiguous)
            return 'c';
        else
            return 'c';

    }

    /**
     * Infer the order for the ndarray based on the
     * array's strides
     * @param arr the array to get the
     *            ordering for
     * @return the ordering for the given array
     */
    public static char getOrder(INDArray arr) {
        return getOrder(arr.shape(), arr.stride(), arr.elementStride());
    }

    /**
     * Convert the given index (such as 1,1)
     * to a linear index
     * @param shape the shape of the indexes to convert
     * @param indices the index to convert
     * @return the linear index given the shape
     * and indices
     */
    public static long sub2Ind(int[] shape, int[] indices) {
        long index = 0;
        int shift = 1;
        for (int i = 0; i < shape.length; i++) {
            index += shift * indices[i];
            shift *= shape[i];
        }
        return index;
    }

    /**
     * Convert a linear index to
     * the equivalent nd index
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @param numIndices the number of total indices (typically prod of shape(
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2sub(int[] shape, long index, long numIndices) {
        long denom = numIndices;
        int[] ret = new int[shape.length];
        for (int i = ret.length - 1; i >= 0; i--) {
            denom /= shape[i];
            if (index / denom >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Dimension can not be >= Integer.MAX_VALUE");
            ret[i] = (int) (index / denom);
            index %= denom;

        }
        return ret;
    }


    public static long[] ind2sub(long[] shape, long index, long numIndices) {
        long denom = numIndices;
        long[] ret = new long[shape.length];
        for (int i = ret.length - 1; i >= 0; i--) {
            denom /= shape[i];
            if (index / denom >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Dimension can not be >= Integer.MAX_VALUE");
            ret[i] = (index / denom);
            index %= denom;
        }
        return ret;
    }

    /**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2sub(int[] shape, long index) {
        return ind2sub(shape, index, ArrayUtil.prodLong(shape));
    }

    public static long[] ind2sub(long[] shape, long index) {
        return ind2sub(shape, index, ArrayUtil.prodLong(shape));
    }

    /**
     * Convert a linear index to
     * the equivalent nd index based on the shape of the specified ndarray.
     * Infers the number of indices from the specified shape.
     *
     * @param arr the array to compute the indexes
     *            based on
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    public static long[] ind2sub(INDArray arr, long index) {
        if (arr.rank() == 1)
            return new long[]{(int) index};
        return ind2sub(arr.shape(), index, ArrayUtil.prodLong(arr.shape()));
    }



    /**
     * Convert a linear index to
     * the equivalent nd index
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @param numIndices the number of total indices (typically prod of shape(
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2subC(int[] shape, long index, long numIndices) {
        long denom = numIndices;
        int[] ret = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            denom /= shape[i];
            if (index / denom >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Dimension can not be >= Integer.MAX_VALUE");
            ret[i] = (int) (index / denom);
            index %= denom;

        }
        return ret;
    }

    public static long[] ind2subC(long[] shape, long index, long numIndices) {
        long denom = numIndices;
        long[] ret = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            denom /= shape[i];
            if (index / denom >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Dimension can not be >= Integer.MAX_VALUE");
            ret[i] = index / denom;
            index %= denom;

        }
        return ret;
    }


    /**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2subC(int[] shape, long index) {
        return ind2subC(shape, index, ArrayUtil.prodLong(shape));
    }

    public static long[] ind2subC(long[] shape, long index) {
        return ind2subC(shape, index, ArrayUtil.prodLong(shape));
    }

    /**
     * Convert a linear index to
     * the equivalent nd index based on the shape of the specified ndarray.
     * Infers the number of indices from the specified shape.
     *
     * @param arr the array to compute the indexes
     *            based on
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    public static long[] ind2subC(INDArray arr, long index) {
        if (arr.rank() == 1)
            return new long[]{index};
        return ind2subC(arr.shape(), index, ArrayUtil.prodLong(arr.shape()));
    }

    /**
     * Compute the offset for the given array
     * given the indices
     * @param arr the array to compute the offset for
     * @param indexes the indexes along each dimension to create the offset for
     * @return the offset for the given array and indexes
     */
    public static long offsetFor(INDArray arr, int[] indexes) {
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(Shape.toIndexes(indexes));
        return resolution.getOffset();
    }



    /**
     * Assert the both shapes are the same length
     * and shape[i] < lessThan[i]
     * @param shape the shape to check
     * @param lessThan the shape to assert against
     */
    public static void assertShapeLessThan(int[] shape, int[] lessThan) {
        if (shape.length != lessThan.length) {
            throw new IllegalArgumentException("Shape length must be == less than length");
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] >= lessThan[i])
                throw new IllegalStateException("Shape[" + i + "] should be less than lessThan[" + i + "]");
        }
    }

    public static void assertShapeLessThan(long[] shape, long[] lessThan) {
        if (shape.length != lessThan.length) {
            throw new IllegalArgumentException("Shape length must be == less than length");
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] >= lessThan[i])
                throw new IllegalStateException("Shape[" + i + "] should be less than lessThan[" + i + "]");
        }
    }



    /**
     * Convert the given int indexes
     * to nd array indexes
     * @param indices the indices to convert
     * @return the converted indexes
     */
    public static INDArrayIndex[] toIndexes(int[] indices) {
        INDArrayIndex[] ret = new INDArrayIndex[indices.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = new NDArrayIndex(indices[i]);
        return ret;
    }


    public static int[] newStrides(int[] strides, int newLength, INDArrayIndex[] indexes) {
        if (strides.length > newLength) {
            int[] newStrides = new int[strides.length - 1];
            for (int i = 0; i < newStrides.length; i++) {
                newStrides[i] = strides[i + 1];
            }
            strides = newStrides;
        }

        return strides;
    }



    /** Check if strides are in order suitable for non-strided mmul etc.
     * Returns true if c order and strides are descending [100,10,1] etc
     * Returns true if f order and strides are ascending [1,10,100] etc
     * False otherwise.
     * @return true if c+descending, f+ascending, false otherwise
     */
    public static boolean strideDescendingCAscendingF(INDArray array) {
        if(array.rank() <= 1)
            return true;
        long[] strides = array.stride();
        if (array.isVector() && strides[0] == 1 && strides[1] == 1)
            return true;
        char order = array.ordering();

        if (order == 'c') { //Expect descending. [100,10,1] etc
            for (int i = 1; i < strides.length; i++)
                if (strides[i - 1] <= strides[i])
                    return false;
            return true;
        } else if (order == 'f') {//Expect ascending. [1,10,100] etc
            for (int i = 1; i < strides.length; i++)
                if (strides[i - 1] >= strides[i])
                    return false;
            return true;
        } else if (order == 'a') {
            return true;
        } else {
            throw new RuntimeException("Invalid order: not c or f (is: " + order + ")");
        }
    }

    /**
     * Gets the rank given the shape info buffer
     * @param buffer the buffer to get the rank for
     * @return the rank for the shape buffer
     */
    public static int length(IntBuffer buffer) {
        int ret = 1;
        IntBuffer shape = Shape.shapeOf(buffer);
        int rank = Shape.rank(buffer);
        for (int i = 0; i < rank; i++)
            ret *= shape.get(i);
        return ret;
    }


    /**
     * Gets the rank given the shape info buffer
     * @param buffer the buffer to get the rank for
     * @return the rank for the shape buffer
     */
    public static long length(DataBuffer buffer) {
        long ret = 1;
        val rr = buffer.asLong();
        DataBuffer shape = Shape.shapeOf(buffer);
        int rank = Shape.rank(buffer);
        for (int i = 0; i < rank; i++)
            ret *= shape.getLong(i);

        return ret;
    }


    public static long length(int[] buffer) {
        long ret = 1;
        int limit = Shape.rank(buffer) + 1;
        for (int i = 1; i < limit; i++)
            ret *= buffer[i];
        return ret;
    }

    public static long length(long[] buffer) {
        long ret = 1;
        int limit = Shape.rank(buffer) + 1;
        for (int i = 1; i < limit; i++)
            ret *= buffer[i];
        return ret;
    }

    /**
     * Gets the rank given the shape info buffer
     * @param buffer the buffer to get the rank for
     * @return the rank for the shape buffer
     */
    public static int rank(DataBuffer buffer) {
        return buffer.getInt(0);
    }

    /**
     * Gets the rank given the shape info buffer
     * @param buffer the buffer to get the rank for
     * @return the rank for the shape buffer
     */
    public static int rank(IntBuffer buffer) {
        val buffer2 = (Buffer) buffer;
        val ret = (IntBuffer) buffer2.position(0);
        return ret.get(0);
    }

    public static int rank(LongBuffer buffer) {
        val buffer2 = (Buffer) buffer;
        val ret = (LongBuffer) buffer2.position(0);
        return (int) ret.get(0);
    }

    public static int rank(long[] buffer) {
        return (int) buffer[0];
    }

    public static int rank(int[] buffer) {
        return buffer[0];
    }

    /**
     * Get the size of the specified dimension. Equivalent to shape()[dimension]
     * @param buffer       The buffer to get the
     * @param dimension    The dimension to get.
     * @return             The size of the specified dimension
     */
    public static int size(IntBuffer buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer.get(1 + dimension);
    }

    public static long size(LongBuffer buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer.get(1 + dimension);
    }

    /**
     * Get the size of the specified dimension. Equivalent to shape()[dimension]
     * @param buffer       The buffer to get the shape from
     * @param dimension    The dimension to get.
     * @return             The size of the specified dimension
     */
    public static int size(DataBuffer buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer.getInt(1 + dimension);
    }

    public static int size(int[] buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer[1 + dimension];
    }

    public static long size(long[] buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer[1 + dimension];
    }

    /**
     * Get the size of the specified dimension. Identical to Shape.size(...), but does not perform any input validation
     * @param buffer       The buffer to get the shape from
     * @param dimension    The dimension to get.
     * @return             The size of the specified dimension
     */
    public static int sizeUnsafe(DataBuffer buffer, int dimension) {
        return buffer.getInt(1 + dimension);
    }

    public static int sizeUnsafe(int[] buffer, int dimension) {
        return buffer[1 + dimension];
    }

    public static long sizeUnsafe(long[] buffer, int dimension) {
        return buffer[1 + dimension];
    }

    /**
     * Get array shape from the buffer, as an int[]
     * @param buffer    Buffer to get the shape from
     * @return          Shape array
     */
    public static long[] shape(IntBuffer buffer) {
        val ret = new long[rank(buffer)];
        for (int i = 0; i < ret.length; i++)
            ret[i] = buffer.get(1 + i);
        return ret;
    }

    public static long[] shape(LongBuffer buffer) {
        val ret = new long[rank(buffer)];
        for (int i = 0; i < ret.length; i++)
            ret[i] = buffer.get(1 + i);
        return ret;
    }

    /**
     * Get array shape from the buffer, as an int[]
     * @param buffer    Buffer to get the shape from
     * @return          Shape array
     */
    public static long[] shape(DataBuffer buffer) {
        val ret = new long[rank(buffer)];
        for (int i = 0; i < ret.length; i++)
            ret[i] = buffer.getInt(1 + i);
        return ret;
    }

    /**
     * Get array shape from an int[]
     * @param buffer    Buffer to get the shape from
     * @return          Shape array
     */
    public static int[] shape(int[] buffer) {
        int[] ret = new int[rank(buffer)];
        for (int i = 0; i < ret.length; i++)
            ret[i] = buffer[1 + i];
        return ret;
    }

    public static long[] shape(long[] buffer) {
        long[] ret = new long[rank(buffer)];
        for (int i = 0; i < ret.length; i++)
            ret[i] = buffer[1 + i];
        return ret;
    }

    /**
     * Get the stride of the specified dimension
     * @param buffer       The buffer to get the stride from
     * @param dimension    The dimension to get.
     * @return             The stride of the specified dimension
     */
    public static int stride(IntBuffer buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer.get(1 + rank + dimension);
    }

    public static long stride(LongBuffer buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer.get(1 + rank + dimension);
    }

    /**
     * Get the stride of the specified dimension
     * @param buffer       The buffer to get the stride from
     * @param dimension    The dimension to get.
     * @return             The stride of the specified dimension
     */
    public static int stride(DataBuffer buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer.getInt(1 + rank + dimension);
    }

    public static int stride(int[] buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer[1 + rank + dimension];
    }

    public static long stride(long[] buffer, int dimension) {
        int rank = rank(buffer);
        if (dimension >= rank)
            throw new IllegalArgumentException("Invalid dimension " + dimension + " for rank " + rank + " array");
        return buffer[1 + rank + dimension];
    }

    /**
     * Get array shape from the buffer, as an int[]
     * @param buffer    Buffer to get the shape from
     * @return          Shape array
     */
    public static long[] strideArr(DataBuffer buffer) {
        val ret = new long[rank(buffer)];
        DataBuffer stride = Shape.stride(buffer);
        for (int i = 0; i < ret.length; i++)
            ret[i] = stride.getInt(i);
        return ret;
    }

    /**
     * Get the stride of the specified dimension, without any input validation
     * @param buffer       The buffer to get the stride from
     * @param dimension    The dimension to get.
     * @param rank         Rank of the array
     * @return             The stride of the specified dimension
     */
    public static int strideUnsafe(DataBuffer buffer, int dimension, int rank) {
        return buffer.getInt(1 + rank + dimension);
    }

    public static int strideUnsafe(int[] buffer, int dimension, int rank) {
        return buffer[1 + rank + dimension];
    }

    public static long strideUnsafe(long[] buffer, int dimension, int rank) {
        return buffer[1 + rank + dimension];
    }

    /**
     * Return the shape info length
     * given the rank
     * @param rank the rank to get the length for
     * @return rank * 2 + 4
     */
    public static int shapeInfoLength(int rank) {
        return rank * 2 + 4;
    }

    public static int shapeInfoLength(long[] shape) {
        return shapeInfoLength((int) shape[0]);
    }

    /**
     * Get the stride for the given
     * shape information buffer
     * @param buffer
     * @return
     */
    public static IntBuffer stride(IntBuffer buffer) {
        int rank = rank(buffer);
        val buffer2 = (Buffer) buffer;
        val ret = (IntBuffer) buffer2.position(1 + rank);
        return ret.slice();
    }

    public static LongBuffer stride(LongBuffer buffer) {
        int rank = rank(buffer);
        val buffer2 = (Buffer) buffer;
        val ret = (LongBuffer) buffer2.position(1 + rank);
        return ret.slice();
    }

    /**
     * Get the shape from
     * the given int buffer
     * @param buffer the buffer to get the shape information for
     * @return
     */
    public static DataBuffer stride(DataBuffer buffer) {
        int rank = rank(buffer);
        return Nd4j.createBuffer(buffer, 1 + rank, rank);
    }

    public static int[] stride(int[] buffer) {
        int rank = rank(buffer);
        int[] ret = new int[rank];
        for (int i = 0; i < rank; i++)
            ret[i] = buffer[1 + rank + i];

        return ret;
    }


    public static long[] stride(long[] buffer) {
        int rank = rank(buffer);
        long[] ret = new long[rank];
        for (int i = 0; i < rank; i++)
            ret[i] = buffer[1 + rank + i];

        return ret;
    }


    /**
     * Get the shape from
     * the given int buffer
     * @param buffer the buffer to get the shape information for
     * @return
     */
    public static DataBuffer shapeOf(DataBuffer buffer) {
        int rank = (int) buffer.getLong(0);
        return Nd4j.createBuffer(buffer, 1, rank);
    }

    /**
     * Get the shape from
     * the given int buffer
     * @param buffer the buffer to get the shape information for
     * @return
     */
    public static IntBuffer shapeOf(IntBuffer buffer) {
        Buffer buffer2 = (Buffer) buffer;
        IntBuffer ret = (IntBuffer) buffer2.position(1);
        return ret.slice();
    }

    public static LongBuffer shapeOf(LongBuffer buffer) {
        Buffer buffer2 = (Buffer) buffer;
        val ret = (LongBuffer) buffer2.position(1);
        return ret.slice();
    }


    public static int[] shapeOf(int[] buffer) {
        val rank = buffer[0];
        return Arrays.copyOfRange(buffer, 1, 1 + rank);
    }

    public static long[] shapeOf(long[] buffer) {
        val rank = (int) buffer[0];
        return Arrays.copyOfRange(buffer, 1, 1 + rank);
    }

    public static int[] stridesOf(int[] buffer) {
        val rank = buffer[0];
        return Arrays.copyOfRange(buffer, 1+rank, 1 + (rank * 2));
    }

    public static long[] stridesOf(long[] buffer) {
        val rank = (int) buffer[0];
        return Arrays.copyOfRange(buffer, 1+rank, 1 + (rank * 2));
    }

    public static int[] flags(DataBuffer buffer) {
        int length = buffer.getInt(0);
        int[] ret = new int[length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = buffer.getInt(1 + i);
        return ret;
    }

    public static int[] sparseOffsets(DataBuffer buffer) {
        int flagsLength = buffer.getInt(0);
        int offLength = buffer.getInt(flagsLength + 1);
        int[] ret = new int[offLength];
        for (int i = 0; i < offLength; i++) {
            ret[i] = buffer.getInt(i + flagsLength + 2);
        }
        return ret;
    }

    public static int[] hiddenDimension(DataBuffer buffer) {
        int flagsLength = buffer.getInt(0);
        int offLength = buffer.getInt(flagsLength + 1);
        int hiddenDimLength = buffer.getInt(flagsLength + offLength + 2);

        int[] ret = new int[hiddenDimLength];
        for (int i = 0; i < hiddenDimLength; i++) {
            ret[i] = buffer.getInt(i + flagsLength + offLength + 3);
        }
        return ret;
    }

    public static int underlyingRank(DataBuffer buffer) {
        int flagsLength = buffer.getInt(0);
        int offLength = buffer.getInt(flagsLength + 1);
        int hiddenDimLength = buffer.getInt(flagsLength + offLength + 2);

        return buffer.getInt(flagsLength + offLength + hiddenDimLength + 3);
    }

    /**
     * Prints the shape
     * for this shape information
     * @param arr the shape information to print
     * @return the shape information to string
     */
    public static String shapeToString(INDArray arr) {
        return shapeToString(arr.shapeInfo());
    }

    /**
     * Prints the shape
     * for this shape information
     * @param buffer the shape information to print
     * @return the shape information to string
     */
    public static String shapeToString(IntBuffer buffer) {
        val shapeBuff = shapeOf(buffer);
        int rank = Shape.rank(buffer);
        val strideBuff = stride(buffer);
        StringBuilder sb = new StringBuilder();
        sb.append("Rank: " + rank + ",");
        sb.append("Offset: " + Shape.offset(buffer) + "\n");
        sb.append(" Order: " + Shape.order(buffer));
        sb.append(" Shape: [");
        for (int i = 0; i < rank; i++) {
            sb.append(shapeBuff.get(i));
            if (i < rank - 1)
                sb.append(",");
        }
        sb.append("], ");

        sb.append(" stride: [");
        for (int i = 0; i < rank; i++) {
            sb.append(strideBuff.get(i));
            if (i < rank - 1)
                sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }

    public static String shapeToString(LongBuffer buffer) {
        val shapeBuff = shapeOf(buffer);
        int rank = Shape.rank(buffer);
        val strideBuff = stride(buffer);
        StringBuilder sb = new StringBuilder();
        sb.append("Rank: " + rank + ",");
        sb.append("Offset: " + Shape.offset(buffer) + "\n");
        sb.append(" Order: " + Shape.order(buffer));
        sb.append(" Shape: [");
        for (int i = 0; i < rank; i++) {
            sb.append(shapeBuff.get(i));
            if (i < rank - 1)
                sb.append(",");
        }
        sb.append("], ");

        sb.append(" stride: [");
        for (int i = 0; i < rank; i++) {
            sb.append(strideBuff.get(i));
            if (i < rank - 1)
                sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }



    /**
     * Get the offset for the buffer
     *
     * PLEASE NOTE: Legacy method. Will return 0 ALWAYS
     * @param buffer the shape info buffer to get the offset for
     * @return
     */
    @Deprecated
    public static int offset(DataBuffer buffer) {
        //throw new UnsupportedOperationException("offset() method should NOT be used");
        return 0;
    }

    public static long options(long[] buffer) {
        int length = shapeInfoLength(rank(buffer));
        long ret = buffer[length - 3];
        return ret;
    }

    public static long extras(long[] buffer) {
        return options(buffer);
    }

    /**
     * Get the offset for the buffer
     *
     * PLEASE NOTE: Legacy method. Will return 0 ALWAYS
     * @param buffer
     * @return
     */
    @Deprecated
    public static int offset(int[] buffer) {
        //throw new UnsupportedOperationException("offset() method should NOT be used");
        return 0;
    }

    @Deprecated
    public static int offset(long[] buffer) {
        //throw new UnsupportedOperationException("offset() method should NOT be used");
        return 0;
    }

    /**
     * Get the offset for the buffer
     * @param buffer the shape info buffer to get the offset for
     * @return
     */
    @Deprecated
    public static int offset(IntBuffer buffer) {
        return 0;
    }

    @Deprecated
    public static long offset(LongBuffer buffer) {
        return 0L;
    }



    /**
     * Get the element wise stride for the
     * shape info buffer
     * @param buffer the buffer to get the element
     *               wise stride from
     * @return the element wise stride for the buffer
     */
    public static int elementWiseStride(DataBuffer buffer) {
        int length2 = shapeInfoLength(buffer.getInt(0));
        return buffer.getInt(length2 - 2);
    }

    /**
     * Get the element wise stride for the
     * shape info buffer
     * @param buffer the buffer to get the element
     *               wise stride from
     * @return the element wise stride for the buffer
     */
    public static int elementWiseStride(IntBuffer buffer) {
        int length2 = shapeInfoLength(buffer.get(0));
        return buffer.get(length2 - 2);
    }

    /**
     * Get the element wise stride for the
     * shape info buffer
     * @param buffer the buffer to get the element
     *               wise stride from
     * @return the element wise stride for the buffer
     */
    public static long elementWiseStride(long[] buffer) {
        int length2 = shapeInfoLength(buffer);
        return buffer[length2 - 2];
    }


    /**
     * Get the element wise stride for the
     * shape info buffer
     * @param buffer the buffer to get the element
     *               wise stride from
     * @return the element wise stride for the buffer
     */
    public static void setElementWiseStride(IntBuffer buffer, int elementWiseStride) {
        int length2 = shapeInfoLength(buffer.get(0));
        //        if (1 > 0) throw new RuntimeException("setElementWiseStride called: [" + elementWiseStride + "], buffer: " + bufferToString(buffer));
        buffer.put(length2 - 2, elementWiseStride);
    }

    /**
     * Get the element wise stride for the
     * shape info buffer
     * @param buffer the buffer to get the element
     *               wise stride from
     * @return the element wise stride for the buffer
     */
    public static void setElementWiseStride(DataBuffer buffer, int elementWiseStride) {
        int length2 = shapeInfoLength(Shape.rank(buffer));
        //if (1 > 0) throw new RuntimeException("setElementWiseStride called: [" + elementWiseStride + "], buffer: " + buffer);
        buffer.put(length2 - 2, elementWiseStride);
    }

    /**
     * Prints the {@link IntBuffer}
     * @param buffer the buffer to print
     * @return the to string for the buffer
     *
     */
    public static String bufferToString(IntBuffer buffer) {
        StringBuilder builder = new StringBuilder();
        int rank = buffer.get(0);
        builder.append("[ ").append(rank).append(", ");
        for (int p = 1; p < rank * 2 + 4; p++) {
            builder.append(buffer.get(p));
            if (p < rank * 2 + 4 - 1)
                builder.append(", ");
        }
        builder.append("]");
        return builder.toString();
    }


    /**
     * Returns the order given the shape information
     * @param buffer the buffer
     * @return
     */
    public static char order(IntBuffer buffer) {
        int length = Shape.shapeInfoLength(Shape.rank(buffer));
        return (char) buffer.get(length - 1);
    }

    public static char order(LongBuffer buffer) {
        int length = Shape.shapeInfoLength(Shape.rank(buffer));
        return (char) buffer.get(length - 1);
    }

    /**
     * Returns the order given the shape information
     * @param buffer the buffer
     * @return
     */
    public static char order(DataBuffer buffer) {
        int length = Shape.shapeInfoLength(Shape.rank(buffer));
        return (char) buffer.getInt(length - 1);
    }

    public static char order(int[] buffer) {
        int length = Shape.shapeInfoLength(Shape.rank(buffer));
        return (char) buffer[length - 1];
    }

    public static char order(long[] buffer) {
        int length = Shape.shapeInfoLength(Shape.rank(buffer));
        return (char) buffer[length - 1];
    }


    /**
     * Returns the order given the shape information
     * @param buffer the buffer
     * @return
     */
    @Deprecated
    public static void setOrder(IntBuffer buffer, char order) {
        int length = Shape.shapeInfoLength(Shape.rank(buffer));
        buffer.put(length - 1, (int) order);
        throw new RuntimeException("setOrder called");
    }

    /**
     * Creates the shape information buffer
     * given the shape,stride
     * @param shape the shape for the buffer
     * @param stride the stride for the buffer
     * @param offset the offset for the buffer
     * @param elementWiseStride the element wise stride for the buffer
     * @param order the order for the buffer
     * @return the shape information buffer given the parameters
     */
    public static DataBuffer createShapeInformation(int[] shape, int[] stride, long offset, int elementWiseStride, char order) {
        if (shape.length != stride.length)
            throw new IllegalStateException("Shape and stride must be the same length");

        int rank = shape.length;
        int shapeBuffer[] = new int[rank * 2 + 4];
        shapeBuffer[0] = rank;
        int count = 1;
        for (int e = 0; e < shape.length; e++)
            shapeBuffer[count++] = shape[e];

        for (int e = 0; e < stride.length; e++)
            shapeBuffer[count++] = stride[e];

        shapeBuffer[count++] = (int) offset;
        shapeBuffer[count++] = elementWiseStride;
        shapeBuffer[count] = (int) order;

        DataBuffer ret = Nd4j.createBufferDetached(shapeBuffer);
        ret.setConstant(true);

        return ret;
    }

    public static DataBuffer createShapeInformation(long[] shape, long[] stride, long offset, long elementWiseStride, char order) {
        offset = 0;

        if (shape.length != stride.length)
            throw new IllegalStateException("Shape and stride must be the same length");

        int rank = shape.length;
        long shapeBuffer[] = new long[rank * 2 + 4];
        shapeBuffer[0] = rank;
        int count = 1;
        for (int e = 0; e < shape.length; e++)
            shapeBuffer[count++] = shape[e];

        for (int e = 0; e < stride.length; e++)
            shapeBuffer[count++] = stride[e];

        shapeBuffer[count++] = (int) offset;
        shapeBuffer[count++] = elementWiseStride;
        shapeBuffer[count] = (int) order;

        DataBuffer ret = Nd4j.createBufferDetached(shapeBuffer);
        ret.setConstant(true);

        return ret;
    }

    public static DataBuffer createSparseInformation(int[] flags, long[] sparseOffsets, int[] hiddenDimensions,
                                                     int underlyingRank) {
        int flagLength = flags.length;
        int offsetsLength = sparseOffsets.length;
        int hiddenDimLength = hiddenDimensions.length;
        int totalLength = flagLength + offsetsLength + hiddenDimLength + 4;


        ArrayList<Integer> accu = new ArrayList<>(totalLength);
        accu.add(flagLength);
        for (int flag : flags) {
            accu.add(flag);
        }
        accu.add(offsetsLength);
        for (long off : sparseOffsets) {
            accu.add((int) off);
        }

        accu.add(hiddenDimLength);

        for (int dim : hiddenDimensions) {
            accu.add(dim);
        }
        accu.add(underlyingRank);

        return Nd4j.createBuffer(Ints.toArray(accu));
    }

    /**
     * Convert an array to a byte buffer
     * @param arr the array
     * @return a direct byte buffer with the array contents
     */
    public static IntBuffer toBuffer(int... arr) {
        ByteBuffer directBuffer = ByteBuffer.allocateDirect(arr.length * 4).order(ByteOrder.nativeOrder());
        IntBuffer buffer = directBuffer.asIntBuffer();
        for (int i = 0; i < arr.length; i++)
            buffer.put(i, arr[i]);

        return buffer;
    }

    /**
     * To String for an int buffer
     * @param buffer
     * @return
     */
    public static String toString(IntBuffer buffer) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < buffer.capacity(); i++) {
            sb.append(buffer.get(i));
            if (i < buffer.capacity() - 1)
                sb.append(",");
        }

        return sb.toString();
    }


    /**
     * To String for an int buffer
     * @param buffer
     * @return
     */
    public static String toString(DataBuffer buffer) {
        return buffer.toString();
    }

    /**
     * Returns true if the given array
     * is meant for the whole dimension
     * @param arr the array to test
     * @return true if arr.length == 1 && arr[0] is Integer.MAX_VALUE
     */
    public static boolean wholeArrayDimension(int... arr) {
        return arr.length == 1 && arr[0] == Integer.MAX_VALUE;
    }

    public static int[] uniquify(int[] array) {
        if (array.length <= 1)
            return array;

        Set<Integer> ints = new LinkedHashSet<>();

        for (val v: array)
            ints.add(v);

        return Ints.toArray(ints);
    }

    public static int[] normalizeAxis(int rank, int... axis) {
        if (axis == null || axis.length == 0)
            return new int[] {Integer.MAX_VALUE};

        // first we should get rid of all negative axis
        int[] tmp = new int[axis.length];

        int cnt = 0;
        for (val v: axis) {
            val t = v < 0 ? v + rank : v;

            if ((t >= rank && t != Integer.MAX_VALUE)|| t < 0)
                throw new ND4JIllegalStateException("Axis array " + Arrays.toString(axis) + " contains values above rank " + rank);

            tmp[cnt++] = t;
        }

        // now we're sorting array
        Arrays.sort(tmp);

        // and getting rid of possible duplicates
        return uniquify(tmp);
    }

    /**
     *
     * Compare the contents of a buffer and
     * an array for equals
     * @param arr the array
     * @param other the buffer
     * @return true if the content equals false otherwise
     */
    public static boolean contentEquals(int[] arr, DataBuffer other) {
        for (int i = 0; i < arr.length; i++) {
            if (other.getInt(i) != arr[i]) {
                return false;
            }
        }
        return true;
    }

    public static boolean contentEquals(long[] arr, DataBuffer other) {
        for (int i = 0; i < arr.length; i++) {
            if (other.getLong(i) != arr[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     *
     * Compare the contents of a buffer and
     * an array for equals
     * @param arr the array
     * @param other the buffer
     * @return true if the content equals false otherwise
     */
    public static boolean contentEquals(int[] arr, IntBuffer other) {
        for (int i = 0; i < arr.length; i++) {
            Buffer buffer2 = (Buffer) other;
            buffer2.position(i);
            if (arr[i] != other.get()) {
                return false;
            }
        }
        return true;
    }

    public static boolean contentEquals(long[] arr, IntBuffer other) {
        for (int i = 0; i < arr.length; i++) {
            Buffer buffer2 = (Buffer) other;
            buffer2.position(i);
            if (arr[i] != other.get()) {
                return false;
            }
        }
        return true;
    }

    /** Are the elements in the buffer contiguous for this NDArray? */
    public static boolean isContiguousInBuffer(INDArray in) {
        long length = in.length();
        long dLength = in.data().length();
        if (length == dLength)
            return true; //full buffer, always contiguous

        char order = in.ordering();

        long[] shape = in.shape();
        long[] stridesIfContiguous;
        if (order == 'f') {
            stridesIfContiguous = ArrayUtil.calcStridesFortran(shape);
        } else if (order == 'c') {
            stridesIfContiguous = ArrayUtil.calcStrides(shape);
        } else if (order == 'a') {
            stridesIfContiguous = new long[] {1, 1};
        } else {
            throw new RuntimeException("Invalid order: not c or f (is: " + order + ")");
        }

        return Arrays.equals(in.stride(), stridesIfContiguous);
    }

    /**
     * This method is used in DL4J LSTM implementation
     * @param input
     * @return
     */
    public static INDArray toMmulCompatible(INDArray input) {
        if (input.rank() != 2)
            throw new IllegalArgumentException("Input must be rank 2 (matrix)");
        //Same conditions as GemmParams.copyIfNecessary()
        boolean doCopy = false;
        if (input.ordering() == 'c' && (input.stride(0) != input.size(1) || input.stride(1) != 1))
            doCopy = true;
        else if (input.ordering() == 'f' && (input.stride(0) != 1 || input.stride(1) != input.size(0)))
            doCopy = true;

        if (doCopy)
            return Shape.toOffsetZeroCopyAnyOrder(input);
        else
            return input;
    }

    /**
     * Return the rank for the given shape
     *
     * @param shape Shape to get the rank for
     * @return Rank, of the array given the shape
     * @throws ND4JIllegalStateException If shape array is null
     */
    public static int rankFromShape(int[] shape){
        if(shape == null){
            throw new ND4JIllegalStateException("Cannot get rank from null shape array");
        }
        return shape.length;
    }

    public static int rankFromShape(long[] shape){
        if(shape == null){
            throw new ND4JIllegalStateException("Cannot get rank from null shape array");
        }
        return shape.length;
    }

    public static void assertBroadcastable(@NonNull INDArray x, @NonNull INDArray y){
        assertBroadcastable(x.shape(), y.shape());
    }

    public static void assertBroadcastable(@NonNull int[] x, @NonNull int[] y){
        if(!areShapesBroadcastable(x, y)){
            throw new ND4JIllegalStateException("Arrays are different shape and are not broadcastable." +
                    " Array 1 shape = " + Arrays.toString(x) + ", array 2 shape = " + Arrays.toString(y));
        }
    }

    public static void assertBroadcastable(@NonNull long[] x, @NonNull long[] y) {
        assertBroadcastable(x, y, null);
    }

    public static void assertBroadcastable(@NonNull long[] x, @NonNull long[] y, Class<?> opClass) {
        if (!areShapesBroadcastable(x, y)) {
            throw new ND4JIllegalStateException("Arrays are different shape and are not broadcastable." +
                    " Array 1 shape = " + Arrays.toString(x) + ", array 2 shape = " + Arrays.toString(y) +
                    (opClass == null ? "" : " - op: " + opClass.getName()));
        }
    }

    public static boolean areShapesBroadcastable(@NonNull int[] x, @NonNull int[] y){
        //Ported from: https://github.com/deeplearning4j/libnd4j/blob/master/include/helpers/impl/ShapeUtils.cpp

        int minRank = Math.min(x.length, y.length);
        for( int i=-1; i>= -minRank; i--){
            if(x[x.length + i] != y[y.length + i] && x[x.length + i] != 1 && y[y.length + i] != 1){
                return false;
            }
        }

        return true;
    }

    public static boolean areShapesBroadcastable(@NonNull long[] x, @NonNull long[] y){
        //Ported from: https://github.com/deeplearning4j/libnd4j/blob/master/include/helpers/impl/ShapeUtils.cpp

        int minRank = Math.min(x.length, y.length);
        for( int i=-1; i>= -minRank; i--){
            if(x[x.length + i] != y[y.length + i] && x[x.length + i] != 1 && y[y.length + i] != 1){
                return false;
            }
        }

        return true;
    }

    /**
     *
     * @param shape
     * @return
     */
    public static long lengthOf(long[] shape) {
        if (shape.length == 0)
            return 1L;
        else
            return ArrayUtil.prodLong(shape);
    }

    public static boolean hasDefaultStridesForShape(INDArray input){
        if(input.rank() == 0)
            return true;
        if(!strideDescendingCAscendingF(input)){
            return false;
        }
        char order = input.ordering();
        long[] defaultStrides;
        if(order == 'f'){
            defaultStrides = ArrayUtil.calcStridesFortran(input.shape());
        } else {
            defaultStrides = ArrayUtil.calcStrides(input.shape());
        }
        return Arrays.equals(input.stride(), defaultStrides);
    }


    public static boolean isEmpty(long[] shapeInfo) {
        return ArrayOptionsHelper.arrayType(shapeInfo) == ArrayType.EMPTY;
    }
}
