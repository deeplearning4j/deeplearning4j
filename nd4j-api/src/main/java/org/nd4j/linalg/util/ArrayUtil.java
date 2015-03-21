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

package org.nd4j.linalg.util;

import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.List;
import java.util.Random;

/**
 * @author Adam Gibson
 */
public class ArrayUtil {


    /**
     * Credit to mikio braun from jblas
     * <p/>
     * Create a random permutation of the numbers 0, ..., size - 1.
     * <p/>
     * see Algorithm P, D.E. Knuth: The Art of Computer Programming, Vol. 2, p. 145
     */
    public static int[] randomPermutation(int size) {
        Random r = new Random();
        int[] result = new int[size];

        for (int j = 0; j < size; j++) {
            result[j] = j + 1;
        }

        for (int j = size - 1; j > 0; j--) {
            int k = r.nextInt(j);
            int temp = result[j];
            result[j] = result[k];
            result[k] = temp;
        }

        return result;
    }

    public static INDArray toNDArray(int[][] nums) {
        if (Nd4j.dataType() == DataBuffer.DOUBLE) {
            double[] doubles = toDoubles(nums);
            INDArray create = Nd4j.create(doubles, new int[]{1, nums.length});
            return create;
        } else {
            float[] doubles = toFloats(nums);
            INDArray create = Nd4j.create(doubles, new int[]{1, nums.length});
            return create;
        }

    }

    public static INDArray toNDArray(int[] nums) {
        if (Nd4j.dataType() == DataBuffer.DOUBLE) {
            double[] doubles = toDoubles(nums);
            INDArray create = Nd4j.create(doubles, new int[]{1, nums.length});
            return create;
        } else {
            float[] doubles = toFloats(nums);
            INDArray create = Nd4j.create(doubles, new int[]{1, nums.length});
            return create;
        }
    }


    public static int[] toInts(INDArray n) {
        if (n instanceof IComplexNDArray)
            throw new IllegalArgumentException("Unable to convert complex array");
        n = n.linearView();
        int[] ret = new int[n.length()];
        for (int i = 0; i < n.length(); i++)
            ret[i] = (int) n.getFloat(i);
        return ret;
    }

    public static int[] toInts(float[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = (int) data[i];
        return ret;
    }

    public static int[] toInts(double[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = (int) data[i];
        return ret;
    }


    public static int prod(int[] mult) {
        if (mult.length < 1)
            return 0;
        int ret = 1;
        for (int i = 0; i < mult.length; i++)
            ret *= mult[i];
        return ret;
    }


    public static boolean equals(float[] data, double[] data2) {
        if (data.length != data2.length)
            return false;
        for (int i = 0; i < data.length; i++) {
            double equals = Math.abs(data2[i] - data[i]);
            if (equals > 1e-6)
                return false;
        }
        return true;
    }


    public static int[] consArray(int a, int[] as) {
        int len = as.length;
        int[] nas = new int[len + 1];
        nas[0] = a;
        System.arraycopy(as, 0, nas, 1, len);
        return nas;
    }


    public static boolean isZero(int[] as) {
        for (int i = 0; i < as.length; i++) {
            if (as[i] != 0) return false;
        }
        return true;
    }

    public static boolean anyMore(int[] target, int[] test) {
        assert target.length == test.length : "Unable to compare: different sizes";
        for (int i = 0; i < target.length; i++) {
            if (target[i] > test[i])
                return true;
        }
        return false;
    }


    public static boolean anyLess(int[] target, int[] test) {
        assert target.length == test.length : "Unable to compare: different sizes";
        for (int i = 0; i < target.length; i++) {
            if (target[i] < test[i])
                return true;
        }
        return false;
    }

    public static int dotProduct(int[] xs, int[] ys) {
        int result = 0;
        int n = xs.length;

        if (ys.length != n)
            throw new IllegalArgumentException("Different array sizes");

        for (int i = 0; i < n; i++) {
            result += xs[i] * ys[i];
        }
        return result;
    }


    public static int[] empty() {
        return new int[0];
    }


    public static int[] of(int... arr) {
        return arr;
    }

    public static int[] copy(int[] copy) {
        int[] ret = new int[copy.length];
        System.arraycopy(copy, 0, ret, 0, ret.length);
        return ret;
    }


    public static double[] doubleCopyOf(float[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = data[i];
        return ret;
    }

    public static float[] floatCopyOf(double[] data) {
        if (data.length == 0)
            return new float[1];
        float[] ret = new float[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = (float) data[i];
        return ret;
    }


    /**
     * Returns a subset of an array from 0 to "to"
     *
     * @param data the data to getFromOrigin a subset of
     * @param to   the end point of the data
     * @return the subset of the data specified
     */
    public static double[] range(double[] data, int to) {
        return range(data, to, 1);
    }


    /**
     * Returns a subset of an array from 0 to "to"
     * using the specified stride
     *
     * @param data   the data to getFromOrigin a subset of
     * @param to     the end point of the data
     * @param stride the stride to go through the array
     * @return the subset of the data specified
     */
    public static double[] range(double[] data, int to, int stride) {
        return range(data, to, stride, 1);
    }


    /**
     * Returns a subset of an array from 0 to "to"
     * using the specified stride
     *
     * @param data                  the data to getFromOrigin a subset of
     * @param to                    the end point of the data
     * @param stride                the stride to go through the array
     * @param numElementsEachStride the number of elements to collect at each stride
     * @return the subset of the data specified
     */
    public static double[] range(double[] data, int to, int stride, int numElementsEachStride) {
        double[] ret = new double[to];
        int count = 0;
        for (int i = 0; i < data.length; i += stride) {
            for (int j = 0; j < numElementsEachStride; j++) {
                if (i + j >= data.length || count >= ret.length)
                    break;
                ret[count++] = data[i + j];
            }
        }
        return ret;
    }


    public static int[] toArray(List<Integer> list) {
        int[] ret = new int[list.size()];
        for (int i = 0; i < list.size(); i++)
            ret[i] = list.get(i);
        return ret;
    }


    public static double[] toArrayDouble(List<Double> list) {
        double[] ret = new double[list.size()];
        for (int i = 0; i < list.size(); i++)
            ret[i] = list.get(i);
        return ret;

    }


    /**
     * Generate an int array ranging from
     * from to to.
     * if from is > to this method will
     * count backwards
     *
     * @param from      the from
     * @param to        the end point of the data
     * @param increment the amount to increment by
     * @return the int array with a length equal to absoluteValue(from - to)
     */
    public static int[] range(int from, int to, int increment) {
        int diff = Math.abs(from - to);
        int[] ret = new int[diff];
        if (from < to) {
            int count = 0;
            for (int i = from; i < to; i += increment) {
                ret[count++] = i;
            }
        } else if (from > to) {
            int count = 0;
            for (int i = from; i > to; i -= increment)
                ret[count++] = i;
        }

        return ret;
    }

    /**
     * Generate an int array ranging from
     * from to to.
     * if from is > to this method will
     * count backwards
     *
     * @param from the from
     * @param to   the end point of the data
     * @return the int array with a length equal to absoluteValue(from - to)
     */
    public static int[] range(int from, int to) {
        if (from == to)
            return new int[0];
        return range(from, to, 1);
    }

    public static double[] toDoubles(int[] ints) {
        double[] ret = new double[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (double) ints[i];
        return ret;
    }

    public static double[] toDoubles(float[] ints) {
        double[] ret = new double[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (double) ints[i];
        return ret;
    }

    public static float[] toFloats(int[][] ints) {
        return toFloats(Ints.concat(ints));
    }

    public static double[] toDoubles(int[][] ints) {
        return toDoubles(Ints.concat(ints));
    }


    public static float[] toFloats(int[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (float) ints[i];
        return ret;
    }

    public static float[] toFloats(double[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (float) ints[i];
        return ret;
    }


    /**
     * Return a copy of this array with the
     * given index omitted
     *
     * @param data     the data to copy
     * @param index    the index of the item to remove
     * @param newValue the newValue to replace
     * @return the new array with the omitted
     * item
     */
    public static int[] replace(int[] data, int index, int newValue) {
        int[] copy = copy(data);
        copy[index] = newValue;
        return copy;
    }

    /**
     * Return a copy of this array with the
     * given index omitted
     *
     * @param data  the data to copy
     * @param index the index of the item to remove
     * @return the new array with the omitted
     * item
     */
    public static int[] removeIndex(int[] data, int index) {
        if (index >= data.length)
            throw new IllegalArgumentException("Unable to remove index " + index + " was >= data.length");

        if (data == null)
            return null;
        if (data.length < 1)
            return data;
        if (index < 0)
            return data;

        int len = data.length;
        int[] result = new int[len - 1];
        System.arraycopy(data, 0, result, 0, index);
        System.arraycopy(data, index + 1, result, index, len - index - 1);
        return result;
    }


    /**
     * Returns the array with the item in index
     * removed, if the array is empty it will return the array itself
     *
     * @param data  the data to remove data from
     * @param index the index of the item to remove
     * @return a copy of the array with the removed item,
     * or the array itself if empty
     */
    public static Integer[] removeIndex(Integer[] data, int index) {
        if (data == null)
            return null;
        if (data.length < 1)
            return data;
        int len = data.length;
        Integer[] result = new Integer[len - 1];
        System.arraycopy(data, 0, result, 0, index);
        System.arraycopy(data, index + 1, result, index, len - index - 1);
        return result;
    }


    /**
     * Computes the standard packed array strides for a given shape.
     *
     * @param shape    the shape of a matrix:
     * @param startNum the start number for the strides
     * @return the strides for a matrix of n dimensions
     */
    public static int[] calcStridesFortran(int[] shape, int startNum) {
        int dimensions = shape.length;
        int[] stride = new int[dimensions];
        int st = startNum;
        for (int j = 0; j < stride.length; j++) {
            stride[j] = st;
            st *= shape[j];
        }

        return stride;
    }

    /**
     * Computes the standard packed array strides for a given shape.
     *
     * @param shape the shape of a matrix:
     * @return the strides for a matrix of n dimensions
     */
    public static int[] calcStridesFortran(int[] shape) {
        return calcStridesFortran(shape, 1);
    }


    /**
     * Computes the standard packed array strides for a given shape.
     *
     * @param shape      the shape of a matrix:
     * @param startValue the startValue for the strides
     * @return the strides for a matrix of n dimensions
     */
    public static int[] calcStrides(int[] shape, int startValue) {
        int dimensions = shape.length;
        int[] stride = new int[dimensions];
        int st = startValue;
        for (int j = dimensions - 1; j >= 0; j--) {
            stride[j] = st;
            st *= shape[j];
        }

        return stride;
    }


    public static int[] plus(int[] ints, int mult) {
        int[] ret = new int[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = ints[i] + mult;
        return ret;
    }

    public static int[] times(int[] ints, int mult) {
        int[] ret = new int[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = ints[i] * mult;
        return ret;
    }

    public static int[] times(int[] ints, int[] mult) {
        assert ints.length == mult.length : "Ints and mult must be the same length";
        int[] ret = new int[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = ints[i] * mult[i];
        return ret;
    }


    /**
     * For use with row vectors to ensure consistent strides
     * with varying offsets
     *
     * @param arr the array to getScalar the stride for
     * @return the stride
     */
    public static int nonOneStride(int[] arr) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i] != 1)
                return arr[i];
        return 1;
    }


    /**
     * Computes the standard packed array strides for a given shape.
     *
     * @param shape the shape of a matrix:
     * @return the strides for a matrix of n dimensions
     */
    public static int[] calcStrides(int[] shape) {
        return calcStrides(shape, 1);
    }


    /**
     * Create a backwards copy of the given array
     *
     * @param e the array to createComplex a reverse clone of
     * @return the reversed copy
     */
    public static int[] reverseCopy(int[] e) {
        if (e.length < 1)
            return e;

        int[] copy = new int[e.length];
        for (int i = 0; i <= e.length / 2; i++) {
            int temp = e[i];
            copy[i] = e[e.length - i - 1];
            copy[e.length - i - 1] = temp;
        }
        return copy;
    }


    public static double[] read(int length, DataInputStream dis) throws IOException {
        double[] ret = new double[length];
        for (int i = 0; i < length; i++)
            ret[i] = dis.readDouble();
        return ret;
    }


    public static void write(double[] data, DataOutputStream dos) throws IOException {
        for (int i = 0; i < data.length; i++)
            dos.writeDouble(data[i]);
    }

    public static double[] readDouble(int length, DataInputStream dis) throws IOException {
        double[] ret = new double[length];
        for (int i = 0; i < length; i++)
            ret[i] = dis.readDouble();
        return ret;
    }


    public static float[] readFloat(int length, DataInputStream dis) throws IOException {
        float[] ret = new float[length];
        for (int i = 0; i < length; i++)
            ret[i] = dis.readFloat();
        return ret;
    }


    public static void write(float[] data, DataOutputStream dos) throws IOException {
        for (int i = 0; i < data.length; i++)
            dos.writeFloat(data[i]);
    }


    public static void assertSquare(double[]... d) {
        if (d.length > 2) {
            for (int i = 0; i < d.length; i++) {
                assertSquare(d[i]);
            }
        } else {
            int firstLength = d[0].length;
            for (int i = 1; i < d.length; i++) {
                assert d[i].length == firstLength;
            }
        }
    }


    /**
     * Reverse the passed in array in place
     *
     * @param e the array to reverse
     */
    public static void reverse(int[] e) {
        for (int i = 0; i <= e.length / 2; i++) {
            int temp = e[i];
            e[i] = e[e.length - i - 1];
            e[e.length - i - 1] = temp;
        }

    }


    public static float[] reverseCopy(float[] e) {
        float[] copy = new float[e.length];
        for (int i = 0; i <= e.length / 2; i++) {
            float temp = e[i];
            copy[i] = e[e.length - i - 1];
            copy[e.length - i - 1] = temp;
        }
        return copy;

    }


    public static <E> E[] reverseCopy(E[] e) {
        E[] copy = (E[]) new Object[e.length];
        for (int i = 0; i <= e.length / 2; i++) {
            E temp = e[i];
            copy[i] = e[e.length - i - 1];
            copy[e.length - i - 1] = temp;
        }
        return copy;

    }

    public static <E> void reverse(E[] e) {
        for (int i = 0; i <= e.length / 2; i++) {
            E temp = e[i];
            e[i] = e[e.length - i - 1];
            e[e.length - i - 1] = temp;
        }

    }

    public static float[] flatten(float[][] arr) {
        float[] ret = new float[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++)
            for (int j = 0; j < arr[i].length; j++)
                ret[count++] = arr[i][j];
        return ret;
    }

    public static int[] flatten(int[][] arr) {
        int[] ret = new int[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++)
            for (int j = 0; j < arr[i].length; j++)
                ret[count++] = arr[i][j];
        return ret;
    }

    public static double[] flatten(double[][] arr) {
        double[] ret = new double[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++)
            for (int j = 0; j < arr[i].length; j++)
                ret[count++] = arr[i][j];
        return ret;
    }

    public static double[][] toDouble(int[][] arr) {
        double[][] ret = new double[arr.length][arr[0].length];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++)
                ret[i][j] = arr[i][j];
        }
        return ret;
    }


    /**
     * Combines a applyTransformToDestination of int arrays in to one flat int array
     *
     * @param nums the int arrays to combineDouble
     * @return one combined int array
     */
    public static float[] combineFloat(List<float[]> nums) {
        int length = 0;
        for (int i = 0; i < nums.size(); i++)
            length += nums.get(i).length;
        float[] ret = new float[length];
        int count = 0;
        for (float[] i : nums) {
            for (int j = 0; j < i.length; j++) {
                ret[count++] = i[j];
            }
        }

        return ret;
    }


    /**
     * Combines a applyTransformToDestination of int arrays in to one flat int array
     *
     * @param nums the int arrays to combineDouble
     * @return one combined int array
     */
    public static float[] combine(List<float[]> nums) {
        int length = 0;
        for (int i = 0; i < nums.size(); i++)
            length += nums.get(i).length;
        float[] ret = new float[length];
        int count = 0;
        for (float[] i : nums) {
            for (int j = 0; j < i.length; j++) {
                ret[count++] = i[j];
            }
        }

        return ret;
    }

    /**
     * Combines a applyTransformToDestination of int arrays in to one flat int array
     *
     * @param nums the int arrays to combineDouble
     * @return one combined int array
     */
    public static double[] combineDouble(List<double[]> nums) {
        int length = 0;
        for (int i = 0; i < nums.size(); i++)
            length += nums.get(i).length;
        double[] ret = new double[length];
        int count = 0;
        for (double[] i : nums) {
            for (int j = 0; j < i.length; j++) {
                ret[count++] = i[j];
            }
        }

        return ret;
    }

    /**
     * Combines a applyTransformToDestination of int arrays in to one flat int array
     *
     * @param ints the int arrays to combineDouble
     * @return one combined int array
     */
    public static double[] combine(float[]... ints) {
        int length = 0;
        for (int i = 0; i < ints.length; i++)
            length += ints[i].length;
        double[] ret = new double[length];
        int count = 0;
        for (float[] i : ints) {
            for (int j = 0; j < i.length; j++) {
                ret[count++] = i[j];
            }
        }

        return ret;
    }

    /**
     * Combines a applyTransformToDestination of int arrays in to one flat int array
     *
     * @param ints the int arrays to combineDouble
     * @return one combined int array
     */
    public static int[] combine(int[]... ints) {
        int length = 0;
        for (int i = 0; i < ints.length; i++)
            length += ints[i].length;
        int[] ret = new int[length];
        int count = 0;
        for (int[] i : ints) {
            for (int j = 0; j < i.length; j++) {
                ret[count++] = i[j];
            }
        }

        return ret;
    }

    public static <E> E[] combine(E[]... arrs) {
        int length = 0;
        for (int i = 0; i < arrs.length; i++)
            length += arrs[i].length;

        E[] ret = (E[]) Array.newInstance(arrs[0][0].getClass(), length);
        int count = 0;
        for (E[] i : arrs) {
            for (int j = 0; j < i.length; j++) {
                ret[count++] = i[j];
            }
        }

        return ret;
    }


    public static int[] toOutcomeArray(int outcome, int numOutcomes) {
        int[] nums = new int[numOutcomes];
        nums[outcome] = 1;
        return nums;
    }

    public static double[] toDouble(int[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = data[i];
        return ret;
    }

    public static float[] copy(float[] data) {
        float[] result = new float[data.length];
        System.arraycopy(data, 0, result, 0, data.length);
        return result;
    }
}
