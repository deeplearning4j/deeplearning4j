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

package org.nd4j.linalg.util;

import com.google.common.primitives.Ints;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.*;

/**
 * @author Adam Gibson
 */
public class ArrayUtil {


    private ArrayUtil() {}

    /**
     * Repeat a value n times
     * @param n the number of times to repeat
     * @param toReplicate the value to repeat
     * @return an array of length n filled with the
     * given value
     */
    public static int[] nTimes(int n,int toReplicate) {
        int[] ret = new int[n];
        Arrays.fill(ret,toReplicate);
        return ret;
    }


    /**
     * Returns true if all of the elements in the
     * given int array are unique
     * @param toTest the array to test
     * @return true if all o fthe items
     * are unique false otherwise
     */
    public static boolean allUnique(int[] toTest) {
        Set<Integer> set = new HashSet<>();
        for(int i : toTest) {
            if(!set.contains(i))
                set.add(i);
            else
                return false;
        }

        return true;
    }

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


    /**
     * Calculate the offset for a given stride array
     * @param stride the stride to use
     * @param i the offset to calculate for
     * @return the offset for the given
     * stride
     */
    public static int offsetFor(int[] stride,int i) {
        int ret = 0;
        for(int j = 0; j < stride.length; j++)
            ret += (i * stride[j]);
        return ret;

    }
    /**
     * Sum of an int array
     * @param add the elements
     *            to calculate the sum for
     * @return the sum of this array
     */
    public static int sum(List<Integer> add) {
        if (add.size() < 1)
            return 0;
        int ret = 0;
        for (int i = 0; i < add.size(); i++)
            ret += add.get(i);
        return ret;
    }
    /**
     * Sum of an int array
     * @param add the elements
     *            to calculate the sum for
     * @return the sum of this array
     */
    public static int sum(int[] add) {
        if (add.length < 1)
            return 0;
        int ret = 0;
        for (int i = 0; i < add.length; i++)
            ret += add[i];
        return ret;
    }
    /**
     * Product of an int array
     * @param mult the elements
     *            to calculate the sum for
     * @return the product of this array
     */
    public static int prod(List<Integer> mult) {
        if (mult.size() < 1)
            return 0;
        int ret = 1;
        for (int i = 0; i < mult.size(); i++)
            ret *= mult.get(i);
        return ret;
    }


    /**
     * Product of an int array
     * @param mult the elements
     *            to calculate the sum for
     * @return the product of this array
     */
    public static int prod(int...mult) {
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


    /**
     * Returns true if any of the elements are zero
     * @param as
     * @return
     */
    public static boolean isZero(int[] as) {
        for (int i = 0; i < as.length; i++) {
            if (as[i] == 0) return true;
        }
        return false;
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


    /**
     * Compute the offset
     * based on teh shape strides and offsets
     * @param shape the shape to compute
     * @param offsets the offsets to compute
     * @param strides the strides to compute
     * @return the offset for the given shape,offset,and strides
     */
    public static int calcOffset(List<Integer> shape,List<Integer> offsets,List<Integer> strides) {
        if(shape.size() != offsets.size() || shape.size() != strides.size())
            throw new IllegalArgumentException("Shapes,strides, and offsets must be the same size");
        int ret = 0;
        for(int i = 0; i < offsets.size(); i++) {
            //we should only do this in the general case, not on vectors
            //the reason for this is we force everything including scalars
            //to be 2d
            if(shape.get(i) == 1 && offsets.size() > 2 && i > 0)
                continue;
            ret += offsets.get(i) * strides.get(i);
        }

        return ret;
    }


    /**
     * Compute the offset
     * based on teh shape strides and offsets
     * @param shape the shape to compute
     * @param offsets the offsets to compute
     * @param strides the strides to compute
     * @return the offset for the given shape,offset,and strides
     */
    public static int calcOffset(int[] shape,int[] offsets,int[] strides) {
        if(shape.length != offsets.length || shape.length!= strides.length)
            throw new IllegalArgumentException("Shapes,strides, and offsets must be the same size");

        int ret = 0;
        for(int i = 0; i < offsets.length; i++) {
            if(shape[i] == 1)
                continue;
            ret += offsets[i] * strides[i];
        }

        return ret;
    }

    /**
     *
     * @param xs
     * @param ys
     * @return
     */
    public static int dotProduct(List<Integer> xs, List<Integer> ys) {
        int result = 0;
        int n = xs.size();

        if (ys.size() != n)
            throw new IllegalArgumentException("Different array sizes");

        for (int i = 0; i < n; i++) {
            result += xs.get(i) * ys.get(i);
        }
        return result;
    }

    /**
     *
     * @param xs
     * @param ys
     * @return
     */
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
        double[] ret = new double[to / stride];
        if(ret.length < 1)
            ret = new double[1];
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
        int[] ret = new int[diff / increment];
        if(ret.length < 1)
            ret = new int[1];

        if (from < to) {
            int count = 0;
            for (int i = from; i < to; i += increment) {
                if (count >= ret.length)
                    break;
                ret[count++] = i;
            }
        } else if (from > to) {
            int count = 0;
            for (int i = from - 1; i >= to; i -= increment) {
                if (count >= ret.length)
                    break;
                ret[count++] = i;
            }
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
     * Return a copy of this array with only the
     * given index(es) remaining
     *
     * @param data  the data to copy
     * @param index the index of the item to remove
     * @return the new array with the omitted
     * item
     */
    public static int[] keep(int[] data, int...index) {
        if(index.length == data.length)
            return data;

        int[] ret = new int[index.length];
        int count = 0;
        for(int i = 0; i < data.length; i++)
            if(Ints.contains(index,i))
                ret[count++] = data[i];

        return ret;
    }


    /**
     * Return a copy of this array with the
     * given index omitted
     *
     * PLEASE NOTE: index to be omitted must exist in source array.
     *
     * @param data  the data to copy
     * @param index the index of the item to remove
     * @return the new array with the omitted
     * item
     */
    public static int[] removeIndex(int[] data, int...index) {
        if(index.length >= data.length)
            throw new IllegalStateException("Illegal remove: indexes.length > data.length");
        int offset = 0;
        /*
            workaround for non-existant indexes (such as Integer.MAX_VALUE)


        for (int i = 0; i < index.length; i ++) {
            if (index[i] >= data.length || index[i] < 0) offset++;
        }
        */

        int[] ret = new int[data.length - index.length + offset];
        int count = 0;
        for(int i = 0; i < data.length; i++)
            if(!Ints.contains(index,i)) {
                ret[count++] = data[i];
            }

        return ret;
    }


    //Credit: http://stackoverflow.com/questions/15533854/converting-byte-array-to-double-array

    /**
     *
     * @param doubleArray
     * @return
     */
    public static byte[] toByteArray(double[] doubleArray) {
        int times = Double.SIZE / Byte.SIZE;
        byte[] bytes = new byte[doubleArray.length * times];
        for(int i = 0;i<doubleArray.length;i++){
            ByteBuffer.wrap(bytes, i*times, times).putDouble(doubleArray[i]);
        }
        return bytes;
    }

    /**
     *
     * @param byteArray
     * @return
     */
    public static double[] toDoubleArray(byte[] byteArray) {
        int times = Double.SIZE / Byte.SIZE;
        double[] doubles = new double[byteArray.length / times];
        for(int i=0;i<doubles.length;i++){
            doubles[i] = ByteBuffer.wrap(byteArray, i*times, times).getDouble();
        }
        return doubles;
    }


    /**
     *
     * @param doubleArray
     * @return
     */
    public static byte[] toByteArray(float[] doubleArray) {
        int times = Float.SIZE / Byte.SIZE;
        byte[] bytes = new byte[doubleArray.length * times];
        for(int i = 0;i<doubleArray.length;i++){
            ByteBuffer.wrap(bytes, i*times, times).putFloat(doubleArray[i]);
        }
        return bytes;
    }

    /**
     *
     * @param byteArray
     * @return
     */
    public static float[] toFloatArray(byte[] byteArray) {
        int times = Float.SIZE / Byte.SIZE;
        float[] doubles = new float[byteArray.length / times];
        for(int i=0;i<doubles.length;i++){
            doubles[i] = ByteBuffer.wrap(byteArray, i*times, times).getFloat();
        }
        return doubles;
    }

    /**
     *
     * @param intArray
     * @return
     */
    public static byte[] toByteArray(int[] intArray) {
        int times = Integer.SIZE / Byte.SIZE;
        byte[] bytes = new byte[intArray.length * times];
        for(int i = 0;i<intArray.length;i++){
            ByteBuffer.wrap(bytes, i*times, times).putInt(intArray[i]);
        }
        return bytes;
    }

    /**
     *
     * @param byteArray
     * @return
     */
    public static int[] toIntArray(byte[] byteArray) {
        int times = Integer.SIZE / Byte.SIZE;
        int[] ints = new int[byteArray.length / times];
        for(int i=0;i<ints.length;i++){
            ints[i] = ByteBuffer.wrap(byteArray, i * times, times).getInt();
        }
        return ints;
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
        if(data == null)
            return null;

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
     * Create a copy of the given array
     * starting at the given index with the given length.
     *
     * The intent here is for striding.
     *
     * For example in slicing, you want the major stride to be first.
     * You achieve this by taking the last index
     * of the matrix's stride and putting
     * this as the first stride of the new ndarray
     * for slicing.
     *
     * All of the elements except the copied elements are
     * initialized as the given value
     * @param valueStarting  the starting value
     * @param copy the array to copy
     * @param idxFrom the index to start at in the from array
     * @param idxAt the index to start at in the return array
     * @param length the length of the array to create
     * @return the given array
     */
    public static int[] valueStartingAt(int valueStarting,int[] copy,int idxFrom,int idxAt,int length) {
        int[] ret = new int[length];
        Arrays.fill(ret, valueStarting);
        for(int i = 0; i < length; i++) {
            if(i + idxFrom >= copy.length || i + idxAt >= ret.length)
                break;
            ret[i + idxAt] = copy[i + idxFrom];
        }

        return ret;
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
        if(shape.length == 2 && shape[0] == 1 || shape[1] == 1) {
            int[] ret = new int[2];
            Arrays.fill(ret,startNum);
            return ret;

        }

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
        if(shape.length == 2 && shape[0] == 1 || shape[1] == 1) {
            int[] ret = new int[2];
            Arrays.fill(ret,startValue);
            return ret;

        }


        int dimensions = shape.length;
        int[] stride = new int[dimensions];

        int st = startValue;
        for (int j = dimensions - 1; j >= 0; j--) {
            stride[j] = st;
            st *= shape[j];
        }



        return stride;
    }


    /**
     * Returns true if the given
     * two arrays are reverse copies of each other
     * @param first
     * @param second
     * @return
     */
    public static boolean isInverse(int[] first,int[] second) {
        int backWardCount = second.length - 1;
        for(int i = 0; i < first.length; i++) {
            if(first[i] != second[backWardCount--])
                return false;
        }
        return true;
    }

    public static int[] plus(int[] ints, int mult) {
        int[] ret = new int[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = ints[i] + mult;
        return ret;
    }


    public static int[] plus(int[] ints, int[] mult) {
        if(ints.length != mult.length)
            throw new IllegalArgumentException("Both arrays must have the same length");
        int[] ret = new int[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = ints[i] + mult[i];
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
     * @param arr the array to get the stride for
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
     * Multiply the given array
     * by the given scalar
     * @param arr the array to multily
     * @param mult the scalar to multiply by
     */
    public static void multiplyBy(int[] arr,int mult) {
        for(int i = 0; i < arr.length; i++)
            arr[i] *= mult;

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


    public static List<double[]> zerosMatrix(int...dimensions) {
        List<double[]> ret = new ArrayList<>();
        for(int i = 0; i < dimensions.length; i++) {
            ret.add(new double[dimensions[i]]);
        }
        return ret;
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

    /**
     * Convert a 2darray in to a flat
     * array (row wise)
     * @param arr the array to flatten
     * @return a flattened representation of the array
     */
    public static double[] flatten(double[][] arr) {
        double[] ret = new double[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++)
            for (int j = 0; j < arr[i].length; j++)
                ret[count++] = arr[i][j];
        return ret;
    }

    /**
     * Cast an int array to a double array
     * @param arr the array to cast
     * @return the elements of this
     * array cast as an int
     */
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
     * Combines a apply of int arrays in to one flat int array
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
     * Combines a apply of int arrays in to one flat int array
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
     * Combines a apply of int arrays in to one flat int array
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
     * Combines a apply of int arrays in to one flat int array
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

    public static double[] copy(double[] data) {
        double[] result = new double[data.length];
        System.arraycopy(data, 0, result, 0, data.length);
        return result;
    }


    /** Convert an arbitrary-dimensional rectangular double array to flat vector.<br>
     * Can pass double[], double[][], double[][][], etc.
     */
    public static double[] flattenDoubleArray( Object doubleArray ){
        if( doubleArray instanceof double[] ) return (double[])doubleArray;

        LinkedList<Object> stack = new LinkedList<>();
        stack.push(doubleArray);

        int[] shape = arrayShape(doubleArray);
        int length = ArrayUtil.prod(shape);
        double[] flat = new double[length];
        int count = 0;

        while(!stack.isEmpty()){
            Object current = stack.pop();
            if( current instanceof double[] ){
                double[] arr = (double[])current;
                for( int i=0; i<arr.length; i++ ) flat[count++] = arr[i];
            } else if(current instanceof Object[] ){
                Object[] o = (Object[])current;
                for( int i=o.length-1; i>=0; i-- ) stack.push(o[i]);
            } else throw new IllegalArgumentException("Base array is not double[]");
        }

        if( count != flat.length ) throw new IllegalArgumentException("Fewer elements than expected. Array is ragged?");
        return flat;
    }

    /** Convert an arbitrary-dimensional rectangular float array to flat vector.<br>
     * Can pass float[], float[][], float[][][], etc.
     */
    public static float[] flattenFloatArray( Object floatArray ){
        if( floatArray instanceof float[] ) return (float[])floatArray;

        LinkedList<Object> stack = new LinkedList<>();
        stack.push(floatArray);

        int[] shape = arrayShape(floatArray);
        int length = ArrayUtil.prod(shape);
        float[] flat = new float[length];
        int count = 0;

        while(!stack.isEmpty()){
            Object current = stack.pop();
            if( current instanceof float[] ){
                float[] arr = (float[])current;
                for( int i=0; i<arr.length; i++ ) flat[count++] = arr[i];
            } else if(current instanceof Object[] ){
                Object[] o = (Object[])current;
                for( int i=o.length-1; i>=0; i-- ) stack.push(o[i]);
            } else throw new IllegalArgumentException("Base array is not float[]");
        }

        if( count != flat.length ) throw new IllegalArgumentException("Fewer elements than expected. Array is ragged?");
        return flat;
    }

    /** Calculate the shape of an arbitrary multi-dimensional array. Assumes:<br>
     * (a) array is rectangular (not ragged) and first elements (i.e., array[0][0][0]...) are non-null <br>
     * (b) First elements have > 0 length. So array[0].length > 0, array[0][0].length > 0, etc.<br>
     * Can pass any Java array type: double[], Object[][][], float[][], etc.<br>
     * Length of returned array is number of dimensions; returned[i] is size of ith dimension.
     */
    public static int[] arrayShape(Object array){
        int nDimensions = 0;
        Class<?> c = array.getClass().getComponentType();
        while( c != null ){
            nDimensions++;
            c = c.getComponentType();
        }


        int[] shape = new int[nDimensions];
        Object current = array;
        for( int i=0; i<shape.length-1; i++ ){
            shape[i] = ((Object[])current).length;
            current = ((Object[])current)[0];
        }

        if(current instanceof Object[]) {
            shape[shape.length-1] = ((Object[])current).length;
        }
        else if(current instanceof double[]) {
            shape[shape.length-1] = ((double[])current).length;
        }
        else if(current instanceof float[]) {
            shape[shape.length-1] = ((float[])current).length;
        }
        else if(current instanceof long[]) {
            shape[shape.length-1] = ((long[])current).length;
        }
        else if(current instanceof int[]) {
            shape[shape.length-1] = ((int[])current).length;
        }
        else if(current instanceof byte[]) {
            shape[shape.length-1] = ((byte[])current).length;
        }
        else if(current instanceof char[]) {
            shape[shape.length-1] = ((char[])current).length;
        }
        else if( current instanceof boolean[] ){
            shape[shape.length-1] = ((boolean[])current).length;
        }
        else if( current instanceof short[] ){
            shape[shape.length-1] = ((short[])current).length;
        }
        else
            throw new IllegalStateException("Unknown array type");	//Should never happen
        return shape;
    }


    /** Returns the maximum value in the array */
    public static int max(int[] in){
        int max = Integer.MIN_VALUE;
        for( int i=0; i<in.length; i++ ) if(in[i]>max) max = in[i];
        return max;
    }

    /** Returns the minimum value in the array */
    public static int min(int[] in){
        int min = Integer.MAX_VALUE;
        for( int i=0; i<in.length; i++ ) if(in[i]<min) min = in[i];
        return min;
    }

    /** Returns the index of the maximum value in the array.
     * If two entries have same maximum value, index of the first one is returned. */
    public static int argMax(int[] in){
        int maxIdx = 0;
        for( int i=1; i<in.length; i++ ) if(in[i]>in[maxIdx]) maxIdx = i;
        return maxIdx;
    }

    /** Returns the index of the minimum value in the array.
     * If two entries have same minimum value, index of the first one is returned. */
    public static int argMin(int[] in){
        int minIdx = 0;
        for( int i=1; i<in.length; i++ ) if(in[i]<in[minIdx]) minIdx = i;
        return minIdx;
    }

    /** Returns the index of the maximum value in the array.
     * If two entries have same maximum value, index of the first one is returned. */
    public static int argMax(long[] in){
        int maxIdx = 0;
        for( int i=1; i<in.length; i++ ) if(in[i]>in[maxIdx]) maxIdx = i;
        return maxIdx;
    }

    /** Returns the index of the minimum value in the array.
     * If two entries have same minimum value, index of the first one is returned. */
    public static int argMin(long[] in){
        int minIdx = 0;
        for( int i=1; i<in.length; i++ ) if(in[i]<in[minIdx]) minIdx = i;
        return minIdx;
    }

    public static int argMinOfMax(int[] first, int[] second){
        int minIdx = 0;
        int maxAtMinIdx = Math.max(first[0],second[0]);
        for( int i=1; i<first.length; i++ ){
            int maxAtIndex = Math.max(first[i],second[i]);
            if(maxAtMinIdx > maxAtIndex){
                maxAtMinIdx = maxAtIndex;
                minIdx = i;
            }
        }
        return minIdx;
    }

    public static int argMinOfMax(int[]... arrays){
        int minIdx = 0;
        int maxAtMinIdx = Integer.MAX_VALUE;

        for( int i=0; i<arrays[0].length; i++ ){
            int maxAtIndex = Integer.MIN_VALUE;
            for( int j=0; j<arrays.length; j++ ){
                maxAtIndex = Math.max(maxAtIndex,arrays[j][i]);
            }

            if(maxAtMinIdx > maxAtIndex){
                maxAtMinIdx = maxAtIndex;
                minIdx = i;
            }
        }
        return minIdx;
    }

    public static int argMinOfSum(int[] first, int[] second){
        int minIdx = 0;
        int sumAtMinIdx = first[0]+second[0];
        for( int i=1; i<first.length; i++ ){
            int sumAtIndex = first[i]+second[i];
            if(sumAtMinIdx > sumAtIndex){
                sumAtMinIdx = sumAtIndex;
                minIdx = i;
            }
        }
        return minIdx;
    }
}
