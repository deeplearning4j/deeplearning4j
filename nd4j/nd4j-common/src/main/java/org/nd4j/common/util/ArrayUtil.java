/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.util;

import org.nd4j.common.primitives.Pair;
import org.nd4j.shade.guava.primitives.Ints;
import org.nd4j.shade.guava.primitives.Longs;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.common.base.Preconditions;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.util.*;

/**
 * @author Adam Gibson
 */
public class ArrayUtil {


    private ArrayUtil() {}



    /**
     * Flattens a potentially multi-dimensional primitive array into a 1D primitive array
     * and returns it along with the original shape. Assumes a regular (non-jagged) array.
     *
     * @param multiDimArray The input array (e.g., int[][], double[][][]). Must not be null
     * and must be an array with a primitive component type.
     * @return A Pair where:
     * - First element: int[] representing the original shape of the input array.
     * - Second element: Object representing the flattened 1D primitive array
     * (e.g., int[], double[], boolean[]). The caller needs to cast this.
     * @throws IllegalArgumentException if the input is not a supported array type or is jagged.
     */
    public static Pair<int[], Object> flattenDeep(Object multiDimArray) {
        Preconditions.checkNotNull(multiDimArray, "Input array cannot be null");
        Preconditions.checkArgument(multiDimArray.getClass().isArray(), "Input must be an array");

        List<Integer> shapeList = new ArrayList<>();
        Object currentLevel = multiDimArray;
        Class<?> componentType = null;

        // 1. Determine shape and component type
        while (currentLevel.getClass().isArray()) {
            int length = Array.getLength(currentLevel);
            shapeList.add(length);
            if (length == 0) {
                // Handle empty dimension - find component type and stop dimension search
                componentType = currentLevel.getClass().getComponentType();
                while(componentType.isArray()){
                    shapeList.add(0); // Add zero for remaining dimensions
                    componentType = componentType.getComponentType();
                }
                break; // Shape determined
            }
            currentLevel = Array.get(currentLevel, 0); // Get first element to descend
            if (currentLevel == null) {
                // Handle null elements if necessary, assume non-jagged for now
                throw new IllegalArgumentException("Array contains null elements, cannot determine shape reliably.");
            }
            componentType = currentLevel.getClass(); // Update component type guess
        }

        // Final component type check
        if(componentType.isArray()) {
            // This happens for things like int[2][0][] - needs careful handling
            // For simplicity, let's assume regular arrays for now
            throw new IllegalArgumentException("Deep component type determination failed or jagged array encountered.");
        }
        if (!componentType.isPrimitive()) {
            throw new IllegalArgumentException("Only arrays with primitive component types are supported (e.g., int, double, boolean). Found: " + componentType);
        }

        int[] shape = Ints.toArray(shapeList); // Convert List<Integer> to int[]
        long totalElementsLong = 1;
        for (int dim : shape) {
            try { totalElementsLong = Math.multiplyExact(totalElementsLong, dim); }
            catch (ArithmeticException e) { throw new IllegalArgumentException("Array size exceeds Long.MAX_VALUE"); }
        }

        if (totalElementsLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Flattened array size exceeds Integer.MAX_VALUE, not supported by standard Java array indexing.");
        }
        int totalElements = (int) totalElementsLong;


        // 2. Create flattened array
        Object flattenedArray = Array.newInstance(componentType, totalElements);

        // 3. Flatten data (recursive helper)
        if (totalElements > 0) { // Avoid recursion if array is empty
            flattenRecursive(multiDimArray, shape, new int[shape.length], flattenedArray, 0);
        }

        // 4. Return shape and flattened data
        return Pair.create(shape, flattenedArray);
    }

    // Recursive helper to copy elements in C-order
    private static int flattenRecursive(Object sourceArray, int[] shape, int[] indices, Object flatArray, int flatIndex) {
        if (indices.length == shape.length) { // Base case: reached individual element
            Array.set(flatArray, flatIndex, sourceArray);
            return flatIndex + 1;
        }

        int currentDim = indices.length; // The dimension we are currently iterating over
        int dimSize = shape[currentDim];

        for (int i = 0; i < dimSize; i++) {
            indices[currentDim] = i; // Set index for current dimension
            Object subArray = Array.get(sourceArray, i); // Get sub-array or element

            // Create a copy of indices for recursive call if needed (or manage index restoration)
            // For simplicity, we can pass indices down and rely on the loop structure.
            // However, passing a copy might be safer if modifications were complex.
            // Let's assume direct passing works here.

            // Recurse to the next dimension
            flatIndex = flattenRecursive(subArray, shape, Arrays.copyOf(indices, currentDim + 1), flatArray, flatIndex);
        }
        return flatIndex;
    }

    // Overload for initial call without indices array needed externally
    private static int flattenRecursive(Object sourceArray, int[] shape, Object flatArray, int flatIndex) {
        return flattenRecursive(sourceArray, shape, new int[0], flatArray, flatIndex);
    }

    /**
     * Converts a byte array to a boolean array.
     * @param input the input byte array
     * @return a boolean array with true values for nonzero bytes and false values for zero bytes
     */
    public static boolean[] toBooleanArray(byte[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] != 0;
        }
        return output;
    }

    /**
     * Converts a short array to a boolean array.
     * @param input the input short array
     * @return a boolean array with true values for nonzero shorts and false values for zero shorts
     */
    public static boolean[] toBooleanArray(short[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] != 0;
        }
        return output;
    }

    /**
     * Converts an int array to a boolean array.
     * @param input the input int array
     * @return a boolean array with true values for nonzero integers and false values for zero integers
     */
    public static boolean[] toBooleanArray(int[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] != 0;
        }
        return output;
    }

    /**
     * Converts a long array to a boolean array.
     * @param input the input long array
     * @return a boolean array with true values for nonzero longs and false values for zero longs
     */
    public static boolean[] toBooleanArray(long[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] != 0L;
        }
        return output;
    }

    /**
     * Converts a float array to a boolean array.
     * @param input the input float array
     * @return a boolean array with true values for nonzero floats and false values for zero floats
     */
    public static boolean[] toBooleanArray(float[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] != 0.0f;
        }
        return output;
    }

    /**
     * Converts a double array to a boolean array.
     * @param input the input double array
     * @return a boolean array with true values for nonzero doubles and false values for zero doubles
     */
    public static boolean[] toBooleanArray(double[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] != 0.0;
        }
        return output;
    }


    /**
     * Create a boolean array from a float array.
     * @param elements the elements to create
     * @return the returned float array
     */
    public static boolean[] fromFloat(float[] elements) {
        boolean[] ret = new boolean[elements.length];
        for(int i = 0; i < elements.length; i++) {
            ret[i] = elements[i] == 0.0f ? false : true;
        }

        return ret;
    }

    /**
     * Generate an array with n elements of the same specified value.
     * @param element the element to create n copies of
     * @param n the number of elements
     * @return the created array
     */
    public static boolean[] nTimes(boolean element,int n) {
        boolean[] ret = new boolean[n];
        //boolean values  default to false. Only need to iterate when true.
        if(element) {
            for (int i = 0; i < n; i++) {
                ret[i] = element;
            }
        }
        return ret;
    }




    /**
     * Converts a short array to an int array.
     * @param input the input short array
     * @return an int array with each element equal to the corresponding short value
     */
    public static int[] toIntArray(short[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a short array to an int array.
     * @param input the input short array
     * @return an int array with each element equal to the corresponding int value
     */
    public static int[] toIntArray(boolean[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] ? 1 : 0;
        }
        return output;
    }



    /**
     * Converts a char array to an int array.
     * @param input the input char array
     * @return an int array with each element equal to the corresponding char value
     */
    public static int[] toIntArray(char[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts an int array to an int array (i.e., returns a copy of the input array).
     * @param input the input int array
     * @return a new int array with the same values as the input array
     */
    public static int[] toIntArray(int[] input) {
        int[] output = new int[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    /**
     * Converts a long array to an int array.
     * @param input the input long array
     * @return an int array with each element equal to the corresponding long value cast to an int
     */
    public static int[] toIntArray(long[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (int) input[i];
        }
        return output;
    }

    /**
     * Converts a float array to an int array.
     * @param input the input float array
     * @return an int array with each element equal to the corresponding float value cast to an int
     */
    public static int[] toIntArray(float[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (int) input[i];
        }
        return output;
    }

    /**
     * Converts a double array to an int array.
     * @param input the input double array
     * @return an int array with each element equal to the corresponding double value cast to an int
     */
    public static int[] toIntArray(double[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (int) input[i];
        }
        return output;
    }



    /**
     * Converts a short array to a double array.
     * @param input the input short array
     * @return a double array with each element equal to the corresponding short value
     */
    public static double[] toDoubleArray(short[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a char array to a double array.
     * @param input the input char array
     * @return a double array with each element equal to the corresponding char value
     */
    public static double[] toDoubleArray(char[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a boolean array to a double array.
     * @param input the input boolean array
     * @return a double array with each element equal to the corresponding double value
     */
    public static double[] toDoubleArray(boolean[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] ? 1.0 : 0.0;
        }
        return output;
    }


    /**
     * Converts an int array to a double array.
     * @param input the input int array
     * @return a double array with each element equal to the corresponding int value
     */
    public static double[] toDoubleArray(int[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a long array to a double array.
     * @param input the input long array
     * @return a double array with each element equal to the corresponding long value
     */
    public static double[] toDoubleArray(long[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a float array to a double array.
     * @param input the input float array
     * @return a double array with each element equal to the corresponding float value
     */
    public static double[] toDoubleArray(float[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a double array to a double array (i.e., returns a copy of the input array).
     * @param input the input double array
     * @return a new double array with the same values as the input array
     */
    public static double[] toDoubleArray(double[] input) {
        double[] output = new double[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    /**
     * Converts a byte array to a long array.
     * @param input the input byte array
     * @return a long array with each element equal to the corresponding byte value
     */
    public static long[] toLongArray(byte[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a byte array to a long array.
     * @param input the input boolean array
     * @return a long array with each element equal to the corresponding long value
     */
    public static long[] toLongArray(boolean[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] ? 1 : 0;
        }
        return output;
    }

    /**
     * Converts a short array to a long array.
     * @param input the input short array
     * @return a long array with each element equal to the corresponding short value
     */
    public static long[] toLongArray(short[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a char array to a long array.
     * @param input the input char array
     * @return a long array with each element equal to the corresponding char value
     */
    public static long[] toLongArray(char[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts an int array to a long array.
     * @param input the input int array
     * @return a long array with each element equal to the corresponding int value
     */
    public static long[] toLongArrayInt(int[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a float array to a long array.
     * @param input the input float array
     * @return a long array with each element equal to the corresponding float value
     */
    public static long[] toLongArrayFloat(float[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (long) input[i];
        }
        return output;
    }

    /**
     * Converts a double array to a long array.
     * @param input the input double array
     * @return a long array with each element equal to the corresponding double value
     */
    public static long[] toLongArray(double[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (long) input[i];
        }
        return output;
    }

    /**
     * Converts a long array to a long array (i.e., returns a copy of the input array).
     * @param input the input long array
     * @return a new long array with the same values as the input array
     */
    public static long[] toLongArray(long[] input) {
        long[] output = new long[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }


    /**
     * Converts a short array to a float array.
     * @param input the input short array
     * @return a float array with each element equal to the corresponding short value
     */
    public static float[] toFloatArray(short[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a boolean array to a float array.
     * @param input the input boolean array
     * @return a float array with each element equal to the corresponding boolean value
     */
    public static float[] toFloatArray(boolean[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] ? 1.0f : 0.0f;
        }
        return output;
    }

    /**
     * Converts a char array to a float array.
     * @param input the input char array
     * @return a float array with each element equal to the corresponding char value
     */
    public static float[] toFloatArray(char[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts an int array to a float array.
     * @param input the input int array
     * @return a float array with each element equal to the corresponding int value
     */
    public static float[] toFloatArray(int[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a long array to a float array.
     * @param input the input long array
     * @return a float array with each element equal to the corresponding long value
     */
    public static float[] toFloatArray(long[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * Converts a double array to a float array.
     * @param input the input double array
     * @return a float array with each element equal to the corresponding double value
     */
    public static float[] toFloatArray(double[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) input[i];
        }
        return output;
    }

    /**
     * Converts a float array to a float array (i.e., returns a copy of the input array).
     * @param input the input float array
     * @return a new float array with the same values as the input array
     */
    public static float[] toFloatArray(float[] input) {
        float[] output = new float[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    /**
     * Concat all the elements
     * @param arrs the input arrays
     * @return
     * @param <T>
     */
    public static <T> T[] concat(Class<T> clazz,T[]...arrs) {
        int totalLength = 0;
        for(T[] arr : arrs) totalLength += arr.length;
        T[] ret  = (T[]) Array.newInstance(clazz,totalLength);
        int count = 0;
        for(T[] arr : arrs) {
            for(T input : arr) {
                ret[count] = input;
                count++;
            }
        }

        return ret;
    }

    /**
     * Returns true if any array elements are negative.
     * If the array is null, it returns false
     * @param arr the array to test
     * @return
     */
    public static boolean containsAnyNegative(int[] arr) {
        if(arr == null)
            return false;

        for(int i = 0; i < arr.length; i++) {
            if(arr[i] < 0)
                return true;
        }
        return false;
    }

    public static boolean containsAnyNegative(long[] arr) {
        if(arr == null)
            return false;

        for(int i = 0; i < arr.length; i++) {
            if(arr[i] < 0)
                return true;
        }
        return false;
    }

    public static boolean contains(int[] arr, int value){
        if(arr == null)
            return false;
        for( int i : arr ) {
            if (i == value)
                return true;
        }
        return false;
    }

    public static boolean contains(long[] arr, int value){
        if(arr == null)
            return false;
        for( long i : arr ) {
            if (i == value)
                return true;
        }
        return false;
    }

    /**
     *
     * @param arrs
     * @param check
     * @return
     */
    public static boolean anyLargerThan(int[] arrs, int check) {
        for(int i = 0; i < arrs.length; i++) {
            if(arrs[i] > check)
                return true;
        }

        return false;
    }


    /**
     *
     * @param arrs
     * @param check
     * @return
     */
    public static boolean anyLessThan(int[] arrs, int check) {
        for(int i = 0; i < arrs.length; i++) {
            if(arrs[i] < check)
                return true;
        }

        return false;
    }


    /**
     * Convert a int array to a string array
     * @param arr the array to convert
     * @return the equivalent string array
     */
    public static String[] convertToString(int[] arr) {
        Preconditions.checkNotNull(arr);
        String[] ret = new String[arr.length];
        for(int i = 0; i < arr.length; i++) {
            ret[i] = String.valueOf(arr[i]);
        }

        return ret;
    }


    /**
     * Proper comparison contains for list of int
     * arrays
     * @param list the to search
     * @param target the target int array
     * @return whether the given target
     * array is contained in the list
     */
    public static boolean listOfIntsContains(List<int[]> list,int[] target) {
        for(int[] arr : list)
            if(Arrays.equals(target,arr))
                return true;
        return false;
    }

    /**
     * Repeat a value n times
     * @param n the number of times to repeat
     * @param toReplicate the value to repeat
     * @return an array of length n filled with the
     * given value
     */
    public static int[] nTimes(int n, int toReplicate) {
        int[] ret = new int[n];
        Arrays.fill(ret, toReplicate);
        return ret;
    }

    public static long[] nTimes(long n, long toReplicate) {
        if (n > Integer.MAX_VALUE)
            throw new RuntimeException("Index overflow in nTimes");
        val ret = new long[(int) n];
        Arrays.fill(ret, toReplicate);
        return ret;
    }

    public static <T> T[] nTimes(int n, T toReplicate, Class<T> tClass){
        Preconditions.checkState(n>=0, "Invalid number of times to replicate: must be >= 0, got %s", n);
        T[] out = (T[])Array.newInstance(tClass, n);
        for( int i=0; i<n; i++ ){
            out[i] = toReplicate;
        }
        return out;
    }

    /**
     * Returns true if all the elements in the
     * given int array are unique
     * @param toTest the array to test
     * @return true if all the items
     * are unique false otherwise
     */
    public static boolean allUnique(int[] toTest) {
        Set<Integer> set = new HashSet<>();
        for (int i : toTest) {
            if (!set.contains(i))
                set.add(i);
            else
                return false;
        }

        return true;
    }

    /**
     * Credit to mikio braun from jblas
     * <p>
     * Create a random permutation of the numbers 0, ..., size - 1.
     * </p>
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

    public static short toBFloat16(short data) {
        // Assume the short is already a float in 16-bit half-precision format
        int sign = data >>> 15;
        int exp = (data >>> 10) & 0x1F;
        int fraction = data & 0x3FF;

        // Adjust exponent from half-precision bias (15) to bfloat16 bias (127)
        exp = exp - 15 + 127;

        // Check for underflow and overflow
        if (exp < 0) {  // Underflow
            exp = 0;
            fraction = 0;
        } else if (exp > 255) {  // Overflow
            exp = 255;
            fraction = 0;
        }

        // Truncate fraction to fit into 7 bits
        fraction >>>= 3;

        // Recombine bits
        short bfloat16 = (short) ((sign << 15) | (exp << 7) | fraction);

        return bfloat16;
    }


    public static float bfloat16ToFloat(short b) {
        int sign = b >>> 15;
        int exp = (b >>> 7) & 0xFF;
        int fraction = b & 0x7F;

        // Extend fraction part to 23 bits
        fraction <<= 16;

        // Recombine bits
        int floatBits = (sign << 31) | (exp << 23) | fraction;

        return Float.intBitsToFloat(floatBits);
    }

    public static double bfloat16ToDouble(short b) {
        // Convert bfloat16 to float then to double
        return (double) bfloat16ToFloat(b);
    }


    public static short longToBFloat16(long l) {
        // Convert long to float then to bfloat16
        return toBFloat16((double) l);
    }

    // Reverse conversions from bfloat16 to types

    public static long bfloat16ToLong(short b) {
        // Convert bfloat16 to float then to long
        return (long) bfloat16ToFloat(b);
    }

    public static int bfloat16ToInt(short b) {
        // Convert bfloat16 to float then to int
        return (int) bfloat16ToFloat(b);
    }

    public static short bfloat16ToShort(short b) {
        // Convert the bfloat16 to a half-precision float
        int sign = b >>> 15;
        int exp = (b >>> 7) & 0xFF;
        int fraction = b & 0x7F;

        // Truncate the exponent and extend fraction to fit half-precision
        exp >>>= 3;
        fraction <<= 3;

        // Recombine bits
        return (short) ((sign << 15) | (exp << 10) | fraction);
    }

    public static short toBFloat16(float data) {
        int floatBits = Float.floatToRawIntBits(data);
        int sign = floatBits >>> 31;
        int exp = (floatBits >>> 23) & 0xFF;
        int fraction = floatBits & 0x7FFFFF;

        // Truncate fraction part to 7 bits
        fraction >>>= 16;

        // Recombine bits
        short bfloat16 = (short) ((sign << 15) | (exp << 7) | fraction);

        return bfloat16;
    }

    public static short toBFloat16(double data) {
        return toBFloat16((float) data);
    }

    public static short toHalf(float data) {
        return fromFloat(data);
    }

    public static short toHalf(double data) {
        return fromFloat((float) data);
    }

    public static short[] toHalfs(boolean[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = fromFloat(data[i] ? 1 : 0);
        }
        return ret;
    }
    public static short[] toHalfs(byte[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = fromFloat(data[i]);
        }
        return ret;
    }
    public static short[] toHalfs(short[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = fromFloat(data[i]);
        }
        return ret;
    }

    public static short[] toHalfs(float[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = fromFloat(data[i]);
        }
        return ret;
    }

    public static short[] toHalfs(int[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = fromFloat((float) data[i]);
        }
        return ret;
    }

    public static short[] toHalfs(long[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = fromFloat((float) data[i]);
        }
        return ret;
    }

    public static short[] toBfloats(double[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = toBFloat16(data[i]);
        }
        return ret;
    }
    public static short[] toBfloats(boolean[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = toBFloat16(data[i] ? 1.0 : 0.0);
        }
        return ret;
    }
    public static short[] toBfloats(byte[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = toBFloat16(data[i]);
        }
        return ret;
    }
    public static short[] toBfloats(short[] data) {
        float[] ret = new float[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = toBFloat16((float) data[i]);
        }
        return ArrayUtil.toShorts(ret);
    }

    public static short[] toBfloats(float[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = toBFloat16(data[i]);
        }
        return ret;
    }

    public static short[] toBfloats(int[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = toBFloat16(data[i]);
        }
        return ret;
    }

    public static short[] toBfloats(long[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = toBFloat16((float) data[i]);
        }
        return ret;
    }

    public static long[] toLongs(byte[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data[i];
        }
        return ret;
    }

    public static long[] toLongs(boolean[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data[i] ? 1 : 0;
        }
        return ret;
    }

    public static long[] toLongs(short[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data[i];
        }
        return ret;
    }

    public static long[] toLongs(int[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data[i];
        }
        return ret;
    }

    public static long[] toLongs(float[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = (long) data[i];
        }
        return ret;
    }

    public static long[] toLongs(double[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = (long) data[i];
        }
        return ret;
    }

    public static short[] toHalfs(double[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = fromFloat((float) data[i]);
        }
        return ret;
    }

    public static short fromFloat(float v) {
        if (Float.isNaN(v))
            return (short) 0x7fff;
        if (v == Float.POSITIVE_INFINITY)
            return (short) 0x7c00;
        if (v == Float.NEGATIVE_INFINITY)
            return (short) 0xfc00;
        if (v == 0.0f)
            return (short) 0x0000;
        if (v == -0.0f)
            return (short) 0x8000;
        if (v > 65504.0f)
            return 0x7bff; // max value supported by half float
        if (v < -65504.0f)
            return (short) (0x7bff | 0x8000);
        if (v > 0.0f && v < 5.96046E-8f)
            return 0x0001;
        if (v < 0.0f && v > -5.96046E-8f)
            return (short) 0x8001;

        final int f = Float.floatToIntBits(v);

        return (short) (((f >> 16) & 0x8000) | ((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00)
                | ((f >> 13) & 0x03ff));
    }


    public static int[] toInts(boolean[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = (int) (data[i] ? 1 : 0);
        return ret;
    }

    public static int[] toInts(byte[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = (int) data[i];
        return ret;
    }

    public static int[] toInts(short[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = (int) data[i];
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

    public static byte[] toBytes(int[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) {
            retVal[i] = (byte) array[i];
        }
        return retVal;
    }

    public static byte[] toBytes(short[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) {
            retVal[i] = (byte) array[i];
        }
        return retVal;
    }

    public static byte[] toBytes(boolean[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) {
            retVal[i] = (byte)  (array[i] ? 1 : 0);
        }
        return retVal;
    }

    public static byte[] toBytes(float[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) {
            retVal[i] = (byte) array[i];
        }
        return retVal;
    }

    public static byte[] toBytes(double[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) {
            retVal[i] = (byte) array[i];
        }
        return retVal;
    }

    public static byte[] toBytes(long[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) {
            retVal[i] = (byte) array[i];
        }
        return retVal;
    }

    public static int[] toInts(long[] array) {
        int[] retVal = new int[array.length];

        for (int i = 0; i < array.length; i++) {
            retVal[i] = (int) array[i];
        }

        return retVal;
    }


    public static int[] mod(int[] input,int mod) {
        int[] ret = new int[input.length];
        for(int i = 0; i < ret.length; i++) {
            ret[i]  = input[i] % mod;
        }

        return ret;
    }


    /**
     * Calculate the offset for a given stride array
     * @param stride the stride to use
     * @param i the offset to calculate for
     * @return the offset for the given
     * stride
     */
    public static int offsetFor(int[] stride, int i) {
        int ret = 0;
        for (int j = 0; j < stride.length; j++)
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
        if (add.isEmpty())
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

    public static long sumLong(long... add) {
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
        if (mult.isEmpty())
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
    public static int prod(long... mult) {
        if (mult.length < 1)
            return 0;
        int ret = 1;
        for (int i = 0; i < mult.length; i++)
            ret *= mult[i];
        return ret;
    }


    /**
     * Product of an int array
     * @param mult the elements
     *            to calculate the sum for
     * @return the product of this array
     */
    public static int prod(int... mult) {
        if (mult.length < 1)
            return 0;
        int ret = 1;
        for (int i = 0; i < mult.length; i++)
            ret *= mult[i];
        return ret;
    }

    /**
     * Product of an int array
     * @param mult the elements
     *            to calculate the sum for
     * @return the product of this array
     */
    public static long prodLong(List<? extends Number> mult) {
        if (mult.isEmpty())
            return 0;
        long ret = 1;
        for (int i = 0; i < mult.size(); i++)
            ret *= mult.get(i).longValue();
        return ret;
    }


    /**
     * Product of an int array
     * @param mult the elements
     *            to calculate the sum for
     * @return the product of this array
     */
    public static long prodLong(int... mult) {
        if (mult.length < 1)
            return 0;
        long ret = 1;
        for (int i = 0; i < mult.length; i++)
            ret *= mult[i];
        return ret;
    }

    public static long prodLong(long... mult) {
        if (mult.length < 1)
            return 0;
        long ret = 1;
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
            if (as[i] == 0)
                return true;
        }
        return false;
    }

    public static boolean isZero(long[] as) {
        for (int i = 0; i < as.length; i++) {
            if (as[i] == 0L)
                return true;
        }
        return false;
    }

    public static boolean anyMore(int[] target, int[] test) {
        Preconditions.checkArgument(target.length == test.length, "Unable to compare: different sizes: length %s vs. %s", target.length, test.length);
        for (int i = 0; i < target.length; i++) {
            if (target[i] > test[i])
                return true;
        }
        return false;
    }


    public static boolean anyLess(int[] target, int[] test) {
        Preconditions.checkArgument(target.length == test.length, "Unable to compare: different sizes: length %s vs. %s", target.length, test.length);
        for (int i = 0; i < target.length; i++) {
            if (target[i] < test[i])
                return true;
        }
        return false;
    }

    public static boolean lessThan(int[] target, int[] test) {
        Preconditions.checkArgument(target.length == test.length, "Unable to compare: different sizes: length %s vs. %s", target.length, test.length);
        for (int i = 0; i < target.length; i++) {
            if (target[i] < test[i])
                return true;
            if (target[i] > test[i])
                return false;
        }
        return false;
    }

    public static boolean greaterThan(int[] target, int[] test) {
        Preconditions.checkArgument(target.length == test.length, "Unable to compare: different sizes: length %s vs. %s", target.length, test.length);
        for (int i = 0; i < target.length; i++) {
            if (target[i] > test[i])
                return true;
            if (target[i] < test[i])
                return false;
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
    public static int calcOffset(List<Integer> shape, List<Integer> offsets, List<Integer> strides) {
        if (shape.size() != offsets.size() || shape.size() != strides.size())
            throw new IllegalArgumentException("Shapes,strides, and offsets must be the same size");
        int ret = 0;
        for (int i = 0; i < offsets.size(); i++) {
            //we should only do this in the general case, not on vectors
            //the reason for this is we force everything including scalars
            //to be 2d
            if (shape.get(i) == 1 && offsets.size() > 2 && i > 0)
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
    public static int calcOffset(int[] shape, int[] offsets, int[] strides) {
        if (shape.length != offsets.length || shape.length != strides.length)
            throw new IllegalArgumentException("Shapes,strides, and offsets must be the same size");

        int ret = 0;
        for (int i = 0; i < offsets.length; i++) {
            if (shape[i] == 1)
                continue;
            ret += offsets[i] * strides[i];
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
    public static long calcOffset(long[] shape, long[] offsets, long[] strides) {
        if (shape.length != offsets.length || shape.length != strides.length)
            throw new IllegalArgumentException("Shapes,strides, and offsets must be the same size");

        long ret = 0;
        for (int i = 0; i < offsets.length; i++) {
            if (shape[i] == 1)
                continue;
            ret += offsets[i] * strides[i];
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
    public static long calcOffsetLong(List<Integer> shape, List<Integer> offsets, List<Integer> strides) {
        if (shape.size() != offsets.size() || shape.size() != strides.size())
            throw new IllegalArgumentException("Shapes,strides, and offsets must be the same size");
        long ret = 0;
        for (int i = 0; i < offsets.size(); i++) {
            //we should only do this in the general case, not on vectors
            //the reason for this is we force everything including scalars
            //to be 2d
            if (shape.get(i) == 1 && offsets.size() > 2 && i > 0)
                continue;
            ret += (long) offsets.get(i) * strides.get(i);
        }

        return ret;
    }


    public static long calcOffsetLong2(List<Long> shape, List<Long> offsets, List<Long> strides) {
        if (shape.size() != offsets.size() || shape.size() != strides.size())
            throw new IllegalArgumentException("Shapes,strides, and offsets must be the same size");
        long ret = 0;
        for (int i = 0; i < offsets.size(); i++) {
            //we should only do this in the general case, not on vectors
            //the reason for this is we force everything including scalars
            //to be 2d
            if (shape.get(i) == 1 && offsets.size() > 2 && i > 0)
                continue;
            ret += (long) offsets.get(i) * strides.get(i);
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
    public static long calcOffsetLong(int[] shape, int[] offsets, int[] strides) {
        if (shape.length != offsets.length || shape.length != strides.length)
            throw new IllegalArgumentException("Shapes,strides, and offsets must be the same size");

        long ret = 0;
        for (int i = 0; i < offsets.length; i++) {
            if (shape[i] == 1)
                continue;
            ret += (long) offsets[i] * strides[i];
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

    /**
     *
     * @param xs
     * @param ys
     * @return
     */
    public static long dotProductLong(List<Integer> xs, List<Integer> ys) {
        long result = 0;
        int n = xs.size();

        if (ys.size() != n)
            throw new IllegalArgumentException("Different array sizes");

        for (int i = 0; i < n; i++) {
            result += (long) xs.get(i) * ys.get(i);
        }
        return result;
    }

    /**
     *
     * @param xs
     * @param ys
     * @return
     */
    public static long dotProductLong2(List<Long> xs, List<Long> ys) {
        long result = 0;
        int n = xs.size();

        if (ys.size() != n)
            throw new IllegalArgumentException("Different array sizes");

        for (int i = 0; i < n; i++) {
            result += (long) xs.get(i) * ys.get(i);
        }
        return result;
    }

    /**
     *
     * @param xs
     * @param ys
     * @return
     */
    public static long dotProductLong(int[] xs, int[] ys) {
        long result = 0;
        int n = xs.length;

        if (ys.length != n)
            throw new IllegalArgumentException("Different array sizes");

        for (int i = 0; i < n; i++) {
            result += (long) xs[i] * ys[i];
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

    public static long[] copy(long[] copy) {
        long[] ret = new long[copy.length];
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
     * Returns a subset of an array from 0 to "to" (exclusive)
     *
     * @param data the data to getFromOrigin a subset of
     * @param to   the end point of the data
     * @return the subset of the data specified
     */
    public static double[] range(double[] data, int to) {
        return range(data, to, 1);
    }


    /**
     * Returns a subset of an array from 0 to "to" (exclusive) using the specified stride
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
        if (ret.length < 1)
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

    public static List<Long> toList(long... ints) {
        if(ints == null){
            return null;
        }
        List<Long> ret = new ArrayList<>();
        for (long anInt : ints) {
            ret.add(anInt);
        }
        return ret;
    }


    public static List<Integer> toList(int... ints) {
        if(ints == null){
            return null;
        }
        List<Integer> ret = new ArrayList<>();
        for (int anInt : ints) {
            ret.add(anInt);
        }
        return ret;
    }

    public static int[] toArray(List<Integer> list) {
        int[] ret = new int[list.size()];
        for (int i = 0; i < list.size(); i++)
            ret[i] = list.get(i);
        return ret;
    }

    public static long[] toArrayLong(List<Long> list) {
        long[] ret = new long[list.size()];
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
     * Generate an int array ranging from "from" to "to".
     * The total number of elements is (from-to)/increment - i.e., range(0,2,1) returns [0,1]
     * If from is > to this method will count backwards
     *
     * @param from      the from
     * @param to        the end point of the data
     * @param increment the amount to increment by
     * @return the int array with a length equal to absoluteValue(from - to)
     */
    public static int[] range(int from, int to, int increment) {
        int diff = Math.abs(from - to);
        int[] ret = new int[diff / increment];
        if (ret.length < 1)
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


    public static long[] range(long from, long to, long increment) {
        long diff = Math.abs(from - to);
        long[] ret = new long[(int) (diff / increment)];
        if (ret.length < 1)
            ret = new long[1];

        if (from < to) {
            int count = 0;
            for (long i = from; i < to; i += increment) {
                if (count >= ret.length)
                    break;
                ret[count++] = i;
            }
        } else if (from > to) {
            int count = 0;
            for (int i = (int) from - 1; i >= to; i -= increment) {
                if (count >= ret.length)
                    break;
                ret[count++] = i;
            }
        }

        return ret;
    }

    /**
     * Generate an int array ranging from "from" to "to".
     * The total number of elements is (from-to) - i.e., range(0,2) returns [0,1]
     * If from is > to this method will count backwards
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

    public static long[] range(long from, long to) {
        if (from == to)
            return new long[0];
        return range(from, to, 1);
    }

    public static double[] toDoubles(int[] ints) {
        double[] ret = new double[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (double) ints[i];
        return ret;
    }

    public static double[] toDoubles(long[] ints) {
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


    public static float[] toFloats(boolean[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (float) (ints[i] ? 1 : 0);
        return ret;
    }


    public static float[] toFloats(byte[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (float) ints[i];
        return ret;
    }

    public static double[] toDoubles(int[][] ints) {
        return toDoubles(Ints.concat(ints));
    }


    public static short[] toShorts(long[] ints) {
        val ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (short) ints[i];
        return ret;
    }

    public static short[] toShorts(byte[] ints) {
        val ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (short) ints[i];
        return ret;
    }


    public static short[] toShorts(boolean[] ints) {
        val ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (short) (ints[i] ? 1 : 0);
        return ret;
    }

    public static short[] toShorts(int[] ints) {
        val ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (short) ints[i];
        return ret;
    }

    public static short[] toShorts(float[] ints) {
        val ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (short) ints[i];
        return ret;
    }

    public static short[] toShorts(double[] ints) {
        val ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (short) ints[i];
        return ret;
    }


    public static float[] toFloats(short[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (float) ints[i];
        return ret;
    }

    public static float[] toFloats(int[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++)
            ret[i] = (float) ints[i];
        return ret;
    }

    public static float[] toFloats(long[] ints) {
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



    public static int[] cutBelowZero(int[] data) {
        val ret = new int[data.length];
        for (int i = 0; i < data.length; i++)
            ret[i] = data[i] < 0 ? 0 : data[i];
        return ret;
    }

    public static long[] cutBelowZero(long[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < data.length; i++)
            ret[i] = data[i] < 0 ? 0 : data[i];
        return ret;
    }

    public static short[] cutBelowZero(short[] data) {
        val ret = new short[data.length];
        for (int i = 0; i < data.length; i++)
            ret[i] = data[i] < 0 ? 0 : data[i];
        return ret;
    }

    public static byte[] cutBelowZero(byte[] data) {
        val ret = new byte[data.length];
        for (int i = 0; i < data.length; i++)
            ret[i] = data[i] < 0 ? 0 : data[i];
        return ret;
    }


    public static float[] cutBelowZero(float[] data) {
        val ret = new float[data.length];
        for (int i = 0; i < data.length; i++)
            ret[i] = data[i] < 0 ? 0 : data[i];
        return ret;
    }


    public static double[] cutBelowZero(double[] data) {
        val ret = new double[data.length];
        for (int i = 0; i < data.length; i++)
            ret[i] = data[i] < 0 ? 0 : data[i];
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
    public static int[] keep(int[] data, int... index) {
        if (index.length == data.length)
            return data;

        int[] ret = new int[index.length];
        int count = 0;
        for (int i = 0; i < data.length; i++)
            if (Ints.contains(index, i))
                ret[count++] = data[i];

        return ret;
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
    public static long[] keep(long[] data, long... index) {
        if (index.length == data.length)
            return data;

        long[] ret = new long[index.length];
        int count = 0;
        for (int i = 0; i < data.length; i++)
            if (Longs.contains(index, i))
                ret[count++] = data[i];

        return ret;
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
    public static long[] keep(long[] data, int... index) {
        if (index.length == data.length)
            return data;

        long[] ret = new long[index.length];
        int count = 0;
        for (int i = 0; i < data.length; i++)
            if (Ints.contains(index, i))
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
    public static long[] removeIndex(long[] data, long... index) {
        if(data.length < 1)
            return data;

        if (index.length >= data.length) {
            throw new IllegalStateException("Illegal remove: indexes.length > data.length (index.length="
                    + index.length + ", data.length=" + data.length + ")");
        }

        int offset = 0;


        long[] ret = new long[data.length - index.length + offset];
        int count = 0;
        for (int i = 0; i < data.length; i++)
            if (!Longs.contains(index, i)) {
                ret[count++] = data[i];
            }

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
    public static int[] removeIndex(int[] data, int... index) {
        if (index.length >= data.length) {
            throw new IllegalStateException("Illegal remove: indexes.length > data.length (index.length="
                    + index.length + ", data.length=" + data.length + ")");
        }
        int offset = 0;


        int[] ret = new int[data.length - index.length + offset];
        int count = 0;
        for (int i = 0; i < data.length; i++)
            if (!Ints.contains(index, i)) {
                ret[count++] = data[i];
            }

        return ret;
    }

    public static long[] removeIndex(long[] data, int... index) {
        if (index.length >= data.length) {
            throw new IllegalStateException("Illegal remove: indexes.length >= data.length (index.length="
                    + index.length + ", data.length=" + data.length + ")");
        }
        int offset = 0;
        long[] ret = new long[data.length - index.length + offset];
        int count = 0;
        for (int i = 0; i < data.length; i++)
            if (!Ints.contains(index, i)) {
                ret[count++] = data[i];
            }

        return ret;
    }



    /**
     * Zip 2 arrays in to:
     *
     * @param as
     * @param bs
     * @return
     */
    public static int[][] zip(int[] as, int[] bs) {
        int[][] result = new int[as.length][2];
        for (int i = 0; i < result.length; i++) {
            result[i] = new int[] {as[i], bs[i]};
        }

        return result;
    }

    /**
     * Get the tensor matrix multiply shape
     * @param aShape the shape of the first array
     * @param bShape the shape of the second array
     * @param axes the axes to do the multiply
     * @return the shape for tensor matrix multiply
     */
    public static long[] getTensorMmulShape(long[] aShape, long[] bShape, int[][] axes) {

        int validationLength = Math.min(axes[0].length, axes[1].length);
        for (int i = 0; i < validationLength; i++) {
            if (aShape[axes[0][i]] != bShape[axes[1][i]])
                throw new IllegalArgumentException(
                        "Size of the given axes a" + " t each dimension must be the same size.");
            if (axes[0][i] < 0)
                axes[0][i] += aShape.length;
            if (axes[1][i] < 0)
                axes[1][i] += bShape.length;

        }

        List<Integer> listA = new ArrayList<>();
        for (int i = 0; i < aShape.length; i++) {
            if (!Ints.contains(axes[0], i))
                listA.add(i);
        }



        List<Integer> listB = new ArrayList<>();
        for (int i = 0; i < bShape.length; i++) {
            if (!Ints.contains(axes[1], i))
                listB.add(i);
        }


        int n2 = 1;
        int aLength = Math.min(aShape.length, axes[0].length);
        for (int i = 0; i < aLength; i++) {
            n2 *= aShape[axes[0][i]];
        }

        //if listA and listB are empty these do not initialize.
        //so initializing with {1} which will then get overridden if not empty
        long[] oldShapeA;
        if (listA.size() == 0) {
            oldShapeA = new long[] {1};
        } else {
            oldShapeA = Longs.toArray(listA);
            for (int i = 0; i < oldShapeA.length; i++)
                oldShapeA[i] = aShape[(int) oldShapeA[i]];
        }

        int n3 = 1;
        int bNax = Math.min(bShape.length, axes[1].length);
        for (int i = 0; i < bNax; i++) {
            n3 *= bShape[axes[1][i]];
        }


        long[] oldShapeB;
        if (listB.isEmpty()) {
            oldShapeB = new long[] {1};
        } else {
            oldShapeB = Longs.toArray(listB);
            for (int i = 0; i < oldShapeB.length; i++)
                oldShapeB[i] = bShape[(int) oldShapeB[i]];
        }


        long[] aPlusB = Longs.concat(oldShapeA, oldShapeB);
        return aPlusB;
    }

    /**
     * Permute the given input
     * switching the dimensions of the input shape
     * array with in the order of the specified
     * dimensions
     * @param shape the shape to permute
     * @param dimensions the dimensions
     * @return
     */
    public static int[] permute(int[] shape, int[] dimensions) {
        int[] ret = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            ret[i] = shape[dimensions[i]];
        }

        return ret;
    }


    public static long[] permute(long[] shape, long[] dimensions) {
        val ret = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            ret[i] = shape[(int) dimensions[i]];
        }

        return ret;
    }
    public static long[] permute(long[] shape, int[] dimensions) {
        val ret = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            ret[i] = shape[dimensions[i]];
        }

        return ret;
    }


    /**
     * Original credit: https://github.com/alberts/array4j/blob/master/src/main/java/net/lunglet/util/ArrayUtils.java
     * @param a
     * @return
     */
    public static int[] argsort(int[] a) {
        return argsort(a, true);
    }


    /**
     *
     * @param a
     * @param ascending
     * @return
     */
    public static int[] argsort(final int[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Ints.compare(a[i1], a[i2]);
            }
        });

        int[] ret = new int[indexes.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = indexes[i];

        return ret;
    }



    /**
     * Convert all dimensions in the specified
     * axes array to be positive
     * based on the specified range of values
     * @param range
     * @param axes
     * @return
     */
    public static int[] convertNegativeIndices(int range, int[] axes) {
        int[] axesRet = ArrayUtil.range(0, range);
        int[] newAxes = ArrayUtil.copy(axes);
        for (int i = 0; i < axes.length; i++) {
            newAxes[i] = axes[axesRet[i]];
        }

        return newAxes;
    }



    /**
     * Generate an array from 0 to length
     * and generate take a subset
     * @param length the length to generate to
     * @param from the begin of the interval to take
     * @param to the end of the interval to take
     * @return the generated array
     */
    public static int[] copyOfRangeFrom(int length, int from, int to) {
        return Arrays.copyOfRange(ArrayUtil.range(0, length), from, to);

    }

    //Credit: https://stackoverflow.com/questions/15533854/converting-byte-array-to-double-array

    /**
     *
     * @param doubleArray
     * @return
     */
    public static byte[] toByteArray(double[] doubleArray) {
        int times = Double.SIZE / Byte.SIZE;
        byte[] bytes = new byte[doubleArray.length * times];
        for (int i = 0; i < doubleArray.length; i++) {
            ByteBuffer.wrap(bytes, i * times, times).putDouble(doubleArray[i]);
        }
        return bytes;
    }

    /**
     * Note this byte array conversion is a simple cast and not a true
     * cast. Use {@link #toByteArraySimple(long[])} for a true cast.
     * @param longArray
     * @return
     */
    public static byte[] toByteArraySimple(long[] longArray) {
        byte[] bytes = new byte[longArray.length];
        for (int i = 0; i < longArray.length; i++) {
            bytes[i] = (byte) longArray[i];
        }
        return bytes;
    }
    /**
     *
     * @param longArray
     * @return
     */
    public static byte[] toByteArray(long[] longArray) {
        int times = Long.SIZE / Byte.SIZE;
        byte[] bytes = new byte[longArray.length * times];
        for (int i = 0; i < longArray.length; i++) {
            ByteBuffer.wrap(bytes, i * times, times).putLong(longArray[i]);
        }
        return bytes;
    }


    /**
     *
     * @param byteArray
     * @return
     */
    public static double[] toDoubleArraySimple(byte[] byteArray) {
        double[] doubles = new double[byteArray.length];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = (double) byteArray[i];
        }
        return doubles;
    }
    /**
     *
     * @param byteArray
     * @return
     */
    public static double[] toDoubleArray(byte[] byteArray) {
        int times = Double.SIZE / Byte.SIZE;
        double[] doubles = new double[byteArray.length / times];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = ByteBuffer.wrap(byteArray, i * times, times).getDouble();
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
        for (int i = 0; i < doubleArray.length; i++) {
            ByteBuffer.wrap(bytes, i * times, times).putFloat(doubleArray[i]);
        }
        return bytes;
    }

    public static long[] toLongArray(int[] intArray) {
        if(intArray == null)
            return null;
        long[] ret = new long[intArray.length];
        for (int i = 0; i < intArray.length; i++) {
            ret[i] = intArray[i];
        }
        return ret;
    }

    public static long[] toLongArray(float[] array) {
        val ret = new long[array.length];
        for (int i = 0; i < array.length; i++) {
            ret[i] = (long) array[i];
        }
        return ret;
    }

    /**
     *
     * @param byteArray
     * @return
     */
    public static float[] toFloatArray(byte[] byteArray) {
        int times = Float.SIZE / Byte.SIZE;
        float[] doubles = new float[byteArray.length / times];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = ByteBuffer.wrap(byteArray, i * times, times).getFloat();
        }
        return doubles;
    }


    /**
     *
     * @param byteArray
     * @return
     */
    public static float[] toFloatArraySimple(byte[] byteArray) {
        float[] doubles = new float[byteArray.length];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = byteArray[i];
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
        for (int i = 0; i < intArray.length; i++) {
            ByteBuffer.wrap(bytes, i * times, times).putInt(intArray[i]);
        }
        return bytes;
    }


    /**
     *
     * @param byteArray
     * @return
     */
    public static int[] toIntArraySimple(byte[] byteArray) {
        int[] ints = new int[byteArray.length];
        for (int i = 0; i < ints.length; i++) {
            ints[i] =   byteArray[i];
        }
        return ints;
    }

    /**
     *
     * @param byteArray
     * @return
     */
    public static int[] toIntArray(byte[] byteArray) {
        int times = Integer.SIZE / Byte.SIZE;
        int[] ints = new int[byteArray.length / times];
        for (int i = 0; i < ints.length; i++) {
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
        if (data == null)
            return null;

        if (index >= data.length)
            throw new IllegalArgumentException("Unable to remove index " + index + " was >= data.length");
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


    public static long[] removeIndex(long[] data, long index) {
        return removeIndex(data,(int) index);
    }

    public static long[] removeIndex(long[] data, int index) {
        if (data == null)
            return null;

        if (index >= data.length)
            throw new IllegalArgumentException("Unable to remove index " + index + " was >= data.length");
        if (data.length < 1)
            return data;
        if (index < 0)
            return data;

        int len = data.length;
        long[] result = new long[len - 1];
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
    public static int[] valueStartingAt(int valueStarting, int[] copy, int idxFrom, int idxAt, int length) {
        int[] ret = new int[length];
        Arrays.fill(ret, valueStarting);
        for (int i = 0; i < length; i++) {
            if (i + idxFrom >= copy.length || i + idxAt >= ret.length)
                break;
            ret[i + idxAt] = copy[i + idxFrom];
        }

        return ret;
    }

    public static BigInteger[] toBigInteger(byte[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) {
            ret[i] = BigInteger.valueOf(input[i]);
        }
        return ret;
    }
    public static BigInteger[] toBigInteger(short[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) {
            ret[i] = BigInteger.valueOf(input[i]);
        }
        return ret;
    }

    public static BigInteger[] toBigInteger(long[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) {
            ret[i] = BigInteger.valueOf(input[i]);
        }
        return ret;
    }
    public static BigInteger[] toBigInteger(boolean[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) {
            ret[i] = BigInteger.valueOf(BigInteger.valueOf(input[i] ? 1 : 0).longValue());
        }
        return ret;
    }


    public static BigInteger[] toBigInteger(float[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) {
            ret[i] = BigInteger.valueOf((long) input[i]);
        }
        return ret;
    }

    public static BigInteger[] toBigInteger(double[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) {
            ret[i] = BigInteger.valueOf((long) input[i]);
        }
        return ret;
    }

    public static BigInteger[] toBigInteger(int[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) {
            ret[i] = BigInteger.valueOf(input[i]);
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
        if (shape.length == 2 && (shape[0] == 1 || shape[1] == 1)) {
            int[] ret = new int[2];
            Arrays.fill(ret, startNum);
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
     * @param shape    the shape of a matrix:
     * @param startNum the start number for the strides
     * @return the strides for a matrix of n dimensions
     */
    public static long[] calcStridesFortran(long[] shape, int startNum) {
        if (shape.length == 2 && (shape[0] == 1 || shape[1] == 1)) {
            long[] ret = new long[2];
            Arrays.fill(ret, startNum);
            return ret;
        }

        int dimensions = shape.length;
        long[] stride = new long[dimensions];
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

    public static long[] calcStridesFortran(long[] shape) {
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
        if (shape.length == 2 && (shape[0] == 1 || shape[1] == 1)) {
            int[] ret = new int[2];
            Arrays.fill(ret, startValue);
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
     * Computes the standard packed array strides for a given shape.
     *
     * @param shape      the shape of a matrix:
     * @param startValue the startValue for the strides
     * @return the strides for a matrix of n dimensions
     */
    public static long[] calcStrides(long[] shape, int startValue) {
        if(shape == null)
            return null;
        if (shape.length == 2 && (shape[0] == 1 || shape[1] == 1)) {
            long[] ret = new long[2];
            Arrays.fill(ret, startValue);
            return ret;
        }


        int dimensions = shape.length;
        long[] stride = new long[dimensions];

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
    public static boolean isInverse(int[] first, int[] second) {
        int backWardCount = second.length - 1;
        for (int i = 0; i < first.length; i++) {
            if (first[i] != second[backWardCount--])
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
        if (ints.length != mult.length)
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
        Preconditions.checkArgument(ints.length == mult.length, "Ints and mult must be the same length");
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

    public static long[] calcStrides(long[] shape) {
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

    public static long[] reverseCopy(long[] e) {
        if (e.length < 1)
            return e;

        long[] copy = new long[e.length];
        for (int i = 0; i <= e.length / 2; i++) {
            long temp = e[i];
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
                Preconditions.checkState(d[i].length == firstLength);
            }
        }
    }


    /**
     * Multiply the given array
     * by the given scalar
     * @param arr the array to multily
     * @param mult the scalar to multiply by
     */
    public static void multiplyBy(int[] arr, int mult) {
        for (int i = 0; i < arr.length; i++)
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

    public static void reverse(long[] e) {
        for (int i = 0; i <= e.length / 2; i++) {
            long temp = e[i];
            e[i] = e[e.length - i - 1];
            e[e.length - i - 1] = temp;
        }
    }


    public static List<double[]> zerosMatrix(long... dimensions) {
        List<double[]> ret = new ArrayList<>();
        for (int i = 0; i < dimensions.length; i++) {
            ret.add(new double[(int) dimensions[i]]);
        }
        return ret;
    }

    public static List<double[]> zerosMatrix(int... dimensions) {
        List<double[]> ret = new ArrayList<>();
        for (int i = 0; i < dimensions.length; i++) {
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

    public static boolean[] flatten(boolean[][] arr) {
        if(arr.length == 0 || arr[0].length == 0)
            return new boolean[0];
        boolean[] ret = new boolean[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, ret, count, arr[i].length);
            count += arr[i].length;
        }
        return ret;
    }

    public static String[] flatten(String[][] arr) {
        if(arr.length == 0 || arr[0].length == 0)
            return new String[0];
        String[] ret = new String[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, ret, count, arr[i].length);
            count += arr[i].length;
        }
        return ret;
    }

    public static String[] flatten(String[][][] arr) {
        if(arr.length == 0 || arr[0].length == 0 || arr[0][0].length == 0)
            return new String[0];
        String[] ret = new String[arr.length * arr[0].length * arr[0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }
        return ret;
    }

    public static boolean[] flatten(boolean[][][] arr) {
        if(arr.length == 0 || arr[0].length == 0 || arr[0][0].length == 0)
            return new boolean[0];
        boolean[] ret = new boolean[arr.length * arr[0].length * arr[0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }
        return ret;
    }

    public static float[] flatten(float[][] arr) {
        if(arr.length == 0 || arr[0].length == 0)
            return new float[0];
        float[] ret = new float[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, ret, count, arr[i].length);
            count += arr[i].length;
        }
        return ret;
    }


    public static float[] flatten(float[][][] arr) {
        if (arr.length == 0 || arr[0].length == 0 || arr[0][0].length == 0)
            return new float[0];
        float[] ret = new float[arr.length * arr[0].length * arr[0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }

        return ret;
    }

    public static double[] flatten(double[][][] arr) {
        if(arr.length == 0 || arr[0].length == 0 || arr[0][0].length == 0)
            return new double[0];
        double[] ret = new double[arr.length * arr[0].length * arr[0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }
        return ret;
    }

    public static int[] flatten(int[][][] arr) {
        if(arr.length == 0 || arr[0].length == 0 || arr[0][0].length == 0)
            return new int[0];
        int[] ret = new int[arr.length * arr[0].length * arr[0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }
        return ret;
    }

    public static short[] flatten(short[][][] arr) {
        if(arr.length == 0 || arr[0].length == 0 || arr[0][0].length == 0)
            return new short[0];
        val ret = new short[arr.length * arr[0].length * arr[0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }
        return ret;
    }

    public static byte[] flatten(byte[][][] arr) {
        if(arr.length == 0 || arr[0].length == 0 || arr[0][0].length == 0)
            return new byte[0];
        val ret = new byte[arr.length * arr[0].length * arr[0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }
        return ret;
    }

    public static long[] flatten(long[][][][] arr) {
        val ret = new long[arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    System.arraycopy(arr[i][j][k], 0, ret, count, arr[0][0][0].length);
                    count += arr[0][0][0].length;
                }
            }
        }

        return ret;
    }

    public static short[] flatten(short[][][][] arr) {
        val ret = new short[arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    System.arraycopy(arr[i][j][k], 0, ret, count, arr[0][0][0].length);
                    count += arr[0][0][0].length;
                }
            }
        }

        return ret;
    }

    public static byte[] flatten(byte[][][][] arr) {
        val ret = new byte[arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    System.arraycopy(arr[i][j][k], 0, ret, count, arr[0][0][0].length);
                    count += arr[0][0][0].length;
                }
            }
        }

        return ret;
    }

    public static boolean[] flatten(boolean[][][][] arr) {
        val ret = new boolean[arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    System.arraycopy(arr[i][j][k], 0, ret, count, arr[0][0][0].length);
                    count += arr[0][0][0].length;
                }
            }
        }

        return ret;
    }

    public static float[] flatten(float[][][][] arr) {
        float[] ret = new float[arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    System.arraycopy(arr[i][j][k], 0, ret, count, arr[0][0][0].length);
                    count += arr[0][0][0].length;
                }
            }
        }

        return ret;
    }

    public static double[] flatten(double[][][][] arr) {
        double[] ret = new double[arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    System.arraycopy(arr[i][j][k], 0, ret, count, arr[0][0][0].length);
                    count += arr[0][0][0].length;
                }
            }
        }

        return ret;
    }

    public static int[] flatten(int[][][][] arr) {
        int[] ret = new int[arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    System.arraycopy(arr[i][j][k], 0, ret, count, arr[0][0][0].length);
                    count += arr[0][0][0].length;
                }
            }
        }

        return ret;
    }


    public static int[] flatten(int[][] arr) {
        if(arr.length == 0 || arr[0].length == 0 )
            return new int[0];
        int[] ret = new int[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, ret, count, arr[i].length);
            count += arr[i].length;
        }
        return ret;
    }

    public static short[] flatten(short[][] arr) {
        if(arr.length == 0 || arr[0].length == 0 )
            return new short[0];
        val ret = new short[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, ret, count, arr[i].length);
            count += arr[i].length;
        }
        return ret;
    }

    public static byte[] flatten(byte[][] arr) {
        if(arr.length == 0 || arr[0].length == 0 )
            return new byte[0];
        val ret = new byte[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, ret, count, arr[i].length);
            count += arr[i].length;
        }
        return ret;
    }

    public static long[] flatten(long[][] arr) {
        if(arr.length == 0 || arr[0].length == 0 )
            return new long[0];
        long[] ret = new long[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, ret, count, arr[i].length);
            count += arr[i].length;
        }
        return ret;
    }

    public static long[] flatten(long[][][] arr) {
        if(arr.length == 0 || arr[0].length == 0 || arr[0][0].length == 0)
            return new long[0];
        long[] ret = new long[arr.length * arr[0].length * arr[0][0].length];

        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.arraycopy(arr[i][j], 0, ret, count, arr[0][0].length);
                count += arr[0][0].length;
            }
        }
        return ret;
    }


    /**
     * Convert a 2darray in to a flat
     * array (row wise)
     * @param arr the array to flatten
     * @return a flattened representation of the array
     */
    public static double[] flatten(double[][] arr) {
        if(arr.length == 0 || arr[0].length == 0 )
            return new double[0];
        double[] ret = new double[arr.length * arr[0].length];
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, ret, count, arr[i].length);
            count += arr[i].length;
        }
        return ret;
    }

    /**
     * Convert a 2darray in to a flat
     * array (row wise)
     * @param arr the array to flatten
     * @return a flattened representation of the array
     */
    public static double[] flattenF(double[][] arr) {
        double[] ret = new double[arr.length * arr[0].length];
        int count = 0;
        for (int j = 0; j < arr[0].length; j++)
            for (int i = 0; i < arr.length; i++)
                ret[count++] = arr[i][j];
        return ret;
    }

    public static float[] flattenF(float[][] arr) {
        float[] ret = new float[arr.length * arr[0].length];
        int count = 0;
        for (int j = 0; j < arr[0].length; j++)
            for (int i = 0; i < arr.length; i++)
                ret[count++] = arr[i][j];
        return ret;
    }

    public static int[] flattenF(int[][] arr) {
        int[] ret = new int[arr.length * arr[0].length];
        int count = 0;
        for (int j = 0; j < arr[0].length; j++)
            for (int i = 0; i < arr.length; i++)
                ret[count++] = arr[i][j];
        return ret;
    }


    public static long[] flattenF(long[][] arr) {
        long[] ret = new long[arr.length * arr[0].length];
        int count = 0;
        for (int j = 0; j < arr[0].length; j++)
            for (int i = 0; i < arr.length; i++)
                ret[count++] = arr[i][j];
        return ret;
    }

    public static int[][] reshapeInt(int[] in, int rows, int cols){
        int[][] out = new int[rows][cols];
        int x = 0;
        for(int i=0; i<rows; i++ ){
            for( int j=0; j<cols; j++ ){
                out[i][j] = in[x++];
            }
        }
        return out;
    }

    public static int[][][] reshapeInt(int[] in, int d0, int d1, int d2){
        int[][][] out = new int[d0][d1][d2];
        int x = 0;
        for(int i=0; i<d0; i++ ){
            for( int j=0; j<d1; j++ ){
                for( int k=0; k<d2; k++ ) {
                    out[i][j][k] = in[x++];
                }
            }
        }
        return out;
    }

    public static double[][] reshapeDouble(double[] in, int rows, int cols){
        double[][] out = new double[rows][cols];
        int x = 0;
        for(int i=0; i<rows; i++ ){
            for( int j=0; j<cols; j++ ){
                out[i][j] = in[x++];
            }
        }
        return out;
    }

    public static double[][][] reshapeDouble(double[] in, int d0, int d1, int d2){
        double[][][] out = new double[d0][d1][d2];
        int x = 0;
        for(int i=0; i<d0; i++ ){
            for( int j=0; j<d1; j++ ){
                for( int k=0; k<d2; k++ ) {
                    out[i][j][k] = in[x++];
                }
            }
        }
        return out;
    }

    public static long[][] reshapeLong(long[] in, int rows, int cols){
        long[][] out = new long[rows][cols];
        int x = 0;
        for(int i=0; i<rows; i++ ){
            for( int j=0; j<cols; j++ ){
                out[i][j] = in[x++];
            }
        }
        return out;
    }

    public static long[][][] reshapeLong(long[] in, int d0, int d1, int d2){
        long[][][] out = new long[d0][d1][d2];
        int x = 0;
        for(int i=0; i<d0; i++ ){
            for( int j=0; j<d1; j++ ){
                for( int k=0; k<d2; k++ ) {
                    out[i][j][k] = in[x++];
                }
            }
        }
        return out;
    }

    public static boolean[][] reshapeBoolean(boolean[] in, int rows, int cols){
        boolean[][] out = new boolean[rows][cols];
        int x = 0;
        for(int i=0; i<rows; i++ ){
            for( int j=0; j<cols; j++ ){
                out[i][j] = in[x++];
            }
        }
        return out;
    }

    public static boolean[][][] reshapeBoolean(boolean[] in, int d0, int d1, int d2){
        boolean[][][] out = new boolean[d0][d1][d2];
        int x = 0;
        for(int i = 0; i < d0; i++) {
            for( int j = 0; j < d1; j++) {
                for( int k = 0; k < d2; k++) {
                    out[i][j][k] = in[x++];
                }
            }
        }
        return out;
    }

    public static <T> T[][] reshapeObject(T[] in, int rows, int cols){
        Object[][] out = new Object[rows][cols];
        int x = 0;
        for(int i = 0; i < rows; i++) {
            for( int j = 0; j<cols; j++) {
                out[i][j] = in[x++];
            }
        }
        return (T[][])out;
    }

    public static <T> T[][][] reshapeObject(T[] in, int d0, int d1, int d2){
        Object[][][] out = new Object[d0][d1][d2];
        int x = 0;
        for(int i = 0; i < d0; i++) {
            for( int j = 0; j < d1; j++) {
                for( int k=0; k < d2; k++) {
                    out[i][j][k] = in[x++];
                }
            }
        }
        return (T[][][])out;
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

    /**
     * Combines a apply of long arrays in to one flat long array
     *
     * @param ints the int arrays to combineDouble
     * @return one combined int array
     */
    public static long[] combine(long[]... ints) {
        int length = 0;
        for (int i = 0; i < ints.length; i++)
            length += ints[i].length;
        long[] ret = new long[length];
        int count = 0;
        for (long[] i : ints) {
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

    public static double[] toDouble(boolean[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = data[i] ? 1.0 : 0.0;
        return ret;
    }

    public static double[] toDouble(byte[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = data[i];
        return ret;
    }
    public static double[] toDouble(int[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = data[i];
        return ret;
    }

    public static double[] toDouble(long[] data) {
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
    public static double[] flattenDoubleArray(Object doubleArray) {
        if (doubleArray instanceof double[])
            return (double[]) doubleArray;

        LinkedList<Object> stack = new LinkedList<>();
        stack.push(doubleArray);

        int[] shape = arrayShape(doubleArray);
        int length = ArrayUtil.prod(shape);
        double[] flat = new double[length];
        int count = 0;

        while (!stack.isEmpty()) {
            Object current = stack.pop();
            if (current instanceof double[]) {
                double[] arr = (double[]) current;
                for (int i = 0; i < arr.length; i++)
                    flat[count++] = arr[i];
            } else if (current instanceof Object[]) {
                Object[] o = (Object[]) current;
                for (int i = o.length - 1; i >= 0; i--)
                    stack.push(o[i]);
            } else
                throw new IllegalArgumentException("Base array is not double[]");
        }

        if (count != flat.length)
            throw new IllegalArgumentException("Fewer elements than expected. Array is ragged?");
        return flat;
    }

    /** Convert an arbitrary-dimensional rectangular float array to flat vector.<br>
     * Can pass float[], float[][], float[][][], etc.
     */
    public static float[] flattenFloatArray(Object floatArray) {
        if (floatArray instanceof float[])
            return (float[]) floatArray;

        LinkedList<Object> stack = new LinkedList<>();
        stack.push(floatArray);

        int[] shape = arrayShape(floatArray);
        int length = ArrayUtil.prod(shape);
        float[] flat = new float[length];
        int count = 0;

        while (!stack.isEmpty()) {
            Object current = stack.pop();
            if (current instanceof float[]) {
                float[] arr = (float[]) current;
                for (int i = 0; i < arr.length; i++)
                    flat[count++] = arr[i];
            } else if (current instanceof Object[]) {
                Object[] o = (Object[]) current;
                for (int i = o.length - 1; i >= 0; i--)
                    stack.push(o[i]);
            } else
                throw new IllegalArgumentException("Base array is not float[]");
        }

        if (count != flat.length)
            throw new IllegalArgumentException("Fewer elements than expected. Array is ragged?");
        return flat;
    }

    /** Calculate the shape of an arbitrary multi-dimensional array. Assumes:<br>
     * (a) array is rectangular (not ragged) and first elements (i.e., array[0][0][0]...) are non-null <br>
     * (b) First elements have > 0 length. So array[0].length > 0, array[0][0].length > 0, etc.<br>
     * Can pass any Java array opType: double[], Object[][][], float[][], etc.<br>
     * Length of returned array is number of dimensions; returned[i] is size of ith dimension.
     */
    public static int[] arrayShape(Object array) {
        return arrayShape(array, false);
    }

    /** Calculate the shape of an arbitrary multi-dimensional array.<br>
     * Note that the method assumes the array is rectangular (not ragged) and first elements (i.e., array[0][0][0]...) are non-null <br>
     * Note also that if allowSize0Dims is true, any elements are length 0, all subsequent dimensions will be reported as 0.
     * i.e., a double[3][0][2] would be reported as shape [3,0,0]. If allowSize0Dims is false, an exception will be thrown for this case instead.
     * Can pass any Java array opType: double[], Object[][][], float[][], etc.<br>
     * Length of returned array is number of dimensions; returned[i] is size of ith dimension.
     */
    public static int[] arrayShape(Object array, boolean allowSize0Dims) {
        int nDimensions = 0;
        Class<?> c = array.getClass().getComponentType();
        while (c != null) {
            nDimensions++;
            c = c.getComponentType();
        }

        int[] shape = new int[nDimensions];
        Object current = array;
        for (int i = 0; i < shape.length - 1; i++) {
            shape[i] = ((Object[]) current).length;
            if(shape[i] == 0){
                if(allowSize0Dims){
                    return shape;
                }
                throw new IllegalStateException("Cannot calculate array shape: Array has size 0 for dimension " + i );
            }
            current = ((Object[]) current)[0];
        }

        if (current instanceof Object[]) {
            shape[shape.length - 1] = ((Object[]) current).length;
        } else if (current instanceof double[]) {
            shape[shape.length - 1] = ((double[]) current).length;
        } else if (current instanceof float[]) {
            shape[shape.length - 1] = ((float[]) current).length;
        } else if (current instanceof long[]) {
            shape[shape.length - 1] = ((long[]) current).length;
        } else if (current instanceof int[]) {
            shape[shape.length - 1] = ((int[]) current).length;
        } else if (current instanceof byte[]) {
            shape[shape.length - 1] = ((byte[]) current).length;
        } else if (current instanceof char[]) {
            shape[shape.length - 1] = ((char[]) current).length;
        } else if (current instanceof boolean[]) {
            shape[shape.length - 1] = ((boolean[]) current).length;
        } else if (current instanceof short[]) {
            shape[shape.length - 1] = ((short[]) current).length;
        } else
            throw new IllegalStateException("Unknown array type"); //Should never happen
        return shape;
    }


    /** Returns the maximum value in the array */
    public static int max(int[] in) {
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < in.length; i++)
            if (in[i] > max)
                max = in[i];
        return max;
    }

    /** Returns the minimum value in the array */
    public static int min(int[] in) {
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < in.length; i++)
            if (in[i] < min)
                min = in[i];
        return min;
    }

    /** Returns the index of the maximum value in the array.
     * If two entries have same maximum value, index of the first one is returned. */
    public static int argMax(int[] in) {
        int maxIdx = 0;
        for (int i = 1; i < in.length; i++)
            if (in[i] > in[maxIdx])
                maxIdx = i;
        return maxIdx;
    }

    /** Returns the index of the minimum value in the array.
     * If two entries have same minimum value, index of the first one is returned. */
    public static int argMin(int[] in) {
        int minIdx = 0;
        for (int i = 1; i < in.length; i++)
            if (in[i] < in[minIdx])
                minIdx = i;
        return minIdx;
    }

    /** Returns the index of the maximum value in the array.
     * If two entries have same maximum value, index of the first one is returned. */
    public static int argMax(long[] in) {
        int maxIdx = 0;
        for (int i = 1; i < in.length; i++)
            if (in[i] > in[maxIdx])
                maxIdx = i;
        return maxIdx;
    }

    /** Returns the index of the minimum value in the array.
     * If two entries have same minimum value, index of the first one is returned. */
    public static int argMin(long[] in) {
        int minIdx = 0;
        for (int i = 1; i < in.length; i++)
            if (in[i] < in[minIdx])
                minIdx = i;
        return minIdx;
    }

    /**
     *
     * @return
     */
    public static int[] buildHalfVector(Random rng, int length) {
        int[] result = new int[length];
        List<Integer> indexes = new ArrayList<>();

        // we add indexes from second half only
        for (int i = result.length - 1; i >= result.length / 2; i--) {
            indexes.add(i);
        }

        Collections.shuffle(indexes, rng);

        for (int i = 0; i < result.length; i++) {
            if (i < result.length / 2) {
                result[i] = indexes.get(0);
                indexes.remove(0);
            } else
                result[i] = -1;
        }

        return result;
    }

    public static int[] buildInterleavedVector(Random rng, int length) {
        int[] result = new int[length];

        List<Integer> indexes = new ArrayList<>();
        List<Integer> odds = new ArrayList<>();

        // we add odd indexes only to list
        for (int i = 1; i < result.length; i += 2) {
            indexes.add(i);
            odds.add(i - 1);
        }

        Collections.shuffle(indexes, rng);

        // now all even elements will be interleaved with odd elements
        for (int i = 0; i < result.length; i++) {
            if (i % 2 == 0 && !indexes.isEmpty()) {
                int idx = indexes.get(0);
                indexes.remove(0);
                result[i] = idx;
            } else
                result[i] = -1;
        }

        // for odd tad numbers, we add special random clause for last element
        if (length % 2 != 0) {
            int rndClause = odds.get(rng.nextInt(odds.size()));
            int tmp = result[rndClause];
            result[rndClause] = result[result.length - 1];
            result[result.length - 1] = tmp;
        }


        return result;
    }

    public static long[] buildInterleavedVector(Random rng, long length) {
        if (length > Integer.MAX_VALUE) {
            throw new RuntimeException("Integer overflow");
        }
        val result = new long[(int) length];

        List<Integer> indexes = new ArrayList<>();
        List<Integer> odds = new ArrayList<>();

        // we add odd indexes only to list
        for (int i = 1; i < result.length; i += 2) {
            indexes.add(i);
            odds.add(i - 1);
        }

        Collections.shuffle(indexes, rng);

        // now all even elements will be interleaved with odd elements
        for (int i = 0; i < result.length; i++) {
            if (i % 2 == 0 && !indexes.isEmpty()) {
                int idx = indexes.get(0);
                indexes.remove(0);
                result[i] = idx;
            } else
                result[i] = -1;
        }

        // for odd tad numbers, we add special random clause for last element
        if (length % 2 != 0) {
            int rndClause = odds.get(rng.nextInt(odds.size()));
            long tmp = result[rndClause];
            result[rndClause] = result[result.length - 1];
            result[result.length - 1] = tmp;
        }


        return result;
    }

    protected static <T extends Object> void swap(List<T> objects, int idxA, int idxB) {
        T tmpA = objects.get(idxA);
        T tmpB = objects.get(idxB);
        objects.set(idxA, tmpB);
        objects.set(idxB, tmpA);
    }

    public static <T extends Object> void shuffleWithMap(List<T> objects, int[] map) {
        for (int i = 0; i < map.length; i++) {
            if (map[i] >= 0) {
                swap(objects, i, map[i]);
            }
        }
    }

    public static int argMinOfMax(int[] first, int[] second) {
        int minIdx = 0;
        int maxAtMinIdx = Math.max(first[0], second[0]);
        for (int i = 1; i < first.length; i++) {
            int maxAtIndex = Math.max(first[i], second[i]);
            if (maxAtMinIdx > maxAtIndex) {
                maxAtMinIdx = maxAtIndex;
                minIdx = i;
            }
        }
        return minIdx;
    }

    public static long argMinOfMax(long[] first, long[] second) {
        long minIdx = 0;
        long maxAtMinIdx = Math.max(first[0], second[0]);
        for (int i = 1; i < first.length; i++) {
            long maxAtIndex = Math.max(first[i], second[i]);
            if (maxAtMinIdx > maxAtIndex) {
                maxAtMinIdx = maxAtIndex;
                minIdx = i;
            }
        }
        return minIdx;
    }

    public static int argMinOfMax(int[]... arrays) {
        int minIdx = 0;
        int maxAtMinIdx = Integer.MAX_VALUE;

        for (int i = 0; i < arrays[0].length; i++) {
            int maxAtIndex = Integer.MIN_VALUE;
            for (int j = 0; j < arrays.length; j++) {
                maxAtIndex = Math.max(maxAtIndex, arrays[j][i]);
            }

            if (maxAtMinIdx > maxAtIndex) {
                maxAtMinIdx = maxAtIndex;
                minIdx = i;
            }
        }
        return minIdx;
    }

    public static long argMinOfMax(long[]... arrays) {
        int minIdx = 0;
        long maxAtMinIdx = Long.MAX_VALUE;

        for (int i = 0; i < arrays[0].length; i++) {
            long maxAtIndex = Long.MIN_VALUE;
            for (int j = 0; j < arrays.length; j++) {
                maxAtIndex = Math.max(maxAtIndex, arrays[j][i]);
            }

            if (maxAtMinIdx > maxAtIndex) {
                maxAtMinIdx = maxAtIndex;
                minIdx = i;
            }
        }
        return minIdx;
    }

    public static int argMinOfSum(int[] first, int[] second) {
        int minIdx = 0;
        int sumAtMinIdx = first[0] + second[0];
        for (int i = 1; i < first.length; i++) {
            int sumAtIndex = first[i] + second[i];
            if (sumAtMinIdx > sumAtIndex) {
                sumAtMinIdx = sumAtIndex;
                minIdx = i;
            }
        }
        return minIdx;
    }

    public static <K, V extends Comparable<? super V>> Map<K, V> sortMapByValue(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new LinkedList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
            @Override
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        Map<K, V> result = new LinkedHashMap<>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }


    public static <T> T getRandomElement(List<T> list) {
        if (list.isEmpty())
            return null;

        return list.get(RandomUtils.nextInt(0, list.size()));
    }

    /**
     * Convert an int
     * @param bool
     * @return
     */
    public static int fromBoolean(boolean bool) {
        return bool ? 1 : 0;
    }

    public static long[] toPrimitives(Long[] array) {
        val res = new long[array.length];
        for (int e = 0; e < array.length; e++)
            res[e] = array[e];

        return res;
    }

    public static int[] toPrimitives(Integer[] array) {
        val res = new int[array.length];
        for (int e = 0; e < array.length; e++)
            res[e] = array[e];

        return res;
    }

    public static short[] toPrimitives(Short[] array) {
        val res = new short[array.length];
        for (int e = 0; e < array.length; e++)
            res[e] = array[e];

        return res;
    }

    public static byte[] toPrimitives(Byte[] array) {
        val res = new byte[array.length];
        for (int e = 0; e < array.length; e++)
            res[e] = array[e];

        return res;
    }

    public static float[] toPrimitives(Float[] array) {
        val res = new float[array.length];
        for (int e = 0; e < array.length; e++)
            res[e] = array[e];

        return res;
    }

    public static double[] toPrimitives(Double[] array) {
        val res = new double[array.length];
        for (int e = 0; e < array.length; e++)
            res[e] = array[e];

        return res;
    }

    public static boolean[] toPrimitives(Boolean[] array) {
        val res = new boolean[array.length];
        for (int e = 0; e < array.length; e++)
            res[e] = array[e];

        return res;
    }

    public static long[][] toPrimitives(Long[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new long[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                res[i][j] = array[i][j];

        return res;
    }

    public static int[][] toPrimitives(Integer[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new int[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                res[i][j] = array[i][j];

        return res;
    }

    public static short[][] toPrimitives(Short[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new short[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                res[i][j] = array[i][j];

        return res;
    }

    public static byte[][] toPrimitives(Byte[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new byte[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                res[i][j] = array[i][j];

        return res;
    }

    public static double[][] toPrimitives(Double[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new double[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                res[i][j] = array[i][j];

        return res;
    }

    public static float[][] toPrimitives(Float[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new float[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                res[i][j] = array[i][j];

        return res;
    }

    public static boolean [][] toPrimitives(Boolean[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new boolean[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                res[i][j] = array[i][j];

        return res;
    }

    public static long[][][] toPrimitives(Long[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new long[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    res[i][j][k] = array[i][j][k];

        return res;
    }

    public static int[][][] toPrimitives(Integer[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new int[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    res[i][j][k] = array[i][j][k];

        return res;
    }

    public static short[][][] toPrimitives(Short[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new short[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    res[i][j][k] = array[i][j][k];

        return res;
    }

    public static byte[][][] toPrimitives(Byte[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new byte[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    res[i][j][k] = array[i][j][k];

        return res;
    }

    public static double[][][] toPrimitives(Double[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new double[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    res[i][j][k] = array[i][j][k];

        return res;
    }

    public static float[][][] toPrimitives(Float[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new float[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    res[i][j][k] = array[i][j][k];

        return res;
    }

    public static boolean[][][] toPrimitives(Boolean[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new boolean[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    res[i][j][k] = array[i][j][k];

        return res;
    }

    public static long[][][][] toPrimitives(Long[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new long[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        res[i][j][k][l] = array[i][j][k][l];

        return res;
    }

    public static int[][][][] toPrimitives(Integer[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new int[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        res[i][j][k][l] = array[i][j][k][l];

        return res;
    }

    public static short[][][][] toPrimitives(Short[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new short[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        res[i][j][k][l] = array[i][j][k][l];

        return res;
    }

    public static byte[][][][] toPrimitives(Byte[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new byte[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        res[i][j][k][l] = array[i][j][k][l];

        return res;
    }

    public static double[][][][] toPrimitives(Double[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new double[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        res[i][j][k][l] = array[i][j][k][l];

        return res;
    }

    public static float[][][][] toPrimitives(Float[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new float[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        res[i][j][k][l] = array[i][j][k][l];

        return res;
    }

    public static boolean[][][][] toPrimitives(Boolean[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new boolean[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; j < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++)
                        res[i][j][k][l] = array[i][j][k][l];

        return res;
    }


    /**
     * Assert that the specified array is not ragged (i.e., is rectangular).<br>
     * Can be used to check Object arrays with any number of dimensions (up to rank 4), or primitive arrays with rank 2 or higher<br>
     * An IllegalStateException is thrown if the array is ragged
     *
     * @param array Array to check
     */
    public static <T> void assertNotRagged(T[] array){
        Class<?> c = array.getClass().getComponentType();
        int[] arrayShape = ArrayUtil.arrayShape(array, true);
        int rank = arrayShape.length;

        if(rank == 1){
            //Rank 1 cannot be ragged
            return;
        }

        if(rank >= 2){
            for( int i = 1; i < arrayShape[0]; i++) {
                Object subArray = array[i];
                int len = arrayLength(subArray);
                Preconditions.checkState(arrayShape[1] == len, "Ragged array detected: array[0].length=%s does not match array[%s].length=%s", arrayShape[1], i, len);
            }
            if(rank == 2)
                return;
        }
        if(rank >= 3){

            for( int i = 0; i < arrayShape[0]; i++) {
                for( int j = 0; j < arrayShape[1]; j++) {
                    Object subArray = ((Object[][])array)[i][j];
                    int len = arrayLength(subArray);
                    Preconditions.checkState(arrayShape[2] == len, "Ragged array detected: array[0][0].length=%s does not match array[%s][%s].length=%s", arrayShape[2], i, j, len);
                }
            }

            if(rank == 3)
                return;
        }
        if(rank >= 4){
            for( int i = 0; i < arrayShape[0]; i++) {
                for( int j = 0; j<arrayShape[1]; j++) {
                    for( int k = 0; k < arrayShape[2]; k++) {
                        Object subArray = ((Object[][][])array)[i][j][k];
                        int len = arrayLength(subArray);
                        Preconditions.checkState(arrayShape[3] == len, "Ragged array detected: array[0][0][0].length=%s does not match array[%s][%s][%s].length=%s",
                                arrayShape[3], i, j, k, len);
                    }
                }
            }
        }
    }

    /**
     * Calculate the length of the object or primitive array. If
     * @param current
     * @return
     */
    public static int arrayLength(Object current){
        if (current instanceof Object[]) {
            return ((Object[]) current).length;
        } else if (current instanceof double[]) {
            return ((double[]) current).length;
        } else if (current instanceof float[]) {
            return ((float[]) current).length;
        } else if (current instanceof long[]) {
            return ((long[]) current).length;
        } else if (current instanceof int[]) {
            return ((int[]) current).length;
        } else if (current instanceof byte[]) {
            return ((byte[]) current).length;
        } else if (current instanceof char[]) {
            return ((char[]) current).length;
        } else if (current instanceof boolean[]) {
            return ((boolean[]) current).length;
        } else if (current instanceof short[]) {
            return ((short[]) current).length;
        } else
            throw new IllegalStateException("Unknown array type (or not an array): " + current.getClass()); //Should never happen
    }

    /**
     * Compute the inverse permutation indices for a permutation operation<br>
     * Example: if input is [2, 0, 1] then output is [1, 2, 0]<br>
     * The idea is that x.permute(input).permute(invertPermutation(input)) == x
     *
     * @param input 1D indices for permutation
     * @return 1D inverted permutation
     */
    public static int[] invertPermutation(int... input){
        int[] target = new int[input.length];

        for(int i = 0 ; i < input.length ; i++){
            target[input[i]] = i;
        }

        return target;
    }

    /**
     * @see #invertPermutation(int...)
     *
     * @param input 1D indices for permutation
     * @return 1D inverted permutation
     */
    public static long[] invertPermutation(long... input){
        long[] target = new long[input.length];

        for(int i = 0 ; i < input.length ; i++){
            target[(int) input[i]] = i;
        }

        return target;
    }

    /**
     * Is this shape an empty shape?
     * Shape is considered to be an empty shape if it contains any zeros.
     * Note: a length 0 shape is NOT considered empty (it's rank 0 scalar)
     * @param shape Shape to check
     * @return True if shape contains zeros
     */
    public static boolean isEmptyShape(long[] shape) {
        for( long l : shape){
            if(l == 0)
                return true;
        }
        return false;
    }

    /**
     * Is this shape an empty shape?
     * Shape is considered to be an empty shape if it contains any zeros.
     * Note: a length 0 shape is NOT considered empty (it's rank 0 scalar)
     * @param shape Shape to check
     * @return True if shape contains zeros
     */
    public static boolean isEmptyShape(int[] shape) {
        for( int i : shape){
            if(i == 0)
                return true;
        }
        return false;
    }

    public static <T> T[] filterNull(T... in) {
        int count = 0;
        for( int i=0; i<in.length; i++ ) {
            if (in[i] != null) count++;
        }
        T[] out = (T[]) Array.newInstance(in.getClass().getComponentType(), count);
        int j=0;
        for( int i=0; i<in.length; i++) {
            if(in[i] != null) {
                out[j++] = in[i];
            }
        }
        return out;
    }

    public static int indexOf(String[] outNames, String varName) {
        int ret = -1;
        for(int i = 0; i < outNames.length; i++) {
            if(outNames[i].equals(varName)) {
                ret = i;
                break;
            }
        }
        return ret;
    }

    public static int[] toInt(short[] v) {
        int[] ret = new int[v.length];
        for(int i = 0; i < v.length; i++) {
            ret[i] = v[i];
        }

        return ret;
    }

    public static int[] toInt(byte[] v) {
        int[] ret = new int[v.length];
        for(int i = 0; i < v.length; i++) {
            ret[i] = v[i];
        }

        return ret;
    }

    public static int[] toInt(char[] v) {
        int[] ret = new int[v.length];
        for(int i = 0; i < v.length; i++) {
            ret[i] = v[i];
        }

        return ret;
    }

    public static double[] toDouble(float[] v) {
        double[] ret = new double[v.length];
        for(int i = 0; i < v.length; i++) {
            ret[i] = v[i];
        }

        return ret;
    }
}
