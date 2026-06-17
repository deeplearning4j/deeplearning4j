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

import lombok.val;
import org.nd4j.shade.guava.primitives.Ints;

import java.math.BigInteger;
import java.nio.ByteBuffer;

/**
 * Type-conversion utilities for primitive arrays.
 *
 * <p>Handles conversions across the full matrix of primitive types:
 * boolean, byte, short, int, long, float, double, half (FP16), bfloat16,
 * as well as boxed-to-primitive (toPrimitives) and byte-buffer
 * encode/decode variants.
 *
 * <p>Methods here were extracted from {@link ArrayUtil} to improve cohesion.
 * {@link ArrayUtil} retains one-line delegates annotated {@code @Deprecated}
 * for backward compatibility.
 */
public final class ArrayTypeConverters {

    private ArrayTypeConverters() {}

    // -------------------------------------------------------------------------
    // toBoolean*
    // -------------------------------------------------------------------------

    public static boolean[] toBooleanArray(byte[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] != 0;
        return output;
    }

    public static boolean[] toBooleanArray(short[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] != 0;
        return output;
    }

    public static boolean[] toBooleanArray(int[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] != 0;
        return output;
    }

    public static boolean[] toBooleanArray(long[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] != 0L;
        return output;
    }

    public static boolean[] toBooleanArray(float[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] != 0.0f;
        return output;
    }

    public static boolean[] toBooleanArray(double[] input) {
        boolean[] output = new boolean[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] != 0.0;
        return output;
    }

    /** Converts a float array to boolean: nonzero → true. */
    public static boolean[] fromFloat(float[] elements) {
        boolean[] ret = new boolean[elements.length];
        for (int i = 0; i < elements.length; i++) ret[i] = elements[i] != 0.0f;
        return ret;
    }

    /** Single-value boolean from int (0 → false, nonzero → true). */
    public static int fromBoolean(boolean bool) {
        return bool ? 1 : 0;
    }

    // -------------------------------------------------------------------------
    // toInt* / toInts
    // -------------------------------------------------------------------------

    public static int[] toIntArray(short[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static int[] toIntArray(boolean[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] ? 1 : 0;
        return output;
    }

    public static int[] toIntArray(char[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static int[] toIntArray(int[] input) {
        int[] output = new int[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    public static int[] toIntArray(long[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (int) input[i];
        return output;
    }

    public static int[] toIntArray(float[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (int) input[i];
        return output;
    }

    public static int[] toIntArray(double[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (int) input[i];
        return output;
    }

    /** Decode a raw byte[] (4 bytes per int, big-endian via ByteBuffer) → int[]. */
    public static int[] toIntArray(byte[] byteArray) {
        int times = Integer.SIZE / Byte.SIZE;
        int[] ints = new int[byteArray.length / times];
        for (int i = 0; i < ints.length; i++)
            ints[i] = ByteBuffer.wrap(byteArray, i * times, times).getInt();
        return ints;
    }

    /** Simple element-wise byte → int cast (no byte-buffer decoding). */
    public static int[] toIntArraySimple(byte[] byteArray) {
        int[] ints = new int[byteArray.length];
        for (int i = 0; i < ints.length; i++) ints[i] = byteArray[i];
        return ints;
    }

    public static int[] toInts(boolean[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i] ? 1 : 0;
        return ret;
    }

    public static int[] toInts(byte[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i];
        return ret;
    }

    public static int[] toInts(short[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i];
        return ret;
    }

    public static int[] toInts(float[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = (int) data[i];
        return ret;
    }

    public static int[] toInts(double[] data) {
        int[] ret = new int[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = (int) data[i];
        return ret;
    }

    public static int[] toInts(long[] array) {
        int[] retVal = new int[array.length];
        for (int i = 0; i < array.length; i++) retVal[i] = (int) array[i];
        return retVal;
    }

    public static int[] toInt(short[] v) {
        int[] ret = new int[v.length];
        for (int i = 0; i < v.length; i++) ret[i] = v[i];
        return ret;
    }

    public static int[] toInt(byte[] v) {
        int[] ret = new int[v.length];
        for (int i = 0; i < v.length; i++) ret[i] = v[i];
        return ret;
    }

    public static int[] toInt(char[] v) {
        int[] ret = new int[v.length];
        for (int i = 0; i < v.length; i++) ret[i] = v[i];
        return ret;
    }

    // -------------------------------------------------------------------------
    // toDouble* / toDoubles
    // -------------------------------------------------------------------------

    public static double[] toDoubleArray(short[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static double[] toDoubleArray(char[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static double[] toDoubleArray(boolean[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] ? 1.0 : 0.0;
        return output;
    }

    public static double[] toDoubleArray(int[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static double[] toDoubleArray(long[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static double[] toDoubleArray(float[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static double[] toDoubleArray(double[] input) {
        double[] output = new double[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    /** Decode a raw byte[] (8 bytes per double, big-endian via ByteBuffer) → double[]. */
    public static double[] toDoubleArray(byte[] byteArray) {
        int times = Double.SIZE / Byte.SIZE;
        double[] doubles = new double[byteArray.length / times];
        for (int i = 0; i < doubles.length; i++)
            doubles[i] = ByteBuffer.wrap(byteArray, i * times, times).getDouble();
        return doubles;
    }

    /** Simple element-wise byte → double cast (no byte-buffer decoding). */
    public static double[] toDoubleArraySimple(byte[] byteArray) {
        double[] doubles = new double[byteArray.length];
        for (int i = 0; i < doubles.length; i++) doubles[i] = byteArray[i];
        return doubles;
    }

    public static double[] toDoubles(int[] ints) {
        double[] ret = new double[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = ints[i];
        return ret;
    }

    public static double[] toDoubles(long[] ints) {
        double[] ret = new double[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = ints[i];
        return ret;
    }

    public static double[] toDoubles(float[] ints) {
        double[] ret = new double[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = ints[i];
        return ret;
    }

    public static double[] toDoubles(int[][] ints) {
        return toDoubles(Ints.concat(ints));
    }

    public static double[] toDouble(boolean[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i] ? 1.0 : 0.0;
        return ret;
    }

    public static double[] toDouble(byte[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i];
        return ret;
    }

    public static double[] toDouble(int[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i];
        return ret;
    }

    public static double[] toDouble(long[] data) {
        double[] ret = new double[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i];
        return ret;
    }

    public static double[] toDouble(float[] v) {
        double[] ret = new double[v.length];
        for (int i = 0; i < v.length; i++) ret[i] = v[i];
        return ret;
    }

    // -------------------------------------------------------------------------
    // toLong* / toLongs
    // -------------------------------------------------------------------------

    public static long[] toLongArray(byte[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static long[] toLongArray(boolean[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] ? 1 : 0;
        return output;
    }

    public static long[] toLongArray(short[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static long[] toLongArray(char[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static long[] toLongArrayInt(int[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static long[] toLongArrayFloat(float[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (long) input[i];
        return output;
    }

    public static long[] toLongArray(double[] input) {
        long[] output = new long[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (long) input[i];
        return output;
    }

    public static long[] toLongArray(long[] input) {
        long[] output = new long[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    public static long[] toLongArray(int[] intArray) {
        if (intArray == null) return null;
        long[] ret = new long[intArray.length];
        for (int i = 0; i < intArray.length; i++) ret[i] = intArray[i];
        return ret;
    }

    public static long[] toLongArray(float[] array) {
        val ret = new long[array.length];
        for (int i = 0; i < array.length; i++) ret[i] = (long) array[i];
        return ret;
    }

    public static long[] toLongs(byte[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i];
        return ret;
    }

    public static long[] toLongs(boolean[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i] ? 1 : 0;
        return ret;
    }

    public static long[] toLongs(short[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i];
        return ret;
    }

    public static long[] toLongs(int[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = data[i];
        return ret;
    }

    public static long[] toLongs(float[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = (long) data[i];
        return ret;
    }

    public static long[] toLongs(double[] data) {
        val ret = new long[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = (long) data[i];
        return ret;
    }

    // -------------------------------------------------------------------------
    // toFloat* / toFloats
    // -------------------------------------------------------------------------

    public static float[] toFloatArray(short[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static float[] toFloatArray(boolean[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] ? 1.0f : 0.0f;
        return output;
    }

    public static float[] toFloatArray(char[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static float[] toFloatArray(int[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static float[] toFloatArray(long[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i];
        return output;
    }

    public static float[] toFloatArray(double[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (float) input[i];
        return output;
    }

    public static float[] toFloatArray(float[] input) {
        float[] output = new float[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    /** Decode a raw byte[] (4 bytes per float, big-endian via ByteBuffer) → float[]. */
    public static float[] toFloatArray(byte[] byteArray) {
        int times = Float.SIZE / Byte.SIZE;
        float[] doubles = new float[byteArray.length / times];
        for (int i = 0; i < doubles.length; i++)
            doubles[i] = ByteBuffer.wrap(byteArray, i * times, times).getFloat();
        return doubles;
    }

    /** Simple element-wise byte → float cast (no byte-buffer decoding). */
    public static float[] toFloatArraySimple(byte[] byteArray) {
        float[] doubles = new float[byteArray.length];
        for (int i = 0; i < doubles.length; i++) doubles[i] = byteArray[i];
        return doubles;
    }

    public static float[] toFloats(int[][] ints) {
        return toFloats(Ints.concat(ints));
    }

    public static float[] toFloats(boolean[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = ints[i] ? 1f : 0f;
        return ret;
    }

    public static float[] toFloats(byte[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = ints[i];
        return ret;
    }

    public static float[] toFloats(short[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = ints[i];
        return ret;
    }

    public static float[] toFloats(int[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = ints[i];
        return ret;
    }

    public static float[] toFloats(long[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = ints[i];
        return ret;
    }

    public static float[] toFloats(double[] ints) {
        float[] ret = new float[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = (float) ints[i];
        return ret;
    }

    // -------------------------------------------------------------------------
    // toShort* / toShorts
    // -------------------------------------------------------------------------

    public static short[] toShorts(long[] ints) {
        val ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = (short) ints[i];
        return ret;
    }

    public static short[] toShorts(byte[] ints) {
        val ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = (short) ints[i];
        return ret;
    }

    public static short[] toShorts(boolean[] ints) {
        short[] ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = (short) (ints[i] ? 1 : 0);
        return ret;
    }

    public static short[] toShorts(int[] ints) {
        short[] ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = (short) ints[i];
        return ret;
    }

    public static short[] toShorts(float[] ints) {
        short[] ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = (short) ints[i];
        return ret;
    }

    public static short[] toShorts(double[] ints) {
        short[] ret = new short[ints.length];
        for (int i = 0; i < ints.length; i++) ret[i] = (short) ints[i];
        return ret;
    }

    // -------------------------------------------------------------------------
    // toByte* / toBytes / toByteArray*
    // -------------------------------------------------------------------------

    public static byte[] toBytes(int[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) retVal[i] = (byte) array[i];
        return retVal;
    }

    public static byte[] toBytes(short[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) retVal[i] = (byte) array[i];
        return retVal;
    }

    public static byte[] toBytes(boolean[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) retVal[i] = (byte) (array[i] ? 1 : 0);
        return retVal;
    }

    public static byte[] toBytes(float[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) retVal[i] = (byte) array[i];
        return retVal;
    }

    public static byte[] toBytes(double[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) retVal[i] = (byte) array[i];
        return retVal;
    }

    public static byte[] toBytes(long[] array) {
        val retVal = new byte[array.length];
        for (int i = 0; i < array.length; i++) retVal[i] = (byte) array[i];
        return retVal;
    }

    /** Encode double[] → raw bytes (8 bytes per element, big-endian). */
    public static byte[] toByteArray(double[] doubleArray) {
        int times = Double.SIZE / Byte.SIZE;
        byte[] bytes = new byte[doubleArray.length * times];
        for (int i = 0; i < doubleArray.length; i++)
            ByteBuffer.wrap(bytes, i * times, times).putDouble(doubleArray[i]);
        return bytes;
    }

    /** Encode long[] → raw bytes (8 bytes per element, big-endian). */
    public static byte[] toByteArray(long[] longArray) {
        int times = Long.SIZE / Byte.SIZE;
        byte[] bytes = new byte[longArray.length * times];
        for (int i = 0; i < longArray.length; i++)
            ByteBuffer.wrap(bytes, i * times, times).putLong(longArray[i]);
        return bytes;
    }

    /** Simple element-wise long → byte cast (not a full 8-byte-per-element encode). */
    public static byte[] toByteArraySimple(long[] longArray) {
        byte[] bytes = new byte[longArray.length];
        for (int i = 0; i < longArray.length; i++) bytes[i] = (byte) longArray[i];
        return bytes;
    }

    /** Encode float[] → raw bytes (4 bytes per element, big-endian). */
    public static byte[] toByteArray(float[] doubleArray) {
        int times = Float.SIZE / Byte.SIZE;
        byte[] bytes = new byte[doubleArray.length * times];
        for (int i = 0; i < doubleArray.length; i++)
            ByteBuffer.wrap(bytes, i * times, times).putFloat(doubleArray[i]);
        return bytes;
    }

    /** Encode int[] → raw bytes (4 bytes per element, big-endian). */
    public static byte[] toByteArray(int[] intArray) {
        int times = Integer.SIZE / Byte.SIZE;
        byte[] bytes = new byte[intArray.length * times];
        for (int i = 0; i < intArray.length; i++)
            ByteBuffer.wrap(bytes, i * times, times).putInt(intArray[i]);
        return bytes;
    }

    // -------------------------------------------------------------------------
    // Half-precision (FP16) conversions
    // -------------------------------------------------------------------------

    /**
     * Convert a single float to its IEEE 754 half-precision (FP16) bit pattern,
     * stored as a short.
     */
    public static short fromFloat(float v) {
        if (Float.isNaN(v))              return (short) 0x7fff;
        if (v == Float.POSITIVE_INFINITY) return (short) 0x7c00;
        if (v == Float.NEGATIVE_INFINITY) return (short) 0xfc00;
        if (v == 0.0f)                   return (short) 0x0000;
        if (v == -0.0f)                  return (short) 0x8000;
        if (v > 65504.0f)                return 0x7bff;
        if (v < -65504.0f)               return (short) (0x7bff | 0x8000);
        if (v > 0.0f && v < 5.96046E-8f) return 0x0001;
        if (v < 0.0f && v > -5.96046E-8f) return (short) 0x8001;
        final int f = Float.floatToIntBits(v);
        return (short) (((f >> 16) & 0x8000)
                | ((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00)
                | ((f >> 13) & 0x03ff));
    }

    public static short toHalf(float data) {
        return fromFloat(data);
    }

    public static short toHalf(double data) {
        return fromFloat((float) data);
    }

    public static short[] toHalfs(boolean[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = fromFloat(data[i] ? 1 : 0);
        return ret;
    }

    public static short[] toHalfs(byte[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = fromFloat(data[i]);
        return ret;
    }

    public static short[] toHalfs(short[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = fromFloat(data[i]);
        return ret;
    }

    public static short[] toHalfs(float[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = fromFloat(data[i]);
        return ret;
    }

    public static short[] toHalfs(int[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = fromFloat((float) data[i]);
        return ret;
    }

    public static short[] toHalfs(long[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = fromFloat((float) data[i]);
        return ret;
    }

    public static short[] toHalfs(double[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = fromFloat((float) data[i]);
        return ret;
    }

    // -------------------------------------------------------------------------
    // BFloat16 conversions
    // -------------------------------------------------------------------------

    /**
     * Convert a half-precision (FP16) short bit pattern to bfloat16 bit pattern.
     * Adjusts the exponent bias from FP16 (15) to BF16 (127).
     */
    public static short toBFloat16(short data) {
        int sign = data >>> 15;
        int exp  = (data >>> 10) & 0x1F;
        int fraction = data & 0x3FF;
        exp = exp - 15 + 127;
        if (exp < 0)   { exp = 0;   fraction = 0; }
        else if (exp > 255) { exp = 255; fraction = 0; }
        fraction >>>= 3;
        return (short) ((sign << 15) | (exp << 7) | fraction);
    }

    public static short toBFloat16(float data) {
        int floatBits = Float.floatToRawIntBits(data);
        int sign     = floatBits >>> 31;
        int exp      = (floatBits >>> 23) & 0xFF;
        int fraction = floatBits & 0x7FFFFF;
        fraction >>>= 16;
        return (short) ((sign << 15) | (exp << 7) | fraction);
    }

    public static short toBFloat16(double data) {
        return toBFloat16((float) data);
    }

    public static short longToBFloat16(long l) {
        return toBFloat16((double) l);
    }

    public static float bfloat16ToFloat(short b) {
        int sign     = b >>> 15;
        int exp      = (b >>> 7) & 0xFF;
        int fraction = b & 0x7F;
        fraction <<= 16;
        int floatBits = (sign << 31) | (exp << 23) | fraction;
        return Float.intBitsToFloat(floatBits);
    }

    public static double bfloat16ToDouble(short b) {
        return bfloat16ToFloat(b);
    }

    public static long bfloat16ToLong(short b) {
        return (long) bfloat16ToFloat(b);
    }

    public static int bfloat16ToInt(short b) {
        return (int) bfloat16ToFloat(b);
    }

    public static short bfloat16ToShort(short b) {
        int sign     = b >>> 15;
        int exp      = (b >>> 7) & 0xFF;
        int fraction = b & 0x7F;
        exp      >>>= 3;
        fraction <<= 3;
        return (short) ((sign << 15) | (exp << 10) | fraction);
    }

    public static short[] toBfloats(double[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = toBFloat16(data[i]);
        return ret;
    }

    public static short[] toBfloats(boolean[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = toBFloat16(data[i] ? 1.0 : 0.0);
        return ret;
    }

    public static short[] toBfloats(byte[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = toBFloat16(data[i]);
        return ret;
    }

    public static short[] toBfloats(short[] data) {
        float[] ret = new float[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = toBFloat16((float) data[i]);
        return ArrayTypeConverters.toShorts(ret);
    }

    public static short[] toBfloats(float[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = toBFloat16(data[i]);
        return ret;
    }

    public static short[] toBfloats(int[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = toBFloat16(data[i]);
        return ret;
    }

    public static short[] toBfloats(long[] data) {
        short[] ret = new short[data.length];
        for (int i = 0; i < ret.length; i++) ret[i] = toBFloat16((float) data[i]);
        return ret;
    }

    // -------------------------------------------------------------------------
    // BigInteger conversions
    // -------------------------------------------------------------------------

    public static BigInteger[] toBigInteger(byte[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) ret[i] = BigInteger.valueOf(input[i]);
        return ret;
    }

    public static BigInteger[] toBigInteger(short[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) ret[i] = BigInteger.valueOf(input[i]);
        return ret;
    }

    public static BigInteger[] toBigInteger(long[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) ret[i] = BigInteger.valueOf(input[i]);
        return ret;
    }

    public static BigInteger[] toBigInteger(boolean[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) ret[i] = BigInteger.valueOf(input[i] ? 1L : 0L);
        return ret;
    }

    public static BigInteger[] toBigInteger(float[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) ret[i] = BigInteger.valueOf((long) input[i]);
        return ret;
    }

    public static BigInteger[] toBigInteger(double[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) ret[i] = BigInteger.valueOf((long) input[i]);
        return ret;
    }

    public static BigInteger[] toBigInteger(int[] input) {
        BigInteger[] ret = new BigInteger[input.length];
        for (int i = 0; i < input.length; i++) ret[i] = BigInteger.valueOf(input[i]);
        return ret;
    }

    // -------------------------------------------------------------------------
    // toPrimitives (boxed → primitive, up to rank 4)
    // -------------------------------------------------------------------------

    public static long[] toPrimitives(Long[] array) {
        val res = new long[array.length];
        for (int e = 0; e < array.length; e++) res[e] = array[e];
        return res;
    }

    public static int[] toPrimitives(Integer[] array) {
        val res = new int[array.length];
        for (int e = 0; e < array.length; e++) res[e] = array[e];
        return res;
    }

    public static short[] toPrimitives(Short[] array) {
        val res = new short[array.length];
        for (int e = 0; e < array.length; e++) res[e] = array[e];
        return res;
    }

    public static byte[] toPrimitives(Byte[] array) {
        val res = new byte[array.length];
        for (int e = 0; e < array.length; e++) res[e] = array[e];
        return res;
    }

    public static float[] toPrimitives(Float[] array) {
        val res = new float[array.length];
        for (int e = 0; e < array.length; e++) res[e] = array[e];
        return res;
    }

    public static double[] toPrimitives(Double[] array) {
        val res = new double[array.length];
        for (int e = 0; e < array.length; e++) res[e] = array[e];
        return res;
    }

    public static boolean[] toPrimitives(Boolean[] array) {
        val res = new boolean[array.length];
        for (int e = 0; e < array.length; e++) res[e] = array[e];
        return res;
    }

    public static long[][] toPrimitives(Long[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new long[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++) res[i][j] = array[i][j];
        return res;
    }

    public static int[][] toPrimitives(Integer[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new int[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++) res[i][j] = array[i][j];
        return res;
    }

    public static short[][] toPrimitives(Short[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new short[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++) res[i][j] = array[i][j];
        return res;
    }

    public static byte[][] toPrimitives(Byte[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new byte[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++) res[i][j] = array[i][j];
        return res;
    }

    public static double[][] toPrimitives(Double[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new double[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++) res[i][j] = array[i][j];
        return res;
    }

    public static float[][] toPrimitives(Float[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new float[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++) res[i][j] = array[i][j];
        return res;
    }

    public static boolean[][] toPrimitives(Boolean[][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new boolean[array.length][array[0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++) res[i][j] = array[i][j];
        return res;
    }

    public static long[][][] toPrimitives(Long[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new long[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++) res[i][j][k] = array[i][j][k];
        return res;
    }

    public static int[][][] toPrimitives(Integer[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new int[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++) res[i][j][k] = array[i][j][k];
        return res;
    }

    public static short[][][] toPrimitives(Short[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new short[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++) res[i][j][k] = array[i][j][k];
        return res;
    }

    public static byte[][][] toPrimitives(Byte[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new byte[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++) res[i][j][k] = array[i][j][k];
        return res;
    }

    public static double[][][] toPrimitives(Double[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new double[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++) res[i][j][k] = array[i][j][k];
        return res;
    }

    public static float[][][] toPrimitives(Float[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new float[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++) res[i][j][k] = array[i][j][k];
        return res;
    }

    public static boolean[][][] toPrimitives(Boolean[][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new boolean[array.length][array[0].length][array[0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++) res[i][j][k] = array[i][j][k];
        return res;
    }

    public static long[][][][] toPrimitives(Long[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new long[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++) res[i][j][k][l] = array[i][j][k][l];
        return res;
    }

    public static int[][][][] toPrimitives(Integer[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new int[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++) res[i][j][k][l] = array[i][j][k][l];
        return res;
    }

    public static short[][][][] toPrimitives(Short[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new short[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++) res[i][j][k][l] = array[i][j][k][l];
        return res;
    }

    public static byte[][][][] toPrimitives(Byte[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new byte[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++) res[i][j][k][l] = array[i][j][k][l];
        return res;
    }

    public static double[][][][] toPrimitives(Double[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new double[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++) res[i][j][k][l] = array[i][j][k][l];
        return res;
    }

    public static float[][][][] toPrimitives(Float[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new float[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++) res[i][j][k][l] = array[i][j][k][l];
        return res;
    }

    public static boolean[][][][] toPrimitives(Boolean[][][][] array) {
        ArrayUtil.assertNotRagged(array);
        val res = new boolean[array.length][array[0].length][array[0][0].length][array[0][0][0].length];
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[0].length; j++)
                for (int k = 0; k < array[0][0].length; k++)
                    for (int l = 0; l < array[0][0][0].length; l++) res[i][j][k][l] = array[i][j][k][l];
        return res;
    }
}
