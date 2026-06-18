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

import org.nd4j.common.base.Preconditions;

import java.util.List;

/**
 * Numeric computation utilities for primitive arrays.
 *
 * <p>Covers: element-wise reduce operations (sum, product), range generation,
 * comparison predicates, arithmetic helpers (mod, multiplyBy), index search
 * (argMax, argMin, max, min), shape validation (assertNotRagged, assertSquare),
 * and stride/offset calculations.
 *
 * <p>Methods here were extracted from {@link ArrayUtil} to improve cohesion.
 * {@link ArrayUtil} retains one-line delegates annotated {@code @Deprecated}
 * for backward compatibility.
 */
public final class ArrayMathUtils {

    private ArrayMathUtils() {}

    // -------------------------------------------------------------------------
    // sum / sumLong
    // -------------------------------------------------------------------------

    public static int sum(List<Integer> add) {
        if (add.isEmpty()) return 0;
        int ret = 0;
        for (int i = 0; i < add.size(); i++) ret += add.get(i);
        return ret;
    }

    public static int sum(int[] add) {
        if (add.length < 1) return 0;
        int ret = 0;
        for (int v : add) ret += v;
        return ret;
    }

    public static long sumLong(long... add) {
        if (add.length < 1) return 0;
        long ret = 0;
        for (long v : add) ret += v;
        return ret;
    }

    // -------------------------------------------------------------------------
    // prod / prodLong
    // -------------------------------------------------------------------------

    public static int prod(List<Integer> mult) {
        if (mult.isEmpty()) return 0;
        int ret = 1;
        for (int i = 0; i < mult.size(); i++) ret *= mult.get(i);
        return ret;
    }

    public static int prod(long... mult) {
        if (mult.length < 1) return 0;
        int ret = 1;
        for (long v : mult) ret *= v;
        return ret;
    }

    public static int prod(int... mult) {
        if (mult.length < 1) return 0;
        int ret = 1;
        for (int v : mult) ret *= v;
        return ret;
    }

    public static long prodLong(List<? extends Number> mult) {
        if (mult.isEmpty()) return 0;
        long ret = 1;
        for (int i = 0; i < mult.size(); i++) {
            long val = mult.get(i).longValue();
            if (val < 0) continue; // skip -1 sentinel (unknown/dynamic dim)
            ret *= val;
        }
        return ret;
    }

    public static long prodLong(int... mult) {
        if (mult.length < 1) return 0;
        long ret = 1;
        for (int v : mult) {
            if (v < 0) continue;
            ret *= v;
        }
        return ret;
    }

    public static long prodLong(long... mult) {
        if (mult.length < 1) return 0;
        long ret = 1;
        for (long v : mult) {
            if (v < 0) continue;
            ret *= v;
        }
        return ret;
    }

    // -------------------------------------------------------------------------
    // range generation
    // -------------------------------------------------------------------------

    /**
     * Returns a subset of {@code data} from index 0 to {@code to} (exclusive).
     */
    public static double[] range(double[] data, int to) {
        return range(data, to, 1);
    }

    public static double[] range(double[] data, int to, int stride) {
        return range(data, to, stride, 1);
    }

    public static double[] range(double[] data, int to, int stride, int numElementsEachStride) {
        double[] ret = new double[to / stride];
        if (ret.length < 1) ret = new double[1];
        int count = 0;
        for (int i = 0; i < data.length; i += stride) {
            for (int j = 0; j < numElementsEachStride; j++) {
                if (i + j >= data.length || count >= ret.length) break;
                ret[count++] = data[i + j];
            }
        }
        return ret;
    }

    /**
     * Generate an int array from {@code from} to {@code to} (exclusive) with the given increment.
     * Counts backwards when {@code from > to}.
     */
    public static int[] range(int from, int to, int increment) {
        int diff = Math.abs(from - to);
        int[] ret = new int[diff / increment];
        if (ret.length < 1) ret = new int[1];
        if (from < to) {
            int count = 0;
            for (int i = from; i < to; i += increment) {
                if (count >= ret.length) break;
                ret[count++] = i;
            }
        } else if (from > to) {
            int count = 0;
            for (int i = from - 1; i >= to; i -= increment) {
                if (count >= ret.length) break;
                ret[count++] = i;
            }
        }
        return ret;
    }

    public static long[] range(long from, long to, long increment) {
        long diff = Math.abs(from - to);
        long[] ret = new long[(int) (diff / increment)];
        if (ret.length < 1) ret = new long[1];
        if (from < to) {
            int count = 0;
            for (long i = from; i < to; i += increment) {
                if (count >= ret.length) break;
                ret[count++] = i;
            }
        } else if (from > to) {
            int count = 0;
            for (int i = (int) from - 1; i >= to; i -= increment) {
                if (count >= ret.length) break;
                ret[count++] = i;
            }
        }
        return ret;
    }

    public static int[] range(int from, int to) {
        if (from == to) return new int[0];
        return range(from, to, 1);
    }

    public static long[] range(long from, long to) {
        if (from == to) return new long[0];
        return range(from, to, 1);
    }

    // -------------------------------------------------------------------------
    // Comparison predicates
    // -------------------------------------------------------------------------

    public static boolean greaterThan(int[] target, int[] test) {
        Preconditions.checkArgument(target.length == test.length,
                "Unable to compare: different sizes: length %s vs. %s", target.length, test.length);
        for (int i = 0; i < target.length; i++) {
            if (target[i] > test[i]) return true;
            if (target[i] < test[i]) return false;
        }
        return false;
    }

    public static boolean lessThan(int[] target, int[] test) {
        Preconditions.checkArgument(target.length == test.length,
                "Unable to compare: different sizes: length %s vs. %s", target.length, test.length);
        for (int i = 0; i < target.length; i++) {
            if (target[i] < test[i]) return true;
            if (target[i] > test[i]) return false;
        }
        return false;
    }

    public static boolean anyLargerThan(int[] arrs, int check) {
        for (int v : arrs) if (v > check) return true;
        return false;
    }

    public static boolean anyLessThan(int[] arrs, int check) {
        for (int v : arrs) if (v < check) return true;
        return false;
    }

    public static boolean anyMore(int[] target, int[] test) {
        Preconditions.checkArgument(target.length == test.length,
                "Unable to compare: different sizes: length %s vs. %s", target.length, test.length);
        for (int i = 0; i < target.length; i++) if (target[i] > test[i]) return true;
        return false;
    }

    public static boolean anyLess(int[] target, int[] test) {
        Preconditions.checkArgument(target.length == test.length,
                "Unable to compare: different sizes: length %s vs. %s", target.length, test.length);
        for (int i = 0; i < target.length; i++) if (target[i] < test[i]) return true;
        return false;
    }

    // -------------------------------------------------------------------------
    // isZero / isInverse
    // -------------------------------------------------------------------------

    /** Returns true if any element of the array is zero. */
    public static boolean isZero(int[] as) {
        for (int v : as) if (v == 0) return true;
        return false;
    }

    public static boolean isZero(long[] as) {
        for (long v : as) if (v == 0L) return true;
        return false;
    }

    /**
     * Returns true if the two arrays are reverse copies of each other
     * (i.e., {@code first[i] == second[n-1-i]} for all i).
     */
    public static boolean isInverse(int[] first, int[] second) {
        int backWardCount = second.length - 1;
        for (int v : first) {
            if (v != second[backWardCount--]) return false;
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Arithmetic helpers
    // -------------------------------------------------------------------------

    /** Multiply every element of {@code arr} in-place by {@code mult}. */
    public static void multiplyBy(int[] arr, int mult) {
        for (int i = 0; i < arr.length; i++) arr[i] *= mult;
    }

    public static int[] mod(int[] input, int mod) {
        int[] ret = new int[input.length];
        for (int i = 0; i < ret.length; i++) ret[i] = input[i] % mod;
        return ret;
    }

    // -------------------------------------------------------------------------
    // argMax / argMin / max / min
    // -------------------------------------------------------------------------

    /** Returns the maximum value in the array. */
    public static int max(int[] in) {
        int max = Integer.MIN_VALUE;
        for (int v : in) if (v > max) max = v;
        return max;
    }

    /** Returns the minimum value in the array. */
    public static int min(int[] in) {
        int min = Integer.MAX_VALUE;
        for (int v : in) if (v < min) min = v;
        return min;
    }

    /** Returns the index of the maximum value. Ties broken by first occurrence. */
    public static int argMax(int[] in) {
        int maxIdx = 0;
        for (int i = 1; i < in.length; i++) if (in[i] > in[maxIdx]) maxIdx = i;
        return maxIdx;
    }

    /** Returns the index of the minimum value. Ties broken by first occurrence. */
    public static int argMin(int[] in) {
        int minIdx = 0;
        for (int i = 1; i < in.length; i++) if (in[i] < in[minIdx]) minIdx = i;
        return minIdx;
    }

    /** Returns the index of the maximum value (long[]). Ties broken by first occurrence. */
    public static int argMax(long[] in) {
        int maxIdx = 0;
        for (int i = 1; i < in.length; i++) if (in[i] > in[maxIdx]) maxIdx = i;
        return maxIdx;
    }

    /** Returns the index of the minimum value (long[]). Ties broken by first occurrence. */
    public static int argMin(long[] in) {
        int minIdx = 0;
        for (int i = 1; i < in.length; i++) if (in[i] < in[minIdx]) minIdx = i;
        return minIdx;
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    /**
     * Assert that every row-array in the variadic argument has the same length
     * (i.e., the matrix is square / rectangular).
     */
    public static void assertSquare(double[]... d) {
        if (d.length > 2) {
            for (double[] row : d) assertSquare(row);
        } else {
            int firstLength = d[0].length;
            for (int i = 1; i < d.length; i++)
                Preconditions.checkState(d[i].length == firstLength);
        }
    }

    // -------------------------------------------------------------------------
    // Offset / stride helpers
    // -------------------------------------------------------------------------

    /**
     * Calculate the linear offset for index {@code i} across all dimensions
     * of {@code stride}.
     */
    public static int offsetFor(int[] stride, int i) {
        int ret = 0;
        for (int j = 0; j < stride.length; j++) ret += (i * stride[j]);
        return ret;
    }

    public static int calcOffset(List<Integer> shape, List<Integer> offsets, List<Integer> strides) {
        if (shape.size() != offsets.size() || shape.size() != strides.size())
            throw new IllegalArgumentException("Shapes, strides, and offsets must be the same size");
        int ret = 0;
        for (int i = 0; i < offsets.size(); i++) {
            if (shape.get(i) == 1 && offsets.size() > 2 && i > 0) continue;
            ret += offsets.get(i) * strides.get(i);
        }
        return ret;
    }

    public static int calcOffset(int[] shape, int[] offsets, int[] strides) {
        if (shape.length != offsets.length || shape.length != strides.length)
            throw new IllegalArgumentException("Shapes, strides, and offsets must be the same size");
        int ret = 0;
        for (int i = 0; i < offsets.length; i++) {
            if (shape[i] == 1) continue;
            ret += offsets[i] * strides[i];
        }
        return ret;
    }

    public static long calcOffset(long[] shape, long[] offsets, long[] strides) {
        if (shape.length != offsets.length || shape.length != strides.length)
            throw new IllegalArgumentException("Shapes, strides, and offsets must be the same size");
        long ret = 0;
        for (int i = 0; i < offsets.length; i++) {
            if (shape[i] == 1) continue;
            ret += offsets[i] * strides[i];
        }
        return ret;
    }

    public static long calcOffsetLong(List<Integer> shape, List<Integer> offsets, List<Integer> strides) {
        if (shape.size() != offsets.size() || shape.size() != strides.size())
            throw new IllegalArgumentException("Shapes, strides, and offsets must be the same size");
        long ret = 0;
        for (int i = 0; i < offsets.size(); i++) {
            if (shape.get(i) == 1 && offsets.size() > 2 && i > 0) continue;
            ret += (long) offsets.get(i) * strides.get(i);
        }
        return ret;
    }

    public static long calcOffsetLong2(List<Long> shape, List<Long> offsets, List<Long> strides) {
        if (shape.size() != offsets.size() || shape.size() != strides.size())
            throw new IllegalArgumentException("Shapes, strides, and offsets must be the same size");
        long ret = 0;
        for (int i = 0; i < offsets.size(); i++) {
            if (shape.get(i) == 1 && offsets.size() > 2 && i > 0) continue;
            ret += offsets.get(i) * strides.get(i);
        }
        return ret;
    }

    public static long calcOffsetLong(int[] shape, int[] offsets, int[] strides) {
        if (shape.length != offsets.length || shape.length != strides.length)
            throw new IllegalArgumentException("Shapes, strides, and offsets must be the same size");
        long ret = 0;
        for (int i = 0; i < offsets.length; i++) {
            if (shape[i] == 1) continue;
            ret += (long) offsets[i] * strides[i];
        }
        return ret;
    }

    // -------------------------------------------------------------------------
    // Dot products
    // -------------------------------------------------------------------------

    public static int dotProduct(List<Integer> xs, List<Integer> ys) {
        int n = xs.size();
        if (ys.size() != n) throw new IllegalArgumentException("Different array sizes");
        int result = 0;
        for (int i = 0; i < n; i++) result += xs.get(i) * ys.get(i);
        return result;
    }

    public static int dotProduct(int[] xs, int[] ys) {
        int n = xs.length;
        if (ys.length != n) throw new IllegalArgumentException("Different array sizes");
        int result = 0;
        for (int i = 0; i < n; i++) result += xs[i] * ys[i];
        return result;
    }

    public static long dotProductLong(List<Integer> xs, List<Integer> ys) {
        int n = xs.size();
        if (ys.size() != n) throw new IllegalArgumentException("Different array sizes");
        long result = 0;
        for (int i = 0; i < n; i++) result += (long) xs.get(i) * ys.get(i);
        return result;
    }

    public static long dotProductLong2(List<Long> xs, List<Long> ys) {
        int n = xs.size();
        if (ys.size() != n) throw new IllegalArgumentException("Different array sizes");
        long result = 0;
        for (int i = 0; i < n; i++) result += xs.get(i) * ys.get(i);
        return result;
    }

    public static long dotProductLong(int[] xs, int[] ys) {
        int n = xs.length;
        if (ys.length != n) throw new IllegalArgumentException("Different array sizes");
        long result = 0;
        for (int i = 0; i < n; i++) result += (long) xs[i] * ys[i];
        return result;
    }
}
