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

package org.deeplearning4j.util;

import org.nd4j.common.base.Preconditions;

import java.util.Arrays;

/**
 * Validation methods for array sizes/shapes and value non-negativeness
 *
 * @author Ryan Nett
 */
public class ValidationUtils {

    private ValidationUtils() {

    }

    /**
     * Checks that the values is >= 0.
     *
     * @param data      An int
     * @param paramName The param name, for error reporting
     */
    public static void validateNonNegative(int data, String paramName) {
        Preconditions.checkArgument(data >= 0,
                "Values for %s must be >= 0, got: %s", paramName, data);
    }


    /**
     * Checks that the values is >= 0.
     *
     * @param data      An int
     * @param paramName The param name, for error reporting
     */
    public static void validateNonNegative(long data, String paramName) {
        Preconditions.checkArgument(data >= 0,
                "Values for %s must be >= 0, got: %s", paramName, data);
    }

    /**
     * Checks that the values is >= 0.
     *
     * @param data      An int
     * @param paramName The param name, for error reporting
     */
    public static void validateNonNegative(double data, String paramName) {
        Preconditions.checkArgument(data >= 0,
                "Values for %s must be >= 0, got: %s", paramName, data);
    }


    /**
     * Checks that all values are >= 0.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     */
    public static void validateNonNegative(long[] data, String paramName) {

        if (data == null) {
            return;
        }

        boolean nonnegative = true;

        for (long value : data) {
            if (value < 0) {
                nonnegative = false;
            }
        }

        Preconditions.checkArgument(nonnegative,
                "Values for %s must be >= 0, got: %s", paramName, data);
    }

    /**
     * Checks that all values are >= 0.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     */
    public static void validateNonNegative(int[] data, String paramName) {

        if (data == null) {
            return;
        }

        boolean nonnegative = true;

        for (int value : data) {
            if (value < 0) {
                nonnegative = false;
            }
        }

        Preconditions.checkArgument(nonnegative,
                "Values for %s must be >= 0, got: %s", paramName, data);
    }

    /**
     * Reformats the input array to a length 1 array and checks that all values are >= 0.
     * <p>
     * If the array is length 1, returns the array
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 1 that represents the input
     */
    public static int[] validate1NonNegative(int[] data, String paramName) {
        validateNonNegative(data, paramName);
        return validate1(data, paramName);
    }

    /**
     * Reformats the input array to a length 1 array.
     * <p>
     * If the array is length 1, returns the array
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 1 that represents the input
     */
    public static int[] validate1(int[] data, String paramName) {
        if (data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1,
                "Need 1 %s value, got %s values: %s",
                paramName, data.length, data);

        return data;
    }










    /**
     * Reformats the input array to a length 1 array and checks that all values are >= 0.
     * <p>
     * If the array is length 1, returns the array
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 1 that represents the input
     */
    public static long[] validate1NonNegativeLong(long[] data, String paramName) {
        validateNonNegative(data, paramName);
        return validate1Long(data, paramName);
    }

    /**
     * Reformats the input array to a length 1 array.
     * <p>
     * If the array is length 1, returns the array
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 1 that represents the input
     */
    public static long[] validate1Long(long[] data, String paramName) {
        if (data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1,
                "Need 1 %s value, got %s values: %s",
                paramName, data.length, data);

        return data;
    }



    /**
     * Reformats the input array to a length 2 array and checks that all values are >= 0.
     * <p>
     * If the array is length 1, returns [a, a]
     * If the array is length 2, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 2 that represents the input
     */
    public static long[] validate2NonNegativeLong(long[] data, boolean allowSz1, String paramName) {
        validateNonNegative(data, paramName);
        return validate2Long(data, allowSz1, paramName);
    }


    /**
     * Reformats the input array to a length 2 array and checks that all values are >= 0.
     * <p>
     * If the array is length 1, returns [a, a]
     * If the array is length 2, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 2 that represents the input
     */
    public static int[] validate2NonNegative(int[] data, boolean allowSz1, String paramName) {
        validateNonNegative(data, paramName);
        return validate2(data, allowSz1, paramName);
    }


    /**
     * Reformats the input array to a length 2 array and checks that all values are >= 0.
     * <p>
     * If the array is length 1, returns [a, a]
     * If the array is length 2, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 2 that represents the input
     */
    public static long[] validate2NonNegative(long[] data, boolean allowSz1, String paramName) {
        validateNonNegative(data, paramName);
        return validate2(data, allowSz1, paramName);
    }


    /**
     * Reformats the input array to a length 2 array.
     * <p>
     * If the array is length 1, returns [a, a]
     * If the array is length 2, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 2 that represents the input
     */
    public static long[] validate2(long[] data, boolean allowSz1, String paramName) {
        if (data == null) {
            return null;
        }


        if (allowSz1) {
            Preconditions.checkArgument(data.length == 1 || data.length == 2,
                    "Need either 1 or 2 %s values, got %s values: %s",
                    paramName, data.length, data);
        } else {
            Preconditions.checkArgument(data.length == 2, "Need 2 %s values, got %s values: %s",
                    paramName, data.length, data);
        }

        if (data.length == 1) {
            return new long[]{data[0], data[0]};
        } else {
            return data;
        }
    }

    /**
     * Reformats the input array to a length 2 array.
     * <p>
     * If the array is length 1, returns [a, a]
     * If the array is length 2, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 2 that represents the input
     */
    public static int[] validate2(int[] data, boolean allowSz1, String paramName) {
        return Arrays.stream(validate2Long(toLongArray(data), allowSz1, paramName)).mapToInt(Math::toIntExact).toArray();
    }

    /**
     * Reformats the input array to a length 2 array.
     * <p>
     * If the array is length 1, returns [a, a]
     * If the array is length 2, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 2 that represents the input
     */
    public static long[] validate2Long(long[] data, boolean allowSz1, String paramName) {
        if (data == null) {
            return null;
        }

        if (allowSz1) {
            Preconditions.checkArgument(data.length == 1 || data.length == 2,
                    "Need either 1 or 2 %s values, got %s values: %s",
                    paramName, data.length, data);
        } else {
            Preconditions.checkArgument(data.length == 2, "Need 2 %s values, got %s values: %s",
                    paramName, data.length, data);
        }

        if (data.length == 1) {
            return new long[]{data[0], data[0]};
        } else {
            return data;
        }
    }



    /**
     * Helper method to convert a 2D int array to a 2D long array.
     *
     * @param intArray The 2D int array to convert.
     * @return The converted 2D long array.
     */
    private static long[][] toLongArray2D(int[][] intArray) {
        if (intArray == null) {
            return null;
        }
        return Arrays.stream(intArray)
                .map(ValidationUtils::toLongArray)
                .toArray(long[][]::new);
    }

    /**
     * Reformats the input array to a 2x2 array and checks that all values are >= 0.
     * <p>
     * If the array is 2x1 ([[a], [b]]), returns [[a, a], [b, b]]
     * If the array is 1x2 ([[a, b]]), returns [[a, b], [a, b]]
     * If the array is 2x2, returns the array
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 2 that represents the input
     */
    public static int[][] validate2x2NonNegative(int[][] data, String paramName) {
        return Arrays.stream(validate2x2NonNegativeLong(toLongArray2D(data), paramName))
                .map(arr -> Arrays.stream(arr).mapToInt(Math::toIntExact).toArray())
                .toArray(int[][]::new);
    }

    /**
     * Reformats the input array to a 2x2 array and checks that all values are >= 0.
     * <p>
     * If the array is 2x1 ([[a], [b]]), returns [[a, a], [b, b]]
     * If the array is 1x2 ([[a, b]]), returns [[a, b], [a, b]]
     * If the array is 2x2, returns the array
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 2 that represents the input
     */
    public static long[][] validate2x2NonNegativeLong(long[][] data, String paramName) {
        for (long[] part : data)
            validateNonNegativeLong(part, paramName);

        return validate2x2Long(data, paramName);
    }

    /**
     * Reformats the input array to a 2x2 array.
     * <p>
     * If the array is 2x1 ([[a], [b]]), returns [[a, a], [b, b]]
     * If the array is 1x2 ([[a, b]]), returns [[a, b], [a, b]]
     * If the array is 2x2, returns the array
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 2 that represents the input
     */
    public static int[][] validate2x2(int[][] data, String paramName) {
        return Arrays.stream(validate2x2Long(toLongArray2D(data), paramName))
                .map(arr -> Arrays.stream(arr).mapToInt(Math::toIntExact).toArray())
                .toArray(int[][]::new);
    }

    /**
     * Reformats the input array to a 2x2 array.
     * <p>
     * If the array is 2x1 ([[a], [b]]), returns [[a, a], [b, b]]
     * If the array is 1x2 ([[a, b]]), returns [[a, b], [a, b]]
     * If the array is 2x2, returns the array
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 2 that represents the input
     */
    public static long[][] validate2x2Long(long[][] data, String paramName) {
        if (data == null) {
            return null;
        }

        Preconditions.checkArgument(
                (data.length == 1 && data[0].length == 2) ||
                        (data.length == 2 &&
                                (data[0].length == 1 || data[0].length == 2) &&
                                (data[1].length == 1 || data[1].length == 2) &&
                                data[0].length == data[1].length
                        ),
                "Value for %s must have shape 2x1, 1x2, or 2x2, got %sx%s shaped array: %s",
                paramName, data.length, data[0].length, data);

        if (data.length == 1) {
            return new long[][]{
                    data[0],
                    data[0]
            };
        } else if (data[0].length == 1) {
            return new long[][]{
                    new long[]{data[0][0], data[0][0]},
                    new long[]{data[1][0], data[1][0]}
            };
        } else {
            return data;
        }
    }

    /**
     * Reformats the input array to a length 3 array and checks that all values >= 0.
     * <p>
     * If the array is length 1, returns [a, a, a]
     * If the array is length 3, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 3 that represents the input
     */
    public static int[] validate3NonNegative(int[] data, String paramName) {
        return Arrays.stream(validate3NonNegativeLong(toLongArray(data), paramName)).mapToInt(Math::toIntExact).toArray();
    }

    /**
     * Reformats the input array to a length 3 array and checks that all values >= 0.
     * <p>
     * If the array is length 1, returns [a, a, a]
     * If the array is length 3, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 3 that represents the input
     */
    public static long[] validate3NonNegativeLong(long[] data, String paramName) {
        validateNonNegativeLong(data, paramName);
        return validate3Long(data, paramName);
    }

    /**
     * Reformats the input array to a length 3 array.
     * <p>
     * If the array is length 1, returns [a, a, a]
     * If the array is length 3, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 3 that represents the input
     */
    public static int[] validate3(int[] data, String paramName) {
        return Arrays.stream(validate3Long(toLongArray(data), paramName)).mapToInt(Math::toIntExact).toArray();
    }

    /**
     * Reformats the input array to a length 3 array.
     * <p>
     * If the array is length 1, returns [a, a, a]
     * If the array is length 3, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 3 that represents the input
     */
    public static long[] validate3Long(long[] data, String paramName) {
        if (data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1 || data.length == 3,
                "Need either 1 or 3 %s values, got %s values: %s",
                paramName, data.length, data);

        if (data.length == 1) {
            return new long[]{data[0], data[0], data[0]};
        } else {
            return data;
        }
    }

    /**
     * Reformats the input array to a length 4 array and checks that all values >= 0.
     * <p>
     * If the array is length 1, returns [a, a, a, a]
     * If the array is length 2, return [a, a, b, b]
     * If the array is length 4, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 4 that represents the input
     */
    public static int[] validate4NonNegative(int[] data, String paramName) {
        return Arrays.stream(validate4NonNegativeLong(toLongArray(data), paramName)).mapToInt(Math::toIntExact).toArray();
    }

    /**
     * Reformats the input array to a length 4 array and checks that all values >= 0.
     * <p>
     * If the array is length 1, returns [a, a, a, a]
     * If the array is length 2, return [a, a, b, b]
     * If the array is length 4, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 4 that represents the input
     */
    public static long[] validate4NonNegativeLong(long[] data, String paramName) {
        validateNonNegativeLong(data, paramName);
        return validate4Long(data, paramName);
    }

    /**
     * Reformats the input array to a length 4 array.
     * <p>
     * If the array is length 1, returns [a, a, a, a]
     * If the array is length 2, return [a, a, b, b]
     * If the array is length 4, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 4 that represents the input
     */
    public static int[] validate4(int[] data, String paramName) {
        return Arrays.stream(validate4Long(toLongArray(data), paramName)).mapToInt(Math::toIntExact).toArray();
    }

    /**
     * Reformats the input array to a length 4 array.
     * <p>
     * If the array is length 1, returns [a, a, a, a]
     * If the array is length 2, return [a, a, b, b]
     * If the array is length 4, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 4 that represents the input
     */
    public static long[] validate4Long(long[] data, String paramName) {
        if (data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1 || data.length == 2 || data.length == 4,
                "Need either 1, 2, or 4 %s values, got %s values: %s",
                paramName, data.length, data);

        if (data.length == 1) {
            return new long[]{data[0], data[0], data[0], data[0]};
        } else if (data.length == 2) {
            return new long[]{data[0], data[0], data[1], data[1]};
        } else {
            return data;
        }
    }

    /**
     * Reformats the input array to a length 6 array and checks that all values >= 0.
     * <p>
     * If the array is length 1, returns [a, a, a, a, a, a]
     * If the array is length 3, return [a, a, b, b, c, c]
     * If the array is length 6, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 6 that represents the input
     */
    public static int[] validate6NonNegative(int[] data, String paramName) {
        return Arrays.stream(validate6NonNegativeLong(toLongArray(data), paramName)).mapToInt(Math::toIntExact).toArray();
    }


    /**
     * Checks that all values in the array are non-negative.
     *
     * @param data      The array to check.
     * @param paramName The parameter name for the array.
     * @throws IllegalArgumentException If any value in the array is negative.
     */
    public static void validateNonNegativeLong(long[] data, String paramName) {
        for (long i : data) {
            if (i < 0) {
                throw new IllegalArgumentException("Invalid value for parameter " + paramName + ": "
                        + Arrays.toString(data) + ". Values must be non-negative.");
            }
        }
    }

    /**
     * Reformats the input array to a length 6 array and checks that all values >= 0.
     * <p>
     * If the array is length 1, returns [a, a, a, a, a, a]
     * If the array is length 3, return [a, a, b, b, c, c]
     * If the array is length 6, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 6 that represents the input
     */
    public static long[] validate6NonNegativeLong(long[] data, String paramName) {
        validateNonNegativeLong(data, paramName);
        return validate6Long(data, paramName);
    }

    /**
     * Reformats the input array to a length 6 array.
     * <p>
     * If the array is length 1, returns [a, a, a, a, a, a]
     * If the array is length 3, return [a, a, b, b, c, c]
     * If the array is length 6, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return An int array of length 6 that represents the input
     */
    public static int[] validate6(int[] data, String paramName) {
        return Arrays.stream(validate6Long(toLongArray(data), paramName)).mapToInt(Math::toIntExact).toArray();
    }

    /**
     * Reformats the input array to a length 6 array.
     * <p>
     * If the array is length 1, returns [a, a, a, a, a, a]
     * If the array is length 3, return [a, a, b, b, c, c]
     * If the array is length 6, returns the array.
     *
     * @param data      An array
     * @param paramName The param name, for error reporting
     * @return A long array of length 6 that represents the input
     */
    public static long[] validate6Long(long[] data, String paramName) {
        if (data == null) {
            return null;
        }

        Preconditions.checkArgument(data.length == 1 || data.length == 3 || data.length == 6,
                "Need either 1, 3, or 6 %s values, got %s values: %s",
                paramName, data.length, data);
        if (data.length == 1) {
            return new long[]{data[0], data[0], data[0], data[0], data[0], data[0]};
        } else if (data.length == 3) {
            return new long[]{data[0], data[0], data[1], data[1], data[2], data[2]};
        } else {
            return data;
        }

    }

    /**
     * Helper method to convert an int array to a long array.
     *
     * @param intArray The int array to convert.
     * @return The converted long array.
     */
    private static long[] toLongArray(int[] intArray) {
        if (intArray == null) {
            return null;
        }
        return Arrays.stream(intArray).asLongStream().toArray();
    }

}

