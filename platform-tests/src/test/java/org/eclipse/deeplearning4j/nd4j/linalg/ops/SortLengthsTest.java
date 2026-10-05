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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TopK;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The sort family orders arrays of every length, in both directions, in every element type, through views and along
 * dimensions. On CUDA a sort of a power-of-two array of 2^19 elements or more left it unordered (each step kernel
 * handled one element per thread while the grid was capped at 512 x 512 threads); the steps of the arbitrary-length
 * network kept their window arithmetic in int (a hang from 2^30 elements, negative positions before that); an empty
 * array divided by zero in the launch sizing (SIGFPE); and sort_by_value read its two buffers as each other's types,
 * which only shows when the types differ (the positions of a non_max_suppression: INT32 indices, FLOAT scores). On the
 * CPU sort_tad_by_key sorted by the values.
 * <p>
 * The arrays hold distinct values (a random permutation of 0 .. n - 1, scaled to halves for the floating-point types),
 * so every ordering is unique and the pair sorts can be checked by position without a stability contract.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class SortLengthsTest extends BaseNd4jTestWithBackends {

    /** nothing, one, a pair, odd, not a power of two, 2^18 (all the old grid covered), 2^19 and 2^20 (powers of two it left half unsorted) and a length past 2^19 that is no power of two */
    private static final int[] LENGTHS = {0, 1, 2, 3, 1000, 1 << 18, 1 << 19, 1 << 20, 3 * (1 << 18) + 5};

    private static final DataType[] TYPES = {DataType.FLOAT, DataType.DOUBLE, DataType.INT32, DataType.INT64};

    @Override
    public long getTimeoutMilliseconds() {
        return 900000;
    }

    // ------------------------------------------------------------------------------------------------ data

    private static int[] permutation(int n, long seed) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++)
            p[i] = i;
        Random r = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /** The value of rank r (0 = smallest) among n distinct values: exact in every element type used here. */
    private static double valueOfRank(int rank, int n, DataType type) {
        return (rank - n / 2) * (type.isFPType() ? 0.5 : 1.0);
    }

    private static INDArray array(double[] values, DataType type) {
        if (values.length == 0)
            return Nd4j.zeros(type, 0);
        return Nd4j.createFromArray(values).castTo(type);
    }

    private static double[] toDoubles(INDArray a) {
        if (a.length() == 0)
            return new double[0];
        return a.dup('c').data().asDouble();
    }

    private static double[] reversed(double[] a) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++)
            r[i] = a[a.length - 1 - i];
        return r;
    }

    private static double[] ranked(int[] perm, DataType type) {
        double[] v = new double[perm.length];
        for (int i = 0; i < v.length; i++)
            v[i] = valueOfRank(perm[i], perm.length, type);
        return v;
    }

    private static INDArray[] exec(DynamicCustomOp op) {
        INDArray[] out = Nd4j.exec(op);
        Nd4j.getExecutioner().commit();
        return out;
    }

    // ------------------------------------------------------------------------------------------------ sort

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sortOrdersEveryLengthInBothDirections(Nd4jBackend backend) {
        for (DataType type : TYPES) {
            for (int n : LENGTHS) {
                if (n == 0)
                    continue;
                int[] perm = permutation(n, 17L * n + type.ordinal());
                double[] values = ranked(perm, type);
                double[] ascending = values.clone();
                Arrays.sort(ascending);
                for (boolean ascendingOrder : new boolean[]{true, false}) {
                    String what = type + ", length " + n + (ascendingOrder ? ", ascending" : ", descending");
                    INDArray x = array(values, type);
                    INDArray sorted = Nd4j.sort(x, ascendingOrder);
                    Nd4j.getExecutioner().commit();
                    assertArrayEquals(ascendingOrder ? ascending : reversed(ascending), toDoubles(sorted), 0.0, what);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void emptyArraysAreLeftAlone(Nd4jBackend backend) {
        for (DataType type : TYPES) {
            INDArray empty = Nd4j.zeros(type, 0);
            assertEquals(0, Nd4j.sort(empty, true).length(), type + ": ascending");
            assertEquals(0, Nd4j.sort(empty, false).length(), type + ": descending");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sortOpOrdersEveryLengthOutOfPlaceAndInPlace(Nd4jBackend backend) {
        for (int n : LENGTHS) {
            if (n == 0)
                continue;
            int[] perm = permutation(n, 31L * n);
            double[] values = ranked(perm, DataType.FLOAT);
            double[] ascending = values.clone();
            Arrays.sort(ascending);
            for (boolean descending : new boolean[]{false, true}) {
                double[] expected = descending ? reversed(ascending) : ascending;
                String what = "legacy_sort, length " + n + (descending ? ", descending" : ", ascending");

                INDArray x = array(values, DataType.FLOAT);
                INDArray out = Nd4j.createUninitialized(DataType.FLOAT, n);
                exec(DynamicCustomOp.builder("legacy_sort").addInputs(x).addOutputs(out)
                        .addBooleanArguments(descending).build());
                assertArrayEquals(expected, toDoubles(out), 0.0, what);
                assertArrayEquals(values, toDoubles(x), 0.0, what + ": the input changed");

                INDArray inPlace = array(values, DataType.FLOAT);
                exec(DynamicCustomOp.builder("legacy_sort").addInputs(inPlace).addOutputs(inPlace).callInplace(true)
                        .addBooleanArguments(descending).build());
                assertArrayEquals(expected, toDoubles(inPlace), 0.0, what + " in place");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void steppedViewsAreSortedThroughTheirStrides(Nd4jBackend backend) {
        // every second element of a vector twice the length: the other elements are not the sort's to move
        for (int n : new int[]{1 << 19, 3 * (1 << 18) + 5, 1000}) {
            int[] perm = permutation(n, 5L * n);
            double[] values = ranked(perm, DataType.DOUBLE);
            double[] ascending = values.clone();
            Arrays.sort(ascending);
            double[] sentinels = new double[n];
            double[] interleaved = new double[2 * n];
            for (int i = 0; i < n; i++) {
                sentinels[i] = -1.0e6 - i;
                interleaved[2 * i] = values[i];
                interleaved[2 * i + 1] = sentinels[i];
            }
            for (boolean ascendingOrder : new boolean[]{true, false}) {
                INDArray base = Nd4j.createFromArray(interleaved);
                INDArray view = base.get(NDArrayIndex.interval(0, 2, 2L * n));
                Nd4j.sort(view, ascendingOrder);
                Nd4j.getExecutioner().commit();
                String what = "length " + n + (ascendingOrder ? ", ascending" : ", descending");
                assertArrayEquals(ascendingOrder ? ascending : reversed(ascending),
                        toDoubles(base.get(NDArrayIndex.interval(0, 2, 2L * n))), 0.0, what + ": the sorted view");
                assertArrayEquals(sentinels, toDoubles(base.get(NDArrayIndex.interval(1, 2, 2L * n))), 0.0,
                        what + ": the elements between");
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ pairs

    @ParameterizedTest
    @MethodSource("configs")
    public void smallPairSortsUseLogicalPositionsForRowColumnAndStridedVectors(Nd4jBackend backend) {
        for (boolean byKey : new boolean[] {true, false})
            for (boolean descending : new boolean[] {false, true})
                for (int layout = 0; layout < 3; layout++) {
                    INDArray first = array(byKey ? new double[] {3, 1, 2} : new double[] {30, 10, 20}, DataType.INT32);
                    INDArray second = array(byKey ? new double[] {30, 10, 20} : new double[] {3, 1, 2}, DataType.DOUBLE);
                    if (layout == 1) {
                        first = first.reshape(1, 3);
                        second = second.reshape(3, 1);
                    } else if (layout == 2) {
                        first = first.reshape(3, 1);
                        second = second.reshape(1, 3);
                    }
                    INDArray firstParent = Nd4j.createFromArray(-9, 0, -9, 0, -9, 0, -9);
                    INDArray secondParent = Nd4j.createFromArray(-9.0, 0, -9, 0, -9, 0, -9);
                    INDArray firstOut = firstParent.get(NDArrayIndex.interval(1, 2, 7)).reshape(first.shape());
                    INDArray secondOut = secondParent.get(NDArrayIndex.interval(1, 2, 7)).reshape(second.shape());
                    exec(DynamicCustomOp.builder(byKey ? "legacy_sort_by_key" : "legacy_sort_by_value")
                            .addInputs(first, second).addOutputs(firstOut, secondOut).addBooleanArguments(descending).build());
                    double[] ordered = descending ? new double[] {3, 2, 1} : new double[] {1, 2, 3};
                    double[] paired = descending ? new double[] {30, 20, 10} : new double[] {10, 20, 30};
                    assertArrayEquals(byKey ? ordered : paired, toDoubles(firstOut), 0.0);
                    assertArrayEquals(byKey ? paired : ordered, toDoubles(secondOut), 0.0);
                    assertArrayEquals(byKey ? ordered : paired,
                            toDoubles(firstParent.get(NDArrayIndex.interval(1, 2, 7))), 0.0);
                    assertArrayEquals(byKey ? paired : ordered,
                            toDoubles(secondParent.get(NDArrayIndex.interval(1, 2, 7))), 0.0);
                    assertArrayEquals(new double[] {-9, -9, -9, -9},
                            toDoubles(firstParent.get(NDArrayIndex.interval(0, 2, 7))), 0.0);
                    assertArrayEquals(new double[] {-9, -9, -9, -9},
                            toDoubles(secondParent.get(NDArrayIndex.interval(0, 2, 7))), 0.0);
                }
    }

    /** keys with the rank of position i in perm[i]; values: the positions. Sorted by key ascending, the value at rank r is the position that held it. */
    private static double[] positionsByRank(int[] perm) {
        double[] byRank = new double[perm.length];
        for (int i = 0; i < perm.length; i++)
            byRank[perm[i]] = i;
        return byRank;
    }

    private static final DataType[][] PAIR_TYPES = {
            {DataType.FLOAT, DataType.INT32},
            {DataType.INT32, DataType.DOUBLE},
            {DataType.DOUBLE, DataType.INT64},
            {DataType.INT64, DataType.FLOAT}};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sortByKeyOrdersTheValuesWithTheKeys(Nd4jBackend backend) {
        for (DataType[] types : PAIR_TYPES) {
            DataType keyType = types[0];
            DataType valueType = types[1];
            for (int n : LENGTHS) {
                if (n == 0)
                    continue;
                int[] perm = permutation(n, 13L * n + keyType.ordinal());
                double[] keys = ranked(perm, keyType);
                double[] positions = new double[n];
                for (int i = 0; i < n; i++)
                    positions[i] = i;
                double[] byRank = positionsByRank(perm);
                double[] keysAscending = keys.clone();
                Arrays.sort(keysAscending);
                for (boolean descending : new boolean[]{false, true}) {
                    String what = "legacy_sort_by_key " + keyType + "/" + valueType + ", length " + n
                            + (descending ? ", descending" : ", ascending");
                    INDArray keysOut = Nd4j.createUninitialized(keyType, n);
                    INDArray valuesOut = Nd4j.createUninitialized(valueType, n);
                    exec(DynamicCustomOp.builder("legacy_sort_by_key")
                            .addInputs(array(keys, keyType), array(positions, valueType))
                            .addOutputs(keysOut, valuesOut).addBooleanArguments(descending).build());
                    assertArrayEquals(descending ? reversed(keysAscending) : keysAscending, toDoubles(keysOut), 0.0,
                            what + ": keys");
                    assertArrayEquals(descending ? reversed(byRank) : byRank, toDoubles(valuesOut), 0.0,
                            what + ": values");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sortByValueOrdersTheKeysWithTheValues(Nd4jBackend backend) {
        // the second array decides; the first holds the positions (INT32 or INT64: not the type of the values, whose
        // bit patterns must not be read as integers: negative FLOAT values order opposite to their bits)
        for (DataType[] types : PAIR_TYPES) {
            DataType positionType = types[1];
            DataType valueType = types[0];
            for (int n : LENGTHS) {
                if (n == 0)
                    continue;
                int[] perm = permutation(n, 19L * n + valueType.ordinal());
                double[] values = ranked(perm, valueType);
                double[] positions = new double[n];
                for (int i = 0; i < n; i++)
                    positions[i] = i;
                double[] byRank = positionsByRank(perm);
                double[] valuesAscending = values.clone();
                Arrays.sort(valuesAscending);
                for (boolean descending : new boolean[]{false, true}) {
                    String what = "legacy_sort_by_value " + positionType + "/" + valueType + ", length " + n
                            + (descending ? ", descending" : ", ascending");
                    INDArray positionsOut = Nd4j.createUninitialized(positionType, n);
                    INDArray valuesOut = Nd4j.createUninitialized(valueType, n);
                    exec(DynamicCustomOp.builder("legacy_sort_by_value")
                            .addInputs(array(positions, positionType), array(values, valueType))
                            .addOutputs(positionsOut, valuesOut).addBooleanArguments(descending).build());
                    assertArrayEquals(descending ? reversed(valuesAscending) : valuesAscending, toDoubles(valuesOut), 0.0,
                            what + ": values");
                    assertArrayEquals(descending ? reversed(byRank) : byRank, toDoubles(positionsOut), 0.0,
                            what + ": positions");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void pairSortsOfSteppedViewsLeaveTheElementsBetween(Nd4jBackend backend) {
        // the outputs are every second element of arrays twice as long: the sort moves those and no others
        for (int n : new int[]{1000, 1 << 19}) {
            int[] perm = permutation(n, 29L * n);
            double[] keys = ranked(perm, DataType.FLOAT);
            double[] positions = new double[n];
            double[] sentinels = new double[n];
            for (int i = 0; i < n; i++) {
                positions[i] = i;
                sentinels[i] = -1.0e6 - i;
            }
            double[] byRank = positionsByRank(perm);
            double[] keysAscending = keys.clone();
            Arrays.sort(keysAscending);
            for (boolean byKey : new boolean[]{true, false}) {
                // by key: (keys, positions) sorted by the first; by value: (positions, keys) sorted by the second
                double[] firstIn = byKey ? keys : positions;
                double[] secondIn = byKey ? positions : keys;
                DataType firstType = byKey ? DataType.FLOAT : DataType.INT32;
                DataType secondType = byKey ? DataType.INT32 : DataType.FLOAT;
                double[] interleavedFirst = new double[2 * n];
                double[] interleavedSecond = new double[2 * n];
                for (int i = 0; i < n; i++) {
                    interleavedFirst[2 * i + 1] = sentinels[i];
                    interleavedSecond[2 * i + 1] = sentinels[i];
                }
                INDArray wideFirst = Nd4j.createFromArray(interleavedFirst).castTo(firstType);
                INDArray wideSecond = Nd4j.createFromArray(interleavedSecond).castTo(secondType);
                INDArray firstOut = wideFirst.get(NDArrayIndex.interval(0, 2, 2L * n));
                INDArray secondOut = wideSecond.get(NDArrayIndex.interval(0, 2, 2L * n));
                String what = (byKey ? "legacy_sort_by_key" : "legacy_sort_by_value") + ", length " + n;
                exec(DynamicCustomOp.builder(byKey ? "legacy_sort_by_key" : "legacy_sort_by_value")
                        .addInputs(array(firstIn, firstType), array(secondIn, secondType))
                        .addOutputs(firstOut, secondOut).addBooleanArguments(false).build());
                double[] expectedKeys = keysAscending;
                double[] expectedPositions = byRank;
                assertArrayEquals(byKey ? expectedKeys : expectedPositions,
                        toDoubles(wideFirst.get(NDArrayIndex.interval(0, 2, 2L * n))), 0.0, what + ": first");
                assertArrayEquals(byKey ? expectedPositions : expectedKeys,
                        toDoubles(wideSecond.get(NDArrayIndex.interval(0, 2, 2L * n))), 0.0, what + ": second");
                assertArrayEquals(sentinels, toDoubles(wideFirst.get(NDArrayIndex.interval(1, 2, 2L * n))), 0.0,
                        what + ": between the first");
                assertArrayEquals(sentinels, toDoubles(wideSecond.get(NDArrayIndex.interval(1, 2, 2L * n))), 0.0,
                        what + ": between the second");
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ along dimensions

    /** rows x cols, every row a permutation of its own */
    private static int[][] rowPermutations(int rows, int cols, long seed) {
        int[][] perms = new int[rows][];
        for (int r = 0; r < rows; r++)
            perms[r] = permutation(cols, seed + 1009L * r);
        return perms;
    }

    private static INDArray matrix(int[][] perms, DataType type) {
        int rows = perms.length;
        int cols = perms[0].length;
        double[] flat = new double[rows * cols];
        for (int r = 0; r < rows; r++)
            System.arraycopy(ranked(perms[r], type), 0, flat, r * cols, cols);
        return Nd4j.createFromArray(flat).castTo(type).reshape('c', rows, cols);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sortAlongADimensionOrdersEveryRowAndColumn(Nd4jBackend backend) {
        // 600 doubles fit the block's shared memory, 5000 do not; 700 rows are a grid of TADs
        int[][] shapes = {{3, 600}, {2, 5000}, {700, 40}};
        for (int[] shape : shapes) {
            int rows = shape[0];
            int cols = shape[1];
            int[][] perms = rowPermutations(rows, cols, 77L * rows + cols);
            for (DataType type : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.INT32}) {
                for (boolean ascendingOrder : new boolean[]{true, false}) {
                    String what = type + " " + rows + " x " + cols + (ascendingOrder ? ", ascending" : ", descending");
                    INDArray sortedRows = Nd4j.sort(matrix(perms, type), 1, ascendingOrder);
                    Nd4j.getExecutioner().commit();
                    for (int r = 0; r < rows; r++) {
                        double[] expected = ranked(perms[r], type);
                        Arrays.sort(expected);
                        assertArrayEquals(ascendingOrder ? expected : reversed(expected),
                                toDoubles(sortedRows.getRow(r)), 0.0, what + ": row " + r);
                    }
                }
            }
        }
        // along the columns: the transpose of the rows' result
        int[][] perms = rowPermutations(4, 1000, 5);
        INDArray byColumns = Nd4j.sort(matrix(perms, DataType.DOUBLE).transpose().dup('c'), 0, true);
        Nd4j.getExecutioner().commit();
        for (int r = 0; r < 4; r++) {
            double[] expected = ranked(perms[r], DataType.DOUBLE);
            Arrays.sort(expected);
            assertArrayEquals(expected, toDoubles(byColumns.getColumn(r)), 0.0, "column " + r);
        }
        // F order: the elements of a row are a stride apart
        int[][] fPerms = rowPermutations(5, 700, 3);
        INDArray fRows = Nd4j.sort(matrix(fPerms, DataType.DOUBLE).dup('f'), 1, true);
        Nd4j.getExecutioner().commit();
        for (int r = 0; r < 5; r++) {
            double[] expected = ranked(fPerms[r], DataType.DOUBLE);
            Arrays.sort(expected);
            assertArrayEquals(expected, toDoubles(fRows.getRow(r)), 0.0, "F order, row " + r);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sortTadByKeyAndByValueOrderEveryRow(Nd4jBackend backend) {
        int[][] shapes = {{4, 300}, {2, 5000}, {50, 7}};
        for (int[] shape : shapes) {
            int rows = shape[0];
            int cols = shape[1];
            int[][] perms = rowPermutations(rows, cols, 41L * rows + cols);
            double[] columnIndex = new double[rows * cols];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    columnIndex[r * cols + c] = c;
            for (boolean descending : new boolean[]{false, true}) {
                String what = rows + " x " + cols + (descending ? ", descending" : ", ascending");

                // keys decide: the second array holds each key's column and follows it
                INDArray keysOut = Nd4j.createUninitialized(DataType.FLOAT, rows, cols);
                INDArray columnsOut = Nd4j.createUninitialized(DataType.INT32, rows, cols);
                exec(DynamicCustomOp.builder("legacy_sort_tad_by_key")
                        .addInputs(matrix(perms, DataType.FLOAT),
                                Nd4j.createFromArray(columnIndex).castTo(DataType.INT32).reshape('c', rows, cols))
                        .addOutputs(keysOut, columnsOut).addIntegerArguments(1).addBooleanArguments(descending).build());

                // values decide: the first array holds each value's column and follows it
                INDArray columnsOut2 = Nd4j.createUninitialized(DataType.INT32, rows, cols);
                INDArray valuesOut = Nd4j.createUninitialized(DataType.FLOAT, rows, cols);
                exec(DynamicCustomOp.builder("legacy_sort_tad_by_value")
                        .addInputs(Nd4j.createFromArray(columnIndex).castTo(DataType.INT32).reshape('c', rows, cols),
                                matrix(perms, DataType.FLOAT))
                        .addOutputs(columnsOut2, valuesOut).addIntegerArguments(1).addBooleanArguments(descending).build());

                for (int r = 0; r < rows; r++) {
                    double[] byRank = positionsByRank(perms[r]);
                    double[] ascending = new double[cols];
                    for (int c = 0; c < cols; c++)
                        ascending[c] = valueOfRank(c, cols, DataType.FLOAT);
                    double[] expectedValues = descending ? reversed(ascending) : ascending;
                    double[] expectedColumns = descending ? reversed(byRank) : byRank;
                    assertArrayEquals(expectedValues, toDoubles(keysOut.getRow(r)), 0.0, what + ": keys, row " + r);
                    assertArrayEquals(expectedColumns, toDoubles(columnsOut.getRow(r)), 0.0,
                            what + ": columns by key, row " + r);
                    assertArrayEquals(expectedValues, toDoubles(valuesOut.getRow(r)), 0.0, what + ": values, row " + r);
                    assertArrayEquals(expectedColumns, toDoubles(columnsOut2.getRow(r)), 0.0,
                            what + ": columns by value, row " + r);
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ users of the sort

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nthElementOfLongVectorsAndWideRows(Nd4jBackend backend) {
        for (int n : new int[]{1000, 1 << 19, 3 * (1 << 18) + 5}) {
            int[] perm = permutation(n, 3L * n);
            INDArray x = array(ranked(perm, DataType.FLOAT), DataType.FLOAT);
            INDArray before = x.dup();
            for (boolean reverse : new boolean[]{false, true}) {
                for (int element : new int[]{0, 1, n / 2, n - 1}) {
                    INDArray[] out = exec(DynamicCustomOp.builder("nth_element").addInputs(x, Nd4j.scalar(element))
                            .addIntegerArguments(reverse ? 1 : 0).build());
                    double expected = valueOfRank(reverse ? n - 1 - element : element, n, DataType.FLOAT);
                    assertEquals(expected, out[0].getDouble(0), 0.0,
                            "nth_element of " + n + " elements, n = " + element + ", reverse = " + reverse);
                }
            }
            assertArrayEquals(toDoubles(before), toDoubles(x), 0.0, "nth_element changed its input");
        }
        // rows: sorted along the last axis, 5000 doubles a row do not fit the block's shared memory
        int[][] perms = rowPermutations(3, 5000, 9);
        INDArray rows = matrix(perms, DataType.DOUBLE);
        for (boolean reverse : new boolean[]{false, true}) {
            INDArray[] out = exec(DynamicCustomOp.builder("nth_element").addInputs(rows, Nd4j.scalar(1234))
                    .addIntegerArguments(reverse ? 1 : 0).build());
            double[] found = toDoubles(out[0]);
            for (int r = 0; r < 3; r++)
                assertEquals(valueOfRank(reverse ? 5000 - 1 - 1234 : 1234, 5000, DataType.DOUBLE), found[r], 0.0,
                        "row " + r + ", reverse = " + reverse);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void topKOfLongVectors(Nd4jBackend backend) {
        int k = 8;
        for (int n : new int[]{1000, (1 << 19) + 1, 3 * (1 << 18) + 5}) {
            int[] perm = permutation(n, 23L * n);
            double[] byRank = positionsByRank(perm);
            INDArray x = array(ranked(perm, DataType.FLOAT), DataType.FLOAT);
            INDArray[] out = exec(new TopK(x, k, true));
            double[] values = toDoubles(out[0]);
            double[] indices = toDoubles(out[1]);
            for (int j = 0; j < k; j++) {
                assertEquals(valueOfRank(n - 1 - j, n, DataType.FLOAT), values[j], 0.0,
                        "top_k of " + n + " elements: value " + j);
                assertEquals(byRank[n - 1 - j], indices[j], 0.0, "top_k of " + n + " elements: index " + j);
            }
        }
    }
}
