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
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.shape.SequenceMask;
import org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentSoftmax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The segment ops (segment_*, unsorted_segment_*, their backprops, segment_softmax) and sequence_mask give the Java
 * reference result on every backend, at the sizes and for the layouts the CUDA launches broke on: one reduction per
 * segment, whatever the number of ids (1025, one block of threads before), the number of classes (767, 1535, 1536:
 * dynamic shared memory per class), the size of the gradient (1024) and of the data (1025 elements); rows x columns
 * and vector inputs; a segment no row maps to (sorted: sum, mean, max, min 0 and prod 1; unsorted: sum, mean, sqrt_n 0,
 * prod 1, max the lowest and min the highest value of the type); every numeric type (HALF/BFLOAT16 sums accumulate in
 * FLOAT, integer sums and products wrap like the narrow type, the maximum of unsigned zeros is zero, a mean is one
 * division of the finished sum, integer means are exact); NaN; ids of any integer type and layout; strided, offset,
 * transposed and F ordered inputs and caller supplied output views; ids that are negative, equal to the number of
 * classes or (sorted ops) out of order are rejected; the maximum and minimum backprops give the gradient to the elements
 * equal to the extremum and to no other.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class SegmentSemanticsTest extends BaseNd4jTestWithBackends {

    private enum Op { SUM, MEAN, MAX, MIN, PROD, SQRT_N }

    private static final DataType[] NUMERIC = {DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT16, DataType.BFLOAT16,
            DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.UINT8, DataType.UINT16,
            DataType.UINT32, DataType.UINT64};
    private static final DataType[] FLOATS = {DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT16, DataType.BFLOAT16};
    private static final long[][] TAILS = {{}, {3}, {2, 2}};

    // ------------------------------------------------------------------------------------------------ helpers

    private static boolean isUnsigned(DataType type) {
        return type == DataType.UINT8 || type == DataType.UINT16 || type == DataType.UINT32
                || type == DataType.UINT64;
    }

    private static boolean isFloating(DataType type) {
        return type == DataType.FLOAT || type == DataType.DOUBLE || type == DataType.FLOAT16
                || type == DataType.BFLOAT16;
    }

    private static String opName(Op op, boolean sorted) {
        String base;
        switch (op) {
            case SUM: base = "sum"; break;
            case MEAN: base = "mean"; break;
            case MAX: base = "max"; break;
            case MIN: base = "min"; break;
            case PROD: base = "prod"; break;
            default: base = "sqrt_n"; break;
        }
        return (sorted ? "segment_" : "unsorted_segment_") + base;
    }

    /** numeric_limits<type>::lowest() as a double (the long types round to the nearest double). */
    private static double lowest(DataType type) {
        switch (type) {
            case FLOAT: return -Float.MAX_VALUE;
            case DOUBLE: return -Double.MAX_VALUE;
            case HALF: return -65504.0;
            case BFLOAT16: return -3.3895313892515355E38;
            case BYTE: return Byte.MIN_VALUE;
            case SHORT: return Short.MIN_VALUE;
            case INT: return Integer.MIN_VALUE;
            case LONG: return (double) Long.MIN_VALUE;
            default: return 0.0;
        }
    }

    /** numeric_limits<type>::max() as a double. */
    private static double highest(DataType type) {
        switch (type) {
            case FLOAT: return Float.MAX_VALUE;
            case DOUBLE: return Double.MAX_VALUE;
            case HALF: return 65504.0;
            case BFLOAT16: return 3.3895313892515355E38;
            case BYTE: return Byte.MAX_VALUE;
            case SHORT: return Short.MAX_VALUE;
            case INT: return Integer.MAX_VALUE;
            case LONG: return (double) Long.MAX_VALUE;
            case UBYTE: return 255.0;
            case UINT16: return 65535.0;
            case UINT32: return 4294967295.0;
            default: return 18446744073709551615.0;
        }
    }

    /** The value of a segment no row maps to. */
    private static double emptyValue(Op op, boolean sorted, DataType type) {
        if (op == Op.PROD) return 1.0;
        if (!sorted && op == Op.MAX) return lowest(type);
        if (!sorted && op == Op.MIN) return highest(type);
        return 0.0;
    }

    private static INDArray idArray(boolean wide, long[] ids) {
        if (wide) return Nd4j.createFromArray(ids);
        int[] narrow = new int[ids.length];
        for (int i = 0; i < ids.length; i++) narrow[i] = (int) ids[i];
        return Nd4j.createFromArray(narrow);
    }

    private static INDArray array(DataType type, double[][] rows, long[] tail) {
        int n = rows.length;
        int k = n == 0 ? 0 : rows[0].length;
        double[] flat = new double[n * k];
        for (int i = 0; i < n; i++) System.arraycopy(rows[i], 0, flat, i * k, k);
        long[] shape = new long[1 + tail.length];
        shape[0] = n;
        System.arraycopy(tail, 0, shape, 1, tail.length);
        return Nd4j.createFromArray(flat).reshape(shape).castTo(type);
    }

    private static int tailLength(long[] tail) {
        int k = 1;
        for (long d : tail) k *= (int) d;
        return k;
    }

    private static double[] values(INDArray array) {
        return array.castTo(DataType.DOUBLE).dup('c').data().asDouble();
    }

    private static INDArray exec(String op, long[] iArgs, INDArray... inputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder(op).addInputs(inputs);
        if (iArgs.length > 0) builder.addIntegerArguments(iArgs);
        return Nd4j.exec(builder.build())[0];
    }

    private static INDArray[] execAll(String op, long[] iArgs, INDArray... inputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder(op).addInputs(inputs);
        if (iArgs.length > 0) builder.addIntegerArguments(iArgs);
        return Nd4j.exec(builder.build());
    }

    private static INDArray run(Op op, boolean sorted, INDArray x, INDArray ids, int classes) {
        return sorted ? exec(opName(op, true), new long[0], x, ids)
                : exec(opName(op, false), new long[]{classes}, x, ids);
    }

    /** One value of a data row: small integers, exact in every type. */
    private static double datum(Random rnd, DataType type, Op op) {
        if (op == Op.PROD) return isUnsigned(type) ? 1 + rnd.nextInt(2) : new double[]{-2, -1, 1, 2}[rnd.nextInt(4)];
        return isUnsigned(type) ? rnd.nextInt(6) : rnd.nextInt(9) - 4;
    }

    private static double[][] data(Random rnd, DataType type, Op op, int n, int k) {
        double[][] x = new double[n][k];
        for (int i = 0; i < n; i++) for (int j = 0; j < k; j++) x[i][j] = datum(rnd, type, op);
        return x;
    }

    /** Sorted ids with gaps (a leading gap too); the number of classes is the last id + 1. */
    private static long[] sortedIds(Random rnd, int n) {
        long[] ids = new long[n];
        long current = rnd.nextInt(3);
        for (int i = 0; i < n; i++) {
            ids[i] = current;
            if (rnd.nextInt(3) == 0) current += 1 + rnd.nextInt(3);
        }
        return ids;
    }

    private static long[] randomIds(Random rnd, int n, int classes) {
        long[] ids = new long[n];
        for (int i = 0; i < n; i++) ids[i] = rnd.nextInt(classes);
        return ids;
    }

    /** An integer sum or product wraps like the narrow type (the values here are small: the long holds them). */
    private static double wrap(double value, DataType type) {
        long l = (long) value;
        switch (type) {
            case BYTE: return (byte) l;
            case SHORT: return (short) l;
            case INT: return (int) l;
            case UBYTE: return l & 0xFFL;
            case UINT16: return l & 0xFFFFL;
            case UINT32: return l & 0xFFFFFFFFL;
            default: return value;
        }
    }

    /** The reference of a segment op over double rows, with the semantics of segment_semantics.h. */
    private static double[][] reference(Op op, boolean sorted, double[][] x, long[] ids, int classes, DataType type) {
        int n = x.length;
        int k = n == 0 ? 0 : x[0].length;
        double[][] out = new double[classes][k];
        for (int s = 0; s < classes; s++) {
            List<double[]> rows = new ArrayList<>();
            for (int i = 0; i < n; i++) if (ids[i] == s) rows.add(x[i]);
            for (int e = 0; e < k; e++) {
                if (rows.isEmpty()) {
                    out[s][e] = emptyValue(op, sorted, type);
                    continue;
                }
                double acc;
                switch (op) {
                    case PROD:
                        acc = 1;
                        for (double[] row : rows) acc *= row[e];
                        break;
                    case MAX:
                        acc = Double.NEGATIVE_INFINITY;
                        for (double[] row : rows) acc = Math.max(acc, row[e]);
                        break;
                    case MIN:
                        acc = Double.POSITIVE_INFINITY;
                        for (double[] row : rows) acc = Math.min(acc, row[e]);
                        break;
                    default:
                        acc = 0;
                        for (double[] row : rows) acc += row[e];
                        if (op == Op.MEAN) acc /= rows.size();
                        if (op == Op.SQRT_N) acc /= Math.sqrt(rows.size());
                        break;
                }
                out[s][e] = (op == Op.SUM || op == Op.PROD) && !isFloating(type) ? wrap(acc, type) : acc;
            }
        }
        return out;
    }

    private static double tolerance(Op op, DataType type) {
        if (op != Op.MEAN && op != Op.SQRT_N) return 0.0;
        switch (type) {
            case DOUBLE: return 1e-12;
            case FLOAT: return 2e-6;
            case HALF: return 2e-3;
            default: return 1.6e-2;
        }
    }

    private static void assertRows(String what, Op op, DataType outType, double[][] expected, INDArray actual) {
        int rows = expected.length;
        int k = rows == 0 ? 0 : expected[0].length;
        double[] got = values(actual);
        assertEquals((long) rows * k, got.length, what + ": number of elements");
        assertEquals(rows, (int) actual.size(0), what + ": rows");
        double tol = tolerance(op, outType);
        for (int s = 0; s < rows; s++) {
            for (int e = 0; e < k; e++) {
                double want = expected[s][e];
                double have = got[s * k + e];
                if (Double.isNaN(want)) {
                    assertTrue(Double.isNaN(have), what + ": [" + s + "," + e + "] expected NaN, got " + have);
                } else {
                    assertTrue(Math.abs(have - want) <= tol * Math.max(1.0, Math.abs(want)),
                            what + ": [" + s + "," + e + "] expected " + want + ", got " + have);
                }
            }
        }
    }

    private static DataType outputType(Op op, DataType inType) {
        return (op == Op.MEAN || op == Op.SQRT_N) && !isFloating(inType) ? DataType.FLOAT : inType;
    }

    private static boolean supports(Op op, boolean sorted, DataType type) {
        if (op == Op.SQRT_N) return !sorted && isFloating(type);
        if (op == Op.MEAN) return isFloating(type) || type == DataType.INT32 || type == DataType.UINT8;
        return true;
    }

    // ------------------------------------------------------------------------------------------- forward ops

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sortedAndUnsortedOpsMatchTheReference(Nd4jBackend backend) {
        Random rnd = new Random(20251004L);
        for (boolean sorted : new boolean[]{true, false}) {
            for (Op op : Op.values()) {
                for (DataType type : NUMERIC) {
                    if (!supports(op, sorted, type)) continue;
                    for (long[] tail : TAILS) {
                        for (int iteration = 0; iteration < 3; iteration++) {
                            int n = 1 + rnd.nextInt(40);
                            int k = tailLength(tail);
                            double[][] x = data(rnd, type, op, n, k);
                            long[] ids;
                            int classes;
                            if (sorted) {
                                ids = sortedIds(rnd, n);
                                classes = (int) ids[n - 1] + 1;
                            } else {
                                classes = 1 + rnd.nextInt(9);
                                ids = randomIds(rnd, n, classes);
                            }
                            INDArray in = array(type, x, tail);
                            INDArray out = run(op, sorted, in, idArray(iteration % 2 == 1, ids), classes);
                            String what = opName(op, sorted) + " " + type + " tail " + Arrays.toString(tail) + " n "
                                    + n + " classes " + classes;
                            assertEquals(outputType(op, type), out.dataType(), what + ": output type");
                            assertRows(what, op, outputType(op, type),
                                    reference(op, sorted, x, ids, classes, type), out);
                        }
                    }
                }
            }
        }
    }

    /** The values a segment no row maps to has, spelled out (a leading, a middle and a trailing class). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void emptySegmentsHaveTheTensorFlowValues(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(1f, 2f, 3f);
        // sorted: classes 0 .. 3, class 0 and 2 have no row
        INDArray sortedIds = Nd4j.createFromArray(1, 1, 3);
        assertArrayEquals(new float[]{0, 2, 0, 3}, exec("segment_max", new long[0], x, sortedIds).data().asFloat(), 0f);
        assertArrayEquals(new float[]{0, 1, 0, 3}, exec("segment_min", new long[0], x, sortedIds).data().asFloat(), 0f);
        assertArrayEquals(new float[]{0, 3, 0, 3}, exec("segment_sum", new long[0], x, sortedIds).data().asFloat(), 0f);
        assertArrayEquals(new float[]{0, 1.5f, 0, 3}, exec("segment_mean", new long[0], x, sortedIds).data().asFloat(),
                0f);
        assertArrayEquals(new float[]{1, 2, 1, 3}, exec("segment_prod", new long[0], x, sortedIds).data().asFloat(),
                0f);
        // unsorted with 5 classes: 0, 2 and 4 have no row
        INDArray ids = Nd4j.createFromArray(1, 1, 3);
        long[] five = {5};
        assertArrayEquals(new float[]{-Float.MAX_VALUE, 2, -Float.MAX_VALUE, 3, -Float.MAX_VALUE},
                exec("unsorted_segment_max", five, x, ids).data().asFloat(), 0f);
        assertArrayEquals(new float[]{Float.MAX_VALUE, 1, Float.MAX_VALUE, 3, Float.MAX_VALUE},
                exec("unsorted_segment_min", five, x, ids).data().asFloat(), 0f);
        assertArrayEquals(new float[]{0, 3, 0, 3, 0}, exec("unsorted_segment_sum", five, x, ids).data().asFloat(), 0f);
        assertArrayEquals(new float[]{0, 1.5f, 0, 3, 0}, exec("unsorted_segment_mean", five, x, ids).data().asFloat(),
                0f);
        assertArrayEquals(new float[]{1, 2, 1, 3, 1}, exec("unsorted_segment_prod", five, x, ids).data().asFloat(), 0f);
        float[] sqrtN = exec("unsorted_segment_sqrt_n", five, x, ids).data().asFloat();
        assertArrayEquals(new float[]{0, (float) (3 / Math.sqrt(2)), 0, 3, 0}, sqrtN, 1e-6f);
        // integer types: the lowest and the highest value of the type (not -max, which is one off), unsigned zero
        INDArray ints = Nd4j.createFromArray(1, 2, 3);
        assertArrayEquals(new int[]{Integer.MIN_VALUE, 2, Integer.MIN_VALUE, 3, Integer.MIN_VALUE},
                exec("unsorted_segment_max", five, ints, ids).data().asInt());
        assertArrayEquals(new int[]{Integer.MAX_VALUE, 1, Integer.MAX_VALUE, 3, Integer.MAX_VALUE},
                exec("unsorted_segment_min", five, ints, ids).data().asInt());
        INDArray bytes = ints.castTo(DataType.UINT8);
        assertArrayEquals(new double[]{0, 2, 0, 3, 0},
                values(exec("unsorted_segment_max", five, bytes, ids)), 0.0);
        assertArrayEquals(new double[]{255, 1, 255, 3, 255},
                values(exec("unsorted_segment_min", five, bytes, ids)), 0.0);
        // sorted unsigned: an empty class is 0 for max and min (a "neutral" of 1 used to show for the maximum)
        assertArrayEquals(new double[]{0, 2, 0, 3}, values(exec("segment_max", new long[0], bytes, sortedIds)), 0.0);
        assertArrayEquals(new double[]{0, 1, 0, 3}, values(exec("segment_min", new long[0], bytes, sortedIds)), 0.0);
    }

    /** Unsigned zeros: the maximum is zero (the old neutral value of a maximum was 1 for the unsigned types). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void maximumOfUnsignedZeros(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.UINT8, DataType.UINT16, DataType.UINT32, DataType.UINT64}) {
            INDArray zeros = Nd4j.zeros(DataType.INT32, 6, 3).castTo(type);
            INDArray ids = Nd4j.createFromArray(0, 0, 1, 1, 1, 3);
            for (double v : values(exec("segment_max", new long[0], zeros, ids)))
                assertEquals(0.0, v, 0.0, "segment_max of " + type + " zeros");
            INDArray unsortedIds = Nd4j.createFromArray(3, 0, 1, 1, 0, 3);
            for (double v : values(exec("unsorted_segment_max", new long[]{5}, zeros, unsortedIds)))
                assertEquals(0.0, v, 0.0, "unsorted_segment_max of " + type + " zeros");
            INDArray mixed = Nd4j.createFromArray(0, 0, 0, 5, 0, 0).castTo(type);
            assertArrayEquals(new double[]{0, 5, 0}, values(exec("segment_max", new long[0], mixed,
                    Nd4j.createFromArray(0, 0, 1, 1, 2, 2))), 0.0, "segment_max " + type);
        }
    }

    /** More ids than a block has threads, more classes than the old launches' shared memory allowed. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void manyIdsAndManyClasses(Nd4jBackend backend) {
        Random rnd = new Random(31L);
        int[][] sizes = {{1025, 767}, {1025, 1535}, {1025, 1536}, {3000, 1536}, {5000, 3}, {2049, 1}, {4097, 2048}};
        for (int[] size : sizes) {
            int n = size[0];
            int classes = size[1];
            for (long[] tail : new long[][]{{}, {2}}) {
                int k = tailLength(tail);
                for (boolean sorted : new boolean[]{true, false}) {
                    long[] ids = new long[n];
                    if (sorted) {
                        for (int i = 0; i < n; i++)
                            ids[i] = n == 1 ? 0 : (long) i * (classes - 1) / (n - 1);
                    } else {
                        ids = randomIds(rnd, n, classes);
                    }
                    int c = sorted ? (int) ids[n - 1] + 1 : classes;
                    for (Op op : new Op[]{Op.SUM, Op.MEAN, Op.MAX, Op.MIN}) {
                        double[][] x = data(rnd, DataType.FLOAT, op, n, k);
                        INDArray in = array(DataType.FLOAT, x, tail);
                        INDArray out = run(op, sorted, in, idArray(n % 2 == 1, ids), c);
                        assertRows(opName(op, sorted) + " n " + n + " classes " + classes + " tail "
                                + Arrays.toString(tail), op, DataType.FLOAT,
                                reference(op, sorted, x, ids, c, DataType.FLOAT), out);
                    }
                }
            }
        }
        // sqrt_n and prod at the same sizes (prod: values 1 and 2 only, the segments are short)
        int n = 1025;
        int classes = 1536;
        long[] ids = randomIds(rnd, n, classes);
        double[][] x = data(rnd, DataType.FLOAT, Op.SUM, n, 1);
        assertRows("unsorted_segment_sqrt_n 1025 x 1536", Op.SQRT_N, DataType.FLOAT,
                reference(Op.SQRT_N, false, x, ids, classes, DataType.FLOAT),
                run(Op.SQRT_N, false, array(DataType.FLOAT, x, new long[0]), idArray(true, ids), classes));
        double[][] ones = data(rnd, DataType.UINT8, Op.PROD, n, 1);
        assertRows("unsorted_segment_prod 1025 x 1536", Op.PROD, DataType.FLOAT,
                reference(Op.PROD, false, ones, ids, classes, DataType.FLOAT),
                run(Op.PROD, false, array(DataType.FLOAT, ones, new long[0]), idArray(false, ids), classes));
    }

    // -------------------------------------------------------------------------------- accumulation precision

    /** HALF and BFLOAT16 sums accumulate in FLOAT: a half-precision running sum stalls at 256 / 2048. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void lowPrecisionSumsAccumulateInFloat(Nd4jBackend backend) {
        for (boolean sorted : new boolean[]{true, false}) {
            INDArray bf16 = Nd4j.ones(DataType.BFLOAT16, 1000, 2);
            INDArray ids = Nd4j.zeros(DataType.INT32, 1000);
            INDArray sum = run(Op.SUM, sorted, bf16, ids, 1);
            assertEquals(1000.0, sum.getDouble(0, 0), 0.0, "bfloat16 sum, sorted " + sorted);
            assertEquals(1000.0, sum.getDouble(0, 1), 0.0, "bfloat16 sum, sorted " + sorted);
            INDArray half = Nd4j.ones(DataType.FLOAT16, 3000, 3);
            INDArray ids3 = Nd4j.zeros(DataType.INT32, 3000);
            INDArray halfSum = run(Op.SUM, sorted, half, ids3, 1);
            assertEquals(3000.0, halfSum.getDouble(0, 0), 0.0, "half sum, sorted " + sorted);
            // a vector, many rows in few classes (contended accumulators)
            INDArray vec = Nd4j.ones(DataType.BFLOAT16, 4000);
            INDArray vecIds = Nd4j.createFromArray(sorted ? sortedTwoClasses(4000) : alternating(4000));
            INDArray vecSum = run(Op.SUM, sorted, vec, vecIds, 2);
            assertEquals(2000.0, vecSum.getDouble(0), 0.0, "bfloat16 vector sum, sorted " + sorted);
            assertEquals(2000.0, vecSum.getDouble(1), 0.0, "bfloat16 vector sum, sorted " + sorted);
        }
    }

    private static int[] sortedTwoClasses(int n) {
        int[] ids = new int[n];
        for (int i = n / 2; i < n; i++) ids[i] = 1;
        return ids;
    }

    private static int[] alternating(int n) {
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) ids[i] = i % 2;
        return ids;
    }

    /** Integer sums and products wrap like the narrow type; integer means are exact (one division of the sum). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void integersWrapAndIntegerMeansAreExact(Nd4jBackend backend) {
        for (boolean sorted : new boolean[]{true, false}) {
            // 300 times 127 in an int8: 38100 mod 256 = 212 = -44
            INDArray big = Nd4j.valueArrayOf(new long[]{300}, 127, DataType.INT8);
            INDArray ids = Nd4j.zeros(DataType.INT32, 300);
            assertEquals(-44.0, run(Op.SUM, sorted, big, ids, 1).getDouble(0), 0.0, "int8 sum wraps, sorted " + sorted);
            // 9 times -3 in an int8: (-3)^9 = -19683 mod 256 = 29
            INDArray factors = Nd4j.valueArrayOf(new long[]{9}, -3, DataType.INT8);
            INDArray ids9 = Nd4j.zeros(DataType.INT32, 9);
            assertEquals(29.0, run(Op.PROD, sorted, factors, ids9, 1).getDouble(0), 0.0,
                    "int8 product wraps, sorted " + sorted);
            // uint8: 20 times 200 = 4000 mod 256 = 160
            INDArray bytes = Nd4j.valueArrayOf(new long[]{20}, 200, DataType.UINT8);
            INDArray ids20 = Nd4j.zeros(DataType.INT32, 20);
            assertEquals(160.0, run(Op.SUM, sorted, bytes, ids20, 1).getDouble(0), 0.0,
                    "uint8 sum wraps, sorted " + sorted);
            // the mean of integers is the exact mean (each element divided first, [1,1,1] had mean 0)
            INDArray ints = Nd4j.createFromArray(1, 1, 1, 2, 2, 5);
            INDArray meanIds = Nd4j.createFromArray(0, 0, 0, 1, 1, 2);
            INDArray mean = run(Op.MEAN, sorted, ints, meanIds, 3);
            assertEquals(DataType.FLOAT, mean.dataType(), "the mean of integers is floating point");
            assertArrayEquals(new float[]{1f, 2f, 5f}, mean.data().asFloat(), 0f, "integer mean, sorted " + sorted);
            INDArray odd = Nd4j.createFromArray(1, 2, 2, 7);
            INDArray oddIds = Nd4j.createFromArray(0, 0, 1, 1);
            assertArrayEquals(new float[]{1.5f, 4.5f}, run(Op.MEAN, sorted, odd, oddIds, 2).data().asFloat(), 0f,
                    "integer mean with a fraction, sorted " + sorted);
            // a caller supplied FLOAT output for an integer input
            INDArray out = Nd4j.create(DataType.FLOAT, 2);
            Nd4j.exec(sorted
                    ? DynamicCustomOp.builder("segment_mean").addInputs(odd, oddIds).addOutputs(out).build()
                    : DynamicCustomOp.builder("unsorted_segment_mean").addInputs(odd, oddIds)
                            .addIntegerArguments(2).addOutputs(out).build());
            assertArrayEquals(new float[]{1.5f, 4.5f}, out.data().asFloat(), 0f, "mean into a FLOAT output");
        }
    }

    /** A NaN makes a maximum or minimum NaN (wherever it is), infinities are ordinary values. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nanAndInfinities(Nd4jBackend backend) {
        float inf = Float.POSITIVE_INFINITY;
        INDArray x = Nd4j.createFromArray(1f, Float.NaN, 3f, 4f, -inf, -inf, 2f);
        INDArray ids = Nd4j.createFromArray(0, 0, 1, 1, 2, 2, 3);
        for (boolean sorted : new boolean[]{true, false}) {
            float[] max = run(Op.MAX, sorted, x, ids, 4).data().asFloat();
            float[] min = run(Op.MIN, sorted, x, ids, 4).data().asFloat();
            float[] sum = run(Op.SUM, sorted, x, ids, 4).data().asFloat();
            assertTrue(Float.isNaN(max[0]) && max[1] == 4f && max[2] == -inf && max[3] == 2f,
                    "max " + Arrays.toString(max) + ", sorted " + sorted);
            assertTrue(Float.isNaN(min[0]) && min[1] == 3f && min[2] == -inf && min[3] == 2f,
                    "min " + Arrays.toString(min) + ", sorted " + sorted);
            assertTrue(Float.isNaN(sum[0]) && sum[1] == 7f && sum[2] == -inf && sum[3] == 2f,
                    "sum " + Arrays.toString(sum) + ", sorted " + sorted);
        }
        // the unsorted maximum of a class with only -inf is -inf; an empty class is the lowest float
        float[] unsortedMax = run(Op.MAX, false, x, ids, 5).data().asFloat();
        assertEquals(-inf, unsortedMax[2], "unsorted max of -inf");
        assertEquals(-Float.MAX_VALUE, unsortedMax[4], "unsorted max of an empty class");
    }

    // ------------------------------------------------------------------------------------------ backprops

    /** The gradient w.r.t. the rows of the input: [n][k] for gradOut [classes][k]. A product's gradient is the product
     * of the other elements of the segment, computed directly (it is not prod / x: that is 0 / 0 for a zero). */
    private static double[][] backpropReference(Op op, double[][] x, long[] ids, double[][] gradOut, int classes) {
        int n = x.length;
        int k = x[0].length;
        double[][] out = new double[n][k];
        for (int e = 0; e < k; e++) {
            double[] extreme = new double[classes];
            int[] count = new int[classes];
            Arrays.fill(extreme, op == Op.MAX ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY);
            for (int i = 0; i < n; i++) {
                int s = (int) ids[i];
                count[s]++;
                extreme[s] = op == Op.MAX ? Math.max(extreme[s], x[i][e]) : Math.min(extreme[s], x[i][e]);
            }
            for (int i = 0; i < n; i++) {
                int s = (int) ids[i];
                double g = gradOut[s][e];
                switch (op) {
                    case SUM: out[i][e] = g; break;
                    case MEAN: out[i][e] = g / count[s]; break;
                    case SQRT_N: out[i][e] = g / Math.sqrt(count[s]); break;
                    case PROD:
                        double others = 1.0;
                        for (int j = 0; j < n; j++) if (j != i && ids[j] == s) others *= x[j][e];
                        out[i][e] = g * others;
                        break;
                    default: out[i][e] = x[i][e] == extreme[s] ? g : 0.0; break;
                }
            }
        }
        return out;
    }

    private void backpropCase(Op op, boolean sorted, DataType type, int n, long[] tail, long[] ids, int classes,
                              Random rnd, String label) {
        int k = tailLength(tail);
        double[][] x = new double[n][k];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                x[i][j] = rnd.nextInt(5) - 2;  // zeros included: the product's gradient is the product of the others
        double[][] g = new double[classes][k];
        for (int s = 0; s < classes; s++) for (int j = 0; j < k; j++) g[s][j] = 1 + rnd.nextInt(5);
        long[] gradShape = new long[1 + tail.length];
        gradShape[0] = classes;
        System.arraycopy(tail, 0, gradShape, 1, tail.length);
        INDArray in = array(type, x, tail);
        INDArray grad = array(type, g, tail);
        String name = opName(op, sorted) + "_bp";
        INDArray idArr = idArray(n % 2 == 0, ids);
        INDArray[] outputs = sorted ? execAll(name, new long[0], in, idArr, grad)
                : execAll(name, new long[]{classes}, in, idArr, grad);
        double[][] expected = backpropReference(op, x, ids, g, classes);
        String what = name + " " + type + " " + label;
        assertEquals(type, outputs[0].dataType(), what + ": output type");
        assertArrayEquals(in.shape(), outputs[0].shape(), what + ": output shape");
        double[] got = values(outputs[0]);
        double tol = (op == Op.MEAN || op == Op.SQRT_N || op == Op.PROD)
                ? (type == DataType.DOUBLE ? 1e-12 : type == DataType.FLOAT ? 2e-6 : 2e-2) : 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                assertTrue(Math.abs(got[i * k + j] - expected[i][j]) <= tol * Math.max(1.0, Math.abs(expected[i][j])),
                        what + ": [" + i + "," + j + "] expected " + expected[i][j] + ", got " + got[i * k + j]);
        assertArrayEquals(values(idArr), values(outputs[1]), 0.0, what + ": the second output carries the ids");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backpropsMatchTheReference(Nd4jBackend backend) {
        Random rnd = new Random(77L);
        for (boolean sorted : new boolean[]{true, false}) {
            for (Op op : Op.values()) {
                if (op == Op.SQRT_N && sorted) continue;
                for (DataType type : FLOATS) {
                    // vector of 2048 ids over 1024 classes (a gradient of 1024: <<<n, 1025>>> was invalid)
                    int n = (type == DataType.FLOAT16 || type == DataType.BFLOAT16) && op == Op.PROD ? 64 : 2048;
                    int classes = n == 2048 ? 1024 : 16;
                    long[] ids = sorted ? sortedIdsFor(n, classes) : randomIds(rnd, n, classes);
                    int c = sorted ? (int) ids[n - 1] + 1 : classes;
                    backpropCase(op, sorted, type, n, new long[0], ids, c, rnd, "vector " + n + " x " + c);
                    // rows x columns past 1024 elements (the swapped launch of the sorted/unsorted TAD kernel)
                    for (long[] shape : new long[][]{{41, 25}, {35, 30}, {2, 600}, {1025, 1}}) {
                        int rows = (int) shape[0];
                        long[] tail = shape[1] == 1 ? new long[0] : new long[]{shape[1]};
                        if ((type == DataType.FLOAT16 || type == DataType.BFLOAT16) && op == Op.PROD && rows > 100)
                            continue;
                        int cl = rows < 5 ? rows : 5;
                        long[] ids2 = sorted ? sortedIdsFor(rows, cl) : randomIds(rnd, rows, cl);
                        int c2 = sorted ? (int) ids2[rows - 1] + 1 : cl;
                        backpropCase(op, sorted, type, rows, tail, ids2, c2, rnd, "matrix " + rows + " x " + shape[1]);
                    }
                }
            }
        }
    }

    /** Non-decreasing ids over classes 0 .. classes - 1 (each used, the first n / classes rows to class 0 ...). */
    private static long[] sortedIdsFor(int n, int classes) {
        long[] ids = new long[n];
        for (int i = 0; i < n; i++) ids[i] = (long) i * classes / n;
        return ids;
    }

    /** The minimum and maximum of HALF, BFLOAT16 and INT8 over rows x columns (contended accumulators), and the
     * minimum's backprop (the gradient came back zero for HALF and BFLOAT16 rows x columns). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void minimumAndMaximumOfSmallTypesAndTheirBackprops(Nd4jBackend backend) {
        Random rnd = new Random(5L);
        for (DataType type : new DataType[]{DataType.FLOAT16, DataType.BFLOAT16, DataType.INT8}) {
            int n = 600;
            int k = 3;
            int classes = 2;
            long[] ids = randomIds(rnd, n, classes);
            double[][] x = new double[n][k];
            for (int i = 0; i < n; i++) for (int j = 0; j < k; j++) x[i][j] = rnd.nextInt(41) - 20;
            INDArray in = array(type, x, new long[]{k});
            for (Op op : new Op[]{Op.MIN, Op.MAX}) {
                INDArray out = run(op, false, in, idArray(true, ids), classes);
                assertRows(opName(op, false) + " " + type + " 600 x 3, 2 classes", op, type,
                        reference(op, false, x, ids, classes, type), out);
            }
            if (type != DataType.INT8) {
                double[][] g = new double[classes][k];
                for (int s = 0; s < classes; s++) for (int j = 0; j < k; j++) g[s][j] = 1 + rnd.nextInt(4);
                INDArray[] bp = execAll("unsorted_segment_min_bp", new long[]{classes}, in, idArray(true, ids),
                        array(type, g, new long[]{k}));
                double[][] expected = backpropReference(Op.MIN, x, ids, g, classes);
                double[] got = values(bp[0]);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < k; j++)
                        assertEquals(expected[i][j], got[i * k + j], 0.0,
                                "unsorted_segment_min_bp " + type + " [" + i + "," + j + "]");
            }
        }
    }

    /** Ties get the whole gradient (an exact comparison), elements merely close to the maximum get none. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void extremumBackpropsCompareExactly(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(5f, 5f, 4.9999995f, 1f, 3f);
        INDArray ids = Nd4j.createFromArray(0, 0, 0, 1, 1);
        INDArray grad = Nd4j.createFromArray(2f, 7f);
        INDArray[] max = execAll("segment_max_bp", new long[0], x, ids, grad);
        assertArrayEquals(new float[]{2f, 2f, 0f, 0f, 7f}, max[0].data().asFloat(), 0f, "max tie");
        INDArray[] min = execAll("unsorted_segment_min_bp", new long[]{2}, x, ids, grad);
        assertArrayEquals(new float[]{0f, 0f, 2f, 7f, 0f}, min[0].data().asFloat(), 0f, "min");
    }

    /** The gradient of a product is the product of the other elements (TensorFlow's three cases): without a zero
     * prod / x; with exactly one zero only that element has a gradient; with two or more none has. prod / x was
     * 0 / 0 = NaN for every zero. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void productBackpropsHandleZeros(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(2f, 3f, 4f, 0f, 5f, 7f, 0f, 0f, 6f);
        INDArray ids = Nd4j.createFromArray(0, 0, 0, 1, 1, 1, 2, 2, 3);
        INDArray grad = Nd4j.createFromArray(1f, 2f, 3f, 4f);
        float[] expected = {12f, 8f, 6f, 70f, 0f, 0f, 0f, 0f, 4f};
        assertArrayEquals(expected, execAll("segment_prod_bp", new long[0], x, ids, grad)[0].data().asFloat(), 0f,
                "segment_prod_bp: no zero, one zero, two zeros, a single element");
        assertArrayEquals(expected,
                execAll("unsorted_segment_prod_bp", new long[]{4}, x, ids, grad)[0].data().asFloat(), 0f,
                "unsorted_segment_prod_bp");
        // unsorted ids: the same segments interleaved
        INDArray shuffledX = Nd4j.createFromArray(6f, 0f, 2f, 0f, 3f, 5f, 0f, 4f, 7f);
        INDArray shuffledIds = Nd4j.createFromArray(3, 2, 0, 1, 0, 1, 2, 0, 1);
        assertArrayEquals(new float[]{4f, 0f, 12f, 70f, 8f, 0f, 0f, 6f, 0f},
                execAll("unsorted_segment_prod_bp", new long[]{4}, shuffledX, shuffledIds, grad)[0].data().asFloat(),
                0f, "unsorted_segment_prod_bp, interleaved ids");
        // columns are independent: column 0 has no zero, column 1 has one (the first row)
        INDArray matrix = Nd4j.createFromArray(new float[][]{{2f, 0f}, {3f, 5f}, {4f, 7f}});
        INDArray matrixIds = Nd4j.createFromArray(0, 0, 0);
        INDArray matrixGrad = Nd4j.createFromArray(new float[][]{{1f, 1f}});
        float[] matrixExpected = {12f, 35f, 8f, 0f, 6f, 0f};
        assertArrayEquals(matrixExpected,
                execAll("segment_prod_bp", new long[0], matrix, matrixIds, matrixGrad)[0].data().asFloat(), 0f,
                "segment_prod_bp, rows x columns");
        assertArrayEquals(matrixExpected,
                execAll("unsorted_segment_prod_bp", new long[]{1}, matrix, matrixIds, matrixGrad)[0].data().asFloat(),
                0f, "unsorted_segment_prod_bp, rows x columns");
        // the same in DOUBLE and HALF
        for (DataType type : new DataType[]{DataType.DOUBLE, DataType.FLOAT16, DataType.BFLOAT16}) {
            assertArrayEquals(new double[]{12, 8, 6, 70, 0, 0, 0, 0, 4},
                    values(execAll("segment_prod_bp", new long[0], x.castTo(type), ids, grad.castTo(type))[0]), 0.0,
                    "segment_prod_bp " + type);
        }
    }

    // ------------------------------------------------------------------------------------------ invalid ids

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void invalidSegmentIdsAreRejected(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(1f, 2f, 3f);
        for (String op : new String[]{"segment_sum", "segment_mean", "segment_max", "segment_min", "segment_prod"}) {
            // out of order
            assertThrows(RuntimeException.class, () -> exec(op, new long[0], x, Nd4j.createFromArray(0, 1, 0)), op);
            // negative (first id)
            assertThrows(RuntimeException.class, () -> exec(op, new long[0], x, Nd4j.createFromArray(-1, 0, 1)), op);
            assertThrows(RuntimeException.class, () -> exec(op, new long[0], x, Nd4j.createFromArray(-2L, 0L, 1L)),
                    op + " int64");
        }
        for (String op : new String[]{"unsorted_segment_sum", "unsorted_segment_mean", "unsorted_segment_max",
                "unsorted_segment_min", "unsorted_segment_prod", "unsorted_segment_sqrt_n"}) {
            // an id equal to the number of classes
            assertThrows(RuntimeException.class, () -> exec(op, new long[]{3}, x, Nd4j.createFromArray(0, 3, 1)), op);
            // an id above it
            assertThrows(RuntimeException.class, () -> exec(op, new long[]{3}, x, Nd4j.createFromArray(0L, 9L, 1L)),
                    op + " int64");
            // negative
            assertThrows(RuntimeException.class, () -> exec(op, new long[]{3}, x, Nd4j.createFromArray(0, -1, 1)), op);
        }
        // the ops still work afterwards
        assertArrayEquals(new float[]{3f, 3f}, exec("segment_sum", new long[0], x, Nd4j.createFromArray(0, 0, 1))
                .data().asFloat(), 0f);
        assertArrayEquals(new float[]{1f, 5f, 0f}, exec("unsorted_segment_sum", new long[]{3}, x,
                Nd4j.createFromArray(0, 1, 1)).data().asFloat(), 0f);
    }

    // ------------------------------------------------------------------------------------ layouts and views

    private static INDArray[] layoutsOf(INDArray base, double[][] x) {
        // base: a C order [n, k] array holding x; returns views and copies of it in other layouts
        int n = x.length;
        int k = x[0].length;
        INDArray parent = Nd4j.create(base.dataType(), n + 4, 2 * k + 3);
        parent.assign(-1);
        INDArray offsetAndStepped = parent.get(NDArrayIndex.interval(2, 2 + n), NDArrayIndex.interval(1, 2, 1 + 2 * k));
        offsetAndStepped.assign(base);
        INDArray transposed = base.dup('c').reshape(n, k);
        INDArray transposedSource = Nd4j.create(base.dataType(), k, n);
        transposedSource.assign(base.transpose());
        return new INDArray[]{base.dup('c'), base.dup('f'), offsetAndStepped, transposedSource.transpose(),
                transposed};
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void viewsAndLayoutsGiveTheResultOfTheirContents(Nd4jBackend backend) {
        Random rnd = new Random(41L);
        int n = 30;
        int k = 4;
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.INT32, DataType.FLOAT16}) {
            for (boolean sorted : new boolean[]{true, false}) {
                for (Op op : new Op[]{Op.SUM, Op.MAX, Op.MIN, Op.MEAN, Op.PROD}) {
                    if (op == Op.MEAN && type == DataType.INT32) continue;
                    double[][] x = data(rnd, type, op, n, k);
                    long[] ids = sorted ? sortedIds(rnd, n) : randomIds(rnd, n, 6);
                    int classes = sorted ? (int) ids[n - 1] + 1 : 6;
                    INDArray plain = array(type, x, new long[]{k});
                    double[][] expected = reference(op, sorted, x, ids, classes, type);
                    INDArray[] layouts = layoutsOf(plain, x);
                    String[] names = {"c order", "f order", "offset and stepped view", "transposed view",
                            "reshaped copy"};
                    // the ids as a stepped view of a wider array, as a row [1, n] and as a column [n, 1]
                    INDArray idsWide = Nd4j.create(DataType.INT32, 2 * n + 3).assign(-7);
                    INDArray idsStepped = idsWide.get(NDArrayIndex.interval(1, 2, 1 + 2 * n));
                    idsStepped.assign(idArray(false, ids));
                    INDArray[] idLayouts = {idArray(true, ids), idsStepped, idArray(false, ids).reshape(1, n),
                            idArray(false, ids).reshape(n, 1)};
                    for (int l = 0; l < layouts.length; l++) {
                        for (int i = 0; i < idLayouts.length; i++) {
                            INDArray out = run(op, sorted, layouts[l], idLayouts[i], classes);
                            assertRows(opName(op, sorted) + " " + type + " " + names[l] + ", ids layout " + i, op,
                                    outputType(op, type), expected, out);
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void callerSuppliedOutputViewsAreWrittenInPlaceAndNothingElse(Nd4jBackend backend) {
        Random rnd = new Random(43L);
        int n = 25;
        int k = 3;
        for (boolean sorted : new boolean[]{true, false}) {
            for (Op op : new Op[]{Op.SUM, Op.MAX, Op.MIN, Op.MEAN, Op.PROD}) {
                double[][] x = data(rnd, DataType.FLOAT, op, n, k);
                long[] ids = sorted ? sortedIds(rnd, n) : randomIds(rnd, n, 5);
                int classes = sorted ? (int) ids[n - 1] + 1 : 5;
                INDArray parent = Nd4j.create(DataType.FLOAT, classes + 2, k + 3).assign(777);
                INDArray out = parent.get(NDArrayIndex.interval(1, 1 + classes), NDArrayIndex.interval(1, 1 + k));
                DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder(opName(op, sorted))
                        .addInputs(array(DataType.FLOAT, x, new long[]{k}), idArray(true, ids)).addOutputs(out);
                if (!sorted) builder.addIntegerArguments(classes);
                Nd4j.exec(builder.build());
                String what = opName(op, sorted) + " into a view";
                assertRows(what, op, DataType.FLOAT, reference(op, sorted, x, ids, classes, DataType.FLOAT), out);
                for (int r = 0; r < classes + 2; r++)
                    for (int c = 0; c < k + 3; c++) {
                        boolean inside = r >= 1 && r < 1 + classes && c >= 1 && c < 1 + k;
                        if (!inside) assertEquals(777.0, parent.getDouble(r, c), 0.0, what + ": parent [" + r + "," + c + "]");
                    }
            }
        }
    }

    /** A [1, N] input (one row of N values): one id, the whole row is the segment. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void aSingleRowOfManyColumns(Nd4jBackend backend) {
        INDArray row = Nd4j.createFromArray(3f, 1f, 4f, 1f, 5f).reshape(1, 5);
        INDArray one = Nd4j.createFromArray(0);
        for (boolean sorted : new boolean[]{true, false}) {
            for (Op op : new Op[]{Op.SUM, Op.MAX, Op.MIN, Op.MEAN, Op.PROD}) {
                INDArray out = run(op, sorted, row, one, 1);
                assertArrayEquals(new long[]{1, 5}, out.shape(), opName(op, sorted) + " shape");
                assertArrayEquals(new float[]{3f, 1f, 4f, 1f, 5f}, out.data().asFloat(), 0f,
                        opName(op, sorted) + " of one row");
            }
            // the same row as class 2 (two classes before it are empty)
            INDArray out = run(Op.SUM, sorted, row, Nd4j.createFromArray(2), 3);
            assertArrayEquals(new float[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3f, 1f, 4f, 1f, 5f}, out.data().asFloat(), 0f);
        }
        // and its backprop
        INDArray[] bp = execAll("segment_sum_bp", new long[0], row, one, Nd4j.createFromArray(1f, 2f, 3f, 4f, 5f)
                .reshape(1, 5));
        assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f}, bp[0].data().asFloat(), 0f);
    }

    // -------------------------------------------------------------------------------- bit-exact accumulation

    /** rows x columns sums are accumulated in row order: bit-identical to a sequential FLOAT sum on every backend. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void rowsByColumnsSumsAccumulateInRowOrder(Nd4jBackend backend) {
        Random rnd = new Random(99L);
        int n = 700;
        int k = 5;
        float[][] x = new float[n][k];
        for (int i = 0; i < n; i++) for (int j = 0; j < k; j++) x[i][j] = (float) (rnd.nextGaussian() * 10);
        INDArray in = Nd4j.createFromArray(x);
        long[] ids = sortedIdsFor(n, 4);
        INDArray out = exec("segment_sum", new long[0], in, idArray(true, ids));
        INDArray mean = exec("segment_mean", new long[0], in, idArray(true, ids));
        for (int s = 0; s < 4; s++) {
            for (int j = 0; j < k; j++) {
                float acc = 0f;
                int count = 0;
                for (int i = 0; i < n; i++) if (ids[i] == s) { acc += x[i][j]; count++; }
                assertEquals(Float.floatToIntBits(acc), Float.floatToIntBits(out.getFloat(s, j)),
                        "sequential float sum of class " + s + " column " + j);
                assertEquals(Float.floatToIntBits(acc / (float) count), Float.floatToIntBits(mean.getFloat(s, j)),
                        "sequential float mean of class " + s + " column " + j);
            }
        }
        // vectors: a fixed reduction tree, the same bits on every run
        INDArray long1 = Nd4j.rand(DataType.FLOAT, 5000).subi(0.5);
        long[] vids = sortedIdsFor(5000, 3);
        float[] first = exec("segment_sum", new long[0], long1, idArray(false, vids)).data().asFloat();
        for (int run = 0; run < 3; run++)
            assertArrayEquals(first, exec("segment_sum", new long[0], long1, idArray(false, vids)).data().asFloat(), 0f,
                    "vector segment_sum is deterministic");
    }

    // --------------------------------------------------------------------------------------- sequence_mask

    private static void assertMask(INDArray mask, long[] lengths, int width, String what) {
        double[] got = values(mask);
        assertEquals((long) lengths.length * width, got.length, what + ": number of elements");
        for (int r = 0; r < lengths.length; r++)
            for (int c = 0; c < width; c++)
                assertEquals(c < lengths[r] ? 1.0 : 0.0, got[r * width + c], 0.0,
                        what + ": [" + r + "," + c + "] for length " + lengths[r]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sequenceMaskOfManyLengths(Nd4jBackend backend) {
        Random rnd = new Random(8L);
        int n = 1025;
        long[] lengths = new long[n];
        for (int i = 0; i < n; i++) lengths[i] = rnd.nextInt(13);
        lengths[17] = 12;
        for (DataType lengthType : new DataType[]{DataType.INT32, DataType.INT64, DataType.UINT8, DataType.INT16}) {
            INDArray in = Nd4j.createFromArray(lengths).castTo(lengthType);
            for (DataType maskType : new DataType[]{DataType.BOOL, DataType.FLOAT, DataType.INT32, DataType.UINT8}) {
                INDArray mask = Nd4j.exec(new SequenceMask(in, maskType))[0];
                assertEquals(maskType, mask.dataType());
                assertArrayEquals(new long[]{n, 12}, mask.shape());
                assertMask(mask, lengths, 12, "lengths " + lengthType + " mask " + maskType);
            }
        }
        // more columns than a block has threads
        INDArray three = Nd4j.createFromArray(1, 3, 2);
        INDArray wide = Nd4j.exec(new SequenceMask(three, 2000, DataType.FLOAT))[0];
        assertArrayEquals(new long[]{3, 2000}, wide.shape());
        assertMask(wide, new long[]{1, 3, 2}, 2000, "maxlen 2000");
        INDArray wideFromInput = Nd4j.exec(new SequenceMask(three, Nd4j.scalar(1500), DataType.INT32))[0];
        assertArrayEquals(new long[]{3, 1500}, wideFromInput.shape());
        assertMask(wideFromInput, new long[]{1, 3, 2}, 1500, "maxlen 1500 as an input");
    }

    /** The width is max(maxlen, longest length) however maxlen is given: the execution used the position of the longest
     * length when a maxlen input was not larger than it (and the first integer argument, the data type, as the
     * width). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sequenceMaskWidthAgreesWithItsShape(Nd4jBackend backend) {
        INDArray lengths = Nd4j.createFromArray(1, 3, 2);
        // a maxlen input smaller than the longest length: the longest length is the width
        INDArray small = Nd4j.exec(new SequenceMask(lengths, Nd4j.scalar(2), DataType.FLOAT))[0];
        assertArrayEquals(new long[]{3, 3}, small.shape());
        assertMask(small, new long[]{1, 3, 2}, 3, "maxlen input 2");
        // equal to it
        INDArray equal = Nd4j.exec(new SequenceMask(lengths, Nd4j.scalar(3), DataType.FLOAT))[0];
        assertMask(equal, new long[]{1, 3, 2}, 3, "maxlen input 3");
        // no maxlen
        INDArray none = Nd4j.exec(new SequenceMask(lengths, DataType.FLOAT))[0];
        assertMask(none, new long[]{1, 3, 2}, 3, "no maxlen");
        // the longest length is first (the position of the maximum is 0, not its value)
        INDArray reversed = Nd4j.createFromArray(5, 2, 1);
        INDArray reversedMask = Nd4j.exec(new SequenceMask(reversed, Nd4j.scalar(4), DataType.FLOAT))[0];
        assertArrayEquals(new long[]{3, 5}, reversedMask.shape());
        assertMask(reversedMask, new long[]{5, 2, 1}, 5, "longest first");
        // legacy integer arguments: maxlen, data type
        INDArray legacy = Nd4j.exec(DynamicCustomOp.builder("sequence_mask").addInputs(lengths)
                .addIntegerArguments(5, DataType.FLOAT.toInt()).build())[0];
        assertEquals(DataType.FLOAT, legacy.dataType());
        assertMask(legacy, new long[]{1, 3, 2}, 5, "integer arguments");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sequenceMaskNegativeLengthsAndLayouts(Nd4jBackend backend) {
        // a negative length is an all-false row (CUDA compared it as an unsigned value: all true)
        for (DataType type : new DataType[]{DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64}) {
            INDArray lengths = Nd4j.createFromArray(-1, 2, -5, 0, 3).castTo(type);
            INDArray mask = Nd4j.exec(new SequenceMask(lengths, 4, DataType.INT32))[0];
            assertArrayEquals(new long[]{5, 4}, mask.shape());
            assertMask(mask, new long[]{-1, 2, -5, 0, 3}, 4, "negative lengths " + type);
        }
        // two dimensional lengths [3, 4] -> [3, 4, 6]
        long[] flat = new long[12];
        for (int i = 0; i < 12; i++) flat[i] = i % 7;
        INDArray lengths2d = Nd4j.createFromArray(flat).reshape(3, 4);
        INDArray mask2d = Nd4j.exec(new SequenceMask(lengths2d, 6, DataType.FLOAT))[0];
        assertArrayEquals(new long[]{3, 4, 6}, mask2d.shape());
        assertMask(mask2d, flat, 6, "[3, 4] lengths");
        // lengths as a stepped view and as an F ordered matrix
        INDArray wide = Nd4j.createFromArray(0L, 99L, 3L, 99L, 1L, 99L, 2L);
        INDArray stepped = wide.get(NDArrayIndex.interval(0, 2, 7));
        INDArray steppedMask = Nd4j.exec(new SequenceMask(stepped, 4, DataType.FLOAT))[0];
        assertMask(steppedMask, new long[]{0, 3, 1, 2}, 4, "stepped lengths");
        INDArray fOrder = lengths2d.dup('f');
        INDArray fMask = Nd4j.exec(new SequenceMask(fOrder, 6, DataType.FLOAT))[0];
        assertMask(fMask, flat, 6, "F ordered lengths");
        // a caller supplied output: a view of a larger array (no other element is touched)
        INDArray parent = Nd4j.create(DataType.FLOAT, 6, 7).assign(9);
        INDArray out = parent.get(NDArrayIndex.interval(1, 5), NDArrayIndex.interval(1, 5));
        Nd4j.exec(DynamicCustomOp.builder("sequence_mask").addInputs(Nd4j.createFromArray(0, 1, 2, 4))
                .addIntegerArguments(4).addOutputs(out).build());
        assertMask(out, new long[]{0, 1, 2, 4}, 4, "mask into a view");
        for (int r = 0; r < 6; r++)
            for (int c = 0; c < 7; c++)
                if (r < 1 || r >= 5 || c < 1 || c >= 5) assertEquals(9.0, parent.getDouble(r, c), 0.0, "parent");
    }

    // ----------------------------------------------------------------------------------------- segment_softmax

    private static double[][] softmaxReference(double[][] logits, long[] ids, int k) {
        int n = logits.length;
        int inner = logits[0].length;
        double[][] out = new double[n][inner];
        for (int s = 0; s < k; s++) {
            for (int f = 0; f < inner; f++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < n; i++) if (ids[i] == s) max = Math.max(max, logits[i][f]);
                double sum = 0;
                for (int i = 0; i < n; i++) if (ids[i] == s) sum += Math.exp(logits[i][f] - max);
                for (int i = 0; i < n; i++) if (ids[i] == s) out[i][f] = Math.exp(logits[i][f] - max) / sum;
            }
        }
        return out;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void segmentSoftmaxOfAnyIdTypeLayoutAndWidth(Nd4jBackend backend) {
        Random rnd = new Random(12L);
        // many features (the grid's second dimension held them: at most 65535) and INT64 ids (read as INT32 before)
        int n = 9;
        int inner = 70001;
        long[] ids = {0, 0, 0, 1, 1, 2, 2, 2, 2};
        double[][] logits = new double[n][inner];
        for (int i = 0; i < n; i++) for (int f = 0; f < inner; f++) logits[i][f] = rnd.nextGaussian();
        INDArray x = array(DataType.FLOAT, logits, new long[]{inner});
        for (boolean wide : new boolean[]{true, false}) {
            INDArray out = Nd4j.exec(new SegmentSoftmax(x, idArray(wide, ids), 3L))[0];
            assertArrayEquals(x.shape(), out.shape());
            double[][] expected = softmaxReference(logits, ids, 3);
            for (int f : new int[]{0, 1, 4095, 65535, 65536, 70000})
                for (int i = 0; i < n; i++)
                    assertEquals(expected[i][f], out.getDouble(i, f), 1e-5, "ids wide " + wide + " [" + i + "," + f + "]");
        }
        // rank 3 logits [6, 3, 4] and a stepped / F ordered view
        int n3 = 6;
        long[] ids3 = {0, 0, 1, 1, 1, 3};
        double[][] flat3 = new double[n3][12];
        for (int i = 0; i < n3; i++) for (int f = 0; f < 12; f++) flat3[i][f] = rnd.nextGaussian() * 3;
        double[][] expected3 = softmaxReference(flat3, ids3, 4);
        INDArray x3 = array(DataType.DOUBLE, flat3, new long[]{3, 4});
        INDArray fOrder = x3.dup('f');
        INDArray parent = Nd4j.create(DataType.DOUBLE, n3 + 2, 3, 9).assign(-5);
        INDArray stepped = parent.get(NDArrayIndex.interval(1, 1 + n3), NDArrayIndex.all(),
                NDArrayIndex.interval(1, 2, 9));
        stepped.assign(x3);
        for (INDArray input : new INDArray[]{x3, fOrder, stepped}) {
            INDArray out = Nd4j.exec(new SegmentSoftmax(input, idArray(true, ids3), 4L))[0];
            double[] got = values(out);
            for (int i = 0; i < n3; i++)
                for (int f = 0; f < 12; f++)
                    assertEquals(expected3[i][f], got[i * 12 + f], 1e-12, "rank 3 [" + i + "," + f + "]");
        }
        // the backprop: dLogits = out * (gradOut - sum(gradOut * out)) within each segment
        INDArray softmax = Nd4j.exec(new SegmentSoftmax(x3, idArray(true, ids3), 4L))[0];
        double[][] grad = new double[n3][12];
        for (int i = 0; i < n3; i++) for (int f = 0; f < 12; f++) grad[i][f] = rnd.nextGaussian();
        INDArray gradArr = array(DataType.DOUBLE, grad, new long[]{3, 4});
        INDArray dLogits = exec("segment_softmax_bp", new long[]{4}, x3, idArray(false, ids3), softmax, gradArr);
        double[] outValues = values(softmax);
        double[] dl = values(dLogits);
        for (int s = 0; s < 4; s++)
            for (int f = 0; f < 12; f++) {
                double dot = 0;
                for (int i = 0; i < n3; i++) if (ids3[i] == s) dot += grad[i][f] * outValues[i * 12 + f];
                for (int i = 0; i < n3; i++)
                    if (ids3[i] == s)
                        assertEquals(outValues[i * 12 + f] * (grad[i][f] - dot), dl[i * 12 + f], 1e-12,
                                "softmax bp [" + i + "," + f + "]");
            }
        // unsorted ids and an id outside [0, K) are rejected
        assertThrows(RuntimeException.class,
                () -> Nd4j.exec(new SegmentSoftmax(x3, idArray(true, new long[]{0, 1, 0, 1, 2, 3}), 4L)));
        assertThrows(RuntimeException.class,
                () -> Nd4j.exec(new SegmentSoftmax(x3, idArray(true, new long[]{0, 0, 1, 1, 1, 4}), 4L)));
    }

    // ------------------------------------------------------------------------------------ CUDA specific

    /** On CUDA the ids are validated from the device with a stream-ordered readback; a long invalid list is found at
     * its first violation (and the next call, with valid ids, still runs on the same stream). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void longInvalidIdListsAreRejectedAtTheFirstViolation(Nd4jBackend backend) {
        assumeTrue(Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA,
                "the device validation (a readback of one report) is a CUDA path");
        int n = 100000;
        long[] ids = sortedIdsFor(n, 5000);
        INDArray x = Nd4j.ones(DataType.FLOAT, n);
        long[] bad = ids.clone();
        bad[73211] = 3;  // out of order
        assertThrows(RuntimeException.class, () -> exec("segment_sum", new long[0], x, idArray(true, bad)));
        INDArray ok = exec("segment_sum", new long[0], x, idArray(true, ids));
        assertEquals(5000, ok.length());
        assertEquals(20.0, ok.getDouble(0), 0.0);
    }
}
