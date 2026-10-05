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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * dynamic_partition splits the slices of its data into partitions by an index array (the data has the shape of the
 * indices followed by the dimensions of a slice): the slices of an index go to that partition's output, in order, and
 * slices of an index outside [0, partitions) go nowhere. dynamic_stitch is the inverse: the slices of its inputs go to
 * the rows of the output their indices name, and the last input wins when an index appears more than once.
 * dynamic_partition_bp is the gradient of the partition: every slice of the data's gradient is the slice of its
 * partition's gradient at the slice's position in the partition, and zero for a slice no partition holds; the gradient
 * of the stitch is a gather of the output's gradient.
 *
 * All of them work on views of any layout, any size (the CUDA kernels used one block per partition scanned by one
 * thread, a launch sized for 256 threads and a stitch of slices that left all but a few elements of every slice
 * unwritten), integer indices of both widths, and every slice rank (the shape of the partitions' outputs was wrong
 * for data beyond two dimensions or of other than C order).
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class DynamicPartitionStitchTest extends BaseNd4jTestWithBackends {

    private static final int[] SIZES = {1, 5, 255, 256, 257, 1000, 4099, 70001};
    private static final DataType[] DATA_TYPES = {DataType.FLOAT, DataType.DOUBLE, DataType.INT32, DataType.INT64};
    private static final DataType[] INDEX_TYPES = {DataType.INT32, DataType.INT64};

    /** What a parent array holds around a view: a value no operation may read. */
    private static double filler(DataType type) {
        return type.isFPType() ? Double.NaN : -7777;
    }

    private static final class Layout {
        final String name;
        final UnaryOperator<INDArray> of;

        Layout(String name, UnaryOperator<INDArray> of) {
            this.name = name;
            this.of = of;
        }
    }

    private static final Layout[] LAYOUTS = {
            new Layout("C order", x -> x.dup('c')),
            new Layout("F order", x -> x.dup('f')),
            new Layout("offset view", x -> {
                long[] s = x.shape().clone();
                s[0] += 1;
                INDArray parent = Nd4j.valueArrayOf(s, filler(x.dataType()), x.dataType());
                INDArrayIndex[] idx = new INDArrayIndex[s.length];
                idx[0] = NDArrayIndex.interval(1, s[0]);
                for (int i = 1; i < s.length; i++)
                    idx[i] = NDArrayIndex.all();
                return parent.get(idx).assign(x);
            }),
            new Layout("stepped view", x -> {
                long[] s = x.shape().clone();
                int last = s.length - 1;
                long n = s[last];
                s[last] = 2 * n + 1;
                INDArray parent = Nd4j.valueArrayOf(s, filler(x.dataType()), x.dataType());
                INDArrayIndex[] idx = new INDArrayIndex[s.length];
                for (int i = 0; i < last; i++)
                    idx[i] = NDArrayIndex.all();
                idx[last] = NDArrayIndex.interval(1, 2, 2 * n + 1);
                return parent.get(idx).assign(x);
            }),
            new Layout("permuted view", x -> {
                long[] s = x.shape();
                int r = s.length;
                long[] reversed = new long[r];
                long[] perm = new long[r];
                for (int i = 0; i < r; i++) {
                    reversed[i] = s[r - 1 - i];
                    perm[i] = r - 1 - i;
                }
                return Nd4j.create(x.dataType(), reversed, 'c').permute(perm).assign(x);
            }),
    };

    private static long size(long... shape) {
        long n = 1;
        for (long s : shape)
            n *= s;
        return n;
    }

    /** Multiples of 1/8 below 128 in size: exact in every type the ops handle. */
    private static INDArray data(Random random, DataType type, long... shape) {
        double[] values = new double[(int) size(shape)];
        for (int i = 0; i < values.length; i++)
            values[i] = Math.floor(random.nextDouble() * 2000 - 1000) / 8;
        return Nd4j.createFromArray(values).reshape(shape).castTo(type);
    }

    /**
     * Partitions in [0, partitions), and with invalid > 0 every invalid-th one an index outside them. With cover, the
     * first partitions indices are 0 .. partitions - 1, so that no partition is empty (n >= partitions).
     */
    private static long[] ids(Random random, int n, int partitions, int invalid, boolean cover) {
        long[] ids = new long[n];
        for (int i = 0; i < n; i++) {
            ids[i] = cover && i < partitions ? i : random.nextInt(partitions);
            if (invalid > 0 && i % invalid == invalid - 1 && !(cover && i < partitions))
                ids[i] = (i % 2 == 0) ? -1 : partitions + (i % 7);
        }
        return ids;
    }

    private static long[] ids(Random random, int n, int partitions, int invalid) {
        return ids(random, n, partitions, invalid, false);
    }

    private static INDArray indexArray(long[] ids, DataType type, long... shape) {
        return Nd4j.createFromArray(ids).reshape(shape).castTo(type);
    }

    /** The elements of an array in C order, whatever its layout. */
    private static double[] flat(INDArray a) {
        return a.dup('c').castTo(DataType.DOUBLE).data().asDouble();
    }

    private static double[][] partitionReference(double[] x, long[] ids, int partitions, int sliceLength) {
        int[] counts = new int[partitions];
        for (long id : ids)
            if (id >= 0 && id < partitions)
                counts[(int) id]++;
        double[][] out = new double[partitions][];
        for (int p = 0; p < partitions; p++)
            out[p] = new double[counts[p] * sliceLength];
        int[] next = new int[partitions];
        for (int e = 0; e < ids.length; e++) {
            int id = (int) ids[e];
            if (id < 0 || id >= partitions)
                continue;
            System.arraycopy(x, e * sliceLength, out[id], next[id]++ * sliceLength, sliceLength);
        }
        return out;
    }

    private static INDArray[] partition(INDArray x, INDArray ids, int partitions) {
        return Nd4j.exec(DynamicCustomOp.builder("dynamic_partition").addInputs(x, ids)
                .addIntegerArguments(partitions).build());
    }

    private static void assertPartitions(INDArray[] actual, double[][] expected, int sliceLength, String what) {
        assertEquals(expected.length, actual.length, what + ": number of partitions");
        for (int p = 0; p < expected.length; p++) {
            assertEquals(expected[p].length / sliceLength, actual[p].length() / sliceLength,
                    what + ": slices in partition " + p);
            if (expected[p].length == 0)
                continue;
            assertArrayEquals(expected[p], flat(actual[p]), 0.0, what + ": partition " + p);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void partitionIsAStableSplit(Nd4jBackend backend) {
        Random random = new Random(11);
        for (DataType type : DATA_TYPES) {
            for (DataType indexType : INDEX_TYPES) {
                for (int n : SIZES) {
                    for (int partitions : new int[]{1, 2, 5}) {
                        long[] ids = ids(random, n, partitions, 0);
                        INDArray x = data(random, type, n);
                        double[] xBefore = flat(x);
                        INDArray[] out = partition(x, indexArray(ids, indexType, n), partitions);
                        String what = type + " data, " + indexType + " indices, " + n + " elements, " + partitions
                                + " partitions";
                        assertPartitions(out, partitionReference(xBefore, ids, partitions, 1), 1, what);
                        assertArrayEquals(xBefore, flat(x), 0.0, what + ": the data changed");
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void partitionMovesWholeSlicesOfAnyShape(Nd4jBackend backend) {
        Random random = new Random(12);
        // (shape of the data, rank of the indices): slices of one, two and three dimensions, and indices of two
        long[][] shapes = {{300, 4}, {300, 3, 2}, {70, 9, 3}, {257, 2, 3, 2}, {64, 17}, {1000, 1}};
        int[] indexRanks = {1, 1, 2, 1, 1, 1};
        for (int s = 0; s < shapes.length; s++) {
            long[] shape = shapes[s];
            int indexRank = indexRanks[s];
            long[] indexShape = new long[indexRank];
            System.arraycopy(shape, 0, indexShape, 0, indexRank);
            int sliceLength = (int) (size(shape) / size(indexShape));
            for (DataType type : new DataType[]{DataType.FLOAT, DataType.INT32}) {
                for (int partitions : new int[]{2, 4}) {
                    long[] ids = ids(random, (int) size(indexShape), partitions, 0);
                    INDArray x = data(random, type, shape);
                    double[] values = flat(x);
                    double[][] expected = partitionReference(values, ids, partitions, sliceLength);
                    for (Layout layout : LAYOUTS) {
                        INDArray[] out = partition(layout.of.apply(x), indexArray(ids, DataType.INT32, indexShape),
                                partitions);
                        String what = type + " data of shape " + Arrays.toString(shape) + " in "
                                + layout.name + ", " + partitions + " partitions";
                        assertPartitions(out, expected, sliceLength, what);
                        for (int p = 0; p < partitions; p++) {
                            long[] outShape = out[p].shape();
                            assertEquals(shape.length - indexRank + 1, outShape.length, what + ": rank of partition " + p);
                            for (int d = 1; d < outShape.length; d++)
                                assertEquals(shape[indexRank + d - 1], outShape[d], what + ": dimension " + d);
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void partitionReadsIndicesOfAnyLayout(Nd4jBackend backend) {
        Random random = new Random(13);
        long[] shape = {40, 13, 2};
        int partitions = 3;
        long[] ids = ids(random, 40 * 13, partitions, 0);
        INDArray x = data(random, DataType.DOUBLE, shape);
        double[][] expected = partitionReference(flat(x), ids, partitions, 2);
        for (DataType indexType : INDEX_TYPES) {
            INDArray indices = indexArray(ids, indexType, 40, 13);
            for (Layout layout : LAYOUTS) {
                INDArray[] out = partition(x.dup(), layout.of.apply(indices), partitions);
                assertPartitions(out, expected, 2, indexType + " indices in " + layout.name);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void indicesOutsideThePartitionsBelongToNone(Nd4jBackend backend) {
        Random random = new Random(14);
        for (int n : new int[]{9, 300, 4099}) {
            for (int sliceLength : new int[]{1, 3}) {
                long[] ids = ids(random, n, 3, 5);
                INDArray x = sliceLength == 1 ? data(random, DataType.FLOAT, n) : data(random, DataType.FLOAT, n, sliceLength);
                INDArray[] out = partition(x, indexArray(ids, DataType.INT64, n), 3);
                assertPartitions(out, partitionReference(flat(x), ids, 3, sliceLength), sliceLength,
                        n + " slices of " + sliceLength);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void partitionsThatGotNothingAreEmpty(Nd4jBackend backend) {
        long[] ids = {0, 2, 2, 0, 2};
        INDArray x = Nd4j.createFromArray(new double[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}});
        INDArray[] out = partition(x, Nd4j.createFromArray(ids), 4);
        assertEquals(4, out.length);
        assertEquals(0, out[1].length());
        assertEquals(0, out[3].length());
        assertArrayEquals(new double[]{1, 2, 7, 8}, flat(out[0]), 0.0);
        assertArrayEquals(new double[]{3, 4, 5, 6, 9, 10}, flat(out[2]), 0.0);
        assertArrayEquals(new long[]{2, 2}, out[0].shape());
        assertArrayEquals(new long[]{3, 2}, out[2].shape());
    }

    private static INDArray[] gradients(Random random, INDArray[] partitions, DataType type) {
        INDArray[] grads = new INDArray[partitions.length];
        for (int p = 0; p < grads.length; p++) {
            // the forward op's own arrays, so that a partition that got nothing keeps the zero-length array the
            // shape function gives it
            grads[p] = partitions[p];
            if (grads[p].length() > 0)
                grads[p].assign(data(random, type, grads[p].shape()));
        }
        return grads;
    }

    private static INDArray partitionBp(INDArray x, INDArray ids, INDArray[] grads, int partitions, INDArray out) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("dynamic_partition_bp")
                .addInputs(x, ids).addInputs(grads).addIntegerArguments(partitions);
        if (out != null)
            builder.addOutputs(out);
        return Nd4j.exec(builder.build())[0];
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void builderChecksAccumulatedInputsAtBuild(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(1.0f);
        INDArray ids = Nd4j.createFromArray(0);
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("dynamic_partition_bp")
                .addInputs(x, ids).addIntegerArguments(1);
        assertThrows(ND4JIllegalStateException.class, builder::build);
        builder.addInputs(Nd4j.createFromArray(2.0f));
        assertEquals(3, builder.build().numInputArguments());
    }

    /** Every slice takes its partition's gradient at its position in the partition; slices of no partition get zero. */
    private static double[] partitionBpReference(long[] ids, double[][] grads, int partitions, int sliceLength) {
        double[] out = new double[ids.length * sliceLength];
        int[] next = new int[partitions];
        for (int e = 0; e < ids.length; e++) {
            int id = (int) ids[e];
            if (id < 0 || id >= partitions)
                continue;
            System.arraycopy(grads[id], next[id]++ * sliceLength, out, e * sliceLength, sliceLength);
        }
        return out;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void partitionGradientIsTheClosedForm(Nd4jBackend backend) {
        Random random = new Random(15);
        long[][] shapes = {{1}, {7}, {255}, {257}, {5000}, {70001}, {300, 3}, {40, 2, 2}, {30, 11, 2}};
        int[] indexRanks = {1, 1, 1, 1, 1, 1, 1, 1, 2};
        for (int s = 0; s < shapes.length; s++) {
            long[] shape = shapes[s];
            long[] indexShape = new long[indexRanks[s]];
            System.arraycopy(shape, 0, indexShape, 0, indexShape.length);
            int sliceLength = (int) (size(shape) / size(indexShape));
            for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                for (DataType indexType : INDEX_TYPES) {
                    int partitions = (int) Math.min(3, size(indexShape));
                    long[] ids = ids(random, (int) size(indexShape), partitions, s % 2 == 0 ? 0 : 6, true);
                    INDArray x = data(random, type, shape);
                    INDArray indices = indexArray(ids, indexType, indexShape);
                    INDArray[] grads = gradients(random, partition(x, indices, partitions), type);
                    double[][] gradValues = new double[partitions][];
                    for (int p = 0; p < partitions; p++)
                        gradValues[p] = grads[p].length() == 0 ? new double[0] : flat(grads[p]);
                    double[] expected = partitionBpReference(ids, gradValues, partitions, sliceLength);
                    String what = type + " gradient of data of shape " + Arrays.toString(shape) + ", "
                            + indexType + " indices";

                    INDArray out = partitionBp(x, indices, grads, partitions, null);
                    assertArrayEquals(shape, out.shape(), what + ": shape");
                    assertEquals(type, out.dataType(), what + ": type");
                    assertArrayEquals(expected, flat(out), 0.0, what);
                    // a second run gives the same (nothing was left behind in the arrays)
                    assertArrayEquals(expected, flat(partitionBp(x, indices, grads, partitions, null)), 0.0,
                            what + ", again");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void partitionGradientWritesEveryElementOfItsOutput(Nd4jBackend backend) {
        Random random = new Random(16);
        int partitions = 3;
        long[] shape = {500, 3};
        long[] ids = ids(random, 500, partitions, 7, true);
        INDArray x = data(random, DataType.DOUBLE, shape);
        INDArray indices = indexArray(ids, DataType.INT32, 500);
        INDArray[] grads = gradients(random, partition(x, indices, partitions), DataType.DOUBLE);
        double[][] gradValues = new double[partitions][];
        for (int p = 0; p < partitions; p++)
            gradValues[p] = flat(grads[p]);
        double[] expected = partitionBpReference(ids, gradValues, partitions, 3);

        // an output of NaN: slices of no partition must be written (zero), and so must all the others
        INDArray out = Nd4j.valueArrayOf(shape, Double.NaN, DataType.DOUBLE);
        partitionBp(x, indices, grads, partitions, out);
        assertArrayEquals(expected, flat(out), 0.0, "output of NaN");

        // an output that is a view of a larger array: the rest of the parent stays as it was
        INDArray parent = Nd4j.valueArrayOf(new long[]{501, 3}, Double.NaN, DataType.DOUBLE);
        INDArray view = parent.get(NDArrayIndex.interval(1, 501), NDArrayIndex.all());
        partitionBp(x, indices, grads, partitions, view);
        assertArrayEquals(expected, flat(view), 0.0, "output view");
        for (int c = 0; c < 3; c++)
            assertTrue(Double.isNaN(parent.getDouble(0, c)), "the first row of the parent was written");

        // gradients and data of every layout
        for (Layout layout : LAYOUTS) {
            INDArray[] laidOut = new INDArray[partitions];
            for (int p = 0; p < partitions; p++)
                laidOut[p] = grads[p].length() == 0 ? grads[p] : layout.of.apply(grads[p]);
            INDArray result = partitionBp(layout.of.apply(x), layout.of.apply(indices), laidOut, partitions, null);
            assertArrayEquals(expected, flat(result), 0.0, "gradients and data in " + layout.name);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void partitionGradientThroughSameDiff(Nd4jBackend backend) {
        DataType previous = Nd4j.defaultFloatingPointType();
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        try {
            Random random = new Random(17);
            int partitions = 3;
            for (long[] shape : new long[][]{{300}, {120, 4}, {40, 3, 2}}) {
                long[] ids = ids(random, (int) shape[0], partitions, 0, true);
                INDArray x = data(random, DataType.DOUBLE, shape);
                int sliceLength = (int) (size(shape) / shape[0]);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", x);
                SDVariable partitionIds = sd.constant("ids", indexArray(ids, DataType.INT32, shape[0]));
                String[] names = {"p0", "p1", "p2"};
                SDVariable[] parts = sd.dynamicPartition(names, in, partitionIds, partitions);
                // the loss weighs every partition by its number: the gradient of a slice is its partition + 1
                SDVariable loss = parts[0].mul(1.0).sum().add(parts[1].mul(2.0).sum()).add(parts[2].mul(3.0).sum());
                loss.rename("loss");
                sd.setLossVariables("loss");

                Map<String, INDArray> grads = sd.calculateGradients(Collections.emptyMap(), "in");
                double[] expected = new double[(int) size(shape)];
                for (int e = 0; e < ids.length; e++)
                    for (int k = 0; k < sliceLength; k++)
                        expected[e * sliceLength + k] = ids[e] + 1;
                assertArrayEquals(expected, flat(grads.get("in")), 1e-12,
                        "gradient of data of shape " + Arrays.toString(shape));
            }
        } finally {
            Nd4j.setDefaultDataTypes(previous, previous);
        }
    }

    private static INDArray stitch(INDArray[] indices, INDArray[] data) {
        return Nd4j.exec(DynamicCustomOp.builder("dynamic_stitch").addInputs(indices).addInputs(data).build())[0];
    }

    /** Row r of the output: the slice of the last input that has an index r (zero where there is none). */
    private static double[] stitchReference(long[][] ids, double[][] data, int rows, int sliceLength) {
        double[] out = new double[rows * sliceLength];
        for (int in = 0; in < ids.length; in++)
            for (int e = 0; e < ids[in].length; e++)
                System.arraycopy(data[in], e * sliceLength, out, (int) ids[in][e] * sliceLength, sliceLength);
        return out;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void stitchMovesWholeSlicesOfAnyShape(Nd4jBackend backend) {
        Random random = new Random(18);
        // (the inputs' leading dimensions, the dimensions of a row): rows of nothing, one, two and three dimensions
        long[][] leading = {{300}, {257}, {70001}, {40, 13}, {200}};
        long[][] rowShapes = {{}, {4}, {3, 2}, {2}, {2, 3, 2}};
        for (int c = 0; c < leading.length; c++) {
            long[] rowShape = rowShapes[c];
            int sliceLength = (int) size(rowShape);
            int parts = 3;
            int perInput = (int) size(leading[c]);
            int rows = perInput * parts;
            // the rows in a random order, dealt out to three inputs
            List<Long> order = new ArrayList<>();
            for (long r = 0; r < rows; r++)
                order.add(r);
            Collections.shuffle(order, random);
            for (DataType type : new DataType[]{DataType.FLOAT, DataType.INT32}) {
                for (DataType indexType : INDEX_TYPES) {
                    long[][] ids = new long[parts][perInput];
                    INDArray[] indexArrays = new INDArray[parts];
                    INDArray[] dataArrays = new INDArray[parts];
                    double[][] dataValues = new double[parts][];
                    for (int p = 0; p < parts; p++) {
                        for (int e = 0; e < perInput; e++)
                            ids[p][e] = order.get(p * perInput + e);
                        indexArrays[p] = indexArray(ids[p], indexType, leading[c]);
                        long[] shape = new long[leading[c].length + rowShape.length];
                        System.arraycopy(leading[c], 0, shape, 0, leading[c].length);
                        System.arraycopy(rowShape, 0, shape, leading[c].length, rowShape.length);
                        INDArray d = data(random, type, shape);
                        dataValues[p] = flat(d);
                        dataArrays[p] = d;
                    }
                    double[] expected = stitchReference(ids, dataValues, rows, sliceLength);
                    String what = type + " rows of " + Arrays.toString(rowShape) + ", " + indexType
                            + " indices of shape " + Arrays.toString(leading[c]);
                    INDArray out = stitch(indexArrays, dataArrays);
                    assertEquals(1 + rowShape.length, out.rank(), what + ": rank");
                    assertEquals(rows, out.size(0), what + ": rows");
                    assertEquals(type, out.dataType(), what + ": type");
                    assertArrayEquals(expected, flat(out), 0.0, what);
                    // the inputs in every layout
                    for (Layout layout : LAYOUTS) {
                        INDArray[] laidOutData = new INDArray[parts];
                        INDArray[] laidOutIndices = new INDArray[parts];
                        for (int p = 0; p < parts; p++) {
                            laidOutData[p] = layout.of.apply(dataArrays[p]);
                            laidOutIndices[p] = layout.of.apply(indexArrays[p]);
                        }
                        assertArrayEquals(expected, flat(stitch(laidOutIndices, laidOutData)), 0.0,
                                what + ", inputs in " + layout.name);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void stitchTakesTheLastInputThatHasAnIndex(Nd4jBackend backend) {
        // index 2 is in all three inputs, index 0 in the first two; later inputs overwrite
        INDArray i0 = Nd4j.createFromArray(0L, 2L, 3L);
        INDArray i1 = Nd4j.createFromArray(2L, 0L);
        INDArray i2 = Nd4j.createFromArray(1L, 2L);
        INDArray d0 = Nd4j.createFromArray(new double[][]{{10, 11}, {20, 21}, {30, 31}});
        INDArray d1 = Nd4j.createFromArray(new double[][]{{40, 41}, {50, 51}});
        INDArray d2 = Nd4j.createFromArray(new double[][]{{60, 61}, {70, 71}});
        INDArray out = stitch(new INDArray[]{i0, i1, i2}, new INDArray[]{d0, d1, d2});
        assertArrayEquals(new double[]{50, 51, 60, 61, 70, 71, 30, 31}, flat(out), 0.0);
        assertArrayEquals(new long[]{4, 2}, out.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void stitchZeroesTheRowsNoIndexNames(Nd4jBackend backend) {
        // rows 0 and 2 are named by no index
        INDArray i0 = Nd4j.createFromArray(1L, 4L);
        INDArray i1 = Nd4j.createFromArray(3L);
        INDArray d0 = Nd4j.createFromArray(new double[][]{{1, 2}, {3, 4}});
        INDArray d1 = Nd4j.createFromArray(new double[][]{{5, 6}});
        double[] expected = {0, 0, 1, 2, 0, 0, 5, 6, 3, 4};

        INDArray stitched = stitch(new INDArray[]{i0, i1}, new INDArray[]{d0, d1});
        assertArrayEquals(new long[]{5, 2}, stitched.shape());
        assertArrayEquals(expected, flat(stitched), 0.0, "output of the op");

        // an output that starts as NaN: the rows no index names must be written too
        INDArray out = Nd4j.valueArrayOf(new long[]{5, 2}, Double.NaN, DataType.DOUBLE);
        Nd4j.exec(DynamicCustomOp.builder("dynamic_stitch").addInputs(i0, i1, d0, d1).addOutputs(out).build());
        assertArrayEquals(expected, flat(out), 0.0, "output of NaN");

        // an output that is a view of a larger array: the rest of the parent stays as it was
        INDArray parent = Nd4j.valueArrayOf(new long[]{6, 2}, Double.NaN, DataType.DOUBLE);
        INDArray view = parent.get(NDArrayIndex.interval(1, 6), NDArrayIndex.all());
        Nd4j.exec(DynamicCustomOp.builder("dynamic_stitch").addInputs(i0, i1, d0, d1).addOutputs(view).build());
        assertArrayEquals(expected, flat(view), 0.0, "output view");
        for (int c = 0; c < 2; c++)
            assertTrue(Double.isNaN(parent.getDouble(0, c)), "the first row of the parent was written");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void stitchingThePartitionsGivesTheDataBack(Nd4jBackend backend) {
        Random random = new Random(19);
        for (int n : new int[]{9, 1025, 70001}) {
            for (long[] rowShape : new long[][]{{}, {3}, {2, 2}}) {
                int partitions = 4;
                long[] ids = ids(random, n, partitions, 0, true);
                long[] shape = new long[1 + rowShape.length];
                shape[0] = n;
                System.arraycopy(rowShape, 0, shape, 1, rowShape.length);
                INDArray x = data(random, DataType.FLOAT, shape);
                INDArray indices = indexArray(ids, DataType.INT32, n);
                INDArray[] parts = partition(x, indices, partitions);
                // where each partition's slices came from: the partition of 0 .. n - 1
                INDArray[] positions = partition(Nd4j.createFromArray(sequence(n)), indices, partitions);
                INDArray back = stitch(positions, parts);
                assertArrayEquals(flat(x), flat(back), 0.0, n + " slices of " + Arrays.toString(rowShape));
            }
        }
    }

    private static int[] sequence(int n) {
        int[] s = new int[n];
        for (int i = 0; i < n; i++)
            s[i] = i;
        return s;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void stitchWithAnUnusedPartitionStillStitchesTheOthers(Nd4jBackend backend) {
        // partitions 1 and 3 got nothing: the stitch still has partitions 0 and 2 to merge, and so does the gradient
        long[] ids = {0, 2, 2, 0, 2};
        INDArray x = Nd4j.createFromArray(new double[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}});
        INDArray indices = Nd4j.createFromArray(ids);
        INDArray[] parts = partition(x, indices, 4);
        INDArray[] positions = partition(Nd4j.createFromArray(sequence(5)).castTo(DataType.INT64), indices, 4);
        assertEquals(0, parts[1].length());
        assertEquals(0, parts[3].length());
        assertArrayEquals(flat(x), flat(stitch(positions, parts)), 0.0);

        INDArray[] grads = gradients(new Random(20), parts, DataType.DOUBLE);
        INDArray gradX = partitionBp(x, indices, grads, 4, null);
        double[][] gradValues = {flat(grads[0]), new double[0], flat(grads[2]), new double[0]};
        assertArrayEquals(partitionBpReference(ids, gradValues, 4, 2), flat(gradX), 0.0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void stitchGradientIsAGatherOfTheOutputsGradient(Nd4jBackend backend) {
        DataType previous = Nd4j.defaultFloatingPointType();
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        try {
            Random random = new Random(21);
            for (long[] rowShape : new long[][]{{}, {3}, {2, 2}}) {
                int rows = 24;
                List<Long> order = new ArrayList<>();
                for (long r = 0; r < rows; r++)
                    order.add(r);
                Collections.shuffle(order, random);
                long[][] ids = new long[2][];
                ids[0] = new long[10];
                ids[1] = new long[14];
                for (int e = 0; e < 10; e++)
                    ids[0][e] = order.get(e);
                for (int e = 0; e < 14; e++)
                    ids[1][e] = order.get(10 + e);
                int sliceLength = (int) size(rowShape);

                SameDiff sd = SameDiff.create();
                SDVariable[] indexVars = new SDVariable[2];
                SDVariable[] dataVars = new SDVariable[2];
                double[][] dataValues = new double[2][];
                for (int p = 0; p < 2; p++) {
                    indexVars[p] = sd.constant("ids" + p, indexArray(ids[p], DataType.INT64, ids[p].length));
                    long[] shape = new long[1 + rowShape.length];
                    shape[0] = ids[p].length;
                    System.arraycopy(rowShape, 0, shape, 1, rowShape.length);
                    INDArray d = data(random, DataType.DOUBLE, shape);
                    dataValues[p] = flat(d);
                    dataVars[p] = sd.var("data" + p, d);
                }
                long[] outShape = new long[1 + rowShape.length];
                outShape[0] = rows;
                System.arraycopy(rowShape, 0, outShape, 1, rowShape.length);
                INDArray weights = data(random, DataType.DOUBLE, outShape);
                SDVariable stitched = sd.dynamicStitch("stitched", indexVars, dataVars);
                SDVariable loss = stitched.mul(sd.constant("weights", weights)).sum();
                loss.rename("loss");
                sd.setLossVariables("loss");

                Map<String, INDArray> grads = sd.calculateGradients(Collections.emptyMap(), "data0", "data1");
                double[] w = flat(weights);
                for (int p = 0; p < 2; p++) {
                    double[] expected = new double[ids[p].length * sliceLength];
                    for (int e = 0; e < ids[p].length; e++)
                        System.arraycopy(w, (int) ids[p][e] * sliceLength, expected, e * sliceLength, sliceLength);
                    assertArrayEquals(expected, flat(grads.get("data" + p)), 1e-12,
                            "gradient of the data of input " + p + ", rows of " + Arrays.toString(rowShape));
                }
            }
        } finally {
            Nd4j.setDefaultDataTypes(previous, previous);
        }
    }
}
