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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * The scatter ops apply update k to the output slice named by index k (index row k for the scatter_nd ops). Where an
 * index repeats, its updates apply one after another in index order, whether or not the op is asked to lock; an
 * index outside the output is skipped; with no indices the output is the input. Every value here is a small integer
 * or a power of two, so each result is exact in any order of accumulation and the expectations are exact.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class ScatterSemanticsTest extends BaseNd4jTestWithBackends {

    private static final String[] OPS = {"scatter_add", "scatter_sub", "scatter_mul", "scatter_div", "scatter_upd",
            "scatter_max", "scatter_min"};
    private static final String[] ND_OPS = {"scatter_nd_update", "scatter_nd_add", "scatter_nd_sub"};
    private static final boolean[] LOCKS = {true, false};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void repeatedIndicesApplyInIndexOrder(Nd4jBackend backend) {
        INDArray ref = Nd4j.createFromArray(new float[][]{{1, -2, 3}, {4, 5, -6}, {-7, 8, 9}, {10, -11, 12},
                {13, 14, -15}, {-16, 17, 18}});
        long[] index = {0, 2, 0, 5, 2, 0};
        INDArray updates = Nd4j.createFromArray(new float[][]{{2, -4, 0.5f}, {4, 0.25f, -2}, {-0.5f, 2, 4},
                {8, -2, 0.5f}, {0.5f, 4, -4}, {-2, -0.5f, 2}});
        checkAllOps("repeated indices", ref, Nd4j.createFromArray(index), updates);
    }

    /** Each layout of the updates the ops accept pairs index k with the k-th update slice. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyAcceptedLayoutPairsIndexAndUpdateSlice(Nd4jBackend backend) {
        // A [1, n] output: its one slice lies along the unit dimension.
        checkAllOps("[1, 4] output", Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}}),
                Nd4j.createFromArray(0L, 0L),
                Nd4j.createFromArray(new float[][]{{2, 4, -2, 0.5f}, {0.5f, -4, 2, 8}}));
        // Vector indices of rank 2 with updates [indices.length] + output.shape[1:].
        checkAllOps("[1, 3] indices", Nd4j.linspace(DataType.FLOAT, 1, 15, 1).reshape(5, 3),
                Nd4j.createFromArray(new long[][]{{1, 3, 1}}),
                Nd4j.createFromArray(new float[][]{{2, 4, 0.5f}, {-2, 8, 4}, {0.25f, -0.5f, 2}}));
        // A vector output with indices and updates of one shape.
        checkAllOps("vector output", Nd4j.createFromArray(1f, 2f, 3f, 4f, 5f, 6f),
                Nd4j.createFromArray(new long[][]{{1, 4}, {1, 0}}),
                Nd4j.createFromArray(new float[][]{{2, 4}, {-0.5f, 8}}));
        // Updates indices.shape + output.shape[1:].
        checkAllOps("[2, 2] indices", Nd4j.linspace(DataType.FLOAT, 1, 8, 1).reshape(4, 2),
                Nd4j.createFromArray(new long[][]{{0, 3}, {3, 1}}),
                Nd4j.linspace(DataType.FLOAT, -4, 8, 1).reshape(2, 2, 2).addi(0.5));
        // A rank 3 output.
        checkAllOps("rank 3 output", Nd4j.linspace(DataType.FLOAT, 1, 12, 1).reshape(3, 2, 2),
                Nd4j.createFromArray(2L, 0L, 2L),
                Nd4j.createFromArray(new float[][][]{{{2, 4}, {-2, 0.5f}}, {{8, -4}, {0.25f, 2}},
                        {{-0.5f, 2}, {4, -8}}}));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void moreUpdatesThanOutputElements(Nd4jBackend backend) {
        INDArray updates = Nd4j.createFromArray(new float[][]{{2, 4}, {-2, 0.5f}, {8, 0.25f}, {4, -4},
                {0.5f, 2}, {-0.5f, 8}, {2, -2}, {0.25f, 4}});
        checkAllOps("8 updates into 2 slices", Nd4j.createFromArray(new float[][]{{1, -2}, {3, 4}}),
                Nd4j.createFromArray(0L, 1L, 0L, 1L, 0L, 1L, 0L, 1L), updates);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void indicesOutsideTheOutputAreSkipped(Nd4jBackend backend) {
        checkAllOps("out of range", Nd4j.createFromArray(new float[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8}}),
                Nd4j.createFromArray(1L, 9L, -1L, 2L),
                Nd4j.createFromArray(new float[][]{{2, 4}, {8, 8}, {-8, -8}, {0.5f, -2}}));
    }

    /** The updates are applied in the output's type. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void updatesOfAnotherTypeAreCast(Nd4jBackend backend) {
        checkAllOps("double updates into float", Nd4j.createFromArray(new float[][]{{1, 2}, {3, 4}, {5, 6}}),
                Nd4j.createFromArray(2L, 0L, 2L),
                Nd4j.createFromArray(new double[][]{{0.5, 2}, {4, -0.25}, {-2, 8}}));
    }

    /** With no indices there is nothing to scatter: the output is the input (scatter_nd's, zeros). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void emptyIndicesLeaveTheInput(Nd4jBackend backend) {
        INDArray ref = Nd4j.createFromArray(new float[][]{{1, 2, 3}, {4, 5, 6}});
        INDArray updates = Nd4j.create(DataType.FLOAT, 0, 3);
        for (String op : OPS) {
            INDArray out = Nd4j.valueArrayOf(new long[]{2, 3}, -9f);
            Nd4j.exec(DynamicCustomOp.builder(op).addInputs(ref, Nd4j.create(DataType.INT64, 0), updates)
                    .addOutputs(out).build());
            assertExact(op, ref, out);
        }
        INDArray noRows = Nd4j.create(DataType.INT64, 0, 1);
        for (String op : ND_OPS) {
            INDArray out = Nd4j.valueArrayOf(new long[]{2, 3}, -9f);
            Nd4j.exec(DynamicCustomOp.builder(op).addInputs(ref, noRows, updates).addOutputs(out).build());
            assertExact(op, ref, out);
        }
        INDArray out = Nd4j.valueArrayOf(new long[]{2, 3}, -9f);
        Nd4j.exec(DynamicCustomOp.builder("scatter_nd").addInputs(noRows, updates, Nd4j.createFromArray(2L, 3L))
                .addOutputs(out).build());
        assertExact("scatter_nd", Nd4j.zeros(DataType.FLOAT, 2, 3), out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void repeatedIndexRowsApplyInOrder(Nd4jBackend backend) {
        INDArray ref = Nd4j.createFromArray(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
        // Index depth 1, rows of ref; and depth 2, single elements.
        INDArray rows = Nd4j.createFromArray(new long[][]{{1}, {3}, {1}, {0}, {1}});
        INDArray rowUpdates = Nd4j.createFromArray(new float[][]{{10, 20, 30}, {40, 50, 60}, {70, 80, 90},
                {-1, -2, -3}, {5, 6, 7}});
        INDArray cells = Nd4j.createFromArray(new long[][]{{0, 1}, {2, 2}, {0, 1}, {3, 0}, {0, 1}, {9, 0}});
        INDArray cellUpdates = Nd4j.createFromArray(100f, 200f, 300f, 400f, 500f, 600f);
        for (boolean lock : LOCKS) {
            for (String op : ND_OPS) {
                assertExact(op + " rows lock " + lock, referenceNd(op, ref, rows, rowUpdates),
                        scatterNd(op, ref, rows, rowUpdates, lock));
                assertExact(op + " cells lock " + lock, referenceNd(op, ref, cells, cellUpdates),
                        scatterNd(op, ref, cells, cellUpdates, lock));
            }
        }
        // scatter_nd adds every update onto zeros.
        INDArray out = Nd4j.valueArrayOf(new long[]{4, 3}, -9f);
        Nd4j.exec(DynamicCustomOp.builder("scatter_nd").addInputs(rows, rowUpdates, Nd4j.createFromArray(4L, 3L))
                .addOutputs(out).build());
        assertExact("scatter_nd", referenceNd("scatter_nd_add", Nd4j.zeros(DataType.FLOAT, 4, 3), rows, rowUpdates),
                out);
    }

    /** get_rows_bp sums the gradient rows of repeated indices in index order, so the sums are reproducible. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void getRowsBackpropAddsRepeatedRowsInIndexOrder(Nd4jBackend backend) {
        int n = 64, width = 33, numRows = 7;
        Nd4j.getRandom().setSeed(7);
        INDArray grad = Nd4j.rand(DataType.FLOAT, n, width).subi(0.5);
        long[] index = new long[n];
        for (int i = 0; i < n; i++) {
            index[i] = (i * 5L) % numRows;
        }
        float[][] expected = new float[numRows][width];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < width; j++) {
                expected[(int) index[i]][j] += grad.getFloat(i, j);
            }
        }
        INDArray out = Nd4j.create(DataType.FLOAT, numRows, width);
        Nd4j.exec(DynamicCustomOp.builder("get_rows_bp").addInputs(grad, Nd4j.createFromArray(index))
                .addIntegerArguments(numRows).addOutputs(out).build());
        assertExact("get_rows_bp", Nd4j.createFromArray(expected), out);
    }

    private static void checkAllOps(String label, INDArray ref, INDArray indices, INDArray updates) {
        long numSlices = ref.size(0);
        long sliceLength = ref.length() / numSlices;
        long[] index = indices.dup().data().asLong();
        INDArray slices = updates.castTo(ref.dataType()).dup('c').reshape(index.length, sliceLength);
        for (String op : OPS) {
            INDArray expected = reference(op, ref, index, slices);
            for (boolean lock : LOCKS) {
                assertExact(label + ": " + op + " lock " + lock, expected, scatter(op, ref, indices, updates, lock));
            }
        }
    }

    private static INDArray scatter(String op, INDArray ref, INDArray indices, INDArray updates, boolean lock) {
        INDArray out = ref.ulike();
        Nd4j.exec(DynamicCustomOp.builder(op).addInputs(ref, indices, updates).addOutputs(out)
                .addBooleanArguments(lock, false).build());
        return out;
    }

    private static INDArray scatterNd(String op, INDArray ref, INDArray indices, INDArray updates, boolean lock) {
        INDArray out = ref.ulike();
        Nd4j.exec(DynamicCustomOp.builder(op).addInputs(ref, indices, updates).addOutputs(out)
                .addBooleanArguments(lock, false).build());
        return out;
    }

    /** The ops' definition: update slice k applies to slice index[k] of ref, one after another in index order. */
    private static INDArray reference(String op, INDArray ref, long[] index, INDArray slices) {
        long numSlices = ref.size(0);
        INDArray out = ref.dup('c').reshape(numSlices, ref.length() / numSlices);
        for (int k = 0; k < index.length; k++) {
            if (index[k] < 0 || index[k] >= numSlices) continue;
            for (int p = 0; p < out.size(1); p++) {
                out.putScalar(index[k], p, apply(op, out.getDouble(index[k], p), slices.getDouble(k, p)));
            }
        }
        return out.reshape(ref.shape());
    }

    /** As reference, for index rows of indexDepth coordinates naming ref's leading dimensions. */
    private static INDArray referenceNd(String op, INDArray ref, INDArray indices, INDArray updates) {
        int indexDepth = (int) indices.size(indices.rank() - 1);
        long numSlices = 1;
        for (int j = 0; j < indexDepth; j++) {
            numSlices *= ref.size(j);
        }
        long sliceLength = ref.length() / numSlices;
        long[] coordinates = indices.dup().data().asLong();
        int rows = coordinates.length / indexDepth;
        INDArray slices = updates.dup('c').reshape(rows, sliceLength);
        INDArray out = ref.dup('c').reshape(numSlices, sliceLength);
        for (int r = 0; r < rows; r++) {
            long slice = 0;
            boolean inRange = true;
            for (int j = 0; j < indexDepth; j++) {
                long coordinate = coordinates[r * indexDepth + j];
                inRange &= coordinate >= 0 && coordinate < ref.size(j);
                slice = slice * ref.size(j) + coordinate;
            }
            if (!inRange) continue;
            for (int p = 0; p < sliceLength; p++) {
                out.putScalar(slice, p, apply(op, out.getDouble(slice, p), slices.getDouble(r, p)));
            }
        }
        return out.reshape(ref.shape());
    }

    private static double apply(String op, double z, double y) {
        switch (op) {
            case "scatter_add":
            case "scatter_nd_add":
                return z + y;
            case "scatter_sub":
            case "scatter_nd_sub":
                return z - y;
            case "scatter_mul":
                return z * y;
            case "scatter_div":
                return z / y;
            case "scatter_upd":
            case "scatter_nd_update":
                return y;
            case "scatter_max":
                return Math.max(z, y);
            case "scatter_min":
                return Math.min(z, y);
            default:
                throw new IllegalArgumentException(op);
        }
    }

    private static void assertExact(String label, INDArray expected, INDArray actual) {
        assertArrayEquals(expected.shape(), actual.shape(), label + " shape");
        assertArrayEquals(expected.dup('c').data().asDouble(), actual.dup('c').data().asDouble(), 0.0, label);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
