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
import org.nd4j.linalg.api.ops.custom.Tri;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * triu, triu_bp and tri against the triangle's definition: triu(k) keeps the entries of each matrix (the last two
 * dimensions) with column - row >= k and zeroes the others, a vector is repeated into the rows of a square matrix first
 * and tri(rows, cols, k) holds ones where column <= row + k and zeros elsewhere. The CPU helper took row 0 and column
 * 1 for every entry of a matrix with one row or one column, restarted the counter that picks the entry of a vector at
 * the start of each thread's chunk (wrong from n * n = 2048 entries, when the work is split), and tri left the entries
 * outside its triangle as they were in an output that is not initialized.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TriangularOpsTest extends BaseNd4jTestWithBackends {

    private static final DataType[] TYPES = {DataType.FLOAT, DataType.DOUBLE};

    /** The values of an array as they are in C order, whatever its layout. */
    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    /** The same values in another layout: F order, or a stepped view into a larger array. */
    private static INDArray layout(INDArray c, int variant) {
        switch (variant) {
            case 1:
                return c.dup('f');
            case 2: {
                long[] shape = c.shape();
                long[] big = new long[shape.length];
                INDArrayIndex[] steps = new INDArrayIndex[shape.length];
                for (int i = 0; i < shape.length; i++) {
                    big[i] = 2 * shape[i];
                    steps[i] = NDArrayIndex.interval(0, 2, big[i]);
                }
                INDArray view = Nd4j.zeros(c.dataType(), big).addi(-7.0).get(steps);
                view.assign(c);
                return view;
            }
            default:
                return c.dup('c');
        }
    }

    private static String layoutName(int variant) {
        return variant == 0 ? "C order" : variant == 1 ? "F order" : "stepped view";
    }

    /** Distinct nonzero values, so that a zero marks a masked entry. */
    private static INDArray values(DataType type, long... shape) {
        long n = 1;
        for (long s : shape)
            n *= s;
        return Nd4j.linspace(1, n, n, DataType.DOUBLE).castTo(type).reshape(shape);
    }

    private static double[] triuReference(double[] x, long[] shape, int k) {
        int cols = (int) shape[shape.length - 1];
        int rows = (int) shape[shape.length - 2];
        double[] expected = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            int row = (i / cols) % rows;
            int col = i % cols;
            expected[i] = col - row >= k ? x[i] : 0.0;
        }
        return expected;
    }

    private static INDArray triu(INDArray x, int k) {
        return Nd4j.exec(DynamicCustomOp.builder("triu").addInputs(x).addIntegerArguments(k).build())[0];
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void triuKeepsTheEntriesOnAndAboveItsDiagonal(Nd4jBackend backend) {
        long[][] shapes = {{3, 5}, {7, 7}, {5, 3}, {1, 6}, {6, 1}, {1, 1}, {2, 3, 4}, {2, 2, 5, 6}, {100, 100},
                {3, 40, 50}};
        for (DataType type : TYPES) {
            for (long[] shape : shapes) {
                INDArray c = values(type, shape);
                for (int k : new int[]{-3, -1, 0, 1, 2, 5}) {
                    double[] expected = triuReference(logical(c), shape, k);
                    for (int variant = 0; variant < 3; variant++) {
                        INDArray result = triu(layout(c, variant), k);
                        assertArrayEquals(shape, result.shape());
                        assertArrayEquals(expected, logical(result), 0.0,
                                type + " triu(" + k + ") of " + Arrays.toString(shape) + " in " + layoutName(variant));
                    }
                }
            }
        }
    }

    /** A vector becomes a square matrix whose every row is the vector, then the triangle is taken. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void triuOfAVectorRepeatsItInEveryRow(Nd4jBackend backend) {
        for (DataType type : TYPES) {
            for (int n : new int[]{1, 2, 5, 33, 64, 70}) {
                INDArray c = values(type, n);
                double[] v = logical(c);
                for (int k : new int[]{-2, 0, 1, 3}) {
                    double[] expected = new double[n * n];
                    for (int row = 0; row < n; row++)
                        for (int col = 0; col < n; col++)
                            expected[row * n + col] = col - row >= k ? v[col] : 0.0;
                    for (int variant : new int[]{0, 2}) {
                        INDArray result = triu(layout(c, variant), k);
                        assertArrayEquals(new long[]{n, n}, result.shape());
                        assertArrayEquals(expected, logical(result), 0.0,
                                type + " triu(" + k + ") of a vector of " + n + " in " + layoutName(variant));
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void triuBackpropagatesThroughTheSameTriangle(Nd4jBackend backend) {
        long[][] shapes = {{4, 6}, {1, 5}, {5, 1}, {2, 30, 40}};
        for (DataType type : TYPES) {
            for (long[] shape : shapes) {
                INDArray input = values(type, shape);
                INDArray grad = values(type, shape).addi(1000.0);
                for (int k : new int[]{-1, 0, 2}) {
                    double[] expected = triuReference(logical(grad), shape, k);
                    for (int variant = 0; variant < 3; variant++) {
                        INDArray result = Nd4j.exec(DynamicCustomOp.builder("triu_bp")
                                .addInputs(input, layout(grad, variant)).addIntegerArguments(k).build())[0];
                        assertArrayEquals(expected, logical(result), 0.0,
                                type + " triu_bp(" + k + ") of " + Arrays.toString(shape) + " in " + layoutName(variant));
                    }
                }
            }
        }
    }

    /** tri writes its whole output: the entries outside the triangle are zeros whatever the output held. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void triFillsOnesBelowItsDiagonalAndZerosElsewhere(Nd4jBackend backend) {
        int[][] sizes = {{1, 5}, {5, 1}, {1, 1}, {4, 4}, {6, 3}, {3, 6}, {70, 70}, {33, 65}};
        for (int[] size : sizes) {
            int rows = size[0], cols = size[1];
            for (int k : new int[]{-2, 0, 1, 3}) {
                double[] expected = new double[rows * cols];
                for (int row = 0; row < rows; row++)
                    for (int col = 0; col < cols; col++)
                        expected[row * cols + col] = col <= row + k ? 1.0 : 0.0;

                INDArray fresh = Nd4j.exec(new Tri(rows, cols, k))[0];
                assertArrayEquals(expected, logical(fresh), 0.0, "tri(" + rows + ", " + cols + ", " + k + ")");

                INDArray recycled = Nd4j.create(DataType.FLOAT, rows, cols).assign(Float.NaN);
                Tri op = new Tri(rows, cols, k);
                op.addOutputArgument(recycled);
                Nd4j.exec(op);
                assertArrayEquals(expected, logical(recycled), 0.0,
                        "tri(" + rows + ", " + cols + ", " + k + ") into an output that held NaN");
            }
        }
    }
}
