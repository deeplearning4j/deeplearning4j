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
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A row [1, N] and a column [N, 1] each count as a vector and hold the same number of elements, but they broadcast
 * to [N, N], not to the shape of either operand. {@code ShapeUtils::evalBroadcastShapeInfo}, which the shape function
 * of every broadcastable op, {@code broadcast_dynamic_shape} and {@code Where} use, answered with the first operand's
 * shape for such a pair, so an op's output was allocated as a row or a column (and the dynamic shape plan sized its
 * slot the same way) while the broadcast writes N * N elements.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class BroadcastVectorShapeTest extends BaseNd4jTestWithBackends {

    private static INDArray row(int n) {
        return Nd4j.linspace(1, n, n, DataType.FLOAT).reshape(1, n);
    }

    private static INDArray column(int n) {
        return Nd4j.linspace(1, n, n, DataType.FLOAT).reshape(n, 1).mul(1.5).add(0.25);
    }

    /** The element of a row or a column at position (i, j) of the matrix it is broadcast to. */
    private static double at(INDArray x, long i, long j) {
        return x.getDouble(x.size(0) == 1 ? 0 : i, x.size(1) == 1 ? 0 : j);
    }

    private static INDArray broadcast(INDArray x, INDArray y, DoubleBinaryOperator f) {
        long rows = Math.max(x.size(0), y.size(0));
        long columns = Math.max(x.size(1), y.size(1));
        INDArray result = Nd4j.create(DataType.FLOAT, rows, columns);
        for (long i = 0; i < rows; i++)
            for (long j = 0; j < columns; j++)
                result.putScalar(i, j, f.applyAsDouble(at(x, i, j), at(y, i, j)));
        return result;
    }

    // ------------------------------------------------------------------------------- broadcast_dynamic_shape

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void broadcastDynamicShapeOfARowAndAColumnIsASquare(Nd4jBackend backend) {
        int[][][] cases = {
                // the shape of x, the shape of y, the shape they broadcast to
                {{2, 1}, {1, 2}, {2, 2}},
                {{1, 2}, {2, 1}, {2, 2}},
                {{1, 5}, {5, 1}, {5, 5}},
                {{5, 1}, {1, 5}, {5, 5}},
                {{1, 1}, {1, 1}, {1, 1}},
        };
        for (int[][] c : cases) {
            INDArray out = Nd4j.create(DataType.INT, 2);
            DynamicCustomOp op = DynamicCustomOp.builder("broadcast_dynamic_shape")
                    .addInputs(Nd4j.createFromArray(c[0]), Nd4j.createFromArray(c[1]))
                    .addOutputs(out)
                    .build();
            Nd4j.getExecutioner().exec(op);
            assertEquals(Nd4j.createFromArray(c[2]), out,
                    "broadcast_dynamic_shape of " + Arrays.toString(c[0]) + " and " + Arrays.toString(c[1]));
        }
    }

    // ------------------------------------------------------------------------------------ broadcastable ops

    private static final class BinaryCase {
        final String name;
        final DoubleBinaryOperator expected;

        BinaryCase(String name, DoubleBinaryOperator expected) {
            this.name = name;
            this.expected = expected;
        }
    }

    private static List<BinaryCase> binaryCases() {
        List<BinaryCase> cases = new ArrayList<>();
        cases.add(new BinaryCase("add", (a, b) -> a + b));
        cases.add(new BinaryCase("subtract", (a, b) -> a - b));
        cases.add(new BinaryCase("multiply", (a, b) -> a * b));
        cases.add(new BinaryCase("divide", (a, b) -> a / b));
        cases.add(new BinaryCase("maximum", Math::max));
        cases.add(new BinaryCase("minimum", Math::min));
        cases.add(new BinaryCase("squaredsubtract", (a, b) -> (a - b) * (a - b)));
        cases.add(new BinaryCase("reversesubtract", (a, b) -> b - a));
        return cases;
    }

    /** A DynamicCustomOp has no Java-side shape inference, so its output shape is the native shape function's. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void broadcastableOpsOutputASquareForARowAndAColumn(Nd4jBackend backend) {
        List<String> failures = new ArrayList<>();
        for (int n : new int[]{2, 5}) {
            INDArray row = row(n);
            INDArray column = column(n);
            for (BinaryCase c : binaryCases()) {
                for (boolean rowFirst : new boolean[]{true, false}) {
                    INDArray x = rowFirst ? row : column;
                    INDArray y = rowFirst ? column : row;
                    String label = c.name + " of " + Arrays.toString(x.shape()) + " and " + Arrays.toString(y.shape());
                    try {
                        List<DataBuffer> descriptors = DynamicCustomOp.builder(c.name).addInputs(x, y).build()
                                .calculateOutputShape();
                        assertEquals(1, descriptors.size(), label + ": the number of outputs");
                        assertArrayEquals(new long[]{n, n}, Shape.shape(descriptors.get(0).asLong()),
                                label + ": the shape the shape function describes");

                        // the executioner allocates the output from that shape
                        INDArray[] out = Nd4j.exec(DynamicCustomOp.builder(c.name).addInputs(x, y).build());
                        Nd4j.getExecutioner().commit();
                        assertEquals(1, out.length, label + ": the number of outputs");
                        assertArrayEquals(new long[]{n, n}, out[0].shape(), label + ": the shape of the output");
                        INDArray expected = broadcast(x, y, c.expected);
                        assertTrue(expected.equalsWithEps(out[0], 1e-5),
                                () -> label + ": the output is " + out[0] + ", expected " + expected);
                    } catch (RuntimeException | AssertionError e) {
                        failures.add(label + ": " + e.getMessage());
                    }
                }
            }
        }
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    /** A row [1, 0] and a column [0, 1] are both empty and broadcast to the empty [0, 0]. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void anEmptyRowAndAnEmptyColumnBroadcastToAnEmptySquare(Nd4jBackend backend) {
        INDArray row = Nd4j.create(DataType.FLOAT, 1, 0);
        INDArray column = Nd4j.create(DataType.FLOAT, 0, 1);
        for (boolean rowFirst : new boolean[]{true, false}) {
            INDArray x = rowFirst ? row : column;
            INDArray y = rowFirst ? column : row;
            List<DataBuffer> descriptors = DynamicCustomOp.builder("add").addInputs(x, y).build()
                    .calculateOutputShape();
            assertEquals(1, descriptors.size());
            assertArrayEquals(new long[]{0, 0}, Shape.shape(descriptors.get(0).asLong()),
                    "add of " + Arrays.toString(x.shape()) + " and " + Arrays.toString(y.shape()));
        }
    }

    // --------------------------------------------------------------------------- SameDiff, plan on and off

    private static final class GraphCase {
        final String name;
        final BinaryOperator<SDVariable> build;
        final DoubleBinaryOperator expected;

        GraphCase(String name, BinaryOperator<SDVariable> build, DoubleBinaryOperator expected) {
            this.name = name;
            this.build = build;
            this.expected = expected;
        }
    }

    /**
     * The dynamic shape plan computes the output shapes of the graph with the native shape functions and sizes the slots
     * from them.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sameDiffBroadcastsARowAndAColumnToASquare(Nd4jBackend backend) {
        List<GraphCase> cases = new ArrayList<>();
        cases.add(new GraphCase("add", (x, y) -> x.add("out", y), (a, b) -> a + b));
        cases.add(new GraphCase("sub", (x, y) -> x.sub("out", y), (a, b) -> a - b));
        cases.add(new GraphCase("mul", (x, y) -> x.mul("out", y), (a, b) -> a * b));
        cases.add(new GraphCase("div", (x, y) -> x.div("out", y), (a, b) -> a / b));

        int n = 5;
        INDArray row = row(n);
        INDArray column = column(n);
        boolean planBefore = InferenceSession.isDynamicShapePlanEnabled();
        List<String> failures = new ArrayList<>();
        try {
            for (boolean plan : new boolean[]{false, true}) {
                InferenceSession.setDynamicShapePlanEnabled(plan);
                for (GraphCase c : cases) {
                    for (boolean rowFirst : new boolean[]{true, false}) {
                        String label = c.name + (rowFirst ? " of a row and a column" : " of a column and a row")
                                + ", dynamic shape plan " + (plan ? "on" : "off");
                        try {
                            SameDiff sd = SameDiff.create();
                            SDVariable r = sd.placeHolder("row", DataType.FLOAT, 1, n);
                            SDVariable col = sd.placeHolder("col", DataType.FLOAT, n, 1);
                            c.build.apply(rowFirst ? r : col, rowFirst ? col : r);
                            Map<String, INDArray> placeholders = new HashMap<>();
                            placeholders.put("row", row);
                            placeholders.put("col", column);
                            INDArray actual = sd.output(placeholders, "out").get("out");
                            Nd4j.getExecutioner().commit();

                            assertArrayEquals(new long[]{n, n}, actual.shape(), label + ": the shape of the output");
                            INDArray expected = rowFirst ? broadcast(row, column, c.expected)
                                    : broadcast(column, row, c.expected);
                            assertTrue(expected.equalsWithEps(actual, 1e-5),
                                    () -> label + ": the output is " + actual + ", expected " + expected);
                        } catch (RuntimeException | AssertionError e) {
                            failures.add(label + ": " + e.getMessage());
                        }
                    }
                }
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(planBefore);
        }
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
