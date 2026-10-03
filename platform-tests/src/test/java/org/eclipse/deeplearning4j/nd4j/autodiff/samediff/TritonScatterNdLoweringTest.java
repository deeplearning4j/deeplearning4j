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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * scatter_nd_update compiled to a Triton section must equal the native op: index row r names the output's leading
 * coordinates, rows apply in order (the last row wins a repeated destination), and a row with a coordinate outside
 * the output is skipped. The lowering it replaced assumed index depth 1, sized its grid by the output (so it never
 * applied updates beyond the output's length, and read indices and updates past their ends when there were fewer),
 * and raced its copy of the input against the scatter.
 */
class TritonScatterNdLoweringTest {

    @ParameterizedTest
    @ValueSource(strings = {"repeatedRows", "elementRows", "moreUpdatesThanOutput", "oneUpdateIntoMany",
            "manyRows"})
    void scatterNdUpdateMatchesTheNativeContract(String name) {
        assumeTrue(Nd4j.backends().isCudaAvailable() && Nd4j.getNativeOps().isTritonAvailable());
        boolean previousAll = Nd4j.getEnvironment().tritonCompileAll();
        String previousTypes = Nd4j.getEnvironment().tritonIncludeTypes();
        try {
            Nd4j.getEnvironment().setTritonCompileAll(true);
            Nd4j.getEnvironment().setTritonIncludeTypes("SCATTER_ND_UPDATE");
            for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.HALF}) {
                run(name, type);
            }
        } finally {
            Nd4j.getEnvironment().setTritonCompileAll(previousAll);
            Nd4j.getEnvironment().setTritonIncludeTypes(previousTypes);
        }
    }

    private static void run(String name, DataType type) {
        long[] refShape;
        long[][] rows;
        switch (name) {
            case "repeatedRows":
                refShape = new long[]{6, 3};
                rows = new long[][]{{1}, {4}, {1}, {0}, {1}};
                break;
            case "elementRows":
                refShape = new long[]{5, 4};
                rows = new long[][]{{0, 1}, {4, 3}, {0, 1}, {9, 0}, {2, 2}, {-1, 2}};
                break;
            case "moreUpdatesThanOutput":
                refShape = new long[]{2, 3};
                rows = new long[][]{{0}, {1}, {0}, {1}, {0}, {1}, {0}};
                break;
            case "oneUpdateIntoMany":
                refShape = new long[]{4096};
                rows = new long[][]{{17}};
                break;
            case "manyRows": {
                refShape = new long[]{8, 2};
                Random random = new Random(3);
                rows = new long[40][1];
                for (long[] row : rows) row[0] = random.nextInt(9);   // 8 is outside the output
                break;
            }
            default:
                throw new IllegalArgumentException(name);
        }
        int depth = rows[0].length;
        long sliceLength = 1;
        for (int d = depth; d < refShape.length; d++) sliceLength *= refShape[d];
        long[] updatesShape = new long[1 + refShape.length - depth];
        updatesShape[0] = rows.length;
        System.arraycopy(refShape, depth, updatesShape, 1, refShape.length - depth);

        String label = name + " " + type;
        try (SameDiff graph = SameDiff.create()) {
            SDVariable ref = graph.placeHolder("ref", type, refShape);
            SDVariable indices = graph.placeHolder("indices", DataType.INT64, rows.length, depth);
            SDVariable updates = graph.placeHolder("updates", type, updatesShape);
            graph.scatterNdUpdate("out", ref, indices, updates);
            graph.setGraphExecutionMode(GraphExecutionMode.TRITON);
            INDArray indexArray = Nd4j.createFromArray(rows);
            for (int iteration = 0; iteration < 6; iteration++) {
                // Small integers and halves: exact in every type, so the comparison is exact.
                INDArray refArray = Nd4j.linspace(DataType.DOUBLE, 1, lengthOf(refShape), 1)
                        .muli(0.5).reshape(refShape).castTo(type);
                INDArray updateArray = Nd4j.linspace(DataType.DOUBLE, -100 - iteration, lengthOf(updatesShape), 1)
                        .reshape(updatesShape).castTo(type);
                double[] expected = reference(refArray, rows, updateArray, refShape, sliceLength);
                try (INDArray output = graph.output(Map.of("ref", refArray, "indices", indexArray,
                        "updates", updateArray), "out").get("out")) {
                    assertArrayEquals(refShape, output.shape(), label);
                    assertArrayEquals(expected, output.dup('c').data().asDouble(), 0.0,
                            label + " iteration " + iteration);
                }
            }
            assertTrue(DspPlanAssertions.getSegmentCompiledBackend(graph, 0).contains("Triton"),
                    label + ": native execution must not conceal a failed Triton compilation");
        }
    }

    /** The native contract: rows in order, the last row wins, rows outside the output skipped. */
    private static double[] reference(INDArray ref, long[][] rows, INDArray updates, long[] refShape,
                                      long sliceLength) {
        double[] out = ref.dup('c').data().asDouble();
        double[] values = updates.dup('c').data().asDouble();
        for (int r = 0; r < rows.length; r++) {
            long slice = 0;
            boolean inRange = true;
            for (int d = 0; d < rows[r].length; d++) {
                inRange &= rows[r][d] >= 0 && rows[r][d] < refShape[d];
                slice = slice * refShape[d] + rows[r][d];
            }
            if (!inRange) continue;
            for (int p = 0; p < sliceLength; p++) {
                out[(int) (slice * sliceLength + p)] = values[(int) (r * sliceLength + p)];
            }
        }
        return out;
    }

    private static long lengthOf(long[] shape) {
        long length = 1;
        for (long dim : shape) length *= dim;
        return length;
    }
}
