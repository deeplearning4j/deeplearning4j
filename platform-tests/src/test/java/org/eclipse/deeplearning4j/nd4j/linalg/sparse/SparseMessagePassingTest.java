/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.sparse;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeAggregate;
import org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeGather;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Forward-correctness and SameDiff gradient-check tests for the MPNN sparse primitives:
 * <ul>
 *   <li>{@code csr_edge_gather}    — gather node features to per-edge vectors ({@link CsrEdgeGather})</li>
 *   <li>{@code csr_edge_aggregate} — scatter-reduce edge messages to per-node ({@link CsrEdgeAggregate})</li>
 * </ul>
 *
 * <h3>Test graph: 3 nodes, 5 directed edges (CSR)</h3>
 * <pre>
 *   Adjacency (as CSR):
 *     node 0 → {node 0, node 2}   (2 out-edges, edges 0 and 1)
 *     node 1 → {node 1}           (1 out-edge,  edge 2)
 *     node 2 → {node 0, node 2}   (2 out-edges, edges 3 and 4)
 *
 *   colIdx = [0, 2, 1, 0, 2]   INT32
 *   rowPtr = [0, 2, 3, 5]      INT32
 * </pre>
 *
 * <h3>Design rules (mirrors SparseGnnTest / SparseBpGradCheckTest)</h3>
 * <ul>
 *   <li>Structural arrays → {@code sd.constant(INT32)} — skipped by GradCheckUtil.</li>
 *   <li>Differentiable arrays → {@code sd.var(DOUBLE)}, well away from zero.</li>
 *   <li>Loss = {@code sd.mean("loss", output)} — single float output; no setLossVariables.</li>
 *   <li>Each test creates a fresh SameDiff and closes it in {@code try/finally}.</li>
 *   <li>{@link #purgeConstantHandlerCache()} runs before every test.</li>
 * </ul>
 */
public class SparseMessagePassingTest extends BaseNd4jTestWithBackends {

    private static final int ROWS = 3;  // number of nodes / segments
    private static final int N    = 3;  // total nodes (= cols of adjacency)
    private static final int NNZ  = 5;  // number of edges

    /**
     * Purge ConstantBuffersCache before every test.
     * Prevents stale-buffer UAF crashes from DeallocatorService (see SparseGradCheckTest
     * for the full root-cause explanation).
     */
    @BeforeEach
    public void purgeConstantHandlerCache() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // Shared CSR helpers
    // -----------------------------------------------------------------------

    private static INDArray makeColIdx() {
        return Nd4j.createFromArray(new int[]{0, 2, 1, 0, 2});
    }

    private static INDArray makeRowPtr() {
        return Nd4j.createFromArray(new int[]{0, 2, 3, 5});
    }

    // -----------------------------------------------------------------------
    // Test 1 — csr_edge_gather: forward correctness
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@code csr_edge_gather}.
     *
     * <p>Node feature matrix (3 nodes × 2 features):
     * <pre>
     *   X = [[1.0, 4.0],   node 0
     *        [2.0, 5.0],   node 1
     *        [3.0, 6.0]]   node 2
     * </pre>
     * Expected gathered edge features ({@code colIdx = [0,2,1,0,2]}):
     * <pre>
     *   edge 0 → node 0: [1.0, 4.0]
     *   edge 1 → node 2: [3.0, 6.0]
     *   edge 2 → node 1: [2.0, 5.0]
     *   edge 3 → node 0: [1.0, 4.0]
     *   edge 4 → node 2: [3.0, 6.0]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrEdgeGatherForward(Nd4jBackend backend) {
        INDArray X = Nd4j.createFromArray(new double[][]{
                {1.0, 4.0},
                {2.0, 5.0},
                {3.0, 6.0}
        });

        INDArray expected = Nd4j.createFromArray(new double[][]{
                {1.0, 4.0},
                {3.0, 6.0},
                {2.0, 5.0},
                {1.0, 4.0},
                {3.0, 6.0}
        });

        INDArray colIdx = makeColIdx();
        CsrEdgeGather op = new CsrEdgeGather(colIdx, X);
        INDArray edgeFeat = Nd4j.exec(op)[0];

        assertTrue(expected.equalsWithEps(edgeFeat, 1e-6),
                "csr_edge_gather forward mismatch:\n  expected=" + expected
                        + "\n  actual=" + edgeFeat);
        long[] expectedShape = {NNZ, 2};
        for (int i = 0; i < 2; i++) {
            assertEquals(expectedShape[i], edgeFeat.size(i),
                    "shape mismatch at dim " + i);
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 — csr_edge_gather: SameDiff gradient check
    // -----------------------------------------------------------------------

    /**
     * SameDiff gradient check for {@code csr_edge_gather}.
     *
     * <p>Differentiates w.r.t. {@code X} [3, 2] (DOUBLE).
     * {@code colIdx} is an INT32 constant skipped by the gradient checker.
     * Loss = mean(edgeFeat[5, 2]) — single output, no setLossVariables needed.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrEdgeGatherGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(31415);

        // Node features: chosen so every element is well away from zero
        INDArray xArr = Nd4j.createFromArray(new double[][]{
                {1.1, 4.4},
                {2.2, 5.5},
                {3.3, 6.6}
        });

        SameDiff sd = SameDiff.create();
        try {
            // Structural constant (INT32 — not differentiated)
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());

            // Differentiable node features (DOUBLE)
            SDVariable X = sd.var("X", xArr);

            // edgeFeat[5, 2] = X[colIdx, :]
            SDVariable edgeFeat = new CsrEdgeGather(sd, colIdx, X, N)
                    .outputVariable();

            // Scalar loss = mean(edgeFeat)
            sd.mean("loss", edgeFeat);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_edge_gather"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — csr_edge_aggregate SUM: forward vs dense reference
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@code csr_edge_aggregate} mode=SUM.
     *
     * <p>Edge messages (5 edges × 2 features):
     * <pre>
     *   edgeMsg = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]
     * </pre>
     * Row 0 edges {0,1}: sum = [3, 30]; row 1 edge {2}: [3, 30]; row 2 edges {3,4}: [9, 90].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrEdgeAggregateSum(Nd4jBackend backend) {
        INDArray edgeMsg = Nd4j.createFromArray(new double[][]{
                {1.0, 10.0},
                {2.0, 20.0},
                {3.0, 30.0},
                {4.0, 40.0},
                {5.0, 50.0}
        });

        INDArray expected = Nd4j.createFromArray(new double[][]{
                {3.0,  30.0},
                {3.0,  30.0},
                {9.0,  90.0}
        });

        INDArray rowPtr = makeRowPtr();
        CsrEdgeAggregate op = new CsrEdgeAggregate(rowPtr, edgeMsg, ROWS, 0);
        INDArray out = Nd4j.exec(op)[0];

        assertTrue(expected.equalsWithEps(out, 1e-6),
                "csr_edge_aggregate SUM mismatch:\n  expected=" + expected
                        + "\n  actual=" + out);
    }

    // -----------------------------------------------------------------------
    // Test 4 — csr_edge_aggregate MEAN: forward vs dense reference
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@code csr_edge_aggregate} mode=MEAN.
     *
     * <p>Using the same edge messages as Test 3:
     * Row 0: deg=2 → mean = [1.5, 15]; row 1: deg=1 → [3, 30]; row 2: deg=2 → [4.5, 45].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrEdgeAggregateMean(Nd4jBackend backend) {
        INDArray edgeMsg = Nd4j.createFromArray(new double[][]{
                {1.0, 10.0},
                {2.0, 20.0},
                {3.0, 30.0},
                {4.0, 40.0},
                {5.0, 50.0}
        });

        INDArray expected = Nd4j.createFromArray(new double[][]{
                {1.5,  15.0},
                {3.0,  30.0},
                {4.5,  45.0}
        });

        INDArray rowPtr = makeRowPtr();
        CsrEdgeAggregate op = new CsrEdgeAggregate(rowPtr, edgeMsg, ROWS, 1);
        INDArray out = Nd4j.exec(op)[0];

        assertTrue(expected.equalsWithEps(out, 1e-6),
                "csr_edge_aggregate MEAN mismatch:\n  expected=" + expected
                        + "\n  actual=" + out);
    }

    // -----------------------------------------------------------------------
    // Test 5 — csr_edge_aggregate MAX: forward vs dense reference
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@code csr_edge_aggregate} mode=MAX.
     *
     * <p>Distinct values per row × feature so the argmax is unambiguous:
     * <pre>
     *   edgeMsg = [[1.1, 20.0], [8.8, 3.3], [5.5, 5.5], [2.2, 9.9], [7.7, 1.1]]
     *   colIdx  = [0, 2, 1, 0, 2]  (irrelevant for aggregate; row structure matters)
     *   row 0 edges {0,1}: max col-0 = max(1.1,8.8)=8.8; col-1=max(20.0,3.3)=20.0
     *   row 1 edge  {2}:   max col-0 = 5.5; col-1 = 5.5
     *   row 2 edges {3,4}: max col-0 = max(2.2,7.7)=7.7; col-1=max(9.9,1.1)=9.9
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrEdgeAggregateMax(Nd4jBackend backend) {
        INDArray edgeMsg = Nd4j.createFromArray(new double[][]{
                {1.1, 20.0},
                {8.8,  3.3},
                {5.5,  5.5},
                {2.2,  9.9},
                {7.7,  1.1}
        });

        INDArray expected = Nd4j.createFromArray(new double[][]{
                {8.8, 20.0},
                {5.5,  5.5},
                {7.7,  9.9}
        });

        INDArray rowPtr = makeRowPtr();
        CsrEdgeAggregate op = new CsrEdgeAggregate(rowPtr, edgeMsg, ROWS, 2);
        INDArray out = Nd4j.exec(op)[0];

        assertTrue(expected.equalsWithEps(out, 1e-6),
                "csr_edge_aggregate MAX mismatch:\n  expected=" + expected
                        + "\n  actual=" + out);
    }

    // -----------------------------------------------------------------------
    // Test 6 — csr_edge_aggregate SUM: gradient check
    // -----------------------------------------------------------------------

    /**
     * SameDiff gradient check for {@code csr_edge_aggregate} mode=SUM.
     *
     * <p>Differentiates w.r.t. {@code edgeMsg} [5, 2] (DOUBLE).
     * {@code rowPtr} is an INT32 constant skipped by the checker.
     * SUM is linear → flat gradient; numerical check is exact.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrEdgeAggregateSumGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(11111);

        INDArray emArr = Nd4j.createFromArray(new double[][]{
                {1.0, 10.0},
                {2.0, 20.0},
                {3.0, 30.0},
                {4.0, 40.0},
                {5.0, 50.0}
        });

        SameDiff sd = SameDiff.create();
        try {
            SDVariable rowPtr  = sd.constant("rowPtr", makeRowPtr());
            SDVariable edgeMsg = sd.var("edgeMsg", emArr);

            SDVariable out = new CsrEdgeAggregate(sd, rowPtr, edgeMsg, ROWS, 0)
                    .outputVariable();

            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_edge_aggregate SUM"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — csr_edge_aggregate MEAN: gradient check
    // -----------------------------------------------------------------------

    /**
     * SameDiff gradient check for {@code csr_edge_aggregate} mode=MEAN.
     *
     * <p>MEAN is linear (÷ constant degree) so the gradient is exact at every step.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrEdgeAggregateMeanGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(22222);

        INDArray emArr = Nd4j.createFromArray(new double[][]{
                {1.0, 10.0},
                {2.0, 20.0},
                {3.0, 30.0},
                {4.0, 40.0},
                {5.0, 50.0}
        });

        SameDiff sd = SameDiff.create();
        try {
            SDVariable rowPtr  = sd.constant("rowPtr", makeRowPtr());
            SDVariable edgeMsg = sd.var("edgeMsg", emArr);

            SDVariable out = new CsrEdgeAggregate(sd, rowPtr, edgeMsg, ROWS, 1)
                    .outputVariable();

            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_edge_aggregate MEAN"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — csr_edge_aggregate MAX: gradient check (distinct values)
    // -----------------------------------------------------------------------

    /**
     * SameDiff gradient check for {@code csr_edge_aggregate} mode=MAX.
     *
     * <p>Edge messages are chosen so that within every row the argmax per feature
     * is strict (no ties), making the indicator function single-valued and the
     * numerical gradient smooth and well-conditioned.
     *
     * <pre>
     *   edgeMsg = [[1.1, 20.1], [8.8, 3.3], [5.5, 5.5], [2.2, 9.9], [7.7, 1.1]]
     *   row 0: col-0 winner = edge 1 (8.8); col-1 winner = edge 0 (20.1)
     *   row 1: single edge 2
     *   row 2: col-0 winner = edge 4 (7.7); col-1 winner = edge 3 (9.9)
     * </pre>
     * All argmaxima are strict → gradient is well-defined everywhere.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrEdgeAggregateMaxGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(33333);

        INDArray emArr = Nd4j.createFromArray(new double[][]{
                {1.1, 20.1},
                {8.8,  3.3},
                {5.5,  5.5},
                {2.2,  9.9},
                {7.7,  1.1}
        });

        SameDiff sd = SameDiff.create();
        try {
            SDVariable rowPtr  = sd.constant("rowPtr", makeRowPtr());
            SDVariable edgeMsg = sd.var("edgeMsg", emArr);

            SDVariable out = new CsrEdgeAggregate(sd, rowPtr, edgeMsg, ROWS, 2)
                    .outputVariable();

            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_edge_aggregate MAX"
            );
        } finally {
            sd.close();
        }
    }
}
