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
import org.nd4j.linalg.api.ops.impl.sparse.CsrRowSoftmax;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSegmentMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Forward-correctness and SameDiff gradient-check tests for the two new GNN sparse ops:
 * <ul>
 *   <li>{@code csr_row_softmax}  — GAT per-row edge-softmax  ({@link CsrRowSoftmax})</li>
 *   <li>{@code csr_segment_max}  — GraphSAGE-max aggregation ({@link CsrSegmentMax})</li>
 * </ul>
 *
 * <h3>Shared 3×3 CSR test pattern (nnz = 5)</h3>
 * <pre>
 *   A = [[1, 0, 2],
 *        [0, 3, 0],
 *        [4, 0, 5]]
 *   values = [1.0, 2.0, 3.0, 4.0, 5.0]   DOUBLE
 *   colIdx = [0, 2, 1, 0, 2]              INT32
 *   rowPtr = [0, 2, 3, 5]                INT32
 * </pre>
 *
 * <h3>Design rules (mirrors SparseGradCheckTest / SparseBpGradCheckTest)</h3>
 * <ul>
 *   <li>Structural arrays (colIdx, rowPtr) → {@code sd.constant(INT32)} — skipped by
 *       GradCheckUtil.</li>
 *   <li>Differentiable value arrays → {@code sd.var(DOUBLE)}, well away from zero.</li>
 *   <li>Loss = {@code sd.mean("loss", &lt;float-output&gt;)} — single output, so
 *       {@code setLossVariables} is not required.</li>
 *   <li>Each test method creates a fresh SameDiff and closes it in a {@code try/finally}
 *       block to prevent inter-test resource leaks.</li>
 *   <li>{@link #purgeConstantHandlerCache()} runs before every test to prevent
 *       stale-buffer UAF crashes from DeallocatorService (see SparseGradCheckTest for
 *       the full root-cause explanation).</li>
 * </ul>
 */
public class SparseGnnTest extends BaseNd4jTestWithBackends {

    /** rows in the test sparse matrix (= number of graph nodes / segments) */
    private static final int ROWS = 3;

    /**
     * Purge ConstantBuffersCache before every test to prevent UAF crashes.
     * See SparseGradCheckTest for the full root-cause explanation.
     */
    @BeforeEach
    public void purgeConstantHandlerCache() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // Shared CSR helpers
    // -----------------------------------------------------------------------

    /** Standard 3×3 test pattern: non-zero values [nnz=5], DOUBLE. */
    private static INDArray makeValues() {
        return Nd4j.createFromArray(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
    }

    /** Standard 3×3 test pattern: column indices [nnz=5], INT32. */
    private static INDArray makeColIdx() {
        return Nd4j.createFromArray(new int[]{0, 2, 1, 0, 2});
    }

    /** Standard 3×3 test pattern: row pointers [rows+1=4], INT32. */
    private static INDArray makeRowPtr() {
        return Nd4j.createFromArray(new int[]{0, 2, 3, 5});
    }

    // -----------------------------------------------------------------------
    // Test 1 — csr_row_softmax: forward correctness
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@code csr_row_softmax}.
     *
     * <p>Verifies that the op outputs a per-row softmax matching a hand-computed
     * reference on the shared 3×3 test pattern.
     *
     * <p>Per-row softmax derivation:
     * <ul>
     *   <li>Row 0, values [1, 2]:  alpha = [e^1/(e^1+e^2), e^2/(e^1+e^2)]</li>
     *   <li>Row 1, values [3]:     alpha = [1.0]</li>
     *   <li>Row 2, values [4, 5]:  alpha = [e^4/(e^4+e^5), e^5/(e^4+e^5)]</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrRowSoftmaxForward(Nd4jBackend backend) {
        // Compute reference per-row softmax
        double e1 = Math.exp(1.0), e2 = Math.exp(2.0);
        double s01 = e1 + e2;
        double a0 = e1 / s01, a1 = e2 / s01;     // row 0

        double a2 = 1.0;                           // row 1 (single entry)

        double e4 = Math.exp(4.0), e5 = Math.exp(5.0);
        double s45 = e4 + e5;
        double a3 = e4 / s45, a4 = e5 / s45;     // row 2

        INDArray expected = Nd4j.createFromArray(a0, a1, a2, a3, a4);

        // Execute op
        INDArray values = makeValues();
        INDArray rowPtr = makeRowPtr();
        CsrRowSoftmax op = new CsrRowSoftmax(values, rowPtr, ROWS);
        INDArray alpha = Nd4j.exec(op)[0];

        assertTrue(expected.equalsWithEps(alpha, 1e-6),
                "csr_row_softmax forward mismatch: expected=" + expected + " actual=" + alpha);
    }

    // -----------------------------------------------------------------------
    // Test 2 — csr_row_softmax: SameDiff gradient check
    // -----------------------------------------------------------------------

    /**
     * SameDiff gradient check for {@code csr_row_softmax}.
     *
     * <p>Differentiates w.r.t. {@code values} [nnz=5] (DOUBLE).
     * {@code rowPtr} is an INT32 constant skipped by the gradient checker.
     * Loss = mean(alpha) — single output, no {@code setLossVariables} needed.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrRowSoftmaxGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(11111);

        SameDiff sd = SameDiff.create();
        try {
            // Structural constant (INT32 — not differentiated)
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            // Differentiable variable (DOUBLE)
            SDVariable values = sd.var("values", makeValues());

            // alpha[nnz=5] = per-row softmax of values
            SDVariable alpha = new CsrRowSoftmax(sd, values, rowPtr, ROWS)
                    .outputVariable();

            // Scalar loss = mean(alpha); single output — no setLossVariables
            sd.mean("loss", alpha);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_row_softmax"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — csr_segment_max: forward correctness
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@code csr_segment_max}.
     *
     * <p>Uses the shared 3×3 CSR sparsity pattern as the adjacency structure, and a
     * small 3-node, 2-feature node-feature matrix:
     * <pre>
     *   X = [[1.0, 6.0],   node 0
     *        [2.0, 4.0],   node 1
     *        [8.0, 3.0]]   node 2
     * </pre>
     * Expected per-row neighbour-max output (rows = 3 segments):
     * <pre>
     *   Row 0 neighbours = {0, 2}: max(X[0], X[2]) = [8, 6]
     *   Row 1 neighbours = {1}:    max(X[1])        = [2, 4]
     *   Row 2 neighbours = {0, 2}: max(X[0], X[2]) = [8, 6]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrSegmentMaxForward(Nd4jBackend backend) {
        INDArray X = Nd4j.createFromArray(new double[][]{
                {1.0, 6.0},
                {2.0, 4.0},
                {8.0, 3.0}
        });

        INDArray expected = Nd4j.createFromArray(new double[][]{
                {8.0, 6.0},
                {2.0, 4.0},
                {8.0, 6.0}
        });

        INDArray colIdx = makeColIdx();
        INDArray rowPtr = makeRowPtr();
        CsrSegmentMax op = new CsrSegmentMax(colIdx, rowPtr, X, ROWS);
        INDArray out = Nd4j.exec(op)[0];

        assertTrue(expected.equalsWithEps(out, 1e-6),
                "csr_segment_max forward mismatch: expected=" + expected + " actual=" + out);
    }

    // -----------------------------------------------------------------------
    // Test 4 — csr_segment_max: SameDiff gradient check
    // -----------------------------------------------------------------------

    /**
     * SameDiff gradient check for {@code csr_segment_max}.
     *
     * <p>Node features {@code X} are chosen so that within every row's neighbourhood
     * each feature dimension has a unique maximum-winner (no ties), making the
     * argmax indicator function single-valued and the numerical gradient smooth:
     * <pre>
     *   X = [[1.1, 6.2],   node 0
     *        [2.3, 4.4],   node 1
     *        [8.5, 3.6]]   node 2
     *
     *   Row 0 neighbours = {0, 2}: feature 0 → X[2]=8.5 wins; feature 1 → X[0]=6.2 wins
     *   Row 1 neighbours = {1}:    feature 0 → X[1]=2.3 wins; feature 1 → X[1]=4.4 wins
     *   Row 2 neighbours = {0, 2}: feature 0 → X[2]=8.5 wins; feature 1 → X[0]=6.2 wins
     * </pre>
     * All per-row per-feature maxima are strict (no ties) so the gradient is
     * well-defined everywhere and numerical differentiation is stable.
     *
     * <p>{@code colIdx} and {@code rowPtr} are INT32 constants skipped by the checker.
     * Differentiates w.r.t. {@code X} [3, 2] (DOUBLE).
     * Loss = mean(out[3, 2]) — single output, no {@code setLossVariables} needed.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrSegmentMaxGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(22222);

        // Node features with distinct per-row-per-feature maxima (no ties)
        INDArray xArr = Nd4j.createFromArray(new double[][]{
                {1.1, 6.2},
                {2.3, 4.4},
                {8.5, 3.6}
        });

        SameDiff sd = SameDiff.create();
        try {
            // Structural constants (INT32 — not differentiated)
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            // Differentiable node features (DOUBLE)
            SDVariable X = sd.var("X", xArr);

            // out[3, 2] = per-row neighbour max of X
            SDVariable out = new CsrSegmentMax(sd, colIdx, rowPtr, X, ROWS)
                    .outputVariable();

            // Scalar loss = mean(out); single output — no setLossVariables needed
            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_segment_max"
            );
        } finally {
            sd.close();
        }
    }
}
