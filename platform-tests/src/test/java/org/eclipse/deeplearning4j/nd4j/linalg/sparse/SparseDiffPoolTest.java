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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.graph.DiffPool;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link DiffPool} — differentiable hierarchical graph pooling
 * (Ying et al. 2018) implemented as a pure SameDiff composition over existing ops.
 *
 * <h3>Test graph</h3>
 * A small undirected chain of 4 nodes (N=4, F=3 features, K=2 clusters):
 * <pre>
 *   0 — 1 — 2 — 3
 *
 *   A = [[0,1,0,0],
 *        [1,0,1,0],
 *        [0,1,0,1],
 *        [0,0,1,0]]
 *
 *   CSR:  aValues  = [1,1,1,1,1,1]  (DOUBLE)
 *         aColIdx  = [1, 0,2, 1,3, 2]  (INT32)
 *         aRowPtr  = [0, 1, 3, 5, 6]   (INT32, length N+1=5)
 * </pre>
 *
 * <h3>Two tests</h3>
 * <ol>
 *   <li>{@code testDiffPoolForwardShapes} — checks output shapes and absence of
 *       NaN / Inf after a forward pass.</li>
 *   <li>{@code testDiffPoolGradCheck} — numerical gradient check via
 *       {@link GradCheckUtil#checkGradients} on {@code H} and
 *       {@code assignLogits} (the two differentiable inputs to DiffPool).</li>
 * </ol>
 *
 * <h3>Notes on setLossVariables</h3>
 * The DiffPool subgraph produces three outputs (Xc, Ac, entropyLoss).  The
 * test combines them into a single scalar {@code loss} and calls
 * {@code sd.setLossVariables("loss")} before running the gradient check.
 * This is required whenever the graph has multiple op outputs that could
 * appear to be terminals; without it GradCheckUtil throws
 * "No loss variables specified".
 */
@Tag(TagNames.SAMEDIFF)
@Tag("sparse")
@Tag("gnn")
public class SparseDiffPoolTest extends BaseNd4jTestWithBackends {

    /** Number of nodes in the test graph. */
    private static final int N = 4;
    /** Node-feature dimension. */
    private static final int F = 3;
    /** Number of target clusters (coarsened nodes). */
    private static final int K = 2;

    /**
     * Purge ConstantBuffersCache before every test.
     *
     * Without this, DeallocatorService.forceFlushAll() (called between tests
     * by BaseND4JTest) frees native buffers that ConstantBuffersCache still
     * holds references to — causing "Ptr data buffer was released!" on the
     * first Nd4j.rand() call in the next test.  The same workaround is used
     * in SparseGradCheckTest and SparseGraphPrepTest.
     */
    @BeforeEach
    public void purgeConstants() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // CSR adjacency helpers: undirected chain 0-1-2-3
    // -----------------------------------------------------------------------

    /** Non-zero values (all 1.0, DOUBLE). */
    private static INDArray makeAValues() {
        return Nd4j.createFromArray(new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    }

    /**
     * Column-index array.
     * Row 0 → col 1; row 1 → cols 0,2; row 2 → cols 1,3; row 3 → col 2.
     */
    private static INDArray makeAColIdx() {
        return Nd4j.createFromArray(new int[]{1,  0, 2,  1, 3,  2});
    }

    /**
     * Row-pointer array (length N+1 = 5).
     * rowPtr[i] = cumulative nnz before row i.
     */
    private static INDArray makeARowPtr() {
        return Nd4j.createFromArray(new int[]{0, 1, 3, 5, 6});
    }

    // -----------------------------------------------------------------------
    // (1) Forward-pass shape and NaN/Inf check
    // -----------------------------------------------------------------------

    /**
     * Verify that {@link DiffPool#apply} produces correctly-shaped outputs
     * with no NaN or Inf values.
     *
     * <ul>
     *   <li>{@code Xc} must have shape {@code [K, F] = [2, 3]}</li>
     *   <li>{@code Ac} must have shape {@code [K, K] = [2, 2]}</li>
     *   <li>{@code entropyLoss} must be a scalar (rank 0)</li>
     *   <li>No element of any output may be NaN or Inf</li>
     *   <li>{@code entropyLoss ≥ 0} (−mean of non-positive per-node entropies)</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDiffPoolForwardShapes(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(42L);

        SameDiff sd = SameDiff.create();
        try {
            // Structural INT32 constants — skipped by GradCheckUtil, not differentiated
            SDVariable aColIdx = sd.constant("aColIdx", makeAColIdx());
            SDVariable aRowPtr = sd.constant("aRowPtr", makeARowPtr());
            // Adjacency values — constant (fixed graph structure)
            SDVariable aValues = sd.constant("aValues", makeAValues());

            // DOUBLE vars: node features and raw cluster-assignment logits
            SDVariable H = sd.var("H",
                    Nd4j.rand(DataType.DOUBLE, N, F).addi(0.1));
            SDVariable assignLogits = sd.var("assignLogits",
                    Nd4j.rand(DataType.DOUBLE, N, K));

            // Apply DiffPool coarsening
            SDVariable[] out = DiffPool.apply(sd, "dp",
                    aValues, aColIdx, aRowPtr, H, assignLogits, N, K);
            SDVariable Xc          = out[0];   // expected [K=2, F=3]
            SDVariable Ac          = out[1];   // expected [K=2, K=2]
            SDVariable entropyLoss = out[2];   // expected scalar

            // Run forward pass — no placeholders (all variables are sd.var()/sd.constant())
            Map<String, INDArray> result = sd.output(
                    new HashMap<>(), Xc.name(), Ac.name(), entropyLoss.name());

            INDArray xcArr = result.get(Xc.name());
            INDArray acArr = result.get(Ac.name());
            INDArray elArr = result.get(entropyLoss.name());

            // ---- shape checks ----
            assertArrayEquals(new long[]{K, F}, xcArr.shape(),
                    "Xc shape should be [K=" + K + ", F=" + F + "]");
            assertArrayEquals(new long[]{K, K}, acArr.shape(),
                    "Ac shape should be [K=" + K + ", K=" + K + "]");
            assertEquals(0, elArr.rank(),
                    "entropyLoss should be a scalar (rank 0)");

            // ---- no NaN or Inf in any output ----
            assertFalse(xcArr.isNaN().any(),
                    "Xc contains NaN");
            assertFalse(xcArr.isInfinite().any(),
                    "Xc contains Inf");
            assertFalse(acArr.isNaN().any(),
                    "Ac contains NaN");
            assertFalse(acArr.isInfinite().any(),
                    "Ac contains Inf");
            assertFalse(elArr.isNaN().any(),
                    "entropyLoss is NaN");
            assertFalse(elArr.isInfinite().any(),
                    "entropyLoss is Inf");

            // ---- sign check: entropy regulariser ≥ 0 ----
            //   Each row of S sums to 1 (softmax), so each s_ik ∈ (0,1].
            //   log(s_ik) ≤ 0 → Σ s_ik log(s_ik) ≤ 0 → -mean(…) ≥ 0.
            assertTrue(elArr.getDouble(0) >= 0.0,
                    "entropyLoss should be non-negative, got " + elArr.getDouble(0));

        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // (2) Gradient check on H and assignLogits
    // -----------------------------------------------------------------------

    /**
     * Numerical gradient check for {@link DiffPool} via
     * {@link GradCheckUtil#checkGradients}.
     *
     * <p>Design:
     * <ul>
     *   <li>{@code H} and {@code assignLogits} are {@code sd.var()} — the two
     *       differentiable inputs GradCheckUtil will perturb.</li>
     *   <li>{@code aValues}, {@code aColIdx}, {@code aRowPtr} are
     *       {@code sd.constant()} — structural/fixed, not differentiated.</li>
     *   <li>Scalar loss = {@code sum(Xc) + sum(Ac) + entropyLoss} combines all
     *       three DiffPool outputs into a single terminal so that gradient
     *       paths from all three outputs are exercised.</li>
     *   <li>{@code sd.setLossVariables("loss")} is called explicitly because
     *       the graph has more than one variable that could appear terminal
     *       before the loss combiner is added — explicit marking is the safe
     *       pattern for multi-output subgraphs (as in CsrSpgemm tests).</li>
     *   <li>DOUBLE dtype throughout for numerical-differentiation accuracy.</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDiffPoolGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345L);

        SameDiff sd = SameDiff.create();
        try {
            // Structural INT32 constants — not differentiated
            SDVariable aColIdx = sd.constant("aColIdx", makeAColIdx());
            SDVariable aRowPtr = sd.constant("aRowPtr", makeARowPtr());
            // Adjacency values — constant (graph structure is not a learnable parameter)
            SDVariable aValues = sd.constant("aValues", makeAValues());

            // Differentiable DOUBLE vars — GradCheckUtil will perturb these numerically
            // Offset away from zero to keep softmax and log numerically well-conditioned
            SDVariable H = sd.var("H",
                    Nd4j.rand(DataType.DOUBLE, N, F).addi(0.5));
            SDVariable assignLogits = sd.var("assignLogits",
                    Nd4j.rand(DataType.DOUBLE, N, K).subi(0.5));

            // Build DiffPool subgraph
            SDVariable[] out = DiffPool.apply(sd, "dp",
                    aValues, aColIdx, aRowPtr, H, assignLogits, N, K);
            SDVariable Xc          = out[0];   // [K, F]
            SDVariable Ac          = out[1];   // [K, K]
            SDVariable entropyLoss = out[2];   // scalar

            // Combined scalar loss — consumes all three DiffPool outputs so that
            // the single terminal is "loss".  This exercises gradient paths through:
            //   • mmul(St, H)               → dXc back to H and assignLogits
            //   • CsrSpmm(A, S)             → dAS back to assignLogits
            //   • mmul(St, AS)              → dAc back to assignLogits
            //   • -mean(Σ S*log(S+ε))       → dEntropy back to assignLogits
            SDVariable sumXc    = sd.math().sum(Xc);       // scalar: reduces [K,F] to []
            SDVariable sumAc    = sd.math().sum(Ac);       // scalar: reduces [K,K] to []
            SDVariable lossXcAc = sd.math().add(sumXc, sumAc);
            // Named "loss" — required for sd.setLossVariables() below
            sd.math().add("loss", lossXcAc, entropyLoss);

            // Explicitly mark "loss" as the differentiation target.
            // Required for multi-output graphs: GradCheckUtil otherwise cannot
            // determine which terminal is the scalar loss.
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "DiffPool gradient check failed — analytical and numerical gradients diverge"
            );

        } finally {
            sd.close();
        }
    }
}
