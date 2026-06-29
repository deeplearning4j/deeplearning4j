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
import org.nd4j.linalg.api.ops.impl.graph.GraphPooling;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Numeric forward-correctness and SameDiff gradient-check tests for the
 * GNN composition API ({@code sd.gnn().*}) and {@link GraphPooling}.
 *
 * <h3>Scope</h3>
 * <ul>
 *   <li>{@link org.nd4j.autodiff.samediff.ops.SDGNN#gcnConv} — GCN layer</li>
 *   <li>{@link org.nd4j.autodiff.samediff.ops.SDGNN#gatConvHead} — single-head GAT
 *       (exercises {@code gather_bp} internally via {@code sd.gather})</li>
 *   <li>{@link org.nd4j.autodiff.samediff.ops.SDGNN#sageMean} — GraphSAGE mean agg</li>
 *   <li>{@link org.nd4j.autodiff.samediff.ops.SDGNN#sageMax}  — GraphSAGE max agg</li>
 *   <li>{@link org.nd4j.autodiff.samediff.ops.SDGNN#sagePool} — GraphSAGE pool agg</li>
 *   <li>{@link org.nd4j.autodiff.samediff.ops.SDGNN#ginConv}  — GIN layer</li>
 *   <li>{@link GraphPooling#globalSumPool}  — graph-level sum readout</li>
 *   <li>{@link GraphPooling#globalMeanPool} — graph-level mean readout</li>
 *   <li>{@link GraphPooling#globalMaxPool}  — graph-level max readout</li>
 * </ul>
 *
 * <p>Complementary to {@link SparseGnnTest}, which covers the underlying primitive
 * ops ({@code CsrRowSoftmax}, {@code CsrSegmentMax}) in isolation.
 *
 * <h3>Common 4-node directed graph (N=4, F=2 input features, nnz=6 edges)</h3>
 * <pre>
 *   Edges: 0→1, 0→2, 1→0, 1→3, 2→0, 3→1
 *
 *   CSR:
 *     colIdx = [1, 2, 0, 3, 0, 1]   INT32  (nnz=6)
 *     rowPtr = [0, 2, 4, 5, 6]      INT32  (N+1=5)
 *     rowIdx = [0, 0, 1, 1, 2, 3]   INT32  (destination node per edge — for GAT)
 * </pre>
 *
 * <h3>Design rules (mirrors SparseGradCheckTest / SparseGnnTest)</h3>
 * <ul>
 *   <li>Structural INT arrays (colIdx, rowPtr, rowIdx, graphIds) →
 *       {@code sd.constant(INT32)} — skipped by GradCheckUtil automatically.</li>
 *   <li>Trainable weight arrays → {@code sd.var(DOUBLE)}, initialised well away
 *       from zero to ensure no dead-relu neurons kill gradients.</li>
 *   <li>Node feature array {@code X} is made a {@code sd.var()} in grad-check tests
 *       so the full Jacobian (w.r.t. inputs <em>and</em> weights) is verified.</li>
 *   <li>Loss = {@code sd.mean("loss", output)} (scalar).  No {@code setLossVariables}
 *       is needed because every GNN op here produces a single DOUBLE output terminal.</li>
 *   <li>Each test creates a fresh SameDiff and closes it in a {@code try/finally}
 *       block to prevent inter-test resource leaks.</li>
 *   <li>{@link #purgeConstantHandlerCache()} runs before every test to prevent
 *       stale-buffer UAF crashes from DeallocatorService (see SparseGradCheckTest
 *       for the full root-cause explanation).</li>
 * </ul>
 *
 * <h3>GradCheckUtil tolerances</h3>
 * All grad-check tests use {@code GradCheckUtil.checkGradients(sd, null)} which
 * applies the library defaults: eps=1e-5, maxRelError=1e-5, minAbsError=1e-6.
 * DOUBLE-precision inputs are required for these tolerances to be achievable.
 */
@Tag(TagNames.SAMEDIFF)
public class SparseGnnLayerTest extends BaseNd4jTestWithBackends {

    // -----------------------------------------------------------------------
    // Graph constants
    // -----------------------------------------------------------------------

    /** Number of nodes. */
    private static final int N = 4;
    /** Input feature dimension. */
    private static final int F = 2;
    /** Output / hidden feature dimension. */
    private static final int H = 2;
    /** Number of non-zero edges in the common graph. */
    private static final int NNZ = 6;

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /**
     * Purge ConstantBuffersCache before every test to prevent UAF crashes from
     * DeallocatorService releasing the native backing of cached constant buffers.
     * See SparseGradCheckTest for the full root-cause explanation.
     */
    @BeforeEach
    public void purgeConstantHandlerCache() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // Shared CSR / feature helpers
    // -----------------------------------------------------------------------

    /** CSR column indices for the common 4-node graph [nnz=6], INT32. */
    private static INDArray makeColIdx() {
        return Nd4j.createFromArray(new int[]{1, 2, 0, 3, 0, 1});
    }

    /** CSR row pointers for the common 4-node graph [N+1=5], INT32. */
    private static INDArray makeRowPtr() {
        return Nd4j.createFromArray(new int[]{0, 2, 4, 5, 6});
    }

    /**
     * Per-edge destination-node indices [nnz=6], INT32.
     * Derived from rowPtr by expansion: edge k in row i ⟹ rowIdx[k] = i.
     * Used as the {@code rowIdx} argument of {@code gatConvHead}.
     */
    private static INDArray makeRowIdx() {
        // rowPtr = [0,2,4,5,6]
        //   row 0: k=0,1 → rowIdx[0]=0, rowIdx[1]=0
        //   row 1: k=2,3 → rowIdx[2]=1, rowIdx[3]=1
        //   row 2: k=4   → rowIdx[4]=2
        //   row 3: k=5   → rowIdx[5]=3
        return Nd4j.createFromArray(new int[]{0, 0, 1, 1, 2, 3});
    }

    /**
     * Node feature matrix [N=4, F=2], DOUBLE.
     * Values are all strictly positive so ReLU does not kill gradients.
     */
    private static INDArray makeX() {
        return Nd4j.createFromArray(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0},
                {7.0, 8.0}
        });
    }

    /**
     * Random weight matrix [rows, cols], DOUBLE, values in [0.2, 1.2].
     * Offset of 0.2 ensures no entry is zero and pre-relu values stay positive
     * when multiplied by the positive node features.
     */
    private static INDArray makeW(long rows, long cols) {
        return Nd4j.rand(DataType.DOUBLE, rows, cols).addi(0.2);
    }

    /**
     * Random bias vector [len], DOUBLE, values in [0.1, 1.1].
     */
    private static INDArray makeBias(long len) {
        return Nd4j.rand(DataType.DOUBLE, len).addi(0.1);
    }

    // -----------------------------------------------------------------------
    // GCN — forward sanity check
    // -----------------------------------------------------------------------

    /**
     * Forward sanity for {@code gcnConv}: with A=I and W=I the output equals X.
     *
     * <p>Uses an identity adjacency (4 self-loops) so A·X·W = X when W is also
     * the identity.  No bias, no ReLU so the output is trivially verifiable.
     *
     * <pre>
     *   identity adjacency:
     *     colIdx_id = [0, 1, 2, 3]    rowPtr_id = [0, 1, 2, 3, 4]
     *     vals_id   = [1.0, 1.0, 1.0, 1.0]
     *   W = eye(2)
     *   expected out[i] = X[i]  for all i ∈ {0,1,2,3}
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGcnConvForward(Nd4jBackend backend) {
        INDArray colIdxId = Nd4j.createFromArray(new int[]{0, 1, 2, 3});
        INDArray rowPtrId = Nd4j.createFromArray(new int[]{0, 1, 2, 3, 4});
        INDArray valsId   = Nd4j.createFromArray(new double[]{1.0, 1.0, 1.0, 1.0});
        INDArray wArr     = Nd4j.eye(F).castTo(DataType.DOUBLE);   // 2×2 identity
        INDArray xArr     = makeX();

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", colIdxId);
            SDVariable rowPtr = sd.constant("rowPtr", rowPtrId);
            SDVariable vals   = sd.constant("vals",   valsId);
            SDVariable X      = sd.constant("X",      xArr);
            SDVariable W      = sd.constant("W",      wArr);

            // applyRelu=false, no bias → out = I · X · I = X
            SDVariable out = sd.gnn().gcnConv(X, W, null, vals, colIdx, rowPtr, N, N, false);

            INDArray result = out.eval();
            assertArrayEquals(new long[]{N, F}, result.shape(), "gcnConv output shape");
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < F; j++) {
                    assertEquals(xArr.getDouble(i, j), result.getDouble(i, j), 1e-6,
                            "gcnConv identity mismatch at [" + i + "," + j + "]");
                }
            }
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GCN — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code gcnConv}: differentiates w.r.t. X, W, bias, and
     * the sparse edge values (aNormVals).
     *
     * <p>Uses the common 4-node graph with random edge weights and random W/bias.
     * ReLU is disabled to avoid dead-neuron issues on arbitrary initialisations.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGcnConvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345L);

        SameDiff sd = SameDiff.create();
        try {
            // Structural constants (INT32 — skipped by GradCheckUtil)
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            // Differentiable variables (DOUBLE)
            // aNormVals: edge weights for the sparse adjacency [nnz=6]
            SDVariable aVals = sd.var("aVals",
                    Nd4j.createFromArray(new double[]{0.5, 0.5, 0.5, 0.5, 1.0, 1.0}));
            SDVariable X    = sd.var("X",    makeX());          // [4, 2]
            SDVariable W    = sd.var("W",    makeW(F, H));      // [2, 2]
            SDVariable bias = sd.var("bias", makeBias(H));      // [2]

            // applyRelu=false keeps gradient clean; bias shifts pre-activation away from 0
            SDVariable out = sd.gnn().gcnConv(X, W, bias, aVals, colIdx, rowPtr, N, N, false);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "gcnConv gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GAT — forward shape check
    // -----------------------------------------------------------------------

    /**
     * Forward shape check for {@code gatConvHead}: output must be [N, H].
     *
     * <p>A full numeric forward reference for GAT is complex (requires computing
     * per-edge attention coefficients by hand), so this test only verifies the
     * output shape and that the computation produces finite values.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatConvHeadShape(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(22222L);

        SameDiff sd = SameDiff.create();
        try {
            // Structural INT constants
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable rowIdx = sd.constant("rowIdx", makeRowIdx());

            // Fixed (non-differentiated) inputs for shape-only check
            SDVariable X      = sd.constant("X",      makeX());
            SDVariable W      = sd.constant("W",      makeW(F, H));
            SDVariable attSrc = sd.constant("attSrc", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1));
            SDVariable attDst = sd.constant("attDst", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1));

            SDVariable out = sd.gnn().gatConvHead(
                    X, W, attSrc, attDst, colIdx, rowPtr, rowIdx, NNZ, N, 0.2);

            INDArray result = out.eval();
            assertArrayEquals(new long[]{N, H}, result.shape(),
                    "gatConvHead output shape mismatch");
            // Verify no NaN/Inf in the output
            assertFalse(result.isNaN().any(), "gatConvHead produced NaN values");
            assertFalse(result.isInfinite().any(), "gatConvHead produced Infinite values");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GAT — gradient check (exercises gather_bp)
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code gatConvHead}: differentiates w.r.t. X, W, attSrc,
     * and attDst.
     *
     * <p>This test is the primary validation of the {@code gather_bp} fix: the
     * implementation of {@code gatConvHead} calls {@code sd.gather(Wh, rowIdx, 0)}
     * to collect per-edge destination-node features.  Correct analytic gradients
     * w.r.t. X and W therefore depend on a working {@code gather} backward pass.
     *
     * <p>LeakyReLU with α=0.2 is used (rather than plain ReLU) so that every
     * negative pre-activation still has a non-zero gradient.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatConvHeadGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(33333L);

        SameDiff sd = SameDiff.create();
        try {
            // Structural INT constants — skipped by GradCheckUtil
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable rowIdx = sd.constant("rowIdx", makeRowIdx());

            // Differentiable variables (DOUBLE)
            SDVariable X      = sd.var("X",      makeX());                                     // [4, 2]
            SDVariable W      = sd.var("W",      makeW(F, H));                                 // [2, 2]
            SDVariable attSrc = sd.var("attSrc", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1)); // [2, 1]
            SDVariable attDst = sd.var("attDst", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1)); // [2, 1]

            SDVariable out = sd.gnn().gatConvHead(
                    X, W, attSrc, attDst, colIdx, rowPtr, rowIdx, NNZ, N, 0.2);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "gatConvHead gradient check failed (possible gather_bp regression)");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GraphSAGE mean — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code sageMean}: differentiates w.r.t. X, W, and bias.
     *
     * <p>{@code sageMean} concatenates X with the mean-aggregated neighbour
     * features → [rows, 2F], then multiplies by W [2F, H] and adds bias [H].
     * ReLU is applied internally; the bias offset of 0.1 keeps all pre-activations
     * positive so no dead neurons suppress gradients.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSageMeanGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(44444L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            // W must be [2F, H] = [4, 2] because sageMean concatenates X and agg
            SDVariable X    = sd.var("X",    makeX());              // [4, 2]
            SDVariable W    = sd.var("W",    makeW(2L * F, H));     // [4, 2]
            SDVariable bias = sd.var("bias", makeBias(H));          // [2]

            SDVariable out = sd.gnn().sageMean(X, W, bias, colIdx, rowPtr, N, N);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "sageMean gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GraphSAGE max — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code sageMax}: differentiates w.r.t. X, W, and bias.
     *
     * <p>The neighbour-max aggregation (via {@code CsrSegmentMax}) is non-smooth at
     * tie points.  The fixed makeX() values [[1,2],[3,4],[5,6],[7,8]] ensure that
     * for every row of our 4-node graph each feature dimension has a unique
     * maximum-winner, making the argmax indicator single-valued everywhere and
     * numerical differentiation stable.
     *
     * <p>Verified unique per-row per-feature maxima:
     * <pre>
     *   node 0 neighbours {1,2}: feat0 → X[2]=5 wins;  feat1 → X[2]=6 wins
     *   node 1 neighbours {0,3}: feat0 → X[3]=7 wins;  feat1 → X[3]=8 wins
     *   node 2 neighbours {0}:   feat0 → X[0]=1 (sole); feat1 → X[0]=2 (sole)
     *   node 3 neighbours {1}:   feat0 → X[1]=3 (sole); feat1 → X[1]=4 (sole)
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSageMaxGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(55555L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            // W must be [2F, H] = [4, 2]
            SDVariable X    = sd.var("X",    makeX());
            SDVariable W    = sd.var("W",    makeW(2L * F, H));
            SDVariable bias = sd.var("bias", makeBias(H));

            SDVariable out = sd.gnn().sageMax(X, W, bias, colIdx, rowPtr, N);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "sageMax gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GraphSAGE pool — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code sagePool}: differentiates w.r.t. X, wPool, bPool,
     * wOut, and bOut.
     *
     * <p>sagePool applies a per-neighbour MLP (wPool/bPool), max-aggregates, then
     * applies a second linear transform (wOut/bOut).  Dimensions used:
     * <ul>
     *   <li>H_pool = 3 (MLP hidden size)</li>
     *   <li>H_out  = 2 (output embedding size)</li>
     *   <li>wOut shape = [F + H_pool, H_out] = [5, 2]</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSagePoolGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(66666L);

        final int H_pool = 3;
        final int H_out  = 2;

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            SDVariable X     = sd.var("X",     makeX());                        // [4, 2]
            SDVariable wPool = sd.var("wPool", makeW(F, H_pool));               // [2, 3]
            SDVariable bPool = sd.var("bPool", makeBias(H_pool));               // [3]
            SDVariable wOut  = sd.var("wOut",  makeW(F + H_pool, H_out));       // [5, 2]
            SDVariable bOut  = sd.var("bOut",  makeBias(H_out));                // [2]

            SDVariable out = sd.gnn().sagePool(
                    X, wPool, bPool, wOut, bOut, colIdx, rowPtr, N, N);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "sagePool gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GIN — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code ginConv}: differentiates w.r.t. X, w1, b1, w2, b2,
     * and the learnable scalar epsilon.
     *
     * <p>GIN: {@code out = MLP((1+ε)·X + Σ_j X_j)}, a 2-layer MLP with ReLU after
     * the first layer.  With positive X and positive weight initialisation the
     * first-layer pre-activations are positive, keeping the ReLU gradient alive.
     *
     * <p>eps is initialised to 0.5 (scalar shape []) so the per-node self-weight
     * is 1.5, well away from zero.
     *
     * <p>Dimensions:
     * <ul>
     *   <li>H_hidden = 3 (hidden dim of MLP)</li>
     *   <li>H_out    = 2 (output dim)</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGinConvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(77777L);

        final int H_hidden = 3;
        final int H_out    = 2;

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            SDVariable X   = sd.var("X",   makeX());                                // [4, 2]
            SDVariable w1  = sd.var("w1",  makeW(F, H_hidden));                     // [2, 3]
            SDVariable b1  = sd.var("b1",  makeBias(H_hidden));                     // [3]
            SDVariable w2  = sd.var("w2",  makeW(H_hidden, H_out));                 // [3, 2]
            SDVariable b2  = sd.var("b2",  makeBias(H_out));                        // [2]
            // Learnable epsilon: scalar DOUBLE variable (shape [])
            SDVariable eps = sd.var("eps",
                    Nd4j.scalar(DataType.DOUBLE, 0.5));                              // []

            SDVariable out = sd.gnn().ginConv(X, w1, b1, w2, b2, eps, colIdx, rowPtr, N, N);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "ginConv gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GIN with LayerNorm — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code ginConv} with {@code layerNorm=true}.
     *
     * <p>Uses the same 4-node graph and positive DOUBLE initialisations as
     * {@link #testGinConvGradCheck}.  After the two-layer MLP the output is
     * normalised over the feature dimension (dim=1) using a parameter-free
     * Layer Normalisation:
     * <pre>
     *   mean = mean(out, keepDims=true, dim=1)
     *   d    = out - mean
     *   var  = mean(d*d, keepDims=true, dim=1)
     *   out  = d / sqrt(var + 1e-5)
     * </pre>
     * The normalisation is a pure SameDiff composition, so gradients flow
     * automatically.  The 1e-5 epsilon keeps the denominator well away from
     * zero and ensures the Jacobian is finite and well-conditioned at the
     * GradCheckUtil default tolerances.
     *
     * <p>Loss = {@code sd.mean("loss", out)} (scalar) — no {@code setLossVariables}
     * required because the graph has a single terminal output variable.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGinConvLayerNormGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(99999L);

        final int H_hidden = 3;
        final int H_out    = 2;

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            SDVariable X   = sd.var("X",   makeX());                                // [4, 2]
            SDVariable w1  = sd.var("w1",  makeW(F, H_hidden));                     // [2, 3]
            SDVariable b1  = sd.var("b1",  makeBias(H_hidden));                     // [3]
            SDVariable w2  = sd.var("w2",  makeW(H_hidden, H_out));                 // [3, 2]
            SDVariable b2  = sd.var("b2",  makeBias(H_out));                        // [2]
            // Learnable epsilon: scalar DOUBLE variable (shape [])
            SDVariable eps = sd.var("eps",
                    Nd4j.scalar(DataType.DOUBLE, 0.5));                              // []

            // layerNorm=true applies parameter-free LN over dim=1 to the MLP output
            SDVariable out = sd.gnn().ginConv(X, w1, b1, w2, b2, eps, colIdx, rowPtr, N, N, true);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "ginConv(layerNorm=true) gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GraphPooling — globalSumPool forward
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@link GraphPooling#globalSumPool}.
     *
     * <p>5 nodes split across 2 graphs:
     * <pre>
     *   nodeFeatures = [[1,2],[3,4],[5,6],[7,8],[9,10]]
     *   graphIds     = [0,0,0,1,1]
     *   numGraphs    = 2
     *
     *   expected out[0] = [1+3+5, 2+4+6] = [9, 12]
     *   expected out[1] = [7+9,   8+10]  = [16, 18]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGlobalSumPoolForward(Nd4jBackend backend) {
        INDArray features = Nd4j.createFromArray(new double[][]{
                {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}
        });
        INDArray graphIds = Nd4j.createFromArray(new int[]{0, 0, 0, 1, 1});

        SameDiff sd = SameDiff.create();
        try {
            SDVariable f   = sd.constant("features", features);
            SDVariable gid = sd.constant("graphIds", graphIds);

            SDVariable out = GraphPooling.globalSumPool(sd, "sumPool", f, gid, 2);

            INDArray result = out.eval();
            assertArrayEquals(new long[]{2, 2}, result.shape(), "globalSumPool output shape");
            assertEquals(9.0,  result.getDouble(0, 0), 1e-6, "sum graph0 feat0");
            assertEquals(12.0, result.getDouble(0, 1), 1e-6, "sum graph0 feat1");
            assertEquals(16.0, result.getDouble(1, 0), 1e-6, "sum graph1 feat0");
            assertEquals(18.0, result.getDouble(1, 1), 1e-6, "sum graph1 feat1");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GraphPooling — globalMeanPool forward
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@link GraphPooling#globalMeanPool}.
     *
     * <pre>
     *   expected out[0] = [9/3,  12/3]  = [3.0, 4.0]
     *   expected out[1] = [16/2, 18/2]  = [8.0, 9.0]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGlobalMeanPoolForward(Nd4jBackend backend) {
        INDArray features = Nd4j.createFromArray(new double[][]{
                {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}
        });
        INDArray graphIds = Nd4j.createFromArray(new int[]{0, 0, 0, 1, 1});

        SameDiff sd = SameDiff.create();
        try {
            SDVariable f   = sd.constant("features", features);
            SDVariable gid = sd.constant("graphIds", graphIds);

            SDVariable out = GraphPooling.globalMeanPool(sd, "meanPool", f, gid, 2);

            INDArray result = out.eval();
            assertArrayEquals(new long[]{2, 2}, result.shape(), "globalMeanPool output shape");
            assertEquals(3.0, result.getDouble(0, 0), 1e-6, "mean graph0 feat0");
            assertEquals(4.0, result.getDouble(0, 1), 1e-6, "mean graph0 feat1");
            assertEquals(8.0, result.getDouble(1, 0), 1e-6, "mean graph1 feat0");
            assertEquals(9.0, result.getDouble(1, 1), 1e-6, "mean graph1 feat1");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GraphPooling — globalMaxPool forward
    // -----------------------------------------------------------------------

    /**
     * Forward correctness for {@link GraphPooling#globalMaxPool}.
     *
     * <pre>
     *   expected out[0] = [max(1,3,5), max(2,4,6)]  = [5.0, 6.0]
     *   expected out[1] = [max(7,9),   max(8,10)]   = [9.0, 10.0]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGlobalMaxPoolForward(Nd4jBackend backend) {
        INDArray features = Nd4j.createFromArray(new double[][]{
                {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}
        });
        INDArray graphIds = Nd4j.createFromArray(new int[]{0, 0, 0, 1, 1});

        SameDiff sd = SameDiff.create();
        try {
            SDVariable f   = sd.constant("features", features);
            SDVariable gid = sd.constant("graphIds", graphIds);

            SDVariable out = GraphPooling.globalMaxPool(sd, "maxPool", f, gid, 2);

            INDArray result = out.eval();
            assertArrayEquals(new long[]{2, 2}, result.shape(), "globalMaxPool output shape");
            assertEquals(5.0,  result.getDouble(0, 0), 1e-6, "max graph0 feat0");
            assertEquals(6.0,  result.getDouble(0, 1), 1e-6, "max graph0 feat1");
            assertEquals(9.0,  result.getDouble(1, 0), 1e-6, "max graph1 feat0");
            assertEquals(10.0, result.getDouble(1, 1), 1e-6, "max graph1 feat1");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GraphPooling — globalSumPool gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@link GraphPooling#globalSumPool}: differentiates w.r.t.
     * the node-feature matrix.
     *
     * <p>Sum-pooling has a trivially correct analytical gradient (dL/dX[i] = dL/dout[g]
     * where g = graphIds[i]), but we verify it numerically to confirm the SameDiff
     * wiring through {@code UnsortedSegmentSum} is correct end-to-end.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGlobalSumPoolGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(88888L);

        INDArray graphIds = Nd4j.createFromArray(new int[]{0, 0, 0, 1, 1});

        SameDiff sd = SameDiff.create();
        try {
            // Differentiable node features [5, 2] — random, offset to be away from zero
            SDVariable features = sd.var("features",
                    Nd4j.rand(DataType.DOUBLE, 5, 2).addi(0.5));
            SDVariable gid = sd.constant("graphIds", graphIds);

            SDVariable out = GraphPooling.globalSumPool(sd, "sumPool", features, gid, 2);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "globalSumPool gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GATv2 — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code gatV2ConvHead}: differentiates w.r.t. X, W and the
     * shared attention vector.  GATv2 applies the LeakyReLU before the attention
     * projection (dynamic attention); LeakyReLU(α=0.2) keeps every gradient alive.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatV2ConvHeadGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(101010L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable rowIdx = sd.constant("rowIdx", makeRowIdx());

            SDVariable X   = sd.var("X",   makeX());                                     // [4, 2]
            SDVariable W   = sd.var("W",   makeW(F, H));                                 // [2, 2]
            SDVariable att = sd.var("att", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1)); // [2, 1]

            SDVariable out = sd.gnn().gatV2ConvHead(X, W, att, colIdx, rowPtr, rowIdx, NNZ, N, 0.2);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "gatV2ConvHead gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // APPNP — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code appnp}: differentiates w.r.t. X, W, bias and the
     * sparse adjacency values across 3 propagation steps (alpha=0.1).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAppnpGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(111111L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable aVals  = sd.var("aVals",
                    Nd4j.createFromArray(new double[]{0.5, 0.5, 0.5, 0.5, 1.0, 1.0}));
            SDVariable X    = sd.var("X",    makeX());          // [4, 2]
            SDVariable W    = sd.var("W",    makeW(F, H));      // [2, 2]
            SDVariable bias = sd.var("bias", makeBias(H));      // [2]

            SDVariable out = sd.gnn().appnp(X, W, bias, aVals, colIdx, rowPtr, N, N, 3, 0.1);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "appnp gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GCNII — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code gcniiConv}: differentiates w.r.t. H, H0, W and the
     * sparse adjacency values.  applyRelu=false keeps the Jacobian linear and clean.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGcniiConvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(121212L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable aVals  = sd.var("aVals",
                    Nd4j.createFromArray(new double[]{0.5, 0.5, 0.5, 0.5, 1.0, 1.0}));
            SDVariable H0 = sd.var("H0", makeX());          // [4, 2] initial representation
            SDVariable H  = sd.var("H",  makeW(N, F));      // [4, 2] current representation
            SDVariable W  = sd.var("W",  makeW(F, F));      // [2, 2]

            SDVariable out = sd.gnn().gcniiConv(H, H0, W, aVals, colIdx, rowPtr, N, N, 0.1, 0.5, false);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "gcniiConv gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // JKNet — gradient checks
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code jkNetConcat}: 3 layer representations concatenated
     * along the feature dim → [rows, 3H].  Differentiates w.r.t. all three.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testJkNetConcatGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(131313L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable h1 = sd.var("h1", makeW(N, H));   // [4, 2]
            SDVariable h2 = sd.var("h2", makeW(N, H));   // [4, 2]
            SDVariable h3 = sd.var("h3", makeW(N, H));   // [4, 2]

            SDVariable out = sd.gnn().jkNetConcat(h1, h2, h3);   // [4, 6]
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "jkNetConcat gradient check failed");
        } finally {
            sd.close();
        }
    }

    /**
     * Gradient check for {@code jkNetMax}: element-wise max across 3 layer
     * representations.  The three inputs occupy disjoint value ranges
     * ([0,1), [2,3), [4,5)) so each element has a unique, margin-separated winner —
     * the argmax indicator is single-valued and numerical differentiation is stable.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testJkNetMaxGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(141414L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable h1 = sd.var("h1", Nd4j.rand(DataType.DOUBLE, N, H));            // [0,1)
            SDVariable h2 = sd.var("h2", Nd4j.rand(DataType.DOUBLE, N, H).addi(2.0));  // [2,3)
            SDVariable h3 = sd.var("h3", Nd4j.rand(DataType.DOUBLE, N, H).addi(4.0));  // [4,5)

            SDVariable out = sd.gnn().jkNetMax(h1, h2, h3);   // [4, 2], h3 wins each element
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "jkNetMax gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // PairNorm — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code pairNorm}: parameter-free centring + rescale over the
     * node dimension.  Differentiates w.r.t. X.  Random positive features keep the
     * mean-subtracted denominator well away from zero.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPairNormGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(151515L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable X = sd.var("X", Nd4j.rand(DataType.DOUBLE, N, F).addi(0.5));   // [4, 2]

            SDVariable out = sd.gnn().pairNorm(X, 2.0);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "pairNorm gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GraphNorm — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code graphNorm}: differentiates w.r.t. X and the learnable
     * gamma, beta and alpha vectors.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphNormGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(161616L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable X     = sd.var("X",     Nd4j.rand(DataType.DOUBLE, N, F).addi(0.5)); // [4, 2]
            SDVariable gamma = sd.var("gamma", Nd4j.rand(DataType.DOUBLE, F).addi(0.5));    // [2]
            SDVariable beta  = sd.var("beta",  Nd4j.rand(DataType.DOUBLE, F).addi(0.1));    // [2]
            SDVariable alpha = sd.var("alpha", Nd4j.rand(DataType.DOUBLE, F).addi(0.1));    // [2]

            SDVariable out = sd.gnn().graphNorm(X, gamma, beta, alpha);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "graphNorm gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // ChebConv — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code chebConv}: K=3 Chebyshev terms over the common 4-node
     * graph (used here as a stand-in scaled Laplacian).  Differentiates w.r.t. X, the
     * three Chebyshev weight matrices, and the Laplacian values.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testChebConvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(171717L);

        final int K = 3;

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx  = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr  = sd.constant("rowPtr", makeRowPtr());
            SDVariable lapVals = sd.var("lapVals",
                    Nd4j.createFromArray(new double[]{0.5, 0.5, 0.5, 0.5, 1.0, 1.0}));
            SDVariable X = sd.var("X", makeX());                                  // [4, 2]

            SDVariable[] weights = new SDVariable[K];
            for (int k = 0; k < K; k++) {
                weights[k] = sd.var("W" + k, makeW(F, H));                        // [2, 2]
            }

            SDVariable out = sd.gnn().chebConv(X, weights, lapVals, colIdx, rowPtr, N, N);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "chebConv gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // RGCN — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code rgcnConv} with 2 relations (each a 1-edge-per-row
     * permutation over the 4 nodes).  Differentiates w.r.t. X, per-relation values,
     * per-relation weights, the self-loop weight and bias.  applyRelu=false keeps the
     * Jacobian clean.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRgcnConvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(181818L);

        SameDiff sd = SameDiff.create();
        try {
            // Relation 0: cycle 0->1->2->3->0 (one edge per row)
            SDVariable r0col = sd.constant("r0col", Nd4j.createFromArray(new int[]{1, 2, 3, 0}));
            SDVariable r0ptr = sd.constant("r0ptr", Nd4j.createFromArray(new int[]{0, 1, 2, 3, 4}));
            SDVariable r0val = sd.var("r0val", Nd4j.createFromArray(new double[]{0.5, 0.6, 0.7, 0.8}));
            // Relation 1: 0->2, 1->3, 2->0, 3->1
            SDVariable r1col = sd.constant("r1col", Nd4j.createFromArray(new int[]{2, 3, 0, 1}));
            SDVariable r1ptr = sd.constant("r1ptr", Nd4j.createFromArray(new int[]{0, 1, 2, 3, 4}));
            SDVariable r1val = sd.var("r1val", Nd4j.createFromArray(new double[]{0.4, 0.5, 0.6, 0.7}));

            SDVariable X     = sd.var("X",     makeX());        // [4, 2]
            SDVariable rW0   = sd.var("rW0",   makeW(F, H));    // [2, 2]
            SDVariable rW1   = sd.var("rW1",   makeW(F, H));    // [2, 2]
            SDVariable selfW = sd.var("selfW", makeW(F, H));    // [2, 2]
            SDVariable bias  = sd.var("bias",  makeBias(H));    // [2]

            SDVariable[] relVals = {r0val, r1val};
            SDVariable[] relCol  = {r0col, r1col};
            SDVariable[] relPtr  = {r0ptr, r1ptr};
            SDVariable[] relW    = {rW0, rW1};

            SDVariable out = sd.gnn().rgcnConv(
                    X, relVals, relCol, relPtr, relW, selfW, bias, N, N, false);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "rgcnConv gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // PNA — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code pnaConv}: 4 aggregators (mean/max/min/std) concatenated
     * → [rows, 4F], projected by W [4F, H].  The fixed makeX() values guarantee a unique
     * per-row per-feature winner for the (non-smooth) min and max aggregators — see
     * {@link #testSageMaxGradCheck} for the winner table — so numerical differentiation
     * is stable.  applyRelu=false keeps the Jacobian clean.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPnaConvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(191919L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            SDVariable X    = sd.var("X",    makeX());             // [4, 2]
            SDVariable W    = sd.var("W",    makeW(4L * F, H));    // [8, 2] — 4 aggregators × F
            SDVariable bias = sd.var("bias", makeBias(H));         // [2]

            SDVariable out = sd.gnn().pnaConv(X, W, bias, colIdx, rowPtr, N, N, false);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "pnaConv gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // VGAE — gradient checks
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code vgaeReparam}: z = mu + exp(0.5*logvar)*noise.  noise is a
     * fixed constant sample so the function is deterministic; differentiates w.r.t. mu, logvar.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVgaeReparamGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(202020L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable mu     = sd.var("mu",     Nd4j.rand(DataType.DOUBLE, N, F).addi(0.1));  // [4,2]
            SDVariable logvar = sd.var("logvar", Nd4j.rand(DataType.DOUBLE, N, F).subi(0.5));  // [4,2]
            SDVariable noise  = sd.constant("noise", Nd4j.rand(DataType.DOUBLE, N, F));        // fixed sample

            SDVariable z = sd.gnn().vgaeReparam(mu, logvar, noise);
            sd.mean("loss", z);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "vgaeReparam gradient check failed");
        } finally {
            sd.close();
        }
    }

    /**
     * Gradient check for {@code vgaeKlLoss}: the scalar KL regulariser; differentiates w.r.t.
     * mu and logvar.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVgaeKlLossGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(212121L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable mu     = sd.var("mu",     Nd4j.rand(DataType.DOUBLE, N, F).addi(0.1));
            SDVariable logvar = sd.var("logvar", Nd4j.rand(DataType.DOUBLE, N, F).subi(0.5));

            SDVariable kl = sd.gnn().vgaeKlLoss(mu, logvar);
            sd.mean("loss", kl);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "vgaeKlLoss gradient check failed");
        } finally {
            sd.close();
        }
    }

    /**
     * Gradient check for {@code innerProductDecoder}: edge logits Z·Z^T; differentiates w.r.t.
     * the node latents Z.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInnerProductDecoderGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(222222L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable z = sd.var("z", Nd4j.rand(DataType.DOUBLE, N, 3).addi(0.1));  // [4,3]

            SDVariable logits = sd.gnn().innerProductDecoder(z);                     // [4,4]
            sd.mean("loss", logits);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "innerProductDecoder gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // Attention readouts — gradient checks
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@link GraphPooling#globalAttentionPool} (gated attention) over the
     * 5-node / 2-graph batch; differentiates w.r.t. node features and the gate weight.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGlobalAttentionPoolGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(232323L);
        INDArray graphIds = Nd4j.createFromArray(new int[]{0, 0, 0, 1, 1});

        SameDiff sd = SameDiff.create();
        try {
            SDVariable features = sd.var("features", Nd4j.rand(DataType.DOUBLE, 5, 2).addi(0.5)); // [5,2]
            SDVariable gateW    = sd.var("gateW",    Nd4j.rand(DataType.DOUBLE, 2, 1).addi(0.1)); // [2,1]
            SDVariable gid      = sd.constant("graphIds", graphIds);

            SDVariable out = GraphPooling.globalAttentionPool(sd, "attnPool", features, gateW, gid, 2);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "globalAttentionPool gradient check failed");
        } finally {
            sd.close();
        }
    }

    /**
     * Gradient check for {@link GraphPooling#globalSoftmaxAttentionPool} (per-graph softmax
     * attention) over the 5-node / 2-graph batch; differentiates w.r.t. node features and the
     * score weight.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGlobalSoftmaxAttentionPoolGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(242424L);
        INDArray graphIds = Nd4j.createFromArray(new int[]{0, 0, 0, 1, 1});

        SameDiff sd = SameDiff.create();
        try {
            SDVariable features = sd.var("features", Nd4j.rand(DataType.DOUBLE, 5, 2).addi(0.5)); // [5,2]
            SDVariable scoreW   = sd.var("scoreW",   Nd4j.rand(DataType.DOUBLE, 2, 1).addi(0.1)); // [2,1]
            SDVariable gid      = sd.constant("graphIds", graphIds);

            SDVariable out = GraphPooling.globalSoftmaxAttentionPool(sd, "softAttnPool", features, scoreW, gid, 2);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "globalSoftmaxAttentionPool gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GGNN — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code ggnn}: 2 propagation/GRU steps over the common graph;
     * differentiates w.r.t. X, the message transform, all 6 GRU gate weights, and the
     * sparse adjacency values.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGgnnGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(252525L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable aVals  = sd.var("aVals",
                    Nd4j.createFromArray(new double[]{0.5, 0.5, 0.5, 0.5, 1.0, 1.0}));
            SDVariable X    = sd.var("X",    makeX());        // [4, 2], H=2
            SDVariable aggW = sd.var("aggW", makeW(H, H));    // [2, 2]
            SDVariable wz = sd.var("wz", makeW(H, H));
            SDVariable uz = sd.var("uz", makeW(H, H));
            SDVariable wr = sd.var("wr", makeW(H, H));
            SDVariable ur = sd.var("ur", makeW(H, H));
            SDVariable wh = sd.var("wh", makeW(H, H));
            SDVariable uh = sd.var("uh", makeW(H, H));

            SDVariable out = sd.gnn().ggnn(X, aggW, wz, uz, wr, ur, wh, uh,
                    aVals, colIdx, rowPtr, N, N, 2);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "ggnn gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // Graph Transformer — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code graphTransformer}: single-head self-attention (full attention,
     * null mask); differentiates w.r.t. X and the Q/K/V/O weights.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphTransformerGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(262626L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable X  = sd.var("X",  makeX());         // [4, 2]
            SDVariable wq = sd.var("wq", makeW(F, H));     // [2, 2]
            SDVariable wk = sd.var("wk", makeW(F, H));
            SDVariable wv = sd.var("wv", makeW(F, H));
            SDVariable wo = sd.var("wo", makeW(H, H));

            SDVariable out = sd.gnn().graphTransformer(X, wq, wk, wv, wo, null, 1.0 / Math.sqrt(H));
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "graphTransformer gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // NNConv (edge-conditioned) — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code nnConv}: the edge network maps per-edge features to a
     * [Fin,Fout] weight matrix applied to neighbour features.  Differentiates w.r.t. X, the
     * edge features, the edge-network weight/bias, the root transform and the output bias.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNnConvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(272727L);
        final int edgeF = 2;
        final long fin = F;     // 2
        final long fout = H;    // 2

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            SDVariable X            = sd.var("X", makeX());                                       // [4, 2]
            SDVariable edgeFeatures = sd.var("edgeFeatures",
                    Nd4j.rand(DataType.DOUBLE, NNZ, edgeF).addi(0.1));                            // [6, 2]
            SDVariable edgeNetW     = sd.var("edgeNetW", makeW(edgeF, fin * fout));               // [2, 4]
            SDVariable edgeNetB     = sd.var("edgeNetB", makeBias(fin * fout));                   // [4]
            SDVariable rootW        = sd.var("rootW", makeW(fin, fout));                          // [2, 2]
            SDVariable bias         = sd.var("bias", makeBias(fout));                             // [2]

            SDVariable out = sd.gnn().nnConv(X, edgeFeatures, edgeNetW, edgeNetB, rootW, bias,
                    colIdx, rowPtr, N, N, fin, fout);
            sd.mean("loss", out);

            assertTrue(GradCheckUtil.checkGradients(sd, null),
                    "nnConv gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // HAN (heterogeneous, 2 meta-paths) — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code han}: 2 meta-paths (the common graph + a permutation), each a
     * node-level GAT, fused by semantic attention.  Differentiates w.r.t. X, both meta-paths'
     * GAT weights/attention vectors, and the semantic-attention weight/bias/query.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHanGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(282828L);
        final int A = 3;   // semantic-attention dim

        SameDiff sd = SameDiff.create();
        try {
            // meta-path 0: common 4-node graph
            SDVariable col0 = sd.constant("col0", makeColIdx());
            SDVariable ptr0 = sd.constant("ptr0", makeRowPtr());
            SDVariable rid0 = sd.constant("rid0", makeRowIdx());
            // meta-path 1: permutation 0->2,1->3,2->0,3->1 (one edge per row)
            SDVariable col1 = sd.constant("col1", Nd4j.createFromArray(new int[]{2, 3, 0, 1}));
            SDVariable ptr1 = sd.constant("ptr1", Nd4j.createFromArray(new int[]{0, 1, 2, 3, 4}));
            SDVariable rid1 = sd.constant("rid1", Nd4j.createFromArray(new int[]{0, 1, 2, 3}));

            SDVariable X   = sd.var("X", makeX());                                           // [4,2]
            SDVariable mW0 = sd.var("mW0", makeW(F, H));
            SDVariable mW1 = sd.var("mW1", makeW(F, H));
            SDVariable as0 = sd.var("as0", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1));
            SDVariable as1 = sd.var("as1", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1));
            SDVariable ad0 = sd.var("ad0", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1));
            SDVariable ad1 = sd.var("ad1", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1));
            SDVariable semW = sd.var("semW", makeW(H, A));                                   // [2,3]
            SDVariable semB = sd.var("semB", makeBias(A));                                   // [3]
            SDVariable semQ = sd.var("semQ", Nd4j.rand(DataType.DOUBLE, A, 1).addi(0.1));    // [3,1]

            SDVariable[] metaW  = {mW0, mW1};
            SDVariable[] attSrc = {as0, as1};
            SDVariable[] attDst = {ad0, ad1};
            SDVariable[] colIdx = {col0, col1};
            SDVariable[] rowPtr = {ptr0, ptr1};
            SDVariable[] rowIdx = {rid0, rid1};
            long[] nnz = {NNZ, 4};

            SDVariable out = sd.gnn().han(X, metaW, attSrc, attDst, colIdx, rowPtr, rowIdx,
                    semW, semB, semQ, nnz, N, 0.2);
            sd.mean("loss", out);
            assertTrue(GradCheckUtil.checkGradients(sd, null), "han gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // Temporal GCN (3 timesteps) — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code temporalGcn}: a weight-shared spatial GCN over 3 timesteps,
     * fused by temporal attention.  Differentiates w.r.t. all three timesteps' features, the
     * shared GCN weight/bias, the sparse adjacency values, and the temporal-attention params.
     * applyRelu=false keeps the Jacobian clean.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTemporalGcnGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(292929L);
        final int A = 3;   // temporal-attention dim

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable aVals  = sd.var("aVals",
                    Nd4j.createFromArray(new double[]{0.5, 0.5, 0.5, 0.5, 1.0, 1.0}));

            SDVariable X0 = sd.var("X0", makeX());          // [4,2]
            SDVariable X1 = sd.var("X1", makeW(N, F));      // [4,2]
            SDVariable X2 = sd.var("X2", makeW(N, F));      // [4,2]
            SDVariable W     = sd.var("W", makeW(F, H));    // [2,2] shared spatial weight
            SDVariable bias  = sd.var("bias", makeBias(H)); // [2]
            SDVariable tempW = sd.var("tempW", makeW(H, A));                              // [2,3]
            SDVariable tempQ = sd.var("tempQ", Nd4j.rand(DataType.DOUBLE, A, 1).addi(0.1)); // [3,1]

            SDVariable[] Xt = {X0, X1, X2};
            SDVariable out = sd.gnn().temporalGcn(Xt, W, bias, aVals, colIdx, rowPtr,
                    tempW, tempQ, N, N, false);
            sd.mean("loss", out);
            assertTrue(GradCheckUtil.checkGradients(sd, null), "temporalGcn gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // R-GAT (relational attention) — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code rgatConvHead}: relation-aware GAT over the common graph with 2
     * relation types. Differentiates w.r.t. X, W, the relation embeddings and the attention vector.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRgatConvHeadGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(303030L);
        final int numRel = 2;
        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx  = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr  = sd.constant("rowPtr", makeRowPtr());
            SDVariable rowIdx  = sd.constant("rowIdx", makeRowIdx());
            SDVariable edgeRel = sd.constant("edgeRel", Nd4j.createFromArray(new int[]{0, 1, 0, 1, 0, 1}));

            SDVariable X      = sd.var("X", makeX());                                          // [4, 2]
            SDVariable W      = sd.var("W", makeW(F, H));                                       // [2, 2]
            SDVariable relEmb = sd.var("relEmb", makeW(numRel, H));                            // [2, 2] (dim = H)
            SDVariable att    = sd.var("att", Nd4j.rand(DataType.DOUBLE, H, 1).addi(0.1));     // [2, 1]

            SDVariable out = sd.gnn().rgatConvHead(X, W, relEmb, edgeRel, att,
                    colIdx, rowPtr, rowIdx, NNZ, N, 0.2);
            sd.mean("loss", out);
            assertTrue(GradCheckUtil.checkGradients(sd, null), "rgatConvHead gradient check failed");
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // HGT (heterogeneous graph transformer) — gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code hgtConvHead}: relation-modulated scaled dot-product attention.
     * Differentiates w.r.t. X, the Q/K/V weights and the relation embeddings.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHgtConvHeadGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(313131L);
        final int numRel = 2;
        final int d = H;
        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx  = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr  = sd.constant("rowPtr", makeRowPtr());
            SDVariable rowIdx  = sd.constant("rowIdx", makeRowIdx());
            SDVariable edgeRel = sd.constant("edgeRel", Nd4j.createFromArray(new int[]{0, 1, 0, 1, 0, 1}));

            SDVariable X      = sd.var("X", makeX());                  // [4, 2]
            SDVariable wq     = sd.var("wq", makeW(F, d));
            SDVariable wk     = sd.var("wk", makeW(F, d));
            SDVariable wv     = sd.var("wv", makeW(F, d));
            SDVariable relEmb = sd.var("relEmb", makeW(numRel, d));    // [2, d]

            SDVariable out = sd.gnn().hgtConvHead(X, wq, wk, wv, relEmb, edgeRel,
                    colIdx, rowPtr, rowIdx, NNZ, N, 1.0 / Math.sqrt(d));
            sd.mean("loss", out);
            assertTrue(GradCheckUtil.checkGradients(sd, null), "hgtConvHead gradient check failed");
        } finally {
            sd.close();
        }
    }
}
