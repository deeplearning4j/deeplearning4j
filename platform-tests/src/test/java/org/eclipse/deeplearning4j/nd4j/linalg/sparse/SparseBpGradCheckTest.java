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
import org.nd4j.linalg.api.ops.impl.sparse.CooToCsr;
import org.nd4j.linalg.api.ops.impl.sparse.CsrToCsc;
import org.nd4j.linalg.api.ops.impl.sparse.CsrToDense;
import org.nd4j.linalg.api.ops.impl.sparse.DenseToCoo;
import org.nd4j.linalg.api.ops.impl.sparse.DenseToCsr;
import org.nd4j.linalg.api.ops.impl.sparse.Sddmm;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SameDiff gradient-check tests for the six sparse _bp ops that are implemented and
 * registered but were not previously validated by a gradcheck:
 * <ul>
 *   <li>csr_to_dense  (GROUP A — fixed-structure)</li>
 *   <li>csr_to_csc    (GROUP A — fixed-structure)</li>
 *   <li>coo_to_csr    (GROUP A — fixed-structure, identity-coalescing)</li>
 *   <li>sddmm         (GROUP A — fixed-structure)</li>
 *   <li>dense_to_csr  (GROUP B — data-dependent pattern; stabilised by fully-dense input)</li>
 *   <li>dense_to_coo  (GROUP B — data-dependent pattern; stabilised by fully-dense input)</li>
 * </ul>
 *
 * <p><b>Design rules (mirrors SparseGradCheckTest):</b>
 * <ul>
 *   <li>Structural arrays (colIdx, rowPtr, COO indices) → {@code sd.constant(INT32/INT64)} and
 *       are skipped by GradCheckUtil.</li>
 *   <li>Differentiable value arrays → {@code sd.var(DOUBLE)}, well away from zero.</li>
 *   <li>Loss = {@code sd.mean("loss", <float-output>)}.</li>
 *   <li>Multi-output ops call {@code sd.setLossVariables("loss")} so the single-terminal
 *       auto-inference is not applied to the INT structural outputs.</li>
 * </ul>
 *
 * <p><b>Shared 3×3 CSR test pattern</b> (nnz=5):
 * <pre>
 *   A = [[1, 0, 2],
 *        [0, 3, 0],
 *        [4, 0, 5]]
 *   values = [1.0, 2.0, 3.0, 4.0, 5.0]  DOUBLE
 *   colIdx = [0, 2, 1, 0, 2]             INT32
 *   rowPtr = [0, 2, 3, 5]               INT32
 * </pre>
 *
 * <p><b>GROUP B note (dense_to_csr / dense_to_coo):</b>
 * These ops extract the non-zero <em>pattern</em> from the dense input at forward time.
 * GradCheckUtil's finite-difference perturbations must not alter the pattern (which would
 * change the output shape and break the comparison).  This is guaranteed by using a
 * fully-dense input — every entry is clearly non-zero — so a small ε perturbation of any
 * element cannot push it to exactly zero under the default threshold=0.0.
 */
public class SparseBpGradCheckTest extends BaseNd4jTestWithBackends {

    private static final int ROWS = 3;
    private static final int COLS = 3;

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
    // Shared helpers for the 3×3 test matrix
    // -----------------------------------------------------------------------

    private static INDArray makeValues() {
        return Nd4j.createFromArray(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
    }

    private static INDArray makeColIdx() {
        return Nd4j.createFromArray(new int[]{0, 2, 1, 0, 2});
    }

    private static INDArray makeRowPtr() {
        return Nd4j.createFromArray(new int[]{0, 2, 3, 5});
    }

    // -----------------------------------------------------------------------
    // GROUP A — Test 1: csr_to_dense_bp  (gather)
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code csr_to_dense}.
     *
     * <p>The forward op scatters {@code values[e]} into {@code dense[i, colIdx[e]]};
     * the backward op ({@code csr_to_dense_bp}) gathers from {@code gradDense} at the
     * same positions.
     *
     * <p>Differentiates w.r.t. {@code values} [nnz=5].
     * {@code colIdx} and {@code rowPtr} are INT32 constants skipped by the checker.
     * Single output ⇒ no {@code setLossVariables} needed.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrToDenseGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(11111);

        SameDiff sd = SameDiff.create();
        try {
            // Structural constants (INT32 — not differentiated)
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            // Differentiable values (DOUBLE, well away from zero)
            SDVariable values = sd.var("values", makeValues());

            // dense [3, 3] = csr_to_dense(values, colIdx, rowPtr, rows=3, cols=3)
            SDVariable dense = new CsrToDense(sd, values, colIdx, rowPtr, ROWS, COLS)
                    .outputVariable();

            // Scalar loss = mean(dense) — single output, no setLossVariables needed
            sd.mean("loss", dense);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_to_dense"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GROUP A — Test 2: csr_to_csc_bp  (inverse permutation)
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code csr_to_csc}.
     *
     * <p>The forward op applies a counting-sort value permutation to convert CSR→CSC.
     * The backward op ({@code csr_to_csc_bp}) applies the inverse permutation by
     * treating the CSC of A as the CSR of Aᵀ and transposing back.
     *
     * <p>CsrToCsc has 3 outputs: [0] cscValues (float), [1] cscRowIdx (INT32),
     * [2] cscColPtr (INT32).  We take the loss on {@code cscValues} (output 0)
     * and call {@code setLossVariables} to prevent the multi-terminal graph from
     * failing single-terminal auto-inference.
     *
     * <p>Differentiates w.r.t. {@code values} [nnz=5].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrToCscGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(22222);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable values = sd.var("values", makeValues());

            // csr_to_csc outputs: [0]=cscValues, [1]=cscRowIdx, [2]=cscColPtr
            SDVariable cscValues = new CsrToCsc(sd, values, colIdx, rowPtr, ROWS, COLS)
                    .outputVariable();   // outputVariable() returns output[0] = cscValues

            sd.mean("loss", cscValues);
            // 3 outputs in graph → must set the loss variable explicitly
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_to_csc"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GROUP A — Test 3: coo_to_csr_bp  (identity coalescing)
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code coo_to_csr}.
     *
     * <p>The forward op converts COO→CSR, potentially coalescing (summing) duplicate
     * entries.  By using a pattern with <em>unique</em> (row, col) pairs sorted in
     * row-major order, the coalescing step is an identity and the value-gradient is a
     * clean copy: {@code dCooValues[e] = gradCsrValues[perm(e)]}.
     *
     * <p>COO encoding of the same 3×3 test matrix (5 unique entries in row-major order):
     * <pre>
     *   indices = [[0,0],[0,2],[1,1],[2,0],[2,2]]   (INT64, [5, 2])
     *   values  = [1.0, 2.0, 3.0, 4.0, 5.0]        (DOUBLE)
     * </pre>
     *
     * <p>CooToCsr has 3 outputs: [0] csrValues (float), [1] csrColIdx (INT32),
     * [2] csrRowPtr (INT32).  Loss is on csrValues; {@code setLossVariables} required.
     *
     * <p>Differentiates w.r.t. {@code cooValues} [nnz=5].
     * {@code cooIndices} is an INT64 constant and is skipped by the checker.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCooToCsrGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(33333);

        // COO index pairs [5, 2] INT64 — sorted, unique (no duplicates → coalescing is identity)
        INDArray cooIndicesArr = Nd4j.create(new double[][]{
                {0, 0}, {0, 2}, {1, 1}, {2, 0}, {2, 2}
        }).castTo(DataType.INT64);

        SameDiff sd = SameDiff.create();
        try {
            // Structural constant (INT64 — not differentiated)
            SDVariable cooIndices = sd.constant("cooIndices", cooIndicesArr);

            // Differentiable COO values (DOUBLE)
            SDVariable cooValues = sd.var("cooValues", makeValues());

            // coo_to_csr outputs: [0]=csrValues, [1]=csrColIdx, [2]=csrRowPtr
            SDVariable csrValues = new CooToCsr(sd, cooIndices, cooValues, ROWS, COLS)
                    .outputVariable();   // output[0] = csrValues

            sd.mean("loss", csrValues);
            // 3 outputs in graph → set loss explicitly
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for coo_to_csr"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GROUP A — Test 4: sddmm_bp  (dD1, dD2)
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code sddmm} (Sampled Dense-Dense Matrix Multiplication).
     *
     * <p>For each non-zero position {@code k} at row {@code i}, column {@code j}:
     * <pre>
     *   out[k] = sum_l  D1[i, l] * D2[j, l]
     * </pre>
     *
     * <p>Sddmm ctor arg order (SameDiff): {@code rowPtr, colIdx, D1, D2, rows, cols}.
     * Note that {@code rowPtr} comes BEFORE {@code colIdx} (unlike the CSR value-first
     * convention used in CsrSpmv/CsrSpmm).
     *
     * <p>Uses the same 3×3 sparsity pattern (nnz=5) as the other tests:
     * <ul>
     *   <li>rowPtr = [0, 2, 3, 5]    (INT32 constant)</li>
     *   <li>colIdx = [0, 2, 1, 0, 2] (INT32 constant)</li>
     *   <li>D1[3, 2] = sd.var        (rows=3, hidden-dim p=2)</li>
     *   <li>D2[3, 2] = sd.var        (cols=3, hidden-dim p=2)</li>
     * </ul>
     *
     * <p>Single output (out[nnz=5]) ⇒ no {@code setLossVariables} needed.
     *
     * <p>Differentiates w.r.t. both {@code D1} and {@code D2}.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSddmmGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(44444);

        SameDiff sd = SameDiff.create();
        try {
            // Structural constants (INT32 — not differentiated)
            // Sddmm expects rowPtr first, then colIdx
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());

            // Differentiable dense factors (DOUBLE, well away from zero)
            SDVariable D1 = sd.var("D1", Nd4j.rand(DataType.DOUBLE, ROWS, 2).addi(0.5));
            SDVariable D2 = sd.var("D2", Nd4j.rand(DataType.DOUBLE, COLS, 2).addi(0.5));

            // out[nnz=5] — single output, no setLossVariables needed
            SDVariable out = new Sddmm(sd, rowPtr, colIdx, D1, D2, ROWS, COLS)
                    .outputVariable();

            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for sddmm"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GROUP B — Test 5: dense_to_csr_bp  (scatter-back)
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code dense_to_csr}.
     *
     * <p>The forward op scans the dense input and compresses non-zero entries into CSR.
     * The backward op ({@code dense_to_csr_bp}) scatters {@code gradValues[e]} back
     * into {@code dDense[i, colIdx[e]]}.
     *
     * <p><b>GROUP B stability requirement:</b>
     * {@code dense_to_csr} extracts the sparsity pattern FROM the dense input at
     * forward-pass time.  If a finite-difference perturbation flipped an entry to zero,
     * the pattern would change (one fewer non-zero), the output shape would differ, and
     * the numerical gradient comparison would fail.  To guarantee pattern stability the
     * input is chosen to be <em>fully dense</em> — every entry is a positive integer in
     * [1, 6] — so no perturbation of size ε = 1e-5 can push any entry to exactly zero
     * under threshold=0.0.
     *
     * <p>Input: dense[2, 3] = [[1,2,3],[4,5,6]] (sd.var, all non-zero).
     * DenseToCsr has 3 outputs: [0] values (float), [1] colIdx (INT32), [2] rowPtr (INT32).
     * Loss is on values (output 0); {@code setLossVariables} required.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDenseToCsrGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(55555);

        // Fully-dense 2x3 input: all entries in [1, 6], none near zero.
        // Guarantees the extracted CSR pattern is stable under gradcheck's ε perturbations.
        INDArray denseArr = Nd4j.createFromArray(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });

        SameDiff sd = SameDiff.create();
        try {
            // Fully-dense input — the only differentiable variable
            SDVariable dense = sd.var("dense", denseArr);

            // dense_to_csr outputs: [0]=values, [1]=colIdx, [2]=rowPtr
            // threshold=0.0 keeps all entries (the default)
            SDVariable csrValues = new DenseToCsr(sd, dense, 0.0)
                    .outputVariable();   // output[0] = values [nnz=6]

            sd.mean("loss", csrValues);
            // 3 outputs in graph → set loss explicitly
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for dense_to_csr"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // GROUP B — Test 6: dense_to_coo_bp  (scatter-back)
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code dense_to_coo}.
     *
     * <p>The forward op scans the dense input and emits (row, col, value) triplets for
     * each non-zero entry.  The backward op ({@code dense_to_coo_bp}) scatters
     * {@code gradValues[e]} back into {@code dDense[indices[e,0], indices[e,1]]}.
     *
     * <p><b>GROUP B stability requirement:</b>
     * Same reasoning as {@code dense_to_csr}: the COO (row, col) index set is
     * determined by the pattern of non-zeros at forward time.  A perturbation that
     * zeroed any entry would shrink {@code indices} and {@code values} by one row,
     * breaking the numerical comparison.  Using a fully-dense input avoids this.
     *
     * <p>Input: dense[2, 3] = [[1,2,3],[4,5,6]] (sd.var, all non-zero).
     * DenseToCoo has 2 outputs: [0] indices [nnz, 2] (INT64 — structural, not differentiated),
     *                            [1] values [nnz] (float — the loss target).
     * Both outputs are graph terminals → {@code setLossVariables("loss")} is required.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDenseToCooGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(66666);

        // Fully-dense 2x3 input: all entries in [1, 6], none near zero.
        // Guarantees the extracted COO pattern is stable under gradcheck's ε perturbations.
        INDArray denseArr = Nd4j.createFromArray(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });

        SameDiff sd = SameDiff.create();
        try {
            // Fully-dense input — the only differentiable variable
            SDVariable dense = sd.var("dense", denseArr);

            // dense_to_coo outputs: [0]=indices (INT64, structural), [1]=values (float)
            // We differentiate through the values output (output[1]).
            SDVariable[] fwdOuts = new DenseToCoo(sd, dense, 0.0).outputVariables();
            SDVariable cooValues = fwdOuts[1];   // output[1] = values [nnz=6]

            sd.mean("loss", cooValues);
            // 2 outputs in graph (indices + values) → set loss explicitly
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for dense_to_coo"
            );
        } finally {
            sd.close();
        }
    }
}
