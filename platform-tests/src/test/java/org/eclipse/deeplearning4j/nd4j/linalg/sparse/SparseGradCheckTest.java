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
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpgemm;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmm;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmv;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SameDiff gradient-check tests for the CSR sparse BLAS ops.
 *
 * <p>Design:
 * <ul>
 *   <li>{@code values} and the dense operand ({@code B} or {@code x}) are {@code sd.var()} —
 *       GradCheckUtil will numerically differentiate w.r.t. these.</li>
 *   <li>{@code colIdx} and {@code rowPtr} are {@code sd.constant()} — they are INT32/structural
 *       and skipped by the gradient checker.</li>
 *   <li>Loss = mean of the op output (scalar).</li>
 *   <li>All values are DOUBLE for numerical-differentiation accuracy.</li>
 * </ul>
 *
 * <p>The sparse matrix used in each test is:
 * <pre>
 *   A = [[1, 0, 2],
 *        [0, 3, 0],
 *        [4, 0, 5]]   (rows=3, cols=3, nnz=5)
 *   values  = [1, 2, 3, 4, 5]   (DOUBLE)
 *   colIdx  = [0, 2, 1, 0, 2]   (INT32)
 *   rowPtr  = [0, 2, 3, 5]      (INT32)
 * </pre>
 */
public class SparseGradCheckTest extends BaseNd4jTestWithBackends {

    /** rows in the test sparse matrix */
    private static final int ROWS = 3;
    /** cols in the test sparse matrix */
    private static final int COLS = 3;

    /**
     * Purge the constant-handler cache before every test method.
     *
     * Root cause: BaseND4JTest.reclaimGpuMemory() calls
     * DeallocatorService.forceFlushAll() after each test. On the CPU backend,
     * that call frees the native OpaqueDataBuffer backing every entry in
     * ConstantBuffersCache.buffersCache — because those buffers are not marked
     * isConstant=true — while the Java DataBuffer wrappers remain in the cache.
     * The next test calls Nd4j.rand(DataType.DOUBLE,...), which internally calls
     * getConstantBuffer([0.0,1.0,...], DOUBLE), finds the stale (released)
     * wrapper in the cache, passes it to actualizePointerAndIndexer(), and
     * crashes with "Ptr data buffer was released!".
     *
     * Purging the cache here ensures each test method starts with an empty
     * cache, so getConstantBuffer() always allocates a fresh buffer rather than
     * returning a previously-released one.
     */
    @BeforeEach
    public void purgeConstantHandlerCache() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    /**
     * Build the INT32 structural arrays for the 3x3 test matrix.
     * They are the same in every test; only {@code values} and the dense operand differ.
     */
    private static INDArray makeColIdx() {
        return Nd4j.createFromArray(new int[]{0, 2, 1, 0, 2});
    }

    private static INDArray makeRowPtr() {
        return Nd4j.createFromArray(new int[]{0, 2, 3, 5});
    }

    /**
     * Initial non-zero values for the sparse matrix.
     * Chosen to be well away from zero so numerical derivatives are stable.
     */
    private static INDArray makeValues() {
        return Nd4j.createFromArray(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
    }

    // -----------------------------------------------------------------------
    // CsrSpmm gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code C = A * B}  (CsrSpmm, non-transpose).
     *
     * <p>Differentiates w.r.t. {@code values} [nnz=5] and {@code B} [3x2], checks
     * analytical grad (from doDiff) matches numerical finite-difference grad.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrSpmmGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        try {
            // Structural constants (INT32 — not differentiated)
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());

            // Differentiable variables (DOUBLE)
            SDVariable values = sd.var("values", makeValues());
            SDVariable B      = sd.var("B", Nd4j.rand(DataType.DOUBLE, COLS, 2).addi(0.1));

            // Build the graph: C [3, 2] = A[3,3] * B[3,2]
            SDVariable C = new CsrSpmm(sd, values, colIdx, rowPtr, B, ROWS, COLS, false)
                    .outputVariable();

            // Scalar loss
            sd.mean("loss", C);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_spmm (non-transpose)"
            );
        } finally {
            sd.close();
        }
    }

    /**
     * Gradient check for {@code C = Aᵀ * B}  (CsrSpmm, transposeA=true).
     *
     * <p>When transposeA=true: B must be [rows=3, n]; output C is [cols=3, n].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrSpmmTransposeGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(54321);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable values = sd.var("values", makeValues());
            // B is [rows, n] for the transpose case
            SDVariable B = sd.var("B", Nd4j.rand(DataType.DOUBLE, ROWS, 2).addi(0.1));

            SDVariable C = new CsrSpmm(sd, values, colIdx, rowPtr, B, ROWS, COLS, true)
                    .outputVariable();
            sd.mean("loss", C);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_spmm (transposeA=true)"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // CsrSpmv gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code y = A * x}  (CsrSpmv, non-transpose).
     *
     * <p>Differentiates w.r.t. {@code values} [nnz=5] and {@code x} [cols=3].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrSpmvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(99999);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable values = sd.var("values", makeValues());
            SDVariable x      = sd.var("x", Nd4j.rand(DataType.DOUBLE, COLS).addi(0.1));

            // y [3] = A[3,3] * x[3]
            SDVariable y = new CsrSpmv(sd, values, colIdx, rowPtr, x, ROWS, COLS, false)
                    .outputVariable();

            // Scalar loss = mean(y)
            sd.mean("loss", y);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_spmv (non-transpose)"
            );
        } finally {
            sd.close();
        }
    }

    /**
     * Gradient check for {@code z = Aᵀ * dy}  (CsrSpmv, transposeA=true).
     *
     * <p>When transposeA=true: input vector has length rows; output has length cols.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrSpmvTransposeGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(11111);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", makeColIdx());
            SDVariable rowPtr = sd.constant("rowPtr", makeRowPtr());
            SDVariable values = sd.var("values", makeValues());
            // For transpose: input length = rows
            SDVariable dy = sd.var("dy", Nd4j.rand(DataType.DOUBLE, ROWS).addi(0.1));

            // z [cols=3] = Aᵀ[3,3] * dy[3]
            SDVariable z = new CsrSpmv(sd, values, colIdx, rowPtr, dy, ROWS, COLS, true)
                    .outputVariable();
            sd.mean("loss", z);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_spmv (transposeA=true)"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // CsrSpgemm gradient check
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code C = A · B} (CsrSpgemm), 3×3 · 3×2 = 3×2.
     *
     * <p>Sparse matrix A (3×3, nnz=5) — same pattern as other tests:
     * <pre>
     *   A = [[1, 0, 2],
     *        [0, 3, 0],
     *        [4, 0, 5]]
     * </pre>
     * Sparse matrix B (3×2, nnz=6 — fully non-zero to maximise gradient signal):
     * <pre>
     *   B = [[b00, b01],
     *        [b10, b11],
     *        [b20, b21]]
     * </pre>
     * C = A·B (3×2, nnz=6 — each row of C touches at least one non-zero from B).
     *
     * <p>Both {@code aValues} and {@code bValues} are DOUBLE {@code sd.var()} nodes;
     * structural int arrays are {@code sd.constant()}.
     * Loss = mean(cValues).  Checks dA and dB via {@link CsrSpgemm#doDiff}.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrSpgemmGradCheck3x3(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(22222);

        // A (3×3):  values=[1,2,3,4,5], colIdx=[0,2,1,0,2], rowPtr=[0,2,3,5]
        final int M = 3, K = 3, N = 2;

        // B (3×2) — all 6 entries non-zero so every gradient component is exercised
        //   bColIdx=[0,1,0,1,0,1], bRowPtr=[0,2,4,6]
        INDArray bColIdxArr = Nd4j.createFromArray(new int[]{0, 1, 0, 1, 0, 1});
        INDArray bRowPtrArr = Nd4j.createFromArray(new int[]{0, 2, 4, 6});

        SameDiff sd = SameDiff.create();
        try {
            // A structural constants
            SDVariable aColIdx = sd.constant("aColIdx", makeColIdx());
            SDVariable aRowPtr = sd.constant("aRowPtr", makeRowPtr());

            // B structural constants
            SDVariable bColIdx = sd.constant("bColIdx", bColIdxArr);
            SDVariable bRowPtr = sd.constant("bRowPtr", bRowPtrArr);

            // Differentiable value arrays (DOUBLE, well away from zero for numerical stability)
            SDVariable aValues = sd.var("aValues", makeValues());
            SDVariable bValues = sd.var("bValues",
                    Nd4j.createFromArray(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));

            // C [3,2] = A[3,3] · B[3,2];  cValues is output[0]
            SDVariable cValues = new CsrSpgemm(sd,
                    aValues, aColIdx, aRowPtr,
                    bValues, bColIdx, bRowPtr,
                    M, K, N).outputVariable();

            // Scalar loss = mean(cValues)
            sd.mean("loss", cValues);
            // csr_spgemm has 3 outputs (cValues, cColIdx, cRowPtr); the unconsumed
            // INT structural outputs are extra graph terminals, so the loss var must
            // be set explicitly (single-terminal auto-inference does not apply here).
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_spgemm 3×3·3×2"
            );
        } finally {
            sd.close();
        }
    }

    /**
     * Gradient check for {@code C = A · B} (CsrSpgemm), 4×3 · 3×2 = 4×2.
     *
     * <p>Sparse matrix A (4×3, nnz=7):
     * <pre>
     *   A = [[1, 0, 2],
     *        [0, 3, 0],
     *        [4, 0, 5],
     *        [0, 6, 7]]
     * </pre>
     * Sparse matrix B (3×2, nnz=6 — fully non-zero).
     * C = A·B (4×2, nnz=8).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrSpgemmGradCheck4x3(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(33333);

        final int M = 4, K = 3, N = 2;

        // A (4×3, nnz=7)
        //   row 0: (0,0)=1, (0,2)=2  → colIdx entries [0,2]
        //   row 1: (1,1)=3           → colIdx entries [1]
        //   row 2: (2,0)=4, (2,2)=5  → colIdx entries [0,2]
        //   row 3: (3,1)=6, (3,2)=7  → colIdx entries [1,2]
        INDArray aColIdxArr = Nd4j.createFromArray(new int[]{0, 2, 1, 0, 2, 1, 2});
        INDArray aRowPtrArr = Nd4j.createFromArray(new int[]{0, 2, 3, 5, 7});

        // B (3×2, nnz=6)
        INDArray bColIdxArr = Nd4j.createFromArray(new int[]{0, 1, 0, 1, 0, 1});
        INDArray bRowPtrArr = Nd4j.createFromArray(new int[]{0, 2, 4, 6});

        SameDiff sd = SameDiff.create();
        try {
            SDVariable aColIdx = sd.constant("aColIdx", aColIdxArr);
            SDVariable aRowPtr = sd.constant("aRowPtr", aRowPtrArr);
            SDVariable bColIdx = sd.constant("bColIdx", bColIdxArr);
            SDVariable bRowPtr = sd.constant("bRowPtr", bRowPtrArr);

            SDVariable aValues = sd.var("aValues",
                    Nd4j.createFromArray(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0));
            SDVariable bValues = sd.var("bValues",
                    Nd4j.createFromArray(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));

            // C [4,2] = A[4,3] · B[3,2]; cValues is output[0]
            SDVariable cValues = new CsrSpgemm(sd,
                    aValues, aColIdx, aRowPtr,
                    bValues, bColIdx, bRowPtr,
                    M, K, N).outputVariable();

            sd.mean("loss", cValues);
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_spgemm 4×3·3×2"
            );
        } finally {
            sd.close();
        }
    }
}
