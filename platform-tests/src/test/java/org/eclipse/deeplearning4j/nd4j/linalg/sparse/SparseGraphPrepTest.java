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
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.api.ops.impl.sparse.CsrDiagMm;
import org.nd4j.linalg.api.ops.impl.sparse.Spdiags;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the new sparse graph-preprocessing ops ({@link Spdiags}, {@link CsrDiagMm})
 * and the corresponding {@link SparseNDArray} ergonomics
 * ({@code addSelfLoops}, {@code laplacian}, {@code normalizeSymmetric},
 * {@code scaleRows}, {@code scaleCols}) plus {@link Nd4j#sparseFromEdges}.
 *
 * <p>Test matrix (3×3, nnz=5 — same as {@link SparseGradCheckTest}):
 * <pre>
 *   A = [[1, 0, 2],
 *        [0, 3, 0],
 *        [4, 0, 5]]
 *   values = [1, 2, 3, 4, 5]  DOUBLE
 *   colIdx = [0, 2, 1, 0, 2]  INT32
 *   rowPtr = [0, 2, 3, 5]     INT32
 * </pre>
 */
public class SparseGraphPrepTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-6;

    private static final int ROWS = 3;
    private static final int COLS = 3;

    /** Shared CSR component arrays (built fresh before each test via helpers below). */

    @BeforeEach
    public void purgeConstantHandlerCache() {
        // Prevents stale ConstantBuffersCache entries from crashing across tests —
        // see SparseGradCheckTest for the full root-cause explanation.
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // CSR helpers for the 3x3 test matrix
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

    /** Build the CSR SparseNDArray for the 3x3 test matrix. */
    private static SparseNDArray makeCsr() {
        return new SparseNDArray(makeValues(), makeColIdx(), makeRowPtr(),
                new long[]{ROWS, COLS}, SparseFormat.CSR);
    }

    /** Assert element-wise equality of two 2D matrices within TOL. */
    private static void assertMatrixEquals(double[][] expected, INDArray actual, String msg) {
        assertEquals(expected.length, actual.rows(), msg + ": row count");
        assertEquals(expected[0].length, actual.columns(), msg + ": col count");
        for (int r = 0; r < expected.length; r++) {
            for (int c = 0; c < expected[r].length; c++) {
                assertEquals(expected[r][c], actual.getDouble(r, c), TOL,
                        msg + " at [" + r + "," + c + "]");
            }
        }
    }

    // -----------------------------------------------------------------------
    // (i) spdiags: output toDense equals diag on diagonal, zero elsewhere
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpdiagsBuildsDiagonalMatrix(Nd4jBackend backend) {
        INDArray diag = Nd4j.createFromArray(new double[]{3.0, 7.0, 2.0, 5.0});
        long n = 4;

        Spdiags op = new Spdiags(diag, n);
        INDArray[] results = Nd4j.exec(op);

        // Wrap outputs as CSR and materialise
        SparseNDArray sparse = new SparseNDArray(
                results[0], results[1], results[2], new long[]{n, n}, SparseFormat.CSR);
        INDArray dense = sparse.toDense();

        // The result must be a 4×4 diagonal matrix
        assertEquals(n, dense.rows(), "rows");
        assertEquals(n, dense.columns(), "cols");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? diag.getDouble(i) : 0.0;
                assertEquals(expected, dense.getDouble(i, j), TOL,
                        "spdiags dense at [" + i + "," + j + "]");
            }
        }
    }

    // -----------------------------------------------------------------------
    // (ii) csr_diag_mm: outValues match Dl * A_dense * Dr
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrDiagMmMatchesDense(Nd4jBackend backend) {
        /*
         * A = [[1,0,2],[0,3,0],[4,0,5]]
         * dl = [2, 3, 4]   →  Dl = diag([2,3,4])
         * dr = [5, 6, 7]   →  Dr = diag([5,6,7])
         * expected = Dl * A * Dr:
         *   [2*1*5, 0,     2*2*7]   = [10,  0, 28]
         *   [0,     3*3*6, 0    ]   = [ 0, 54,  0]
         *   [4*4*5, 0,     4*5*7]   = [80,  0, 140]
         */
        INDArray aValues = makeValues();
        INDArray aColIdx = makeColIdx();
        INDArray aRowPtr = makeRowPtr();
        INDArray dl = Nd4j.createFromArray(new double[]{2.0, 3.0, 4.0});
        INDArray dr = Nd4j.createFromArray(new double[]{5.0, 6.0, 7.0});

        CsrDiagMm op = new CsrDiagMm(aValues, aColIdx, aRowPtr, dl, dr, ROWS, COLS);
        INDArray[] results = Nd4j.exec(op);
        INDArray outValues = results[0]; // [nnz=5]

        // Reconstruct CSR with scaled values (structure unchanged)
        SparseNDArray scaled = new SparseNDArray(
                outValues, aColIdx, aRowPtr, new long[]{ROWS, COLS}, SparseFormat.CSR);
        INDArray dense = scaled.toDense();

        double[][] expected = {
                {10.0,  0.0,  28.0},
                { 0.0, 54.0,   0.0},
                {80.0,  0.0, 140.0}
        };
        assertMatrixEquals(expected, dense, "csr_diag_mm vs Dl*A*Dr");
    }

    // -----------------------------------------------------------------------
    // (iii) addSelfLoops: CSR(A+I) values and toDense == A_dense + I
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddSelfLoops(Nd4jBackend backend) {
        SparseNDArray csr = makeCsr();
        SparseNDArray withLoops = csr.addSelfLoops();

        INDArray dense = withLoops.toDense();

        /*
         * A + I:
         *  [[1+1, 0, 2], [0, 3+1, 0], [4, 0, 5+1]]
         *  = [[2, 0, 2], [0, 4, 0], [4, 0, 6]]
         */
        double[][] expected = {
                {2.0, 0.0, 2.0},
                {0.0, 4.0, 0.0},
                {4.0, 0.0, 6.0}
        };
        assertEquals(SparseFormat.CSR, withLoops.getFormat(), "format must stay CSR");
        assertMatrixEquals(expected, dense, "addSelfLoops A+I");
    }

    // -----------------------------------------------------------------------
    // (iv) laplacian: L = D - A
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLaplacian(Nd4jBackend backend) {
        /*
         * rowSums(A) = [3, 3, 9]
         * D = diag([3,3,9])
         * L = D - A:
         *   [3-1,  0,  0-2]   = [ 2,  0, -2]
         *   [0,   3-3, 0  ]   = [ 0,  0,  0]
         *   [0-4,  0,  9-5]   = [-4,  0,  4]
         */
        SparseNDArray csr = makeCsr();
        SparseNDArray L = csr.laplacian();

        INDArray dense = L.toDense();

        double[][] expected = {
                { 2.0, 0.0, -2.0},
                { 0.0, 0.0,  0.0},
                {-4.0, 0.0,  4.0}
        };
        assertEquals(SparseFormat.CSR, L.getFormat(), "laplacian format must be CSR");
        assertMatrixEquals(expected, dense, "laplacian D-A");
    }

    // -----------------------------------------------------------------------
    // (v) normalizeSymmetric: Â = D̃^{-1/2}(A+I)D̃^{-1/2}
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormalizeSymmetric(Nd4jBackend backend) {
        /*
         * A+I = [[2,0,2],[0,4,0],[4,0,6]]
         * rowSums(A+I) = [4, 4, 10]
         * D̃^{-1/2} = [1/2, 1/2, 1/sqrt(10)]
         *
         * Â = D̃^{-1/2}(A+I)D̃^{-1/2}  (entry-wise: Â[i,j] = (A+I)[i,j] * dInv[i] * dInv[j])
         *
         *  [0,0]: 2 * 0.5 * 0.5             = 0.5
         *  [0,2]: 2 * 0.5 * 1/√10           = 1/√10 ≈ 0.31623
         *  [1,1]: 4 * 0.5 * 0.5             = 1.0
         *  [2,0]: 4 * (1/√10) * 0.5         = 2/√10 ≈ 0.63246
         *  [2,2]: 6 * (1/√10) * (1/√10)     = 6/10 = 0.6
         */
        SparseNDArray csr = makeCsr();
        SparseNDArray norm = csr.normalizeSymmetric();
        INDArray dense = norm.toDense();

        double inv2   = 0.5;
        double inv10  = 1.0 / Math.sqrt(10.0);

        double[][] expected = {
                {2 * inv2  * inv2,   0.0,  2 * inv2  * inv10},
                {0.0,                4 * inv2  * inv2,    0.0},
                {4 * inv10 * inv2,   0.0,  6 * inv10 * inv10}
        };

        assertEquals(SparseFormat.CSR, norm.getFormat(), "normalizeSymmetric format must be CSR");
        assertMatrixEquals(expected, dense, "normalizeSymmetric Â = D̃^{-1/2}(A+I)D̃^{-1/2}");
    }

    // -----------------------------------------------------------------------
    // (vi) sparseFromEdges with duplicate edges → toCsr() sums duplicates
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSparseFromEdgesCoalescesDuplicates(Nd4jBackend backend) {
        /*
         * Edges: (0→1) weight=1.0, (0→1) weight=2.0 (duplicate), (1→0) weight=3.0
         * After coalescing via toCsr():
         *   (0,1) = 1+2 = 3.0
         *   (1,0) = 3.0
         * Dense result (2×2):
         *   [[0, 3],
         *    [3, 0]]
         */
        INDArray src     = Nd4j.createFromArray(new int[]{0, 0, 1}).castTo(DataType.INT64);
        INDArray dst     = Nd4j.createFromArray(new int[]{1, 1, 0}).castTo(DataType.INT64);
        INDArray weights = Nd4j.createFromArray(new double[]{1.0, 2.0, 3.0});

        SparseNDArray coo = Nd4j.sparseFromEdges(src, dst, weights, 2L, 2L);

        assertEquals(SparseFormat.COO, coo.getFormat(), "sparseFromEdges must return COO");
        assertEquals(2L, coo.rows(), "rows");
        assertEquals(2L, coo.cols(), "cols");
        assertEquals(3L, coo.nnz(), "nnz in COO before coalescing");

        // toCsr() triggers coo_to_csr which sums duplicate entries
        SparseNDArray csr = coo.toCsr();
        INDArray dense = csr.toDense();

        double[][] expected = {
                {0.0, 3.0},
                {3.0, 0.0}
        };
        assertMatrixEquals(expected, dense, "sparseFromEdges duplicate coalescing");
    }

    // -----------------------------------------------------------------------
    // (vii-a) Gradient check: spdiags
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code spdiags}: loss = mean(values_output).
     *
     * <p>Since {@code spdiags} has 3 outputs (values, colIdx, rowPtr) the loss variable must
     * be set explicitly via {@link SameDiff#setLossVariables(String...)} — single-terminal
     * auto-inference does not apply for multi-output ops.
     *
     * <p>Expected: {@code d(diag) = grads.get(0) = 1/n} (gradient of mean over n elements).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpdiagGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(77777);

        SameDiff sd = SameDiff.create();
        try {
            // Diagonal values — the only differentiable input
            SDVariable diag = sd.var("diag",
                    Nd4j.createFromArray(new double[]{2.0, 5.0, 3.0}));

            // Build graph: spdiags(diag, 3) → [values, colIdx, rowPtr]
            // outputVariable() returns output[0] = values
            SDVariable values = new Spdiags(sd, diag, 3L).outputVariable();

            // Scalar loss = mean(values)
            sd.mean("loss", values);

            // spdiags has 3 outputs — must set loss explicitly
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for spdiags"
            );
        } finally {
            sd.close();
        }
    }

    // -----------------------------------------------------------------------
    // (vii-b) Gradient check: csr_diag_mm
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code csr_diag_mm}: loss = mean(outValues).
     *
     * <p>Only {@code aValues} is a {@code sd.var()} — the gradient checker differentiates
     * with respect to it.  {@code aColIdx}, {@code aRowPtr} (INT32 structural) and
     * {@code dl}, {@code dr} (diagonal vectors — gradients are returned as zero, as
     * documented in {@link CsrDiagMm}) are {@code sd.constant()}, so the gradient
     * checker skips them.
     *
     * <p>Expected: {@code dA[e] = g[e] * dl[i_e] * dr[j_e]} where
     * {@code g = 1/nnz} (gradient of mean over nnz=5 elements).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrDiagMmGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(88888);

        SameDiff sd = SameDiff.create();
        try {
            // Structural constants (INT32 — not differentiated)
            SDVariable aColIdx = sd.constant("aColIdx", makeColIdx());
            SDVariable aRowPtr = sd.constant("aRowPtr", makeRowPtr());

            // Diagonal vectors — constants (gradients returned as zero; see CsrDiagMm doc)
            SDVariable dl = sd.constant("dl", Nd4j.createFromArray(new double[]{2.0, 3.0, 4.0}));
            SDVariable dr = sd.constant("dr", Nd4j.createFromArray(new double[]{5.0, 6.0, 7.0}));

            // Differentiable values variable (DOUBLE, well away from zero)
            SDVariable aValues = sd.var("aValues", makeValues());

            // Build graph: outValues[nnz] = dl[i] * aValues[e] * dr[j]
            SDVariable outValues = new CsrDiagMm(sd, aValues, aColIdx, aRowPtr, dl, dr, ROWS, COLS)
                    .outputVariable();

            // Scalar loss = mean(outValues) — single output, no setLossVariables needed
            sd.mean("loss", outValues);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_diag_mm"
            );
        } finally {
            sd.close();
        }
    }
}
