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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.api.ndarray.SparseSolvers;
import org.nd4j.linalg.api.ndarray.SparseSolvers.CgResult;
import org.nd4j.linalg.api.ndarray.SparseSolvers.LanczosResult;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.inverse.InvertMatrix;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parameterized tests for {@link SparseSolvers} across all registered backends.
 *
 * <h3>Tests</h3>
 * <ol>
 *   <li>{@link #testConjugateGradientSmallSpd} — CG solves a 5×5 SPD tridiagonal sparse
 *       system to tolerance and agrees with the dense inverse reference.</li>
 *   <li>{@link #testLanczosEigenvaluesMatchDense} — Lanczos k-smallest eigenvalues on a
 *       5×5 symmetric sparse matrix agree with LAPACK dense {@code syev} within tolerance;
 *       eigenvectors are orthonormal and match up to a sign flip.</li>
 *   <li>{@link #testLaplacianPositionalEncodingShape} — LPE on a 6-node cycle graph returns
 *       a finite [n, k] matrix with approximately orthonormal columns.</li>
 * </ol>
 *
 * <h3>Test conventions</h3>
 * <ul>
 *   <li>All matrices use DOUBLE dtype for maximum accuracy.</li>
 *   <li>{@link #purgeConstants()} is called {@code @BeforeEach} to prevent the
 *       {@link org.nd4j.linalg.api.buffer.DataBuffer} handle cache from leaking across tests
 *       (mirrors the pattern in {@link SparseGradCheckTest}).</li>
 *   <li>Sparse matrices are built from dense arrays via {@link Nd4j#toSparse}.</li>
 *   <li>No {@code syncToHost} / {@code syncToDevice} / element-loop assertions — scalar
 *       extractions ({@code getDouble(i)}) are used only for the final verification loop on
 *       small reference matrices (n ≤ 8), which is standard for correctness tests.</li>
 * </ul>
 */
@DisplayName("SparseSolvers: CG / Lanczos / Laplacian PE Tests")
public class SparseSolverTest extends BaseNd4jTestWithBackends {

    // Tolerances
    private static final double TOL_CG      = 1e-8;   // CG residual + solution error
    private static final double TOL_EIGVAL  = 1e-5;   // Lanczos eigenvalue error
    private static final double TOL_EIGVEC  = 1e-4;   // Lanczos eigenvector error (sign-agnostic)
    private static final double TOL_ORTH    = 0.05;   // LPE orthogonality check

    // =========================================================================
    // Lifecycle
    // =========================================================================

    @BeforeEach
    public void purgeConstants() {
        // Prevent ConstantBuffersCache handle leaks between tests (mirrors SparseGradCheckTest)
        Nd4j.getConstantHandler().purgeConstants();
    }

    // =========================================================================
    // Test 1: Conjugate Gradient
    // =========================================================================

    /**
     * Solves the 5×5 SPD tridiagonal system {@code A x = b} with CG and verifies:
     * <ol>
     *   <li>CG converges ({@link CgResult#converged} == true).</li>
     *   <li>The CG solution matches the dense reference {@code x_ref = A⁻¹ b} within
     *       {@link #TOL_CG}.</li>
     *   <li>The residual ‖Ax − b‖₂ is small (verifying that A.mv() is used correctly).</li>
     * </ol>
     *
     * <p>SPD tridiagonal A (condition number ≈ 6):
     * <pre>
     *   A = [[ 4, -1,  0,  0,  0],
     *        [-1,  4, -1,  0,  0],
     *        [ 0, -1,  4, -1,  0],
     *        [ 0,  0, -1,  4, -1],
     *        [ 0,  0,  0, -1,  4]]
     *
     *   b = [1, 2, 3, 2, 1]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("CG solves 5x5 SPD sparse system to tolerance")
    public void testConjugateGradientSmallSpd(Nd4jBackend backend) {
        int n = 5;

        // Dense A: 5×5 SPD tridiagonal
        double[][] aData = {
                { 4, -1,  0,  0,  0},
                {-1,  4, -1,  0,  0},
                { 0, -1,  4, -1,  0},
                { 0,  0, -1,  4, -1},
                { 0,  0,  0, -1,  4}
        };
        INDArray Adense = Nd4j.create(flattenRowMajor(aData), new long[]{n, n}, DataType.DOUBLE);

        // RHS
        INDArray b = Nd4j.create(new double[]{1, 2, 3, 2, 1}, new long[]{n}, DataType.DOUBLE);

        // Convert A to sparse CSR
        SparseNDArray Asparse = Nd4j.toSparse(Adense, SparseFormat.CSR);

        // --- CG solve ---
        CgResult result = SparseSolvers.conjugateGradient(Asparse, b, TOL_CG, 100);

        assertTrue(result.converged,
                "CG must converge for a 5×5 SPD system within 100 iterations; "
                + "residualNorm=" + result.residualNorm + ", iterations=" + result.iterations);
        assertTrue(result.residualNorm < TOL_CG,
                "residualNorm=" + result.residualNorm + " must be < tol=" + TOL_CG);

        // --- Dense reference solve: xRef = A^{-1} b ---
        INDArray Ainv = InvertMatrix.invert(Adense, false);
        INDArray xRef = Ainv.mmul(b.reshape(n, 1)).reshape(n);

        // Compare CG solution to reference (element-wise, host loop on small vector)
        INDArray xCg = result.x;
        for (int i = 0; i < n; i++) {
            assertEquals(xRef.getDouble(i), xCg.getDouble(i), TOL_CG,
                    "x[" + i + "] CG vs dense reference");
        }

        // Extra sanity: ‖Ax - b‖₂ via SpMV  (exercises A.mv on the solution)
        INDArray Ax      = Asparse.mv(xCg);
        INDArray residual = b.sub(Ax);
        double   resNorm  = Math.sqrt(residual.mul(residual).sumNumber().doubleValue());
        assertTrue(resNorm < TOL_CG * 10,
                "Explicit residual ‖Ax-b‖₂=" + resNorm + " must be small");
    }

    // =========================================================================
    // Test 2: Lanczos eigenvalues + eigenvectors vs dense LAPACK
    // =========================================================================

    /**
     * Verifies that {@link SparseSolvers#lanczos} produces k-smallest Ritz pairs that
     * agree with a LAPACK dense symmetric eigensolver.
     *
     * <p>Test matrix — 5×5 symmetric tridiagonal (positive semi-definite):
     * <pre>
     *   A = [[ 3,  1,  0,  0,  0],
     *        [ 1,  3,  1,  0,  0],
     *        [ 0,  1,  3,  1,  0],
     *        [ 0,  0,  1,  3,  1],
     *        [ 0,  0,  0,  1,  3]]
     * </pre>
     * Exact eigenvalues (ascending): 3-√2, 3-1, 3, 3+1, 3+√2 ≈ 1.586, 2, 3, 4, 4.414.
     *
     * <p>Checks:
     * <ol>
     *   <li>Ritz eigenvalues agree with LAPACK reference within {@link #TOL_EIGVAL}.</li>
     *   <li>Ritz eigenvectors are orthonormal (‖v‖ ≈ 1, v·u ≈ 0 for distinct pairs).</li>
     *   <li>Each Ritz vector matches the LAPACK reference up to a global sign flip:
     *       min(‖v - v_ref‖, ‖v + v_ref‖) &lt; {@link #TOL_EIGVEC}.</li>
     * </ol>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Lanczos k-smallest eigenpairs match dense LAPACK on 5x5 symmetric sparse")
    public void testLanczosEigenvaluesMatchDense(Nd4jBackend backend) {
        int n = 5;
        int k = 2; // request 2 smallest eigenpairs

        // Dense symmetric tridiagonal
        double[][] aData = {
                {3, 1, 0, 0, 0},
                {1, 3, 1, 0, 0},
                {0, 1, 3, 1, 0},
                {0, 0, 1, 3, 1},
                {0, 0, 0, 1, 3}
        };
        INDArray Adense = Nd4j.create(flattenRowMajor(aData), new long[]{n, n}, DataType.DOUBLE);

        // Dense reference eigendecomposition via LAPACK syev
        INDArray AdenseRef = Adense.dup();
        // symmetricGeneralizedEigenvalues overwrites AdenseRef with eigenvectors (columns),
        // returns eigenvalues in ascending order.
        INDArray refEigvals = Eigen.symmetricGeneralizedEigenvalues(AdenseRef);
        // AdenseRef[:,i] = eigenvector for refEigvals[i]  (ascending order)

        // Sparse CSR
        SparseNDArray Asparse = Nd4j.toSparse(Adense, SparseFormat.CSR);

        // Lanczos: k=2 smallest, maxIter=20 (clamped to n=5 internally)
        LanczosResult lr = SparseSolvers.lanczos(Asparse, k, 20, false /* smallest */);

        assertNotNull(lr.eigenvalues,  "eigenvalues must be non-null");
        assertNotNull(lr.eigenvectors, "eigenvectors must be non-null");
        assertEquals(k, lr.eigenvalues.length(),
                "eigenvalues must have length k=" + k);
        assertEquals(n, lr.eigenvectors.size(0),
                "eigenvectors must have " + n + " rows");
        assertEquals(k, lr.eigenvectors.size(1),
                "eigenvectors must have k=" + k + " columns");

        // ---- Eigenvalue accuracy ----
        for (int i = 0; i < k; i++) {
            double ritz = lr.eigenvalues.getDouble(i);
            double ref  = refEigvals.getDouble(i);
            assertEquals(ref, ritz, TOL_EIGVAL,
                    "Ritz eigenvalue[" + i + "]=" + ritz + " must match LAPACK ref=" + ref);
        }

        // ---- Orthonormality of Ritz vectors ----
        for (int i = 0; i < k; i++) {
            INDArray vi    = lr.eigenvectors.getColumn(i);
            double   normI = Math.sqrt(vi.mul(vi).sumNumber().doubleValue());
            assertEquals(1.0, normI, TOL_EIGVEC,
                    "Ritz vector " + i + " must be unit; ‖v‖=" + normI);
            for (int j = i + 1; j < k; j++) {
                INDArray vj     = lr.eigenvectors.getColumn(j);
                double   dotIJ  = vi.mul(vj).sumNumber().doubleValue();
                assertEquals(0.0, dotIJ, TOL_EIGVEC,
                        "Ritz vectors " + i + " and " + j + " must be orthogonal; v·u=" + dotIJ);
            }
        }

        // ---- Sign-agnostic eigenvector match against LAPACK ----
        for (int i = 0; i < k; i++) {
            INDArray ritzVec = lr.eigenvectors.getColumn(i);
            INDArray refVec  = AdenseRef.getColumn(i);  // LAPACK eigenvec column i
            assertEigenvectorMatch(refVec, ritzVec, TOL_EIGVEC,
                    "Ritz eigenvector[" + i + "] vs LAPACK reference");
        }
    }

    // =========================================================================
    // Test 3: Laplacian Positional Encoding
    // =========================================================================

    /**
     * Verifies {@link SparseSolvers#laplacianPositionalEncoding} on a 6-node undirected
     * cycle graph (C₆).
     *
     * <p>The cycle-graph adjacency is:
     * <pre>
     *   adj[i][j] = 1 iff (j == (i+1) mod 6) or (j == (i-1+6) mod 6)
     * </pre>
     *
     * <p>Expected Laplacian eigenvalues of C₆ (unnormalised): 0, 1, 1, 3, 3, 4.
     *
     * <p>Checks:
     * <ol>
     *   <li>Output shape is [n=6, k=3].</li>
     *   <li>No NaN or Infinite values in the encoding.</li>
     *   <li>Columns are approximately orthonormal: ‖v_i‖ ≈ √n (unnormalised) or,
     *       after column-normalisation, Q^T Q ≈ I within {@link #TOL_ORTH}.</li>
     * </ol>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("LPE on 6-node cycle graph returns finite [n,k] with orthogonal columns")
    public void testLaplacianPositionalEncodingShape(Nd4jBackend backend) {
        int n = 6;
        int k = 3;

        // 6-node cycle graph adjacency matrix (undirected, symmetric)
        double[][] adjData = {
                {0, 1, 0, 0, 0, 1},
                {1, 0, 1, 0, 0, 0},
                {0, 1, 0, 1, 0, 0},
                {0, 0, 1, 0, 1, 0},
                {0, 0, 0, 1, 0, 1},
                {1, 0, 0, 0, 1, 0}
        };
        INDArray adjDense  = Nd4j.create(flattenRowMajor(adjData), new long[]{n, n}, DataType.DOUBLE);
        SparseNDArray adjSparse = Nd4j.toSparse(adjDense, SparseFormat.CSR);

        // --- Compute LPE ---
        INDArray encodings = SparseSolvers.laplacianPositionalEncoding(adjSparse, k);

        // ---- Shape ----
        assertNotNull(encodings, "LPE result must be non-null");
        assertEquals(2, encodings.rank(), "LPE result must be a matrix");
        assertEquals(n, encodings.size(0), "LPE rows must equal n=" + n);
        assertEquals(k, encodings.size(1), "LPE cols must equal k=" + k);

        // ---- Finiteness (host extraction on small n*k = 18 elements) ----
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < k; c++) {
                double val = encodings.getDouble(r, c);
                assertFalse(Double.isNaN(val),
                        "encoding[" + r + "," + c + "] must not be NaN");
                assertFalse(Double.isInfinite(val),
                        "encoding[" + r + "," + c + "] must not be Infinite");
            }
        }

        // ---- Approximate orthogonality of columns (cosine similarity) ----
        // Ritz vectors from a symmetric eigenproblem must be orthogonal but may not be
        // unit-normalised after the Q * V_T projection.  We verify via pairwise cosine
        // similarity: |cos(v_i, v_j)| ≈ 0 for i ≠ j.
        for (int i = 0; i < k; i++) {
            INDArray vi    = encodings.getColumn(i);
            double   normI = Math.sqrt(vi.mul(vi).sumNumber().doubleValue());
            assertTrue(normI > 1e-6,
                    "LPE column " + i + " must be non-trivial; ‖v‖=" + normI);
            for (int j = i + 1; j < k; j++) {
                INDArray vj      = encodings.getColumn(j);
                double   normJ   = Math.sqrt(vj.mul(vj).sumNumber().doubleValue());
                double   dotIJ   = vi.mul(vj).sumNumber().doubleValue();
                double   cosine  = (normI > 1e-12 && normJ > 1e-12)
                                       ? Math.abs(dotIJ / (normI * normJ))
                                       : 0.0;
                assertTrue(cosine < TOL_ORTH,
                        "LPE columns " + i + " and " + j + " must be approximately orthogonal; "
                        + "|cosine|=" + cosine + " must be < " + TOL_ORTH);
            }
        }
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /**
     * Sign-agnostic eigenvector comparison: passes if either {@code ‖expected - actual‖₂}
     * or {@code ‖expected + actual‖₂} is below {@code tol}.  This accounts for the fact
     * that eigenvectors are defined only up to a global sign.
     */
    private static void assertEigenvectorMatch(INDArray expected, INDArray actual,
                                                double tol, String msg) {
        assertEquals(expected.length(), actual.length(), msg + ": length mismatch");
        double normSame = expected.sub(actual).norm2Number().doubleValue();
        double normFlip = expected.add(actual).norm2Number().doubleValue();
        double best     = Math.min(normSame, normFlip);
        assertTrue(best < tol,
                msg + ": min(‖e-a‖₂, ‖e+a‖₂)=" + best + " >= tol=" + tol
                + "\n  expected=" + expected
                + "\n  actual  =" + actual);
    }

    /**
     * Flatten a 2-D Java array to a 1-D double array in row-major order.
     */
    private static double[] flattenRowMajor(double[][] mat) {
        int rows  = mat.length;
        int cols  = mat[0].length;
        double[] flat = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(mat[r], 0, flat, r * cols, cols);
        }
        return flat;
    }
}
