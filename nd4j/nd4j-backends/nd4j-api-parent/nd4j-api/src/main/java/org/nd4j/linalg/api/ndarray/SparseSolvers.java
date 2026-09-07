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

package org.nd4j.linalg.api.ndarray;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Sparse spectral and linear-solver utilities built as pure Java compositions
 * over {@link SparseNDArray#mv(INDArray)} (GPU-accelerated SpMV) and standard
 * ND4J dense array operations.
 *
 * <h2>Design principles</h2>
 * <ul>
 *   <li>No C++ op additions.  Everything composes over the existing
 *       {@code csr_spmv} op (via {@link SparseNDArray#mv}) and built-in dense ops.</li>
 *   <li>All n-dimensional vector arithmetic executes on the active backend (CPU or GPU)
 *       through {@link INDArray} in-place and functional ops — no host-side element loops.</li>
 *   <li>Scalar quantities (dot products, norms, Lanczos coefficients) are obtained from
 *       INDArray reductions ({@code sumNumber()}).  One device-to-host scalar per dot product
 *       is unavoidable for convergence decisions but is the only host interaction.</li>
 *   <li>No manual {@code syncToHost} / {@code syncToDevice} / {@code tick*} calls.</li>
 *   <li>No silent fallback: on failure the method throws or returns a {@code CgResult}
 *       with {@code converged=false} and a clear residual.</li>
 * </ul>
 *
 * <h2>Methods</h2>
 * <ul>
 *   <li>{@link #conjugateGradient} — standard CG for symmetric positive-definite sparse systems</li>
 *   <li>{@link #lanczos} — Lanczos eigenvalue/eigenvector extraction with full reorthogonalization</li>
 *   <li>{@link #laplacianPositionalEncoding} — graph Laplacian positional encodings</li>
 * </ul>
 *
 * <h2>Dense eigensolver</h2>
 * The small tridiagonal dense eigenproblem inside Lanczos is solved via
 * {@link Eigen#symmetricGeneralizedEigenvalues(INDArray)}, which dispatches to
 * LAPACK {@code dsyev}/{@code ssyev} through the active backend.  The tridiagonal
 * matrix T is small (k × k, where k is the Lanczos step count) so this LAPACK
 * call is cheap relative to the n-dimensional SpMV iterations.
 *
 * <h2>New op registrations</h2>
 * None — zero new ops added; no changes to {@code ImportClassMapping} or
 * {@code DifferentialFunctionClassHolder} required.
 */
public final class SparseSolvers {

    private SparseSolvers() {}

    // =========================================================================
    // Public result types
    // =========================================================================

    /**
     * Result returned by {@link #conjugateGradient}.
     */
    public static final class CgResult {
        /** The solution vector x (best-effort if {@code converged == false}). */
        public final INDArray x;
        /** {@code true} iff ‖r‖₂ fell below {@code tol} within {@code maxIter} steps. */
        public final boolean converged;
        /** Number of CG iterations actually performed. */
        public final int iterations;
        /** Euclidean norm of the final residual ‖r‖₂. */
        public final double residualNorm;

        public CgResult(INDArray x, boolean converged, int iterations, double residualNorm) {
            this.x = x;
            this.converged = converged;
            this.iterations = iterations;
            this.residualNorm = residualNorm;
        }

        @Override
        public String toString() {
            return String.format("CgResult{converged=%b, iter=%d, residual=%.3e}", converged, iterations, residualNorm);
        }
    }

    /**
     * Result returned by {@link #lanczos}.
     */
    public static final class LanczosResult {
        /** Ritz eigenvalues in ascending order, shape [k]. */
        public final INDArray eigenvalues;
        /**
         * Ritz eigenvectors, shape [n, k].  Column {@code i} corresponds to
         * {@code eigenvalues[i]}.  Eigenvectors are defined up to a global sign flip.
         */
        public final INDArray eigenvectors;

        public LanczosResult(INDArray eigenvalues, INDArray eigenvectors) {
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
        }
    }

    // =========================================================================
    // 1.  Conjugate Gradient
    // =========================================================================

    /**
     * Solves the symmetric positive-definite sparse linear system {@code A x = b}
     * via the Conjugate Gradient method.
     *
     * <p><b>Algorithm (standard CG):</b>
     * <pre>
     *   x = 0;  r = b;  p = r;  rs = r·r
     *   for i = 1..maxIter:
     *     Ap    = A.mv(p)            // GPU SpMV
     *     alpha = rs / (p · Ap)
     *     x    += alpha * p
     *     r    -= alpha * Ap
     *     rsNew = r · r
     *     if sqrt(rsNew) &lt; tol: break
     *     p = r + (rsNew/rs) * p
     *     rs = rsNew
     * </pre>
     *
     * <p>All n-dimensional vector operations execute on-device through INDArray ops.
     * {@link SparseNDArray#mv(INDArray)} provides the GPU-accelerated SpMV.
     * Scalar values ({@code alpha}, {@code beta}, the convergence norm) are obtained
     * from single-element reductions — one {@code sumNumber()} per dot product.
     *
     * @param A       symmetric positive-definite CSR sparse matrix [n, n]
     * @param b       right-hand side dense vector [n], same or higher floating dtype as A
     * @param tol     convergence threshold on ‖r‖₂ (must be &gt; 0)
     * @param maxIter maximum number of CG iterations (must be &gt; 0)
     * @return {@link CgResult} with solution, convergence flag, iteration count, and residual norm.
     *         If not converged, {@code x} holds the best-effort iterate.
     * @throws IllegalArgumentException if A is not square, b is the wrong length, or tol/maxIter invalid
     */
    public static CgResult conjugateGradient(SparseNDArray A, INDArray b,
                                              double tol, int maxIter) {
        Preconditions.checkArgument(A.rows() == A.cols(),
                "A must be square: shape=%s", java.util.Arrays.toString(A.getShape()));
        Preconditions.checkArgument(b.isVector() && b.length() == A.rows(),
                "b must be a vector of length rows(A)=%d; got length %d", A.rows(), b.length());
        Preconditions.checkArgument(tol > 0,
                "tol must be positive; got %f", tol);
        Preconditions.checkArgument(maxIter > 0,
                "maxIter must be positive; got %d", maxIter);

        // x0 = 0, r0 = b - A*x0 = b
        INDArray x  = Nd4j.zeros(b.dataType(), b.length());
        INDArray r  = b.dup();
        INDArray p  = r.dup();
        double   rs = dot(r, r);          // ‖r‖² — scalar reduction (host)
        int      iter = 0;

        for (int i = 0; i < maxIter; i++) {
            iter = i + 1;

            INDArray Ap  = A.mv(p);          // GPU SpMV — the heavy matvec
            double   pAp = dot(p, Ap);       // p · Ap — scalar reduction (host)

            if (Math.abs(pAp) < Double.MIN_NORMAL) {
                break;  // near-zero denominator: A is near-singular or p is in null-space
            }
            double alpha = rs / pAp;

            // x += alpha * p  (all on-device)
            x.addi(p.mul(alpha));
            // r -= alpha * Ap
            r.subi(Ap.mul(alpha));

            double rsNew   = dot(r, r);      // ‖r_new‖² — scalar reduction
            double resNorm = Math.sqrt(Math.max(rsNew, 0.0));
            if (resNorm < tol) {
                rs = rsNew;
                break;
            }
            double beta = rsNew / rs;
            // p = r + beta * p  (on-device)
            p = r.add(p.mul(beta));
            rs = rsNew;
        }

        double finalNorm = Math.sqrt(Math.max(rs, 0.0));
        return new CgResult(x, finalNorm < tol, iter, finalNorm);
    }

    // =========================================================================
    // 2.  Lanczos with full reorthogonalization
    // =========================================================================

    /**
     * Computes the {@code k} extreme Ritz pairs of a real symmetric sparse matrix
     * via the Lanczos algorithm with full reorthogonalization (twice-per-step for
     * numerical stability).
     *
     * <h3>Algorithm</h3>
     * <ol>
     *   <li>Start from a random unit vector q₀.</li>
     *   <li>Build the n×m Krylov basis Q and m×m tridiagonal T via:
     *     <pre>
     *       αⱼ = qⱼ · A·qⱼ
     *       v  = A·qⱼ − αⱼ qⱼ − βⱼ₋₁ qⱼ₋₁    // GPU SpMV each step
     *       full reorthogonalization of v against all stored qᵢ (two passes)
     *       βⱼ = ‖v‖;   q_{j+1} = v / βⱼ
     *     </pre>
     *   </li>
     *   <li>Solve the small (m×m) symmetric tridiagonal eigenproblem for T via LAPACK
     *       {@code dsyev} (through {@link Eigen#symmetricGeneralizedEigenvalues}).</li>
     *   <li>Form Ritz vectors: {@code V_ritz = Q · W} where W are the selected
     *       eigenvectors of T.</li>
     * </ol>
     *
     * <p>All n-dimensional vector operations run on-device through INDArray ops.
     * The only host scalars are the per-step dot products / norms (reductions)
     * used to compute αⱼ, βⱼ, and convergence decisions — unavoidable in Lanczos.
     *
     * <p><b>Edge cases:</b>
     * <ul>
     *   <li>If an invariant subspace is detected (βⱼ &lt; 1e-12) iteration stops early;
     *       all converged Ritz pairs are still returned.</li>
     *   <li>If k ≥ m (Lanczos produced fewer steps than k), all available Ritz pairs
     *       are returned (kActual ≤ k).</li>
     * </ul>
     *
     * <p>Lanczos vectors are promoted to DOUBLE for numerical stability even if A's
     * values are FLOAT.  The SpMV result is cast back to DOUBLE after each step.
     *
     * @param A       symmetric CSR sparse matrix [n, n]
     * @param k       number of Ritz pairs to return; must satisfy 0 &lt; k &lt; n
     * @param maxIter maximum Lanczos steps (clamped to n)
     * @param largest if {@code true}, return the k largest Ritz pairs; if {@code false}, the k smallest
     * @return {@link LanczosResult} with eigenvalues [k] (ascending) and eigenvectors [n, k]
     * @throws IllegalArgumentException if A is not square or k is out of range
     */
    public static LanczosResult lanczos(SparseNDArray A, int k,
                                         int maxIter, boolean largest) {
        long n = A.rows();
        Preconditions.checkArgument(A.rows() == A.cols(),
                "A must be square: shape=%s", java.util.Arrays.toString(A.getShape()));
        Preconditions.checkArgument(k > 0 && k < n,
                "k must satisfy 0 < k < n=%d; got k=%d", n, k);
        Preconditions.checkArgument(maxIter > 0,
                "maxIter must be positive; got %d", maxIter);

        // Lanczos vectors are always promoted to DOUBLE for numerical stability
        DataType wdtype  = DataType.DOUBLE;
        DataType adtype  = A.getValues().dataType();   // SpMV operand dtype
        int      m       = (int) Math.min(maxIter, n); // max Lanczos steps

        // Q: n × m matrix that accumulates Lanczos vectors as columns
        INDArray Q = Nd4j.zeros(wdtype, n, m);

        double[] alphaArr = new double[m];
        double[] betaArr  = new double[m]; // betaArr[j] = β_{j+1}; betaArr[m-1] unused

        // Start: random unit vector
        INDArray q = Nd4j.rand(wdtype, n).subi(0.5);
        q.divi(Math.sqrt(dot(q, q)));       // normalize on-device (one scalar reduction)
        Q.putColumn(0, q);

        INDArray qPrev  = Nd4j.zeros(wdtype, n); // q_{-1} = 0
        int      actualM = 1;

        for (int j = 0; j < m; j++) {
            INDArray qj = Q.getColumn(j).dup(); // working copy of current Lanczos vector

            // v = A * qj  via GPU SpMV (cast to A's dtype, cast result back to DOUBLE)
            INDArray qjA = qj.dataType() != adtype ? qj.castTo(adtype) : qj;
            INDArray Aqj = A.mv(qjA);
            INDArray v   = Aqj.dataType() != wdtype ? Aqj.castTo(wdtype) : Aqj;

            // αⱼ = qⱼ · v  (scalar reduction)
            double alphaj = dot(qj, v);
            alphaArr[j] = alphaj;

            // Three-term recurrence: v -= αⱼ qⱼ + β_{j-1} q_{j-1}  (on-device)
            v.subi(qj.mul(alphaj));
            if (j > 0) {
                v.subi(qPrev.mul(betaArr[j - 1]));
            }

            // Full reorthogonalization (two passes — classical Gram-Schmidt, twice)
            for (int pass = 0; pass < 2; pass++) {
                for (int i = 0; i <= j; i++) {
                    INDArray qi   = Q.getColumn(i);   // on-device column view
                    double   proj = dot(qi, v);        // scalar reduction
                    v.subi(qi.mul(proj));              // v -= proj * qi  (on-device)
                }
            }

            actualM = j + 1;
            if (j == m - 1) break; // done with all steps

            // βⱼ = ‖v‖  (scalar reduction)
            double betaj = Math.sqrt(Math.max(dot(v, v), 0.0));
            if (betaj < 1e-12) {
                break;  // lucky breakdown: exact invariant subspace
            }
            betaArr[j] = betaj;

            // q_{j+1} = v / βⱼ  (on-device)
            INDArray qNext = v.div(betaj);
            Q.putColumn(j + 1, qNext);
            qPrev = qj;
        }

        // Build the actualM × actualM symmetric tridiagonal T (small, host-constructed)
        int m2 = actualM;
        INDArray T = Nd4j.zeros(wdtype, m2, m2);
        for (int j = 0; j < m2; j++) {
            T.putScalar(new long[]{j, j}, alphaArr[j]);
            if (j < m2 - 1) {
                T.putScalar(new long[]{j, j + 1}, betaArr[j]);
                T.putScalar(new long[]{j + 1, j}, betaArr[j]);
            }
        }

        // Dense symmetric eigensolver on the small T (LAPACK dsyev via Nd4j)
        // Eigen.symmetricGeneralizedEigenvalues(T) overwrites T with eigenvectors (as columns)
        // and returns eigenvalues in ASCENDING order.
        INDArray Tcopy     = T.dup();
        INDArray allEigvals = Eigen.symmetricGeneralizedEigenvalues(Tcopy);
        // After the call: Tcopy.getColumn(i) = eigenvector for allEigvals[i] (ascending)

        // Select k (or fewer if m2 < k) extreme Ritz pairs
        int kActual = Math.min(k, m2);
        INDArray selEigvals;
        INDArray selEigvecs; // [m2, kActual] columns from Tcopy

        if (largest) {
            // Largest k eigenvalues are at the END of the ascending-sorted array
            selEigvals = allEigvals.get(NDArrayIndex.interval(m2 - kActual, m2)).dup();
            selEigvecs = Tcopy.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(m2 - kActual, m2)).dup();
        } else {
            // Smallest k eigenvalues are at the START
            selEigvals = allEigvals.get(NDArrayIndex.interval(0, kActual)).dup();
            selEigvecs = Tcopy.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(0, kActual)).dup();
        }

        // Ritz vectors = Q[n, m2] * selEigvecs[m2, kActual]  →  [n, kActual]
        // Dense on-device matmul; Q is already on the backend.
        INDArray Qm       = Q.get(NDArrayIndex.all(), NDArrayIndex.interval(0, m2));
        INDArray ritzVecs = Qm.mmul(selEigvecs);   // on-device dense matmul

        return new LanczosResult(selEigvals, ritzVecs);
    }

    // =========================================================================
    // 3.  Laplacian Positional Encoding
    // =========================================================================

    /**
     * Computes Laplacian Positional Encodings (LPE) for graph-transformer models.
     *
     * <p>Given an undirected adjacency matrix in CSR format, computes the
     * (unnormalised) graph Laplacian {@code L = D - A} via
     * {@link SparseNDArray#laplacian()} and returns the {@code k} smallest
     * non-trivial eigenvectors as [n, k] node-level positional features.
     *
     * <p>The trivial first Ritz pair (eigenvalue ≈ 0, eigenvector ∝ ones) is
     * always dropped; the returned encodings correspond to the 2nd through (k+1)-th
     * smallest eigenvectors of L.  These are the standard "Laplacian PE" features
     * used in graph-transformer architectures (e.g., GPS, SAN, Exphormer).
     *
     * <h3>Steps</h3>
     * <ol>
     *   <li>{@code L = adjacency.laplacian()} — sparse Laplacian in CSR format.</li>
     *   <li>{@link #lanczos}(L, k+1, maxIter, false) — k+1 smallest Ritz pairs.</li>
     *   <li>Drop Ritz pair 0 (trivial zero mode), return columns 1..k of eigenvectors.</li>
     * </ol>
     *
     * <p>All heavy matvecs go through {@link SparseNDArray#mv} (GPU SpMV).
     * No host element loops; no manual sync.
     *
     * @param adjacency undirected, non-negative square CSR adjacency matrix [n, n]
     * @param k         number of non-trivial eigenvectors to return; must satisfy 0 &lt; k &lt; n-1
     * @return INDArray of shape [n, k] with Laplacian positional encodings
     * @throws IllegalArgumentException if adjacency is not square or k is out of range
     * @throws IllegalStateException    if Lanczos fails to produce enough Ritz pairs
     */
    public static INDArray laplacianPositionalEncoding(SparseNDArray adjacency, int k) {
        long n = adjacency.rows();
        Preconditions.checkArgument(adjacency.rows() == adjacency.cols(),
                "adjacency must be square; got shape=%s", java.util.Arrays.toString(adjacency.getShape()));
        Preconditions.checkArgument(k > 0 && k < n - 1,
                "k must satisfy 0 < k < n-1=%d; got k=%d", n - 1, k);

        // Build sparse graph Laplacian L = D - A (CSR, same sparsity as A + diagonal)
        SparseNDArray L = adjacency.laplacian();

        // Request k+1 Ritz pairs so we can drop the trivial zero-eigenvalue mode
        int lk      = k + 1;
        // Use at least 2*lk steps for good convergence, capped at n
        int maxIter = (int) Math.min(Math.max(2L * lk, 20L), n);

        LanczosResult result = lanczos(L, lk, maxIter, false /* smallest */);

        long numReturned = result.eigenvectors.size(1);
        if (numReturned < 2) {
            throw new IllegalStateException(
                    "Lanczos returned only " + numReturned + " Ritz pair(s) — cannot drop the "
                  + "trivial zero mode and still provide k=" + k + " encodings. "
                  + "Increase the graph size or reduce k.");
        }

        // Drop Ritz pair 0 (trivial constant eigenvector, eigenvalue ≈ 0)
        // Return eigenvectors 1..min(k, numReturned-1) as positional encodings
        int kActual = (int) Math.min(k, numReturned - 1);
        return result.eigenvectors.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(1, 1 + kActual)).dup();
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /**
     * Inner product of two INDArrays via element-wise multiply + sum.
     * One device-to-host scalar reduction per call — the only host interaction
     * per Lanczos/CG step.  No element-level host loops.
     */
    private static double dot(INDArray a, INDArray b) {
        return a.mul(b).sumNumber().doubleValue();
    }
}
