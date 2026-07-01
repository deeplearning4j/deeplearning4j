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
 *  * specific language governing permissions and limitations under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Graphical Lasso (sparse inverse covariance estimation) via ADMM.
 *
 * <p>Solves the penalized log-likelihood problem:
 * <pre>
 *   maximize   log det Theta - trace(S * Theta) - lambda * ||Theta||_{1,off}
 * </pre>
 * where {@code S} is the empirical covariance matrix and {@code ||.||_{1,off}} is the L1 norm
 * over the off-diagonal entries only. The resulting precision matrix {@code Theta} encodes the
 * sparse conditional independence structure of the variables.
 *
 * <p>Algorithm (Alternating Direction Method of Multipliers, Boyd et al. 2011):
 * <ol>
 *   <li>Theta-update: closed-form proximal step via eigendecomposition of
 *       (rho*Z - rho*U - S), using {@link Eigen#symmetricGeneralizedEigenvalues(INDArray)}
 *       (BLAS {@code syev}). Each eigenvalue d_i maps to
 *       {@code (d_i + sqrt(d_i^2 + 4*rho)) / (2*rho)}.</li>
 *   <li>Z-update: element-wise soft-thresholding of {@code (Theta + U)} with
 *       threshold {@code lambda/rho} on the off-diagonal; diagonal is unpenalized.</li>
 *   <li>U-update: dual residual accumulation {@code U = U + Theta - Z}.</li>
 * </ol>
 *
 * <p>Convergence is declared when the Frobenius norms of the primal residual
 * {@code ||Theta - Z||_F} and the dual residual {@code rho * ||Z - Z_prev||_F} both fall
 * below {@code tol}.
 */
public class GraphicalLasso {

    private GraphicalLasso() {}

    /**
     * Fit a sparse precision matrix from a data matrix using the graphical lasso.
     *
     * <p>Computes the empirical covariance {@code S = (X - mean(X))^T (X - mean(X)) / n}
     * and delegates to {@link #fitFromCovariance(INDArray, double, double, int, double)}.
     *
     * @param data   observations matrix [n, p] (DOUBLE or will be cast)
     * @param lambda L1 regularization strength on off-diagonal entries (> 0)
     * @param rho    ADMM penalty parameter (default 1.0 usually works)
     * @param maxIter maximum ADMM iterations
     * @param tol    convergence tolerance for primal + dual residuals
     * @return sparse precision matrix Theta [p, p]
     */
    public static INDArray fit(INDArray data, double lambda, double rho, int maxIter,
            double tol) {
        Preconditions.checkArgument(data.rank() == 2,
                "data must be rank 2 [n, p]; got shape %s", data.shapeInfoToString());
        Preconditions.checkArgument(lambda > 0, "lambda must be > 0; got %s", lambda);
        Preconditions.checkArgument(rho > 0, "rho must be > 0; got %s", rho);

        INDArray X = data.castTo(DataType.DOUBLE);
        long n = X.rows();

        // Center the data
        INDArray mean = X.mean(0);
        X = X.subRowVector(mean);

        INDArray S = X.transpose().mmul(X).div(n);

        return fitFromCovariance(S, lambda, rho, maxIter, tol);
    }

    /**
     * Fit a sparse precision matrix from a pre-computed covariance matrix.
     *
     * @param S      empirical covariance matrix [p, p], symmetric positive semi-definite
     * @param lambda L1 regularization strength on off-diagonal entries (> 0)
     * @param rho    ADMM penalty parameter (1.0 is a good starting point)
     * @param maxIter maximum ADMM iterations (200-500 is usually sufficient)
     * @param tol    convergence tolerance (e.g. 1e-6)
     * @return sparse precision matrix Theta [p, p]
     */
    public static INDArray fitFromCovariance(INDArray S, double lambda, double rho, int maxIter,
            double tol) {
        Preconditions.checkArgument(S.rank() == 2 && S.rows() == S.columns(),
                "S must be a square matrix [p, p]; got shape %s", S.shapeInfoToString());
        Preconditions.checkArgument(lambda > 0, "lambda must be > 0; got %s", lambda);
        Preconditions.checkArgument(rho > 0, "rho must be > 0; got %s", rho);

        INDArray Sd = S.castTo(DataType.DOUBLE);
        int p = (int) Sd.rows();
        double shrink = lambda / rho;   // soft-threshold level for Z-update

        // Initialize: Theta = I, Z = I, U = 0
        INDArray Theta = Nd4j.eye(p).castTo(DataType.DOUBLE);
        INDArray Z     = Nd4j.eye(p).castTo(DataType.DOUBLE);
        INDArray U     = Nd4j.zeros(DataType.DOUBLE, p, p);

        for (int iter = 0; iter < maxIter; iter++) {
            INDArray ZPrev = Z.dup();

            // ------------------------------------------------------------------
            // Theta update: prox_{1/rho * log det} at (rho*Z - rho*U - S)
            //
            // Let A = rho*(Z - U) - S   [symmetric]
            // Eigendecomp A = V diag(d) V^T  (Eigen overwrites A with eigenvectors)
            // New eigenvalue: d_i -> (-d_i + sqrt(d_i^2 + 4*rho)) / (2*rho)
            // ------------------------------------------------------------------
            INDArray A = Z.sub(U).muli(rho).subi(Sd);
            // symmetrize to guard against floating-point asymmetry
            A = A.add(A.transpose()).divi(2.0);

            INDArray eigenvalues = Eigen.symmetricGeneralizedEigenvalues(A);
            // A is now V (eigenvectors as columns), eigenvalues are in ascending order

            double[] newEig = new double[p];
            for (int i = 0; i < p; i++) {
                double d = eigenvalues.getDouble(i);
                // Proximal operator of -log det: d_new = (d + sqrt(d^2 + 4*rho)) / (2*rho)
                newEig[i] = (d + Math.sqrt(d * d + 4.0 * rho)) / (2.0 * rho);
            }

            // Theta = V * diag(newEig) * V^T
            INDArray D = Nd4j.diag(Nd4j.create(newEig).castTo(DataType.DOUBLE));
            Theta = A.mmul(D).mmul(A.transpose());

            // ------------------------------------------------------------------
            // Z update: soft-threshold (Theta + U), no penalty on diagonal
            // ------------------------------------------------------------------
            INDArray ThetaPlusU = Theta.add(U);
            Z = softThresholdOffDiag(ThetaPlusU, shrink);

            // ------------------------------------------------------------------
            // U update: dual residual accumulation
            // ------------------------------------------------------------------
            U.addi(Theta.sub(Z));

            // ------------------------------------------------------------------
            // Convergence check
            // ------------------------------------------------------------------
            double primalRes = Theta.sub(Z).norm2Number().doubleValue();
            double dualRes = rho * Z.sub(ZPrev).norm2Number().doubleValue();
            if (primalRes < tol && dualRes < tol) {
                break;
            }
        }

        return Theta;
    }

    /**
     * Apply soft-thresholding with {@code threshold} to all off-diagonal entries of {@code M};
     * the diagonal is left unchanged.
     */
    private static INDArray softThresholdOffDiag(INDArray M, double threshold) {
        int p = (int) M.rows();
        double[] flat = new double[p * p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double v = M.getDouble(i, j);
                if (i == j) {
                    flat[i * p + j] = v;
                } else {
                    double av = Math.abs(v);
                    flat[i * p + j] = (av <= threshold) ? 0.0
                            : (v > 0 ? av - threshold : -(av - threshold));
                }
            }
        }
        return Nd4j.create(flat, new long[]{p, p}, DataType.DOUBLE);
    }
}
