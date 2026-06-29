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

package org.eclipse.deeplearning4j.nd4j.linalg.graph;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.graph.BgrlTargetUpdate;
import org.nd4j.linalg.api.ops.impl.graph.GraphicalLasso;
import org.nd4j.linalg.api.ops.impl.graph.MutualInformationGraph;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * CPU correctness tests for BgrlTargetUpdate, GraphicalLasso, and MutualInformationGraph.
 */
@Slf4j
@NativeTag
public class GraphUtilsTest extends BaseNd4jTestWithBackends {

    private DataType initialType;

    @BeforeEach
    public void before() {
        initialType = Nd4j.dataType();
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @AfterEach
    public void after() {
        Nd4j.setDataType(initialType);
    }

    // -------------------------------------------------------------------------
    // BgrlTargetUpdate tests
    // -------------------------------------------------------------------------

    /**
     * EMA update: result[i] = momentum * target[i] + (1 - momentum) * online[i].
     * Verified element-wise against a manually computed expected array.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBgrlEmaUpdateCorrectness(Nd4jBackend backend) {
        double momentum = 0.99;
        INDArray target = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray online = Nd4j.create(new double[]{4.0, 5.0, 6.0});

        INDArray result = BgrlTargetUpdate.emaUpdate(target, online, momentum);

        // expected[i] = 0.99 * target[i] + 0.01 * online[i]
        double[] expected = {0.99 * 1.0 + 0.01 * 4.0,
                             0.99 * 2.0 + 0.01 * 5.0,
                             0.99 * 3.0 + 0.01 * 6.0};
        for (int i = 0; i < 3; i++) {
            assertEquals(expected[i], result.getDouble(i), 1e-10,
                    "EMA mismatch at index " + i);
        }
        // Original arrays must not be mutated
        assertEquals(1.0, target.getDouble(0), 1e-10, "target was mutated");
        assertEquals(4.0, online.getDouble(0), 1e-10, "online was mutated");
    }

    /**
     * In-place EMA update: target is modified in-place to
     * momentum * target + (1 - momentum) * online.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBgrlEmaUpdateInPlace(Nd4jBackend backend) {
        double momentum = 0.9;
        INDArray target = Nd4j.create(new double[]{10.0, 20.0});
        INDArray online = Nd4j.create(new double[]{0.0, 0.0});

        BgrlTargetUpdate.emaUpdateInPlace(target, online, momentum);

        assertEquals(10.0 * 0.9, target.getDouble(0), 1e-10);
        assertEquals(20.0 * 0.9, target.getDouble(1), 1e-10);
    }

    /**
     * EMA with extreme momentum = 1.0 - epsilon should keep target almost unchanged.
     * EMA with extreme momentum ~= 0 should copy online into target.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBgrlEmaEdgeMomentum(Nd4jBackend backend) {
        INDArray target = Nd4j.create(new double[]{1.0, 1.0, 1.0});
        INDArray online = Nd4j.create(new double[]{100.0, 100.0, 100.0});

        // Very high momentum: result ≈ target
        INDArray r1 = BgrlTargetUpdate.emaUpdate(target, online, 0.9999);
        assertTrue(Math.abs(r1.getDouble(0) - 1.0) < 0.1, "high-momentum result should be near target");

        // Very low momentum: result ≈ online
        INDArray r2 = BgrlTargetUpdate.emaUpdate(target, online, 0.0001);
        assertTrue(Math.abs(r2.getDouble(0) - 100.0) < 0.1, "low-momentum result should be near online");
    }

    // -------------------------------------------------------------------------
    // GraphicalLasso tests
    // -------------------------------------------------------------------------

    /**
     * Recover a known chain (AR-1) precision matrix:
     *
     * <pre>
     *   Theta_true = [[ 2, -1,  0],
     *                 [-1,  2, -1],
     *                 [ 0, -1,  2]]
     * </pre>
     *
     * True off-diagonal edges: (0,1) and (1,2); non-edge: (0,2).
     * We generate n=2000 Gaussian samples from the corresponding covariance,
     * then run graphical lasso and verify:
     * - True edges have |Theta[i,j]| > 0.2
     * - Non-edge (0,2) has |Theta[0,2]| < 0.15
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphicalLassoChainPrecision(Nd4jBackend backend) {
        // Build true precision matrix for a chain graph on 3 nodes
        // Theta = [[2,-1,0],[-1,2,-1],[0,-1,2]] (positive definite for a chain)
        // Covariance = Theta^{-1}
        double[][] thetaArr = {{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}};
        // Sigma = Theta^{-1}:  Sigma = [[3/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 3/4]]
        // (exact for the tridiagonal 2-1 precision matrix of order 3, scaled by 1/4)
        double sc = 0.25; // Sigma scale factor makes Theta eigenvalues in [0.6, 3.4]
        double[][] sigma = {
            {3.0 * sc, 2.0 * sc, 1.0 * sc},
            {2.0 * sc, 4.0 * sc, 2.0 * sc},
            {1.0 * sc, 2.0 * sc, 3.0 * sc}
        };

        // Generate n=2000 Gaussian samples from N(0, Sigma) via Cholesky
        int n = 2000;
        int p = 3;
        Random rng = new Random(42L);

        // Cholesky of sigma (lower triangular L such that L L^T = Sigma)
        double[][] L = cholesky(sigma, p);

        double[][] dataMat = new double[n][p];
        for (int i = 0; i < n; i++) {
            double[] z = new double[p];
            for (int j = 0; j < p; j++) z[j] = rng.nextGaussian();
            // x = L z
            for (int row = 0; row < p; row++) {
                for (int col = 0; col <= row; col++) {
                    dataMat[i][row] += L[row][col] * z[col];
                }
            }
        }

        // Flatten into INDArray [n, p]
        double[] flat = new double[n * p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) flat[i * p + j] = dataMat[i][j];
        }
        INDArray data = Nd4j.create(flat, new long[]{n, p}, DataType.DOUBLE);

        // Fit graphical lasso (lambda chosen to recover sparsity but not shrink too much)
        INDArray theta = GraphicalLasso.fit(data, 0.05, 1.0, 300, 1e-5);

        log.info("Recovered Theta:\n{}", theta);

        // True edges should have large magnitude
        double t01 = Math.abs(theta.getDouble(0, 1));
        double t12 = Math.abs(theta.getDouble(1, 2));
        // Non-edge should be near zero
        double t02 = Math.abs(theta.getDouble(0, 2));

        assertTrue(t01 > 0.1, String.format("Edge (0,1) should be nonzero; got |Theta[0,1]|=%.4f", t01));
        assertTrue(t12 > 0.1, String.format("Edge (1,2) should be nonzero; got |Theta[1,2]|=%.4f", t12));
        // The true non-edge should be substantially smaller than the true edges
        assertTrue(t02 < t01 * 0.8 && t02 < t12 * 0.8,
                String.format("Non-edge (0,2)=%.4f should be smaller than edges (0,1)=%.4f, (1,2)=%.4f",
                        t02, t01, t12));

        // Theta should be symmetric
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                assertEquals(theta.getDouble(i, j), theta.getDouble(j, i), 1e-8,
                        "Theta must be symmetric at (" + i + "," + j + ")");
            }
        }
    }

    // -------------------------------------------------------------------------
    // MutualInformationGraph tests
    // -------------------------------------------------------------------------

    /**
     * Two perfectly linearly dependent columns (col1 = 2*col0 + 3) should have
     * very high MI (equal to their marginal entropies, within discretization error),
     * while two independent columns should have MI close to 0.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMutualInformationDependentVsIndependent(Nd4jBackend backend) {
        int n = 1000;
        int numBins = 20;

        // Generate random data
        Random rng = new Random(7L);
        double[] x0 = new double[n];
        double[] x1 = new double[n]; // perfectly dependent: x1 = 2*x0 + 3
        double[] x2 = new double[n]; // independent uniform
        for (int i = 0; i < n; i++) {
            x0[i] = rng.nextDouble() * 10.0;
            x1[i] = 2.0 * x0[i] + 3.0;
            x2[i] = rng.nextDouble() * 10.0;
        }

        double[] flat = new double[n * 3];
        for (int i = 0; i < n; i++) {
            flat[i * 3]     = x0[i];
            flat[i * 3 + 1] = x1[i];
            flat[i * 3 + 2] = x2[i];
        }
        INDArray data = Nd4j.create(flat, new long[]{n, 3}, DataType.DOUBLE);

        INDArray mi = MutualInformationGraph.compute(data, numBins);

        log.info("MI matrix:\n{}", mi);

        double mi01 = mi.getDouble(0, 1); // dependent
        double mi02 = mi.getDouble(0, 2); // independent
        double mi12 = mi.getDouble(1, 2); // independent

        // Dependent pair: MI should be large (close to marginal entropy, > 2 nats)
        assertTrue(mi01 > 2.0,
                String.format("Dependent columns should have high MI; got MI(0,1)=%.4f", mi01));
        // Independent pairs: MI should be much smaller than the dependent pair.
        // With finite-sample histogram estimation there is always positive bias, but the
        // dependent MI should be at least 5x larger than any independent pair's MI.
        assertTrue(mi02 < mi01 / 5.0,
                String.format("Independent MI(0,2)=%.4f should be << dependent MI(0,1)=%.4f", mi02, mi01));
        assertTrue(mi12 < mi01 / 5.0,
                String.format("Independent MI(1,2)=%.4f should be << dependent MI(0,1)=%.4f", mi12, mi01));

        // MI must be symmetric
        assertEquals(mi.getDouble(0, 1), mi.getDouble(1, 0), 1e-12, "MI must be symmetric (0,1)");
        assertEquals(mi.getDouble(0, 2), mi.getDouble(2, 0), 1e-12, "MI must be symmetric (0,2)");

        // Diagonal entries (self-MI = entropy) should be >= all off-diagonal (by MI <= H property)
        double h0 = mi.getDouble(0, 0);
        double h1 = mi.getDouble(1, 1);
        assertTrue(h0 >= mi01 - 1e-9,
                String.format("Self-MI(0)=%.4f should be >= MI(0,1)=%.4f", h0, mi01));
        assertTrue(h1 >= mi01 - 1e-9,
                String.format("Self-MI(1)=%.4f should be >= MI(0,1)=%.4f", h1, mi01));
    }

    /**
     * Verify that MI of identical columns equals the marginal entropy (self-MI).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMutualInformationIdenticalColumns(Nd4jBackend backend) {
        int n = 500;
        int numBins = 15;

        Random rng = new Random(13L);
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = rng.nextDouble();

        double[] flat = new double[n * 2];
        for (int i = 0; i < n; i++) {
            flat[i * 2]     = x[i];
            flat[i * 2 + 1] = x[i]; // identical column
        }
        INDArray data = Nd4j.create(flat, new long[]{n, 2}, DataType.DOUBLE);

        INDArray mi = MutualInformationGraph.compute(data, numBins);

        double mi00 = mi.getDouble(0, 0); // self-MI = entropy of col 0
        double mi11 = mi.getDouble(1, 1); // self-MI = entropy of col 1
        double mi01 = mi.getDouble(0, 1); // MI between identical columns = entropy

        // MI between identical columns == entropy of either (within binning discretization error)
        assertEquals(mi00, mi01, 1e-10,
                "MI of identical cols should equal self-MI of col 0");
        assertEquals(mi11, mi01, 1e-10,
                "MI of identical cols should equal self-MI of col 1");
    }

    // -------------------------------------------------------------------------
    // Helper: Cholesky decomposition for test data generation
    // -------------------------------------------------------------------------

    /** Lower-triangular Cholesky: L s.t. L L^T = A (A must be symmetric pos-def). */
    private static double[][] cholesky(double[][] A, int p) {
        double[][] L = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = A[i][j];
                for (int k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
                if (i == j) {
                    L[i][j] = Math.sqrt(sum);
                } else {
                    L[i][j] = sum / L[j][j];
                }
            }
        }
        return L;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
