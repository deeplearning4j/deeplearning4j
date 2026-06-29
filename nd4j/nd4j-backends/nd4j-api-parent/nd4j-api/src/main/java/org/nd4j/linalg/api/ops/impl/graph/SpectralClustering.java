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
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
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

import java.util.Arrays;
import java.util.Random;

/**
 * Spectral Clustering via the normalized graph Laplacian.
 *
 * <p>Algorithm:
 * <ol>
 *   <li>Build the symmetric normalized Laplacian:
 *       L_sym = I - D^{-1/2} A D^{-1/2}
 *   </li>
 *   <li>Compute all eigenvalues/eigenvectors of L_sym via
 *       {@link Eigen#symmetricGeneralizedEigenvalues(INDArray)} (BLAS syev).
 *       Eigenvalues are returned in ascending order; eigenvectors are stored
 *       as columns of the (overwritten) Laplacian copy.</li>
 *   <li>Take the k eigenvectors corresponding to the k <em>smallest nontrivial</em>
 *       eigenvalues (indices 1..k, skipping the trivial lambda=0).</li>
 *   <li>Row-normalize each node's k-dimensional embedding to unit length.</li>
 *   <li>Assign clusters via Lloyd's k-means on the embedding.</li>
 * </ol>
 *
 * <p>Input: symmetric, non-negative adjacency matrix with zero diagonal.
 * Suitable for undirected, connected graphs. Disconnected graphs are
 * handled (isolated nodes get their own cluster), but results may vary.
 */
public class SpectralClustering {

    private SpectralClustering() {}

    /**
     * Cluster an undirected graph into k groups using spectral clustering.
     *
     * @param adjacency  square symmetric adjacency matrix (DOUBLE or will be cast), shape [n, n]
     * @param k          number of clusters (>= 1)
     * @param maxKmeansIter  maximum k-means iterations (typically 100-300 is sufficient)
     * @param seed       random seed for k-means initialisation
     * @return int array of length n where result[i] is the cluster index of node i (0-based)
     */
    public static int[] cluster(INDArray adjacency, int k, int maxKmeansIter, long seed) {
        Preconditions.checkArgument(adjacency.rank() == 2 && adjacency.rows() == adjacency.columns(),
                "adjacency must be a square matrix; got shape %s", Arrays.toString(adjacency.shape()));
        int n = (int) adjacency.rows();
        Preconditions.checkArgument(k >= 1 && k <= n, "k must be in [1, n]; got k=%d, n=%d", k, n);

        if (k == 1) {
            return new int[n]; // all in one cluster
        }

        // Cast to DOUBLE for numerical stability
        INDArray A = adjacency.castTo(DataType.DOUBLE);

        // --- Step 1: Normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2} ---
        double[] deg = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                deg[i] += A.getDouble(i, j);
            }
        }
        double[] dInvSqrt = new double[n];
        for (int i = 0; i < n; i++) {
            dInvSqrt[i] = deg[i] > 0 ? 1.0 / Math.sqrt(deg[i]) : 0.0;
        }

        // Build L_sym as a Java double[][] first, then create INDArray
        double[] lsymFlat = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double v = -A.getDouble(i, j) * dInvSqrt[i] * dInvSqrt[j];
                if (i == j) v += 1.0;
                lsymFlat[i * n + j] = v;
            }
        }
        INDArray Lsym = Nd4j.create(lsymFlat, new long[]{n, n}, DataType.DOUBLE);

        // --- Step 2: Eigendecomposition ---
        // symmetricGeneralizedEigenvalues overwrites Lsym with eigenvectors (as columns)
        // and returns eigenvalues in ascending order.
        Eigen.symmetricGeneralizedEigenvalues(Lsym); // Lsym now holds eigenvectors as columns

        // --- Step 3: Extract k nontrivial eigenvectors (skip index 0, take 1..k) ---
        // For k clusters we need at least k+1 eigenvectors; n >= k+1 in practice.
        int startIdx = (n > k) ? 1 : 0; // graceful fallback if n == k
        double[][] embedding = new double[n][k];
        for (int c = 0; c < k; c++) {
            int colIdx = startIdx + c;
            if (colIdx >= n) colIdx = n - 1; // clamp (shouldn't happen)
            for (int i = 0; i < n; i++) {
                embedding[i][c] = Lsym.getDouble(i, colIdx);
            }
        }

        // --- Step 4: Row-normalize to unit sphere ---
        for (int i = 0; i < n; i++) {
            double norm = 0;
            for (int c = 0; c < k; c++) {
                norm += embedding[i][c] * embedding[i][c];
            }
            norm = Math.sqrt(norm);
            if (norm > 1e-12) {
                for (int c = 0; c < k; c++) {
                    embedding[i][c] /= norm;
                }
            }
        }

        // --- Step 5: Lloyd's k-means ---
        return lloydsKmeans(embedding, n, k, maxKmeansIter, seed);
    }

    /**
     * Deterministic Lloyd's k-means on a double[][] matrix of n rows, k-dim embeddings.
     * Initialised by picking the first centroid randomly, then choosing subsequent
     * centroids as the point farthest from all current centroids (max-min init,
     * a deterministic variant of k-means++ for reproducibility).
     */
    public static int[] lloydsKmeans(double[][] points, int n, int k, int maxIter, long seed) {
        // --- Initialisation: max-min (greedy farthest-point selection) ---
        Random rng = new Random(seed);
        int[] centroidIdx = new int[k];
        centroidIdx[0] = rng.nextInt(n);
        double[] minDist = new double[n];
        Arrays.fill(minDist, Double.MAX_VALUE);

        for (int ci = 1; ci < k; ci++) {
            // Update min distances from newly added centroid
            int prev = centroidIdx[ci - 1];
            for (int i = 0; i < n; i++) {
                double d = sqDist(points[i], points[prev], k);
                if (d < minDist[i]) minDist[i] = d;
            }
            // Pick the point with maximum min-distance
            int best = 0;
            for (int i = 1; i < n; i++) {
                if (minDist[i] > minDist[best]) best = i;
            }
            centroidIdx[ci] = best;
        }

        // Copy initial centroids
        double[][] centroids = new double[k][k];
        for (int ci = 0; ci < k; ci++) {
            centroids[ci] = Arrays.copyOf(points[centroidIdx[ci]], points[centroidIdx[ci]].length);
        }

        // --- Lloyd's iterations ---
        int[] assignments = new int[n];
        for (int iter = 0; iter < maxIter; iter++) {
            // Assignment step
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int best = 0;
                double bestDist = Double.MAX_VALUE;
                for (int ci = 0; ci < k; ci++) {
                    double d = sqDist(points[i], centroids[ci], k);
                    if (d < bestDist) {
                        bestDist = d;
                        best = ci;
                    }
                }
                if (assignments[i] != best) {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if (!changed) break;

            // Update step: recompute centroids
            double[][] newCentroids = new double[k][k];
            int[] counts = new int[k];
            for (int i = 0; i < n; i++) {
                int ci = assignments[i];
                counts[ci]++;
                for (int d = 0; d < k; d++) {
                    newCentroids[ci][d] += points[i][d];
                }
            }
            for (int ci = 0; ci < k; ci++) {
                if (counts[ci] > 0) {
                    for (int d = 0; d < k; d++) {
                        newCentroids[ci][d] /= counts[ci];
                    }
                } else {
                    // Empty cluster: reinitialise to a random point
                    newCentroids[ci] = Arrays.copyOf(points[rng.nextInt(n)], k);
                }
            }
            centroids = newCentroids;
        }

        return assignments;
    }

    private static double sqDist(double[] a, double[] b, int dims) {
        double sum = 0;
        for (int d = 0; d < dims; d++) {
            double diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    }
}
