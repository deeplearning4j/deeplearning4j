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

package org.eclipse.deeplearning4j.llm.generation;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Lloyd-Max optimal scalar quantizer for the Gaussian distribution N(0, 1/d)
 * arising from random orthogonal rotation of unit-norm vectors.
 *
 * <p>After rotating a d-dimensional unit vector by a random orthogonal matrix,
 * each coordinate is approximately distributed as N(0, 1/d) (accurate for d >= 64).
 * The Lloyd-Max algorithm solves continuous 1-D k-means to find optimal centroids
 * that minimize mean squared quantization error.</p>
 *
 * <p>This is Stage 1 of TurboQuant (ICLR 2026): the MSE-optimal per-coordinate
 * scalar quantizer used before QJL residual correction.</p>
 *
 * <p>Codebooks are cached by (dimension, bits) key for reuse across layers.</p>
 *
 * @see TurboQuantKvCacheManager
 */
public class TurboQuantCodebook {

    private static final int MAX_ITERATIONS = 200;
    private static final double CONVERGENCE_TOL = 1e-10;

    /** Cache of pre-computed codebooks keyed by (d << 32) | bits. */
    private static final ConcurrentHashMap<Long, Codebook> CACHE = new ConcurrentHashMap<>();

    /**
     * A codebook containing optimal centroids and decision boundaries.
     */
    public static class Codebook {
        /** Optimal centroid values, sorted ascending. Length = 2^bits. */
        public final float[] centroids;
        /** Decision boundaries between centroids. Length = 2^bits - 1. */
        public final float[] boundaries;
        /** Expected distortion per coordinate. */
        public final double distortion;

        Codebook(float[] centroids, float[] boundaries, double distortion) {
            this.centroids = centroids;
            this.boundaries = boundaries;
            this.distortion = distortion;
        }
    }

    /**
     * Get or compute the Lloyd-Max codebook for given dimension and bit-width.
     *
     * @param d    vector dimension (determines sigma = 1/sqrt(d))
     * @param bits number of quantization bits (codebook has 2^bits levels)
     * @return the optimal codebook (cached)
     */
    public static Codebook getCodebook(int d, int bits) {
        long key = ((long) d << 32) | bits;
        return CACHE.computeIfAbsent(key, k -> solveLloydMax(d, bits));
    }

    /**
     * Quantize a single value to its nearest centroid index.
     *
     * @param value    the value to quantize
     * @param codebook the codebook to use
     * @return the index of the nearest centroid (0 to 2^bits - 1)
     */
    public static int quantize(float value, Codebook codebook) {
        // Binary search through boundaries
        float[] boundaries = codebook.boundaries;
        int lo = 0;
        int hi = boundaries.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (value > boundaries[mid]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /**
     * Dequantize an index back to its centroid value.
     *
     * @param index    centroid index
     * @param codebook the codebook
     * @return the centroid value
     */
    public static float dequantize(int index, Codebook codebook) {
        return codebook.centroids[index];
    }

    /**
     * Solve the Lloyd-Max optimal quantizer for N(0, 1/d).
     */
    static Codebook solveLloydMax(int d, int bits) {
        int nLevels = 1 << bits;
        double sigma = 1.0 / Math.sqrt(d);

        // Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
        double lo = -3.5 * sigma;
        double hi = 3.5 * sigma;
        double[] centroids = new double[nLevels];
        for (int i = 0; i < nLevels; i++) {
            centroids[i] = lo + (hi - lo) * (i + 0.5) / nLevels;
        }

        double[] boundaries = new double[nLevels - 1];
        double sigma2 = 1.0 / d;

        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            // Step 1: Compute boundaries (midpoints between adjacent centroids)
            for (int i = 0; i < nLevels - 1; i++) {
                boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
            }

            // Step 2: Update centroids as conditional expectations E[X | X in partition_i]
            double maxShift = 0.0;
            for (int i = 0; i < nLevels; i++) {
                double a = (i == 0) ? lo * 3 : boundaries[i - 1];
                double b = (i == nLevels - 1) ? hi * 3 : boundaries[i];

                double numerator = adaptiveSimpson(x -> x * gaussianPdf(x, sigma2), a, b);
                double denominator = adaptiveSimpson(x -> gaussianPdf(x, sigma2), a, b);

                double newCentroid;
                if (denominator > 1e-15) {
                    newCentroid = numerator / denominator;
                } else {
                    newCentroid = centroids[i];
                }

                double shift = Math.abs(newCentroid - centroids[i]);
                if (shift > maxShift) maxShift = shift;
                centroids[i] = newCentroid;
            }

            if (maxShift < CONVERGENCE_TOL) {
                break;
            }
        }

        // Final boundaries
        for (int i = 0; i < nLevels - 1; i++) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        // Compute expected distortion
        double distortion = computeDistortion(centroids, boundaries, sigma2, lo, hi);

        // Convert to float arrays
        float[] fCentroids = new float[nLevels];
        float[] fBoundaries = new float[nLevels - 1];
        for (int i = 0; i < nLevels; i++) {
            fCentroids[i] = (float) centroids[i];
        }
        for (int i = 0; i < nLevels - 1; i++) {
            fBoundaries[i] = (float) boundaries[i];
        }

        return new Codebook(fCentroids, fBoundaries, distortion);
    }

    /**
     * Gaussian PDF: N(0, sigma2) = (1/sqrt(2*pi*sigma2)) * exp(-x^2 / (2*sigma2))
     */
    private static double gaussianPdf(double x, double sigma2) {
        return (1.0 / Math.sqrt(2 * Math.PI * sigma2)) * Math.exp(-x * x / (2 * sigma2));
    }

    /**
     * Compute expected MSE distortion per coordinate.
     */
    private static double computeDistortion(double[] centroids, double[] boundaries,
                                             double sigma2, double lo, double hi) {
        int nLevels = centroids.length;
        double total = 0.0;
        for (int i = 0; i < nLevels; i++) {
            double a = (i == 0) ? lo * 3 : boundaries[i - 1];
            double b = (i == nLevels - 1) ? hi * 3 : boundaries[i];
            final double c = centroids[i];
            total += adaptiveSimpson(x -> (x - c) * (x - c) * gaussianPdf(x, sigma2), a, b);
        }
        return total;
    }

    // ======================== Adaptive Simpson's Rule ========================

    @FunctionalInterface
    private interface DoubleFunction {
        double apply(double x);
    }

    private static final int MAX_RECURSION = 50;
    private static final double SIMPSON_TOL = 1e-12;

    /**
     * Adaptive Simpson's rule for numerical integration.
     */
    private static double adaptiveSimpson(DoubleFunction f, double a, double b) {
        double c = (a + b) / 2.0;
        double fa = f.apply(a);
        double fb = f.apply(b);
        double fc = f.apply(c);
        double whole = (b - a) / 6.0 * (fa + 4 * fc + fb);
        return adaptiveSimpsonRecursive(f, a, b, fa, fb, fc, whole, SIMPSON_TOL, MAX_RECURSION);
    }

    private static double adaptiveSimpsonRecursive(DoubleFunction f, double a, double b,
                                                    double fa, double fb, double fc,
                                                    double whole, double tol, int depth) {
        double c = (a + b) / 2.0;
        double d = (a + c) / 2.0;
        double e = (c + b) / 2.0;
        double fd = f.apply(d);
        double fe = f.apply(e);
        double left = (c - a) / 6.0 * (fa + 4 * fd + fc);
        double right = (b - c) / 6.0 * (fc + 4 * fe + fb);
        double combined = left + right;

        if (depth <= 0 || Math.abs(combined - whole) <= 15 * tol) {
            return combined + (combined - whole) / 15.0;
        }

        return adaptiveSimpsonRecursive(f, a, c, fa, fc, fd, left, tol / 2, depth - 1)
                + adaptiveSimpsonRecursive(f, c, b, fc, fb, fe, right, tol / 2, depth - 1);
    }
}
