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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.kvcache.TurboQuantCodebook;
import org.eclipse.deeplearning4j.llm.generation.kvcache.TurboQuantCodebook.Codebook;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@DisplayName("TurboQuantCodebookTest")
public class TurboQuantCodebookTest {

    @ParameterizedTest
    @CsvSource({
            "64, 2",
            "64, 3",
            "128, 2",
            "128, 3",
            "128, 4"
    })
    @DisplayName("Lloyd-Max convergence: centroids sorted and boundaries between them")
    void testLloydMaxConvergence(int d, int bits) {
        Codebook cb = TurboQuantCodebook.getCodebook(d, bits);
        int nLevels = 1 << bits;

        assertEquals(nLevels, cb.centroids.length, "centroid count");
        assertEquals(nLevels - 1, cb.boundaries.length, "boundary count");

        // Centroids must be strictly sorted ascending
        for (int i = 1; i < cb.centroids.length; i++) {
            assertTrue(cb.centroids[i] > cb.centroids[i - 1],
                    String.format("centroids not sorted: [%d]=%f >= [%d]=%f",
                            i - 1, cb.centroids[i - 1], i, cb.centroids[i]));
        }

        // Boundaries must be strictly sorted ascending
        for (int i = 1; i < cb.boundaries.length; i++) {
            assertTrue(cb.boundaries[i] > cb.boundaries[i - 1],
                    "boundaries not sorted at index " + i);
        }

        // Each boundary must lie between its adjacent centroids
        for (int i = 0; i < cb.boundaries.length; i++) {
            assertTrue(cb.boundaries[i] > cb.centroids[i],
                    String.format("boundary[%d]=%f not > centroid[%d]=%f",
                            i, cb.boundaries[i], i, cb.centroids[i]));
            assertTrue(cb.boundaries[i] < cb.centroids[i + 1],
                    String.format("boundary[%d]=%f not < centroid[%d]=%f",
                            i, cb.boundaries[i], i + 1, cb.centroids[i + 1]));
        }

        // Distortion must be positive and finite
        assertTrue(cb.distortion > 0, "distortion must be positive");
        assertTrue(Double.isFinite(cb.distortion), "distortion must be finite");
    }

    @Test
    @DisplayName("Codebook symmetry: centroids for N(0, sigma^2) are symmetric around 0")
    void testCodebookSymmetry() {
        Codebook cb = TurboQuantCodebook.getCodebook(128, 3);
        int n = cb.centroids.length;

        // For a symmetric Gaussian, centroids should be symmetric: c[i] ≈ -c[n-1-i]
        for (int i = 0; i < n / 2; i++) {
            float expected = -cb.centroids[n - 1 - i];
            float actual = cb.centroids[i];
            assertEquals(expected, actual, 1e-5f,
                    String.format("centroid[%d] should be -%f but is %f", i, cb.centroids[n - 1 - i], actual));
        }

        // Boundaries should also be symmetric: b[i] ≈ -b[n-2-i]
        int nb = cb.boundaries.length;
        for (int i = 0; i < nb / 2; i++) {
            float expected = -cb.boundaries[nb - 1 - i];
            float actual = cb.boundaries[i];
            assertEquals(expected, actual, 1e-5f,
                    String.format("boundary[%d] should be -%f but is %f", i, cb.boundaries[nb - 1 - i], actual));
        }
    }

    @Test
    @DisplayName("Distortion decreases with more bits")
    void testDistortionDecreasesWithBits() {
        int d = 128;
        double dist2 = TurboQuantCodebook.getCodebook(d, 2).distortion;
        double dist3 = TurboQuantCodebook.getCodebook(d, 3).distortion;
        double dist4 = TurboQuantCodebook.getCodebook(d, 4).distortion;

        log.info("Distortion d={}: 2-bit={}, 3-bit={}, 4-bit={}", d, dist2, dist3, dist4);

        assertTrue(dist3 < dist2, "3-bit distortion should be less than 2-bit");
        assertTrue(dist4 < dist3, "4-bit distortion should be less than 3-bit");
    }

    @Test
    @DisplayName("Quantize + dequantize round-trip for Gaussian samples")
    void testQuantizeDequantizeRoundTrip() {
        int d = 128;
        int bits = 3;
        double sigma = 1.0 / Math.sqrt(d);
        Codebook cb = TurboQuantCodebook.getCodebook(d, bits);

        Random rng = new Random(42);
        int N = 10000;
        double mse = 0;

        for (int i = 0; i < N; i++) {
            float val = (float) (rng.nextGaussian() * sigma);
            int idx = TurboQuantCodebook.quantize(val, cb);
            float recon = TurboQuantCodebook.dequantize(idx, cb);
            double err = (val - recon);
            mse += err * err;
        }
        mse /= N;

        log.info("Empirical MSE: {}, codebook distortion: {}", mse, cb.distortion);

        // Empirical MSE should be close to the codebook's theoretical distortion
        // Allow 20% tolerance since empirical estimate has sampling variance
        assertEquals(cb.distortion, mse, cb.distortion * 0.2,
                "Empirical MSE should approximate theoretical distortion");
    }

    @Test
    @DisplayName("Quantize index is in valid range")
    void testQuantizeIndexRange() {
        int d = 128;
        int bits = 3;
        int nLevels = 1 << bits;
        Codebook cb = TurboQuantCodebook.getCodebook(d, bits);

        Random rng = new Random(123);
        double sigma = 1.0 / Math.sqrt(d);

        for (int i = 0; i < 1000; i++) {
            float val = (float) (rng.nextGaussian() * sigma);
            int idx = TurboQuantCodebook.quantize(val, cb);
            assertTrue(idx >= 0 && idx < nLevels,
                    String.format("Index %d out of range [0, %d) for value %f", idx, nLevels, val));
        }

        // Extreme values should map to first/last index
        assertEquals(0, TurboQuantCodebook.quantize(-100f, cb), "Very negative should map to index 0");
        assertEquals(nLevels - 1, TurboQuantCodebook.quantize(100f, cb), "Very positive should map to last index");
    }

    @Test
    @DisplayName("Codebook caching: same (d, bits) returns same object")
    void testCodebookCaching() {
        Codebook cb1 = TurboQuantCodebook.getCodebook(128, 3);
        Codebook cb2 = TurboQuantCodebook.getCodebook(128, 3);
        assertSame(cb1, cb2, "Same (d, bits) should return cached codebook");

        // Different parameters should return different codebooks
        Codebook cb3 = TurboQuantCodebook.getCodebook(64, 3);
        assertNotSame(cb1, cb3, "Different d should return different codebook");

        Codebook cb4 = TurboQuantCodebook.getCodebook(128, 4);
        assertNotSame(cb1, cb4, "Different bits should return different codebook");
    }

    @Test
    @DisplayName("Dequantize maps to exact centroid value")
    void testDequantizeExact() {
        Codebook cb = TurboQuantCodebook.getCodebook(128, 3);
        for (int i = 0; i < cb.centroids.length; i++) {
            assertEquals(cb.centroids[i], TurboQuantCodebook.dequantize(i, cb),
                    "Dequantize should return exact centroid");
        }
    }

    @Test
    @DisplayName("Quantize centroid value maps back to its own index")
    void testQuantizeCentroidsSelfConsistent() {
        Codebook cb = TurboQuantCodebook.getCodebook(128, 3);
        for (int i = 0; i < cb.centroids.length; i++) {
            int idx = TurboQuantCodebook.quantize(cb.centroids[i], cb);
            assertEquals(i, idx,
                    String.format("Quantizing centroid[%d]=%f should return index %d but got %d",
                            i, cb.centroids[i], i, idx));
        }
    }
}
