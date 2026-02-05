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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SamplerUtils.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
class SamplerUtilsTest {

    @BeforeEach
    void setUp() {
        Nd4j.getRandom().setSeed(42);
    }

    // ==================== Temperature Tests ====================

    @Test
    void testApplyTemperature_NoChange() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0});
        INDArray result = SamplerUtils.applyTemperature(logits, 1.0);

        // Temperature 1.0 should return same logits
        assertSame(logits, result);
    }

    @Test
    void testApplyTemperature_HighTemperature() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0});
        INDArray result = SamplerUtils.applyTemperature(logits, 2.0);

        // High temperature divides logits, making distribution flatter
        assertEquals(0.5, result.getDouble(0), 1e-6);
        assertEquals(1.0, result.getDouble(1), 1e-6);
        assertEquals(1.5, result.getDouble(2), 1e-6);
        assertEquals(2.0, result.getDouble(3), 1e-6);
    }

    @Test
    void testApplyTemperature_LowTemperature() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0});
        INDArray result = SamplerUtils.applyTemperature(logits, 0.5);

        // Low temperature multiplies logits, making distribution sharper
        assertEquals(2.0, result.getDouble(0), 1e-6);
        assertEquals(4.0, result.getDouble(1), 1e-6);
        assertEquals(6.0, result.getDouble(2), 1e-6);
        assertEquals(8.0, result.getDouble(3), 1e-6);
    }

    @Test
    void testApplyTemperature_InvalidTemperature() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        assertThrows(IllegalArgumentException.class, () -> SamplerUtils.applyTemperature(logits, 0.0));
        assertThrows(IllegalArgumentException.class, () -> SamplerUtils.applyTemperature(logits, -1.0));
    }

    // ==================== Softmax Tests ====================

    @Test
    void testSoftmax_1D() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray probs = SamplerUtils.softmax(logits);

        // Should sum to 1
        assertEquals(1.0, probs.sumNumber().doubleValue(), 1e-6);
        // All values should be positive
        assertTrue(probs.minNumber().doubleValue() > 0);
        // Shape should be preserved
        assertEquals(1, probs.rank());
        assertEquals(3, probs.length());
        // Higher logit should have higher probability
        assertTrue(probs.getDouble(2) > probs.getDouble(1));
        assertTrue(probs.getDouble(1) > probs.getDouble(0));
    }

    @Test
    void testSoftmax_2D() {
        INDArray logits = Nd4j.create(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
        INDArray probs = SamplerUtils.softmax(logits);

        assertEquals(2, probs.rank());
        // Each row should sum to 1
        assertEquals(1.0, probs.getRow(0).sumNumber().doubleValue(), 1e-6);
        assertEquals(1.0, probs.getRow(1).sumNumber().doubleValue(), 1e-6);
    }

    // ==================== Top-K Tests ====================

    @Test
    void testTopKFilter() {
        INDArray logits = Nd4j.create(new double[]{1.0, 5.0, 2.0, 4.0, 3.0});
        INDArray filtered = SamplerUtils.topKFilter(logits, 3);

        // Top 3 values are 5.0, 4.0, 3.0 at indices 1, 3, 4
        // Indices 0, 2 should be -inf
        assertTrue(Double.isInfinite(filtered.getDouble(0)));
        assertEquals(5.0, filtered.getDouble(1), 1e-6);
        assertTrue(Double.isInfinite(filtered.getDouble(2)));
        assertEquals(4.0, filtered.getDouble(3), 1e-6);
        assertEquals(3.0, filtered.getDouble(4), 1e-6);
    }

    @Test
    void testTopKFilter_KGreaterThanVocab() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray filtered = SamplerUtils.topKFilter(logits, 10);

        // Should return unchanged
        assertEquals(1.0, filtered.getDouble(0), 1e-6);
        assertEquals(2.0, filtered.getDouble(1), 1e-6);
        assertEquals(3.0, filtered.getDouble(2), 1e-6);
    }

    @Test
    void testTopKFilter_Disabled() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray filtered = SamplerUtils.topKFilter(logits, 0);

        // K=0 should return unchanged
        assertSame(logits, filtered);
    }

    // ==================== Top-P Tests ====================

    @Test
    void testTopPFilter() {
        // Create logits where softmax gives clear probabilities
        INDArray logits = Nd4j.create(new double[]{0.0, 0.0, 0.0, 2.0, 4.0});
        INDArray filtered = SamplerUtils.topPFilter(logits, 0.9);

        // After softmax, token 4 has ~0.88, token 3 has ~0.12
        // With p=0.9, we should keep tokens 4 and 3 (cumulative > 0.9)
        // Tokens 0, 1, 2 should be filtered (very low probability)
        assertTrue(Double.isInfinite(filtered.getDouble(0)) || filtered.getDouble(0) < -1e6);
    }

    @Test
    void testTopPFilter_Disabled() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray filtered = SamplerUtils.topPFilter(logits, 1.0);

        // P=1.0 should return unchanged
        assertSame(logits, filtered);
    }

    // ==================== Repetition Penalty Tests ====================

    @Test
    void testRepetitionPenalty() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        int[] generated = {1, 3}; // Penalize indices 1 and 3
        INDArray penalized = SamplerUtils.applyRepetitionPenalty(logits, generated, 2.0);

        // Index 0: unchanged
        assertEquals(1.0, penalized.getDouble(0), 1e-6);
        // Index 1: positive, divided by penalty
        assertEquals(1.0, penalized.getDouble(1), 1e-6); // 2.0 / 2.0
        // Index 2: unchanged
        assertEquals(3.0, penalized.getDouble(2), 1e-6);
        // Index 3: positive, divided by penalty
        assertEquals(2.0, penalized.getDouble(3), 1e-6); // 4.0 / 2.0
        // Index 4: unchanged
        assertEquals(5.0, penalized.getDouble(4), 1e-6);
    }

    @Test
    void testRepetitionPenalty_NoPenalty() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        int[] generated = {0, 1};
        INDArray result = SamplerUtils.applyRepetitionPenalty(logits, generated, 1.0);

        // Penalty=1.0 should return same values
        assertEquals(1.0, result.getDouble(0), 1e-6);
        assertEquals(2.0, result.getDouble(1), 1e-6);
        assertEquals(3.0, result.getDouble(2), 1e-6);
    }

    // ==================== Argmax Tests ====================

    @Test
    void testArgmax() {
        INDArray logits = Nd4j.create(new double[]{1.0, 5.0, 3.0, 2.0});
        int idx = SamplerUtils.argmax(logits);
        assertEquals(1, idx);
    }

    @Test
    void testArgmaxBatch() {
        INDArray logits = Nd4j.create(new double[][]{{1.0, 3.0, 2.0}, {5.0, 1.0, 2.0}});
        int[] indices = SamplerUtils.argmaxBatch(logits);

        assertEquals(2, indices.length);
        assertEquals(1, indices[0]); // First row: max at index 1
        assertEquals(0, indices[1]); // Second row: max at index 0
    }

    // ==================== Multinomial Sampling Tests ====================

    @Test
    void testMultinomialSample_Deterministic() {
        // Create nearly deterministic distribution (one very high prob)
        INDArray probs = Nd4j.create(new double[]{0.001, 0.001, 0.997, 0.001});
        Random random = new Random(42);

        // Most samples should be index 2
        int count2 = 0;
        for (int i = 0; i < 100; i++) {
            if (SamplerUtils.multinomialSample(probs, random) == 2) {
                count2++;
            }
        }
        assertTrue(count2 > 90, "Expected most samples at index 2, got " + count2);
    }

    @Test
    void testMultinomialSample_Uniform() {
        INDArray probs = Nd4j.create(new double[]{0.25, 0.25, 0.25, 0.25});
        Random random = new Random(42);

        int[] counts = new int[4];
        for (int i = 0; i < 1000; i++) {
            counts[SamplerUtils.multinomialSample(probs, random)]++;
        }

        // Each should be roughly 250 (within reasonable variance)
        for (int count : counts) {
            assertTrue(count > 150 && count < 350, "Expected ~250, got " + count);
        }
    }

    // ==================== Log Softmax Tests ====================

    @Test
    void testLogSoftmax() {
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray logProbs = SamplerUtils.logSoftmax(logits);

        // Log probs should be negative (since probs < 1)
        assertTrue(logProbs.maxNumber().doubleValue() <= 0);

        // exp(log_softmax) should equal softmax
        INDArray expLogProbs = Nd4j.math().exp(logProbs);
        INDArray softmax = SamplerUtils.softmax(logits);

        for (int i = 0; i < 3; i++) {
            assertEquals(softmax.getDouble(i), expLogProbs.getDouble(i), 1e-5);
        }
    }

    // ==================== Entropy Tests ====================

    @Test
    void testEntropy_Uniform() {
        // Uniform distribution has maximum entropy
        INDArray uniform = Nd4j.create(new double[]{0.25, 0.25, 0.25, 0.25});
        double entropy = SamplerUtils.entropy(uniform);

        // Max entropy for 4 elements is ln(4) ≈ 1.386
        assertEquals(Math.log(4), entropy, 0.01);
    }

    @Test
    void testEntropy_Peaked() {
        // Very peaked distribution has low entropy
        INDArray peaked = Nd4j.create(new double[]{0.97, 0.01, 0.01, 0.01});
        double entropy = SamplerUtils.entropy(peaked);

        // Should be much lower than uniform
        assertTrue(entropy < 0.5);
    }

    // ==================== Distribution Validation Tests ====================

    @Test
    void testIsValidDistribution() {
        INDArray valid = Nd4j.create(new double[]{0.3, 0.3, 0.4});
        assertTrue(SamplerUtils.isValidDistribution(valid, 1e-6));

        INDArray invalid1 = Nd4j.create(new double[]{0.5, 0.3, 0.3}); // Sum > 1
        assertFalse(SamplerUtils.isValidDistribution(invalid1, 1e-6));

        INDArray invalid2 = Nd4j.create(new double[]{0.5, -0.1, 0.6}); // Negative value
        assertFalse(SamplerUtils.isValidDistribution(invalid2, 1e-6));
    }

    // ==================== Batch Operation Tests ====================

    @Test
    void testBatchOperations() {
        INDArray batchLogits = Nd4j.create(new double[][]{
            {1.0, 2.0, 3.0, 4.0},
            {4.0, 3.0, 2.0, 1.0}
        });

        // Test batch softmax
        INDArray batchProbs = SamplerUtils.softmax(batchLogits);
        assertEquals(1.0, batchProbs.getRow(0).sumNumber().doubleValue(), 1e-6);
        assertEquals(1.0, batchProbs.getRow(1).sumNumber().doubleValue(), 1e-6);

        // Test batch argmax
        int[] argmaxes = SamplerUtils.argmaxBatch(batchLogits);
        assertEquals(3, argmaxes[0]); // Max at index 3
        assertEquals(0, argmaxes[1]); // Max at index 0

        // Test batch top-K
        INDArray batchTopK = SamplerUtils.topKFilter(batchLogits, 2);
        assertEquals(2, batchTopK.rank());
    }
}
