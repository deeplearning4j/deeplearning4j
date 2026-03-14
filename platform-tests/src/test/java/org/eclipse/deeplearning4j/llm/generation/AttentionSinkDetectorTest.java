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

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for AttentionSinkDetector, EvictionPolicy, and SinkAwareEvictionPolicy.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
class AttentionSinkDetectorTest {

    @BeforeEach
    void setUp() {
        Nd4j.getRandom().setSeed(42);
    }

    // ==================== AttentionSinkDetector Tests ====================

    @Test
    void testDefaultConstants() {
        assertEquals(3.0f, AttentionSinkDetector.DEFAULT_THRESHOLD);
        assertEquals(16, AttentionSinkDetector.DEFAULT_MAX_SINKS);
        assertEquals(10, AttentionSinkDetector.DEFAULT_WARMUP_STEPS);
    }

    @Test
    void testDefaultConstructorUsesDefaults() {
        AttentionSinkDetector detector = new AttentionSinkDetector(128);
        assertEquals(128, detector.getMaxSeqLen());
        assertEquals(AttentionSinkDetector.DEFAULT_THRESHOLD, detector.getThreshold());
        assertEquals(AttentionSinkDetector.DEFAULT_MAX_SINKS, detector.getMaxSinks());
        assertEquals(AttentionSinkDetector.DEFAULT_WARMUP_STEPS, detector.getWarmupSteps());
    }

    @Test
    void testFullConstructor() {
        AttentionSinkDetector detector = new AttentionSinkDetector(64, 2.0f, 4, 5);
        assertEquals(64, detector.getMaxSeqLen());
        assertEquals(2.0f, detector.getThreshold());
        assertEquals(4, detector.getMaxSinks());
        assertEquals(5, detector.getWarmupSteps());
    }

    @Test
    void testConstructorValidation_maxSeqLenZero() {
        assertThrows(IllegalArgumentException.class, () -> new AttentionSinkDetector(0));
    }

    @Test
    void testConstructorValidation_maxSeqLenNegative() {
        assertThrows(IllegalArgumentException.class, () -> new AttentionSinkDetector(-1));
    }

    @Test
    void testConstructorValidation_thresholdZero() {
        assertThrows(IllegalArgumentException.class,
                () -> new AttentionSinkDetector(32, 0.0f, 4, 5));
    }

    @Test
    void testConstructorValidation_thresholdNegative() {
        assertThrows(IllegalArgumentException.class,
                () -> new AttentionSinkDetector(32, -1.0f, 4, 5));
    }

    @Test
    void testNoSinksDuringWarmup() {
        int warmupSteps = 5;
        AttentionSinkDetector detector = new AttentionSinkDetector(10, 3.0f, 16, warmupSteps);

        // Feed strong signal at position 0 for fewer than warmupSteps
        for (int step = 0; step < warmupSteps - 1; step++) {
            INDArray weights = Nd4j.zeros(10);
            weights.putScalar(0, 100.0);  // Very high attention at position 0
            for (int i = 1; i < 10; i++) {
                weights.putScalar(i, 1.0);
            }
            detector.updateScores(weights);
        }

        assertEquals(warmupSteps - 1, detector.getStepCount());
        assertTrue(detector.getSinkPositions().isEmpty(),
                "No sinks should be detected during warmup period");
        assertFalse(detector.isSink(0));
    }

    @Test
    void testSinkDetectionAfterWarmup() {
        int warmupSteps = 5;
        AttentionSinkDetector detector = new AttentionSinkDetector(10, 3.0f, 16, warmupSteps);

        // Feed strong signal at position 0 for enough steps to pass warmup
        for (int step = 0; step < warmupSteps + 5; step++) {
            INDArray weights = Nd4j.zeros(10);
            weights.putScalar(0, 50.0);  // Very high attention at position 0
            for (int i = 1; i < 10; i++) {
                weights.putScalar(i, 1.0);
            }
            detector.updateScores(weights);
        }

        assertEquals(warmupSteps + 5, detector.getStepCount());

        Set<Integer> sinks = detector.getSinkPositions();
        assertFalse(sinks.isEmpty(), "Should detect sinks after warmup");
        assertTrue(detector.isSink(0), "Position 0 should be a sink (high attention)");

        // Position 0 cumulative score should be much higher than others
        double score0 = detector.getScore(0);
        double score5 = detector.getScore(5);
        assertTrue(score0 > score5 * 3, "Sink position score should be much higher than non-sink");
    }

    @Test
    void testSinkDetectionWith2dWeights() {
        int warmupSteps = 3;
        AttentionSinkDetector detector = new AttentionSinkDetector(8, 3.0f, 16, warmupSteps);

        // Feed 2D weights [1, seqLen]
        for (int step = 0; step < warmupSteps + 2; step++) {
            INDArray weights = Nd4j.zeros(1, 8);
            weights.putScalar(0, 0, 100.0);
            for (int i = 1; i < 8; i++) {
                weights.putScalar(0, i, 1.0);
            }
            detector.updateScores(weights);
        }

        assertTrue(detector.isSink(0), "Should detect sink from 2D attention weights");
    }

    @Test
    void testMaxSinksCap() {
        // Create detector with maxSinks = 2
        int warmupSteps = 3;
        AttentionSinkDetector detector = new AttentionSinkDetector(20, 2.0f, 2, warmupSteps);

        // Make positions 0, 1, 2 all have very high scores (all above threshold)
        for (int step = 0; step < warmupSteps + 5; step++) {
            INDArray weights = Nd4j.zeros(20);
            weights.putScalar(0, 100.0);
            weights.putScalar(1, 90.0);
            weights.putScalar(2, 80.0);
            for (int i = 3; i < 20; i++) {
                weights.putScalar(i, 1.0);
            }
            detector.updateScores(weights);
        }

        Set<Integer> sinks = detector.getSinkPositions();
        assertTrue(sinks.size() <= 2, "Sink count should be capped at maxSinks=2, got " + sinks.size());
    }

    @Test
    void testResetClearsState() {
        int warmupSteps = 3;
        AttentionSinkDetector detector = new AttentionSinkDetector(10, 3.0f, 16, warmupSteps);

        // Build up state
        for (int step = 0; step < warmupSteps + 5; step++) {
            INDArray weights = Nd4j.zeros(10);
            weights.putScalar(0, 50.0);
            for (int i = 1; i < 10; i++) {
                weights.putScalar(i, 1.0);
            }
            detector.updateScores(weights);
        }

        assertTrue(detector.getStepCount() > 0);
        assertTrue(detector.getScore(0) > 0);

        // Reset
        detector.reset();

        assertEquals(0, detector.getStepCount());
        assertEquals(0.0, detector.getScore(0), 1e-9);
        assertTrue(detector.getSinkPositions().isEmpty(), "Sinks should be cleared after reset");
        assertFalse(detector.isSink(0));
    }

    @Test
    void testGetScoreBoundsCheck() {
        AttentionSinkDetector detector = new AttentionSinkDetector(10);
        assertThrows(IndexOutOfBoundsException.class, () -> detector.getScore(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> detector.getScore(10));
    }

    @Test
    void testCumulativeScoreAccumulation() {
        AttentionSinkDetector detector = new AttentionSinkDetector(4, 3.0f, 16, 100);

        INDArray weights = Nd4j.create(new double[]{2.0, 3.0, 1.0, 4.0});
        detector.updateScores(weights);
        detector.updateScores(weights);

        assertEquals(4.0, detector.getScore(0), 1e-9);
        assertEquals(6.0, detector.getScore(1), 1e-9);
        assertEquals(2.0, detector.getScore(2), 1e-9);
        assertEquals(8.0, detector.getScore(3), 1e-9);
    }

    @Test
    void testWeightsShorterThanMaxSeqLen() {
        AttentionSinkDetector detector = new AttentionSinkDetector(100, 3.0f, 16, 1);

        // Only 5 positions in the weights, but maxSeqLen is 100
        INDArray weights = Nd4j.create(new double[]{50.0, 1.0, 1.0, 1.0, 1.0});
        detector.updateScores(weights);

        assertTrue(detector.isSink(0), "Position 0 should be a sink");
        assertEquals(0.0, detector.getScore(50), 1e-9, "Untouched positions should have score 0");
    }

    @Test
    void testToStringDoesNotThrow() {
        AttentionSinkDetector detector = new AttentionSinkDetector(32);
        assertNotNull(detector.toString());
        assertTrue(detector.toString().contains("AttentionSinkDetector"));
    }

    // ==================== EvictionPolicy / SinkAwareEvictionPolicy Tests ====================

    @Test
    void testSinkAwareEvictionPolicy_filtersOutSinkBlocks() {
        SinkAwareEvictionPolicy policy = new SinkAwareEvictionPolicy(4);

        // Candidates: positions 0, 4, 8, 12, 16 (sorted oldest first)
        int[] candidates = {0, 4, 8, 12, 16};
        // Sinks at positions 0 and 1 -- block containing position 0 spans [0,4)
        Set<Integer> sinks = Set.of(0, 1);

        int[] evicted = policy.selectForEviction(candidates, sinks, 3);

        // Block at position 0 spans [0,4), contains sink at 0 and 1, should be excluded
        for (int pos : evicted) {
            assertNotEquals(0, pos, "Position 0 (block containing sink) should not be evicted");
        }
        assertTrue(evicted.length <= 3);
    }

    @Test
    void testSinkAwareEvictionPolicy_noSinksEvictsOldest() {
        SinkAwareEvictionPolicy policy = new SinkAwareEvictionPolicy(4);

        int[] candidates = {0, 4, 8, 12, 16};
        Set<Integer> sinks = Set.of();

        int[] evicted = policy.selectForEviction(candidates, sinks, 2);

        assertEquals(2, evicted.length);
        // Should select oldest first: 0, 4
        assertEquals(0, evicted[0]);
        assertEquals(4, evicted[1]);
    }

    @Test
    void testSinkAwareEvictionPolicy_emptyInput() {
        SinkAwareEvictionPolicy policy = new SinkAwareEvictionPolicy(4);

        int[] evicted = policy.selectForEviction(new int[0], Set.of(), 5);
        assertEquals(0, evicted.length);

        evicted = policy.selectForEviction(null, Set.of(), 5);
        assertEquals(0, evicted.length);
    }

    @Test
    void testSinkAwareEvictionPolicy_allProtected() {
        SinkAwareEvictionPolicy policy = new SinkAwareEvictionPolicy(4);

        int[] candidates = {0, 4};
        // Sinks cover every block
        Set<Integer> sinks = Set.of(0, 5);  // block [0,4) has sink 0, block [4,8) has sink 5

        int[] evicted = policy.selectForEviction(candidates, sinks, 2);
        assertEquals(0, evicted.length, "No blocks should be evicted when all are protected");
    }

    @Test
    void testSinkAwareEvictionPolicy_zeroBlocksNeeded() {
        SinkAwareEvictionPolicy policy = new SinkAwareEvictionPolicy(4);

        int[] candidates = {0, 4, 8};
        int[] evicted = policy.selectForEviction(candidates, Set.of(), 0);
        assertEquals(0, evicted.length);
    }

    @Test
    void testSinkAwareEvictionPolicy_constructorValidation() {
        assertThrows(IllegalArgumentException.class, () -> new SinkAwareEvictionPolicy(0));
        assertThrows(IllegalArgumentException.class, () -> new SinkAwareEvictionPolicy(-1));
    }

    @Test
    void testSinkAwareEvictionPolicy_name() {
        SinkAwareEvictionPolicy policy = new SinkAwareEvictionPolicy(4);
        assertEquals("SinkAwareEviction", policy.name());
    }

    @Test
    void testSinkAwareEvictionPolicy_partialProtection() {
        // blockSize = 8, candidates at block boundaries
        SinkAwareEvictionPolicy policy = new SinkAwareEvictionPolicy(8);

        // 5 candidate blocks
        int[] candidates = {0, 8, 16, 24, 32};
        // Sink at position 3 protects block [0,8), sink at position 20 protects block [16,24)
        Set<Integer> sinks = Set.of(3, 20);

        int[] evicted = policy.selectForEviction(candidates, sinks, 4);

        // Blocks at 0 and 16 should be protected, leaving 8, 24, 32
        assertEquals(3, evicted.length);
        assertArrayEquals(new int[]{8, 24, 32}, evicted);
    }

    @Test
    void testSinkAwareEvictionPolicy_fewerAvailableThanNeeded() {
        SinkAwareEvictionPolicy policy = new SinkAwareEvictionPolicy(4);

        int[] candidates = {0, 4, 8};
        Set<Integer> sinks = Set.of(0, 4);  // Protects blocks [0,4) and [4,8)

        // Only block at position 8 is available, but we need 3
        int[] evicted = policy.selectForEviction(candidates, sinks, 3);
        assertEquals(1, evicted.length);
        assertEquals(8, evicted[0]);
    }
}
