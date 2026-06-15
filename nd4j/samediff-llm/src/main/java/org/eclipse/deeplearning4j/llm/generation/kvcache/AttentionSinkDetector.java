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

package org.eclipse.deeplearning4j.llm.generation.kvcache;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Runtime detection of high-attention "sink" tokens.
 *
 * <p>Attention sinks are token positions that consistently receive disproportionately
 * high attention weights across many decoding steps. These are typically the BOS token,
 * punctuation, or structural tokens that serve as "anchor points" for the attention
 * mechanism. Evicting these positions from the KV cache causes severe degradation in
 * generation quality.</p>
 *
 * <h3>Detection algorithm</h3>
 * <ol>
 *   <li>At each decoding step, accumulate attention weights summed across all heads</li>
 *   <li>After a warmup period, compute the mean cumulative score across all positions</li>
 *   <li>Mark positions where cumulative score exceeds {@code mean * threshold} as sinks</li>
 *   <li>Cap the total number of sinks at {@code maxSinks}</li>
 * </ol>
 *
 * <h3>Usage with eviction policies</h3>
 * <p>The detected sink positions are passed to {@link EvictionPolicy} implementations
 * (e.g., {@link SinkAwareEvictionPolicy}) to protect important tokens during
 * KV cache eviction.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see EvictionPolicy
 * @see SinkAwareEvictionPolicy
 */
@Slf4j
public class AttentionSinkDetector {

    /** Default threshold multiplier for sink detection. */
    public static final float DEFAULT_THRESHOLD = 3.0f;

    /** Default maximum number of sink positions. */
    public static final int DEFAULT_MAX_SINKS = 16;

    /** Default number of warmup steps before detection activates. */
    public static final int DEFAULT_WARMUP_STEPS = 10;

    @Getter
    private final int maxSeqLen;

    @Getter
    private final float threshold;

    @Getter
    private final int maxSinks;

    @Getter
    private final int warmupSteps;

    /** Cumulative attention scores per position. */
    private final double[] cumulativeScores;

    /** Number of update steps performed. */
    private int stepCount;

    /** Cached sink positions (recomputed on each update after warmup). */
    private Set<Integer> cachedSinkPositions;

    /**
     * Create an attention sink detector with full configuration.
     *
     * @param maxSeqLen   maximum sequence length to track
     * @param threshold   multiplier on mean score for sink classification
     * @param maxSinks    maximum number of sink positions to detect
     * @param warmupSteps number of steps before detection activates
     */
    public AttentionSinkDetector(int maxSeqLen, float threshold, int maxSinks, int warmupSteps) {
        if (maxSeqLen < 1) {
            throw new IllegalArgumentException("maxSeqLen must be >= 1, got " + maxSeqLen);
        }
        if (threshold <= 0) {
            throw new IllegalArgumentException("threshold must be > 0, got " + threshold);
        }
        this.maxSeqLen = maxSeqLen;
        this.threshold = threshold;
        this.maxSinks = maxSinks;
        this.warmupSteps = warmupSteps;
        this.cumulativeScores = new double[maxSeqLen];
        this.stepCount = 0;
        this.cachedSinkPositions = Collections.emptySet();
    }

    /**
     * Create an attention sink detector with default parameters.
     *
     * @param maxSeqLen maximum sequence length to track
     */
    public AttentionSinkDetector(int maxSeqLen) {
        this(maxSeqLen, DEFAULT_THRESHOLD, DEFAULT_MAX_SINKS, DEFAULT_WARMUP_STEPS);
    }

    /**
     * Update cumulative scores with attention weights from one decoding step.
     *
     * <p>The attention weights should be summed across all heads before passing
     * to this method. Expected shape is [seqLen] or [1, seqLen] representing
     * the attention distribution over all positions for the current query.</p>
     *
     * @param attentionWeights attention weights summed across heads, shape [seqLen] or [1, seqLen]
     */
    public void updateScores(INDArray attentionWeights) {
        INDArray weights = attentionWeights.rank() == 2
                ? attentionWeights.reshape(attentionWeights.length())
                : attentionWeights;

        long len = Math.min(weights.length(), maxSeqLen);
        for (int i = 0; i < len; i++) {
            cumulativeScores[i] += weights.getDouble(i);
        }

        stepCount++;

        // Recompute sink positions after warmup
        if (stepCount >= warmupSteps) {
            recomputeSinks((int) len);
        }
    }

    /**
     * Get the set of detected sink positions.
     *
     * <p>Returns an empty set during the warmup period.</p>
     *
     * @return unmodifiable set of sink position indices
     */
    public Set<Integer> getSinkPositions() {
        return cachedSinkPositions;
    }

    /**
     * Check if a specific position is detected as a sink.
     *
     * @param position the sequence position to check
     * @return true if the position is a detected sink
     */
    public boolean isSink(int position) {
        return cachedSinkPositions.contains(position);
    }

    /**
     * Reset all state, clearing cumulative scores and detected sinks.
     */
    public void reset() {
        java.util.Arrays.fill(cumulativeScores, 0.0);
        stepCount = 0;
        cachedSinkPositions = Collections.emptySet();
    }

    /**
     * Get the current step count.
     */
    public int getStepCount() {
        return stepCount;
    }

    /**
     * Get the cumulative score for a specific position.
     *
     * @param position the sequence position
     * @return the cumulative attention score
     */
    public double getScore(int position) {
        if (position < 0 || position >= maxSeqLen) {
            throw new IndexOutOfBoundsException("Position " + position + " out of range [0, " + maxSeqLen + ")");
        }
        return cumulativeScores[position];
    }

    @Override
    public String toString() {
        return String.format("AttentionSinkDetector[maxSeqLen=%d, threshold=%.1f, sinks=%d, steps=%d]",
                maxSeqLen, threshold, cachedSinkPositions.size(), stepCount);
    }

    // ========== Internal Helpers ==========

    /**
     * Recompute sink positions based on current cumulative scores.
     *
     * @param activeLen number of active positions (with nonzero scores)
     */
    private void recomputeSinks(int activeLen) {
        if (activeLen <= 0) {
            cachedSinkPositions = Collections.emptySet();
            return;
        }

        // Compute mean score across active positions
        double sum = 0;
        for (int i = 0; i < activeLen; i++) {
            sum += cumulativeScores[i];
        }
        double mean = sum / activeLen;

        if (mean <= 0) {
            cachedSinkPositions = Collections.emptySet();
            return;
        }

        double sinkThreshold = mean * threshold;

        // Collect all positions exceeding threshold, sorted by score descending
        // We use a simple approach: collect candidates, sort, take top maxSinks
        int[][] candidates = new int[activeLen][1];
        double[] scores = new double[activeLen];
        int numCandidates = 0;

        for (int i = 0; i < activeLen; i++) {
            if (cumulativeScores[i] > sinkThreshold) {
                candidates[numCandidates][0] = i;
                scores[numCandidates] = cumulativeScores[i];
                numCandidates++;
            }
        }

        // Sort by score descending, take top maxSinks
        Set<Integer> sinks = new HashSet<>();
        if (numCandidates <= maxSinks) {
            for (int i = 0; i < numCandidates; i++) {
                sinks.add(candidates[i][0]);
            }
        } else {
            // Simple selection: find top maxSinks by iterating
            boolean[] selected = new boolean[numCandidates];
            for (int s = 0; s < maxSinks; s++) {
                int bestIdx = -1;
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < numCandidates; i++) {
                    if (!selected[i] && scores[i] > bestScore) {
                        bestScore = scores[i];
                        bestIdx = i;
                    }
                }
                if (bestIdx >= 0) {
                    selected[bestIdx] = true;
                    sinks.add(candidates[bestIdx][0]);
                }
            }
        }

        cachedSinkPositions = Collections.unmodifiableSet(sinks);

        log.trace("Recomputed sinks: {} positions (threshold={:.2f}, mean={:.4f})",
                sinks.size(), sinkThreshold, mean);
    }
}
