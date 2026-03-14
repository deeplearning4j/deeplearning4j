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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Heavy Hitter Oracle (H2O) token eviction policy.
 *
 * <p>H2O tracks cumulative attention scores across decode steps to identify "heavy hitter"
 * tokens that consistently receive high attention. When eviction is needed, it retains
 * the top-K tokens by cumulative score plus the most recent tokens (which are needed
 * for local context), and evicts the rest.</p>
 *
 * <p>This is based on the observation that a small subset of tokens (heavy hitters)
 * accumulate disproportionately high attention scores and are critical for generation
 * quality. Evicting low-score tokens has minimal impact on output quality.</p>
 *
 * <h3>Reference</h3>
 * <p>Zhang et al., "H2O: Heavy-Hitter Oracle: Efficient Generative Inference of
 * Large Language Models with Heavy Hitters", NeurIPS 2023.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TokenEvictionPolicy
 * @see EvictablePagedKVCache
 */
@Slf4j
public class H2OEvictionPolicy implements TokenEvictionPolicy {

    @Getter private final int maxSeqLen;
    @Getter private final int numLayers;
    @Getter private final int topK;
    @Getter private final int recentWindow;

    /** Cumulative attention scores per layer: [numLayers][maxSeqLen]. */
    private final float[][] cumulativeScores;

    /**
     * Optional per-layer budget overrides. When set, {@code perLayerBudgets[layerIdx]}
     * is used instead of the budget parameter passed to {@link #selectTokensToEvict}.
     * A value of -1 means "use the default budget".
     */
    @Getter private int[] perLayerBudgets;

    /**
     * Create an H2O eviction policy.
     *
     * @param maxSeqLen    maximum sequence length (determines score array size)
     * @param numLayers    number of decoder layers
     * @param topK         number of heavy-hitter tokens to retain by score
     * @param recentWindow number of most recent tokens to always retain
     */
    public H2OEvictionPolicy(int maxSeqLen, int numLayers, int topK, int recentWindow) {
        if (maxSeqLen < 1) throw new IllegalArgumentException("maxSeqLen must be >= 1, got " + maxSeqLen);
        if (numLayers < 1) throw new IllegalArgumentException("numLayers must be >= 1, got " + numLayers);
        if (topK < 0) throw new IllegalArgumentException("topK must be >= 0, got " + topK);
        if (recentWindow < 0) throw new IllegalArgumentException("recentWindow must be >= 0, got " + recentWindow);

        this.maxSeqLen = maxSeqLen;
        this.numLayers = numLayers;
        this.topK = topK;
        this.recentWindow = recentWindow;
        this.cumulativeScores = new float[numLayers][maxSeqLen];
        this.perLayerBudgets = null;
    }

    /**
     * Set per-layer budget overrides.
     *
     * @param budgets array of length {@code numLayers}; -1 entries mean "use default budget"
     */
    public void setPerLayerBudgets(int[] budgets) {
        if (budgets != null && budgets.length != numLayers) {
            throw new IllegalArgumentException("perLayerBudgets length (" + budgets.length
                    + ") must match numLayers (" + numLayers + ")");
        }
        this.perLayerBudgets = budgets;
    }

    @Override
    public void updateScores(int layerIdx, INDArray attentionWeights) {
        if (layerIdx < 0 || layerIdx >= numLayers) return;

        // attentionWeights shape: [batch, numHeads, queryLen, keyLen]
        // Sum across batch and heads to get per-position importance: [queryLen, keyLen]
        // Then sum across queryLen to get per-key-position score: [keyLen]
        long keyLen = attentionWeights.size(attentionWeights.rank() - 1);
        int len = (int) Math.min(keyLen, maxSeqLen);

        // Sum over batch dimension (dim 0), then heads (dim 0 of result), then queries (dim 0 of result)
        // Flatten to [keyLen] by summing all leading dimensions
        INDArray summed = attentionWeights;
        while (summed.rank() > 1) {
            summed = summed.sum(0);
        }

        // Accumulate into per-position scores
        for (int k = 0; k < len; k++) {
            cumulativeScores[layerIdx][k] += summed.getFloat(k);
        }

        log.trace("H2O updateScores layer={}, keyLen={}", layerIdx, keyLen);
    }

    @Override
    public int[] selectTokensToEvict(int layerIdx, int currentLen, int budget) {
        if (layerIdx < 0 || layerIdx >= numLayers) return new int[0];

        // Apply per-layer budget override if set
        int effectiveBudget = budget;
        if (perLayerBudgets != null && perLayerBudgets[layerIdx] >= 0) {
            effectiveBudget = perLayerBudgets[layerIdx];
        }

        // No eviction needed if within budget
        if (currentLen <= effectiveBudget) {
            return new int[0];
        }

        // Determine which positions are protected by recency
        int recentStart = Math.max(0, currentLen - recentWindow);

        // Build list of candidate positions (not in recent window)
        // and find top-K among them by cumulative score
        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < recentStart; i++) {
            candidates.add(i);
        }

        // Sort candidates by cumulative score descending
        candidates.sort((a, b) -> Float.compare(cumulativeScores[layerIdx][b], cumulativeScores[layerIdx][a]));

        // Keep top-K candidates (heavy hitters) + all recent positions
        int numToKeepFromCandidates = Math.min(topK, candidates.size());
        int totalKept = numToKeepFromCandidates + Math.min(recentWindow, currentLen);

        // If we are still over budget, reduce candidates kept
        if (totalKept > effectiveBudget) {
            numToKeepFromCandidates = Math.max(0, effectiveBudget - Math.min(recentWindow, currentLen));
        }

        // Positions to keep: top-K candidates + recent window
        boolean[] keep = new boolean[currentLen];

        // Mark top-K candidates as kept
        for (int i = 0; i < numToKeepFromCandidates; i++) {
            keep[candidates.get(i)] = true;
        }

        // Mark recent window as kept
        for (int i = recentStart; i < currentLen; i++) {
            keep[i] = true;
        }

        // Collect eviction positions
        List<Integer> evictList = new ArrayList<>();
        for (int i = 0; i < currentLen; i++) {
            if (!keep[i]) {
                evictList.add(i);
            }
        }

        int[] result = new int[evictList.size()];
        for (int i = 0; i < evictList.size(); i++) {
            result[i] = evictList.get(i);
        }

        log.debug("H2O eviction layer={}: currentLen={}, budget={}, evicting {} tokens, keeping {} heavy hitters + {} recent",
                layerIdx, currentLen, effectiveBudget, result.length, numToKeepFromCandidates,
                Math.min(recentWindow, currentLen));

        return result;
    }

    @Override
    public void reset() {
        for (int l = 0; l < numLayers; l++) {
            Arrays.fill(cumulativeScores[l], 0.0f);
        }
        log.debug("H2O eviction policy reset");
    }

    @Override
    public String toString() {
        return String.format("H2OEvictionPolicy[maxSeqLen=%d, layers=%d, topK=%d, recentWindow=%d]",
                maxSeqLen, numLayers, topK, recentWindow);
    }
}
