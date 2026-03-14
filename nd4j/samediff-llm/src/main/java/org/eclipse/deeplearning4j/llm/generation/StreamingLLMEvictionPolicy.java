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

/**
 * StreamingLLM-style token eviction policy.
 *
 * <p>StreamingLLM is a position-based eviction strategy that keeps two disjoint token regions:
 * <ul>
 *   <li><b>Attention sinks</b>: the first {@code numSinks} tokens in the sequence. These initial
 *       tokens accumulate high attention scores regardless of content (the "attention sink" effect)
 *       and are essential for stable generation.</li>
 *   <li><b>Rolling window</b>: the last {@code windowSize} tokens, which provide local context
 *       for the next generation step.</li>
 * </ul>
 *
 * <p>All tokens between the sinks and the rolling window are evicted. This is a zero-overhead
 * policy: it does not track attention scores or maintain any per-token state. Eviction decisions
 * are purely based on position.</p>
 *
 * <h3>Reference</h3>
 * <p>Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TokenEvictionPolicy
 * @see EvictablePagedKVCache
 */
@Slf4j
public class StreamingLLMEvictionPolicy implements TokenEvictionPolicy {

    /** Number of initial "attention sink" tokens to always retain. */
    @Getter private final int numSinks;

    /** Number of most recent tokens to always retain. */
    @Getter private final int windowSize;

    /**
     * Create a StreamingLLM eviction policy.
     *
     * @param numSinks   number of initial tokens to retain as attention sinks (typically 1-4)
     * @param windowSize number of most recent tokens to retain as the rolling window
     */
    public StreamingLLMEvictionPolicy(int numSinks, int windowSize) {
        if (numSinks < 0) throw new IllegalArgumentException("numSinks must be >= 0, got " + numSinks);
        if (windowSize < 1) throw new IllegalArgumentException("windowSize must be >= 1, got " + windowSize);

        this.numSinks = numSinks;
        this.windowSize = windowSize;
    }

    @Override
    public int[] selectTokensToEvict(int layerIdx, int currentLen, int budget) {
        // Evict the range [numSinks, currentLen - windowSize)
        // This keeps the first numSinks tokens and the last windowSize tokens
        int evictStart = numSinks;
        int evictEnd = currentLen - windowSize;

        if (evictStart >= evictEnd || currentLen <= numSinks + windowSize) {
            // Nothing to evict: sequence is short enough to fit entirely
            return new int[0];
        }

        int numToEvict = evictEnd - evictStart;
        int[] result = new int[numToEvict];
        for (int i = 0; i < numToEvict; i++) {
            result[i] = evictStart + i;
        }

        log.debug("StreamingLLM eviction: currentLen={}, evicting positions [{}, {}), keeping {} sinks + {} recent",
                currentLen, evictStart, evictEnd, numSinks, windowSize);

        return result;
    }

    /**
     * No-op: StreamingLLM is purely position-based and does not track attention scores.
     */
    @Override
    public void updateScores(int layerIdx, INDArray attentionWeights) {
        // Intentionally empty: position-based policy, zero overhead
    }

    @Override
    public void reset() {
        // No state to reset
    }

    @Override
    public String toString() {
        return String.format("StreamingLLMEvictionPolicy[sinks=%d, window=%d]", numSinks, windowSize);
    }
}
