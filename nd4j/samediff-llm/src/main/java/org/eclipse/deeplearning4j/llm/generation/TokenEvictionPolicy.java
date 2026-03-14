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

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Policy interface for token-level KV cache eviction.
 *
 * <p>During long-sequence generation, KV cache memory can grow beyond available budget.
 * Eviction policies decide which token positions to remove from the cache to stay within
 * budget while preserving generation quality.</p>
 *
 * <p>Implementations range from simple position-based policies (e.g., sliding window)
 * to attention-score-based policies that track token importance across decode steps.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see H2OEvictionPolicy
 * @see StreamingLLMEvictionPolicy
 * @see EvictablePagedKVCache
 */
public interface TokenEvictionPolicy {

    /**
     * Select token positions to evict from a layer's KV cache.
     *
     * <p>The returned indices are absolute positions in the current KV sequence.
     * The caller is responsible for physically removing the tokens and compacting
     * the cache.</p>
     *
     * @param layerIdx   the decoder layer index
     * @param currentLen current number of tokens in the KV cache for this layer
     * @param budget     maximum number of tokens to retain (evict enough to reach this budget)
     * @return array of token position indices to evict, empty if no eviction needed
     */
    int[] selectTokensToEvict(int layerIdx, int currentLen, int budget);

    /**
     * Update internal importance scores from attention weights.
     *
     * <p>Called after each decode step with the attention weights from the model.
     * Policies that track cumulative attention scores (e.g., H2O) use this to
     * update their token importance estimates.</p>
     *
     * @param layerIdx         the decoder layer index
     * @param attentionWeights attention weight tensor, typically shape
     *                         [batch, numHeads, queryLen, keyLen]
     */
    void updateScores(int layerIdx, INDArray attentionWeights);

    /**
     * Reset all internal state.
     *
     * <p>Called when starting a new generation sequence to clear any accumulated
     * scores or state from the previous sequence.</p>
     */
    void reset();
}
