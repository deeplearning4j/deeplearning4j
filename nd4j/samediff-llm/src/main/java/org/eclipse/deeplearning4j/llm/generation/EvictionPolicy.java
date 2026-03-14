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

import java.util.Set;

/**
 * Interface for KV cache eviction policies.
 *
 * <p>Implementations decide which blocks to evict from the KV cache when
 * memory pressure requires freeing space. Policies can take into account
 * attention sink positions (tokens that should be protected from eviction),
 * access recency, and other heuristics.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see SinkAwareEvictionPolicy
 * @see AttentionSinkDetector
 */
public interface EvictionPolicy {

    /**
     * Select block positions for eviction.
     *
     * <p>Given a set of candidate positions and a set of sink positions that
     * should be protected, return an array of positions to evict. The number
     * of returned positions should be at most {@code blocksNeeded}.</p>
     *
     * @param candidatePositions positions eligible for eviction (sorted by age, oldest first)
     * @param sinkPositions      positions that should be protected from eviction
     * @param blocksNeeded       number of blocks that need to be freed
     * @return array of positions to evict, length <= blocksNeeded
     */
    int[] selectForEviction(int[] candidatePositions, Set<Integer> sinkPositions, int blocksNeeded);

    /**
     * Get the name of this eviction policy.
     *
     * @return policy name for logging and diagnostics
     */
    String name();
}
