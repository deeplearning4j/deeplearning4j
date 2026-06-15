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

import lombok.extern.slf4j.Slf4j;

import java.util.Set;

/**
 * Eviction policy that protects attention sink positions.
 *
 * <p>Filters out candidate positions that contain attention sinks (at the block
 * level: a block is protected if ANY position within it is a sink). From the
 * remaining candidates, selects the oldest {@code blocksNeeded} blocks for
 * eviction.</p>
 *
 * <h3>Block-level protection</h3>
 * <p>Since KV cache blocks span multiple token positions (blockSize tokens each),
 * a block is considered protected if any token position within it is classified as
 * an attention sink. This is conservative but safe: evicting a block that contains
 * even one sink position would degrade attention quality for that head.</p>
 *
 * <h3>Fallback behavior</h3>
 * <p>If there are not enough non-sink candidates to satisfy {@code blocksNeeded},
 * the policy returns as many as available. The caller should handle the case where
 * fewer blocks are returned than requested.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see EvictionPolicy
 * @see AttentionSinkDetector
 */
@Slf4j
public class SinkAwareEvictionPolicy implements EvictionPolicy {

    /** Block size in tokens, used for sink position -> block mapping. */
    private final int blockSize;

    /**
     * Create a sink-aware eviction policy.
     *
     * @param blockSize tokens per KV cache block (used to map positions to blocks)
     */
    public SinkAwareEvictionPolicy(int blockSize) {
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be >= 1, got " + blockSize);
        }
        this.blockSize = blockSize;
    }

    /**
     * Create a sink-aware eviction policy with default block size.
     */
    public SinkAwareEvictionPolicy() {
        this(PagedKVCache.DEFAULT_BLOCK_SIZE);
    }

    @Override
    public int[] selectForEviction(int[] candidatePositions, Set<Integer> sinkPositions, int blocksNeeded) {
        if (candidatePositions == null || candidatePositions.length == 0 || blocksNeeded <= 0) {
            return new int[0];
        }

        // Filter candidates: exclude any position whose block contains a sink
        int[] filtered = new int[candidatePositions.length];
        int filteredCount = 0;

        for (int pos : candidatePositions) {
            if (!blockContainsSink(pos, sinkPositions)) {
                filtered[filteredCount++] = pos;
            }
        }

        // Select the oldest (first in array, since candidates are sorted oldest-first)
        int selectCount = Math.min(filteredCount, blocksNeeded);
        int[] selected = new int[selectCount];
        System.arraycopy(filtered, 0, selected, 0, selectCount);

        if (selectCount < blocksNeeded) {
            log.debug("SinkAwareEvictionPolicy: could only select {} of {} needed blocks " +
                            "({}  candidates protected by {} sink positions)",
                    selectCount, blocksNeeded, candidatePositions.length - filteredCount,
                    sinkPositions.size());
        }

        return selected;
    }

    @Override
    public String name() {
        return "SinkAwareEviction";
    }

    /**
     * Check if the block containing the given position has any sink positions.
     *
     * <p>A block spans positions [blockStart, blockStart + blockSize). If any
     * position in that range is in the sink set, the block is protected.</p>
     *
     * @param position      the candidate position
     * @param sinkPositions the set of detected sink positions
     * @return true if the block is protected
     */
    private boolean blockContainsSink(int position, Set<Integer> sinkPositions) {
        if (sinkPositions.isEmpty()) {
            return false;
        }

        int blockStart = (position / blockSize) * blockSize;
        int blockEnd = blockStart + blockSize;

        for (int sinkPos : sinkPositions) {
            if (sinkPos >= blockStart && sinkPos < blockEnd) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return String.format("SinkAwareEvictionPolicy[blockSize=%d]", blockSize);
    }
}
