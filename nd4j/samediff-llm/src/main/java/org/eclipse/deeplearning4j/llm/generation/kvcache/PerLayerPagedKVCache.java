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

/**
 * Per-layer paged KV cache with independent cache sizes per transformer layer.
 *
 * <p>Creates one {@link PagedKVCache} per layer, each with its own maximum
 * sequence length derived from the per-layer policy. Layers with sliding
 * window policies automatically evict oldest blocks when the window is exceeded.</p>
 *
 * <h3>Sliding window</h3>
 * <p>For layers configured with {@code slidingWindow=true}, the
 * {@link #enforceSlidingWindow(int, int)} method evicts the oldest blocks
 * to keep within the layer's window size. This is called automatically
 * during append if the policy dictates it.</p>
 *
 * <h3>Memory efficiency</h3>
 * <p>By giving different window sizes to different layers, total memory usage
 * can be significantly reduced compared to a uniform maximum sequence length.
 * For example, with a pyramid policy, early layers may only keep the last 128
 * tokens while deep layers keep the full 2048-token context.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see PerLayerKVPolicy
 * @see PagedKVCache
 */
@Slf4j
public class PerLayerPagedKVCache implements AutoCloseable {

    @Getter
    private final PerLayerKVPolicy policy;

    @Getter
    private final int maxBatchSize;

    @Getter
    private final int numKvHeads;

    @Getter
    private final int headDim;

    @Getter
    private final int blockSize;

    /** Per-layer KV caches. */
    private final PagedKVCache[] layerCaches;

    /**
     * Create a per-layer paged KV cache.
     *
     * @param policy       the per-layer KV policy
     * @param maxBatchSize maximum batch size
     * @param numKvHeads   number of KV attention heads
     * @param headDim      dimension per head
     * @param blockSize    tokens per block
     */
    public PerLayerPagedKVCache(PerLayerKVPolicy policy, int maxBatchSize,
                                 int numKvHeads, int headDim, int blockSize) {
        this.policy = policy;
        this.maxBatchSize = maxBatchSize;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.blockSize = blockSize;

        int numLayers = policy.getNumLayers();
        this.layerCaches = new PagedKVCache[numLayers];

        long totalMemory = 0;
        for (int layer = 0; layer < numLayers; layer++) {
            PerLayerKVPolicy.LayerPolicy layerPolicy = policy.getPolicy(layer);
            int maxSeqLen = layerPolicy.getWindowSize();

            layerCaches[layer] = new PagedKVCache(
                    maxBatchSize, maxSeqLen, numKvHeads, headDim,
                    blockSize, layerPolicy.getKvDataType(), 1.1);

            long layerBytes = 2L * layerCaches[layer].getNumBlocks() * blockSize
                    * numKvHeads * headDim * layerPolicy.getKvDataType().width();
            totalMemory += layerBytes;
        }

        log.info("PerLayerPagedKVCache: {} layers, blockSize={}, totalMemory={} MB",
                numLayers, blockSize, totalMemory / (1024 * 1024));
    }

    /**
     * Append new KV entries to a specific layer for a specific sequence.
     *
     * <p>If the layer's policy uses a sliding window and the append would push the
     * sequence beyond the window size, the oldest blocks are evicted first so the
     * invariant {@code sequenceLength <= windowSize + newLen} holds and the append
     * itself always fits inside the window.</p>
     *
     * @param layerIdx  layer index (0-based)
     * @param seqIdx    batch index of the sequence
     * @param newKeys   new key tensor
     * @param newValues new value tensor
     */
    public void append(int layerIdx, int seqIdx, INDArray newKeys, INDArray newValues) {
        validateLayerIdx(layerIdx);

        PerLayerKVPolicy.LayerPolicy layerPolicy = policy.getPolicy(layerIdx);

        // Evict before appending when the POST-append length would exceed the window:
        // evicting before the append keeps the append inside the window instead of
        // leaving the sequence over the window until some later append triggers.
        if (layerPolicy.isSlidingWindow()) {
            long newLen = newKeys.rank() == 4 ? newKeys.size(1) : newKeys.size(0);
            int currentLen = layerCaches[layerIdx].getSequenceLength(seqIdx);
            int totalLen = currentLen + (int) newLen;

            if (totalLen > layerPolicy.getWindowSize()) {
                // Keep at most (windowSize - newLen) existing tokens so the
                // appended tokens fit entirely inside the window.
                int keepTokens = Math.max(0, layerPolicy.getWindowSize() - (int) newLen);
                int blocksToEvict = (layerCaches[layerIdx].getSequenceLength(seqIdx)
                        - keepTokens + blockSize - 1) / blockSize;
                if (blocksToEvict > 0) {
                    enforceSlidingWindow(layerIdx, seqIdx, blocksToEvict);
                }
            }
        }

        layerCaches[layerIdx].append(seqIdx, newKeys, newValues);
    }

    /**
     * Enforce the sliding window for a specific layer and sequence.
     *
     * <p>Evicts the oldest blocks (with page-table shift and length update) so the
     * sequence stays within the layer's configured window size.</p>
     *
     * @param layerIdx layer index
     * @param seqIdx   sequence batch index
     */
    public void enforceSlidingWindow(int layerIdx, int seqIdx) {
        validateLayerIdx(layerIdx);

        PerLayerKVPolicy.LayerPolicy layerPolicy = policy.getPolicy(layerIdx);
        PagedKVCache cache = layerCaches[layerIdx];

        int currentLen = cache.getSequenceLength(seqIdx);
        int windowSize = layerPolicy.getWindowSize();

        if (currentLen <= windowSize) {
            return; // Within bounds
        }

        int blocksToEvict = (currentLen - windowSize + blockSize - 1) / blockSize;
        enforceSlidingWindow(layerIdx, seqIdx, blocksToEvict);
    }

    /**
     * Evict a specific number of oldest blocks from a layer's sequence. This delegates
     * to {@link PagedKVCache#evictOldestBlocks(int, int)}, which shifts the page table
     * and updates the sequence length, so freed blocks can never be double-freed by a
     * later {@link #freeSequence(int)} and appends always land in valid blocks.
     *
     * @param layerIdx        layer index
     * @param seqIdx          sequence batch index
     * @param blocksToEvict   number of oldest blocks to evict
     * @return the number of blocks actually evicted
     */
    public int enforceSlidingWindow(int layerIdx, int seqIdx, int blocksToEvict) {
        validateLayerIdx(layerIdx);

        PagedKVCache cache = layerCaches[layerIdx];
        int evicted = cache.evictOldestBlocks(seqIdx, blocksToEvict);

        if (evicted > 0 && log.isTraceEnabled()) {
            log.trace("Sliding window eviction: layer={}, seqIdx={}, evicted {} blocks, length now {}",
                    layerIdx, seqIdx, evicted, cache.getSequenceLength(seqIdx));
        }
        return evicted;
    }

    /**
     * Free all blocks for a sequence across all layers.
     *
     * @param seqIdx batch index of the sequence
     */
    public void freeSequence(int seqIdx) {
        for (int layer = 0; layer < layerCaches.length; layer++) {
            layerCaches[layer].freeSequence(seqIdx);
        }
    }

    /**
     * Get the KV cache for a specific layer.
     *
     * @param layerIdx layer index
     * @return the layer's paged KV cache
     */
    public PagedKVCache getLayerCache(int layerIdx) {
        validateLayerIdx(layerIdx);
        return layerCaches[layerIdx];
    }

    /**
     * Get the number of layers.
     */
    public int getNumLayers() {
        return layerCaches.length;
    }

    /**
     * Get the total number of free blocks across all layers.
     */
    public int getTotalFreeBlocks() {
        int total = 0;
        for (PagedKVCache cache : layerCaches) {
            total += cache.getNumFreeBlocks();
        }
        return total;
    }

    /**
     * Get the total number of blocks across all layers.
     */
    public int getTotalBlocks() {
        int total = 0;
        for (PagedKVCache cache : layerCaches) {
            total += cache.getNumBlocks();
        }
        return total;
    }

    @Override
    public void close() {
        for (PagedKVCache cache : layerCaches) {
            cache.close();
        }
        log.info("PerLayerPagedKVCache closed: {} layers", layerCaches.length);
    }

    @Override
    public String toString() {
        int totalUsed = getTotalBlocks() - getTotalFreeBlocks();
        return String.format("PerLayerPagedKVCache[layers=%d, blocks=%d/%d used, blockSize=%d]",
                layerCaches.length, totalUsed, getTotalBlocks(), blockSize);
    }

    // ========== Internal Helpers ==========

    private void validateLayerIdx(int layerIdx) {
        if (layerIdx < 0 || layerIdx >= layerCaches.length) {
            throw new IndexOutOfBoundsException(
                    "Layer index " + layerIdx + " out of range [0, " + layerCaches.length + ")");
        }
    }
}
