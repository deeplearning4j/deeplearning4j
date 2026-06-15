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
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * PagedKVCache with token-level eviction support.
 *
 * <p>Extends {@link PagedKVCache} with the ability to evict individual tokens from
 * the cache based on a configurable {@link TokenEvictionPolicy}. When the cache
 * exceeds a token budget, the eviction policy selects which tokens to remove, and
 * this class handles the physical removal and block compaction.</p>
 *
 * <h3>Eviction flow</h3>
 * <ol>
 *   <li>Tokens are appended normally via {@link #append(int, INDArray, INDArray)}</li>
 *   <li>After appending, if the sequence length exceeds the budget and a policy is set,
 *       the policy is consulted for tokens to evict</li>
 *   <li>Evicted tokens are removed from their blocks, remaining tokens shift down</li>
 *   <li>Partially-empty blocks are compacted to free whole blocks back to the pool</li>
 * </ol>
 *
 * <h3>Block compaction</h3>
 * <p>After removing tokens from scattered positions, some blocks may have gaps.
 * {@link #compactBlocks(int)} consolidates tokens from partially-filled blocks
 * into fewer fully-packed blocks, freeing the emptied blocks back to the pool.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see PagedKVCache
 * @see TokenEvictionPolicy
 * @see H2OEvictionPolicy
 * @see StreamingLLMEvictionPolicy
 */
@Slf4j
public class EvictablePagedKVCache extends PagedKVCache {

    /** The eviction policy, or null if eviction is disabled. */
    @Getter @Setter
    private TokenEvictionPolicy evictionPolicy;

    /** Token budget per sequence. When exceeded, eviction is triggered. */
    @Getter @Setter
    private int tokenBudget;

    /**
     * Per-block valid token count. Tracks how many of the {@code blockSize} slots in
     * each physical block actually contain valid tokens. Indexed by physical block ID.
     */
    private final int[] blockValidCounts;

    /**
     * Create an evictable paged KV cache.
     *
     * @param maxBatchSize   maximum number of concurrent sequences
     * @param maxSeqLen      maximum sequence length
     * @param numKvHeads     number of KV heads
     * @param headDim        dimension per head
     * @param blockSize      tokens per block
     * @param dataType       data type for KV tensors
     * @param poolSizeFactor multiplier on pool size for headroom
     * @param tokenBudget    maximum tokens per sequence before eviction triggers
     */
    public EvictablePagedKVCache(int maxBatchSize, int maxSeqLen, int numKvHeads, int headDim,
                                  int blockSize, DataType dataType, double poolSizeFactor,
                                  int tokenBudget) {
        super(maxBatchSize, maxSeqLen, numKvHeads, headDim, blockSize, dataType, poolSizeFactor);
        this.tokenBudget = tokenBudget;
        this.evictionPolicy = null;
        this.blockValidCounts = new int[getNumBlocks()];
    }

    /**
     * Create with default block size and 1.2x pool headroom.
     */
    public EvictablePagedKVCache(int maxBatchSize, int maxSeqLen, int numKvHeads, int headDim,
                                  DataType dataType, int tokenBudget) {
        super(maxBatchSize, maxSeqLen, numKvHeads, headDim, dataType);
        this.tokenBudget = tokenBudget;
        this.evictionPolicy = null;
        this.blockValidCounts = new int[getNumBlocks()];
    }

    /**
     * Append new KV entries for a sequence, triggering eviction if over budget.
     *
     * <p>After the normal append, if the sequence length exceeds {@code tokenBudget}
     * and an eviction policy is set, tokens are evicted and blocks are compacted.</p>
     *
     * @param seqIdx   batch index of the sequence
     * @param newKeys  new key tensor [1, newLen, numKvHeads, headDim] or [newLen, numKvHeads, headDim]
     * @param newValues new value tensor (same shape as newKeys)
     */
    @Override
    public void append(int seqIdx, INDArray newKeys, INDArray newValues) {
        super.append(seqIdx, newKeys, newValues);

        // Update block valid counts for newly written blocks
        updateBlockValidCounts(seqIdx);

        // Check if eviction is needed
        int currentLen = getSequenceLength(seqIdx);
        if (evictionPolicy != null && currentLen > tokenBudget) {
            int[] toEvict = evictionPolicy.selectTokensToEvict(0, currentLen, tokenBudget);
            if (toEvict.length > 0) {
                evictTokens(seqIdx, toEvict);
            }
        }
    }

    /**
     * Evict specific token positions from a sequence's KV cache.
     *
     * <p>Removes the tokens at the given positions by shifting remaining tokens
     * within their blocks, then compacts partially-empty blocks to reclaim memory.</p>
     *
     * @param seqIdx    batch index of the sequence
     * @param positions sorted array of token positions to evict (0-based, ascending)
     */
    public void evictTokens(int seqIdx, int[] positions) {
        if (positions.length == 0) return;
        if (seqIdx < 0 || seqIdx >= getMaxBatchSize()) {
            throw new IllegalArgumentException("seqIdx out of range: " + seqIdx);
        }

        // Sort positions to process in order
        int[] sorted = positions.clone();
        Arrays.sort(sorted);

        log.debug("Evicting {} tokens from seq {}: positions {}", sorted.length, seqIdx,
                sorted.length <= 10 ? Arrays.toString(sorted) : sorted.length + " positions");

        int[] pageTable = getRawPageTable(seqIdx);

        // Remove tokens from blocks, working backwards to preserve indices
        for (int i = sorted.length - 1; i >= 0; i--) {
            int pos = sorted[i];
            int logicalBlock = pos / getBlockSize();
            int offsetInBlock = pos % getBlockSize();

            if (logicalBlock < pageTable.length && pageTable[logicalBlock] >= 0) {
                int blockId = pageTable[logicalBlock];
                removeTokenFromBlock(blockId, offsetInBlock);
            }
        }

        // Update sequence length
        setSequenceLength(seqIdx, getSequenceLength(seqIdx) - sorted.length);

        // Compact blocks to reclaim memory
        compactBlocks(seqIdx);

        log.debug("After eviction: seq {} length={}, freeBlocks={}", seqIdx,
                getSequenceLength(seqIdx), getNumFreeBlocks());
    }

    /**
     * Compact partially-empty blocks for a sequence.
     *
     * <p>Walks through the sequence's blocks and consolidates tokens from
     * partially-filled blocks into earlier blocks that have space. Fully emptied
     * blocks are freed back to the pool.</p>
     *
     * @param seqIdx batch index of the sequence
     */
    public void compactBlocks(int seqIdx) {
        if (seqIdx < 0 || seqIdx >= getMaxBatchSize()) return;

        int blockSize = getBlockSize();
        int[] pageTable = getRawPageTable(seqIdx);

        // Collect all valid tokens from all blocks into a flat list of (key, value) pairs
        List<INDArray> keys = new ArrayList<>();
        List<INDArray> values = new ArrayList<>();

        for (int b = 0; b < pageTable.length; b++) {
            int blockId = pageTable[b];
            if (blockId < 0) break;

            int validCount = blockValidCounts[blockId];
            for (int t = 0; t < validCount; t++) {
                INDArray keySlice = getKeyBlockPool().get(NDArrayIndex.point(blockId),
                        NDArrayIndex.point(t), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray valSlice = getValueBlockPool().get(NDArrayIndex.point(blockId),
                        NDArrayIndex.point(t), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                keys.add(keySlice);
                values.add(valSlice);
            }
        }

        // Free all current blocks
        for (int b = 0; b < pageTable.length; b++) {
            int blockId = pageTable[b];
            if (blockId < 0) break;
            blockValidCounts[blockId] = 0;
            freeBlock(blockId);
            pageTable[b] = -1;
        }

        // Re-pack tokens into fresh blocks
        int tokenIdx = 0;
        int numTokens = keys.size();
        int logicalBlock = 0;

        while (tokenIdx < numTokens) {
            int blockId = allocateBlock();
            pageTable[logicalBlock] = blockId;

            int tokensInBlock = Math.min(blockSize, numTokens - tokenIdx);
            blockValidCounts[blockId] = tokensInBlock;

            for (int t = 0; t < tokensInBlock; t++) {
                INDArray keyDest = getKeyBlockPool().get(NDArrayIndex.point(blockId),
                        NDArrayIndex.point(t), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray valDest = getValueBlockPool().get(NDArrayIndex.point(blockId),
                        NDArrayIndex.point(t), NDArrayIndex.all(), NDArrayIndex.all());
                keyDest.assign(keys.get(tokenIdx));
                valDest.assign(values.get(tokenIdx));
                tokenIdx++;
            }

            logicalBlock++;
        }

        // Close temporary dup arrays
        for (INDArray k : keys) k.close();
        for (INDArray v : values) v.close();

        log.trace("Compacted seq {}: {} tokens in {} blocks", seqIdx, numTokens, logicalBlock);
    }

    /**
     * Remove a single token from a physical block by shifting remaining tokens down.
     *
     * @param blockId       physical block ID
     * @param offsetInBlock position within the block to remove (0-based)
     */
    private void removeTokenFromBlock(int blockId, int offsetInBlock) {
        int validCount = blockValidCounts[blockId];
        if (offsetInBlock >= validCount) return;

        // Shift tokens after the removed position down by one
        for (int t = offsetInBlock; t < validCount - 1; t++) {
            INDArray keySrc = getKeyBlockPool().get(NDArrayIndex.point(blockId),
                    NDArrayIndex.point(t + 1), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray keyDest = getKeyBlockPool().get(NDArrayIndex.point(blockId),
                    NDArrayIndex.point(t), NDArrayIndex.all(), NDArrayIndex.all());
            keyDest.assign(keySrc);

            INDArray valSrc = getValueBlockPool().get(NDArrayIndex.point(blockId),
                    NDArrayIndex.point(t + 1), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray valDest = getValueBlockPool().get(NDArrayIndex.point(blockId),
                    NDArrayIndex.point(t), NDArrayIndex.all(), NDArrayIndex.all());
            valDest.assign(valSrc);
        }

        blockValidCounts[blockId] = validCount - 1;
    }

    /**
     * Update block valid counts based on current sequence lengths.
     */
    private void updateBlockValidCounts(int seqIdx) {
        int blockSize = getBlockSize();
        int seqLen = getSequenceLength(seqIdx);
        int[] pageTable = getRawPageTable(seqIdx);

        for (int b = 0; b < pageTable.length; b++) {
            int blockId = pageTable[b];
            if (blockId < 0) break;

            int blockStartPos = b * blockSize;
            int blockEndPos = Math.min(blockStartPos + blockSize, seqLen);
            blockValidCounts[blockId] = blockEndPos - blockStartPos;
        }
    }

    @Override
    public String toString() {
        return String.format("EvictablePagedKVCache[budget=%d, policy=%s, %s]",
                tokenBudget, evictionPolicy != null ? evictionPolicy.getClass().getSimpleName() : "none",
                super.toString());
    }
}
