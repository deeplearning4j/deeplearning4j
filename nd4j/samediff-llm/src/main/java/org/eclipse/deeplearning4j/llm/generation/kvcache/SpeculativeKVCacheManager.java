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
import org.eclipse.deeplearning4j.llm.generation.speculative.SpeculativeDecodeLoop;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Zero-copy KV cache management for speculative decoding.
 *
 * <p>Speculative decoding proposes multiple tokens at once and verifies them
 * against the target model. If verification fails at position K, all tokens
 * after K must be rolled back. This class provides efficient checkpoint and
 * rollback operations for both static and paged KV caches.</p>
 *
 * <h3>Static KV cache insight</h3>
 * <p>For static KV caches, rollback is zero-copy: simply resetting the
 * cachePosition pointer is sufficient because positions beyond cachePos are
 * masked out by the attention mask. Any stale data in those positions is
 * harmless and will be overwritten on the next append.</p>
 *
 * <h3>Paged KV cache rollback</h3>
 * <p>For paged KV caches, rollback requires freeing blocks that were allocated
 * after the checkpoint. The page table is restored to its checkpointed state,
 * and newly allocated blocks are returned to the free pool.</p>
 *
 * <h3>Tree-based speculative decoding</h3>
 * <p>The {@link #compactAcceptedPath(PagedKVCache, int, int[])} method supports
 * tree-structured speculation where multiple candidate branches are evaluated
 * in parallel. After verification, only the accepted path's blocks are kept
 * and others are freed.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see SpeculativeDecodeLoop
 * @see PagedKVCache
 */
@Slf4j
public class SpeculativeKVCacheManager {

    /** Counter for generating unique checkpoint IDs. */
    private final AtomicInteger nextCheckpointId = new AtomicInteger(0);

    /** Active checkpoints for static KV caches. */
    private final Map<Integer, StaticCheckpoint> staticCheckpoints = new HashMap<>();

    /** Active checkpoints for paged KV caches. */
    private final Map<Integer, PagedCheckpoint> pagedCheckpoints = new HashMap<>();

    // Statistics
    @Getter
    private long checkpointCount;
    @Getter
    private long rollbackCount;
    @Getter
    private long acceptCount;

    /**
     * Checkpoint for static KV caches.
     * Only stores the cache position -- zero data copy.
     */
    private static class StaticCheckpoint {
        final Map<String, INDArray> kvBuffers;
        final int cachePosition;
        final long createdAtNanos;

        StaticCheckpoint(Map<String, INDArray> kvBuffers, int cachePosition) {
            this.kvBuffers = kvBuffers;
            this.cachePosition = cachePosition;
            this.createdAtNanos = System.nanoTime();
        }
    }

    /**
     * Checkpoint for paged KV caches.
     * Stores the page table and context length at checkpoint time.
     */
    private static class PagedCheckpoint {
        final int seqIdx;
        final int[] pageTableSnapshot;
        final int contextLength;
        final int numAllocatedBlocks;
        final long createdAtNanos;

        PagedCheckpoint(int seqIdx, int[] pageTableSnapshot, int contextLength, int numAllocatedBlocks) {
            this.seqIdx = seqIdx;
            this.pageTableSnapshot = pageTableSnapshot;
            this.contextLength = contextLength;
            this.numAllocatedBlocks = numAllocatedBlocks;
            this.createdAtNanos = System.nanoTime();
        }
    }

    /**
     * Create a new speculative KV cache manager.
     */
    public SpeculativeKVCacheManager() {
        // Default constructor
    }

    /**
     * Create a checkpoint for a static KV cache.
     *
     * <p>This is a zero-copy operation: only the cache position is recorded.
     * The KV buffer references are stored for rollback, but no data is copied.</p>
     *
     * @param staticKvBuffers the static KV cache buffers (references, not copies)
     * @param cachePosition   the current cache position
     * @return a checkpoint ID for later use with accept/rollback
     */
    public int checkpoint(Map<String, INDArray> staticKvBuffers, int cachePosition) {
        int id = nextCheckpointId.getAndIncrement();
        staticCheckpoints.put(id, new StaticCheckpoint(staticKvBuffers, cachePosition));
        checkpointCount++;

        log.trace("Static checkpoint {}: cachePosition={}", id, cachePosition);
        return id;
    }

    /**
     * Create a checkpoint for a paged KV cache.
     *
     * <p>Snapshots the page table and context length. Block data is not copied;
     * on rollback, newly allocated blocks are simply freed.</p>
     *
     * @param pagedCache the paged KV cache
     * @param seqIdx     the sequence index to checkpoint
     * @return a checkpoint ID
     */
    public int checkpoint(PagedKVCache pagedCache, int seqIdx) {
        int id = nextCheckpointId.getAndIncrement();

        // Snapshot the page table
        INDArray pageTableArr = pagedCache.getPageTableArray(seqIdx);
        int[] pageTable = pageTableArr.toIntVector();
        pageTableArr.close();

        int contextLen = pagedCache.getSequenceLength(seqIdx);
        int numAllocated = pagedCache.getNumAllocatedBlocks(seqIdx);

        pagedCheckpoints.put(id, new PagedCheckpoint(seqIdx, pageTable, contextLen, numAllocated));
        checkpointCount++;

        log.trace("Paged checkpoint {}: seqIdx={}, contextLen={}, blocks={}", id, seqIdx, contextLen, numAllocated);
        return id;
    }

    /**
     * Accept tokens after successful verification of speculative tokens.
     *
     * <p>For static KV caches, this simply advances the implicit cache position
     * by the number of accepted tokens. The checkpoint is consumed and removed.</p>
     *
     * @param checkpointId the checkpoint to advance from
     * @param numAccepted  number of tokens accepted
     * @return the new cache position after acceptance
     */
    public int acceptTokens(int checkpointId, int numAccepted) {
        StaticCheckpoint cp = staticCheckpoints.remove(checkpointId);
        if (cp != null) {
            int newPosition = cp.cachePosition + numAccepted;
            acceptCount++;
            log.trace("Accepted {} tokens from static checkpoint {}: {} -> {}",
                    numAccepted, checkpointId, cp.cachePosition, newPosition);
            return newPosition;
        }

        PagedCheckpoint pcp = pagedCheckpoints.remove(checkpointId);
        if (pcp != null) {
            int newPosition = pcp.contextLength + numAccepted;
            acceptCount++;
            log.trace("Accepted {} tokens from paged checkpoint {}: {} -> {}",
                    numAccepted, checkpointId, pcp.contextLength, newPosition);
            return newPosition;
        }

        throw new IllegalArgumentException("No checkpoint found with ID: " + checkpointId);
    }

    /**
     * Rollback a static KV cache to a checkpoint.
     *
     * <p>This is a zero-copy operation for static KV caches: only the cache
     * position is rolled back. Positions beyond the checkpoint position are
     * masked by the attention mask, so stale data is harmless.</p>
     *
     * @param checkpointId the checkpoint to roll back to
     * @return the rolled-back cache position
     */
    public int rollback(int checkpointId) {
        StaticCheckpoint cp = staticCheckpoints.remove(checkpointId);
        if (cp == null) {
            throw new IllegalArgumentException("No static checkpoint found with ID: " + checkpointId);
        }

        rollbackCount++;
        log.trace("Rolled back static checkpoint {} to position {}", checkpointId, cp.cachePosition);
        return cp.cachePosition;
    }

    /**
     * Rollback a paged KV cache to a checkpoint.
     *
     * <p>Frees any blocks that were allocated after the checkpoint was taken.
     * Restores the page table and context length to their checkpointed state.</p>
     *
     * @param checkpointId the checkpoint to roll back to
     * @param cache        the paged KV cache to rollback
     * @param seqIdx       the sequence index (must match the checkpoint)
     */
    public void rollbackPaged(int checkpointId, PagedKVCache cache, int seqIdx) {
        PagedCheckpoint cp = pagedCheckpoints.remove(checkpointId);
        if (cp == null) {
            throw new IllegalArgumentException("No paged checkpoint found with ID: " + checkpointId);
        }
        if (cp.seqIdx != seqIdx) {
            throw new IllegalArgumentException("Checkpoint seqIdx (" + cp.seqIdx +
                    ") does not match requested seqIdx (" + seqIdx + ")");
        }

        // Find blocks allocated after the checkpoint
        int currentAllocated = cache.getNumAllocatedBlocks(seqIdx);
        INDArray currentPageTableArr = cache.getPageTableArray(seqIdx);
        int[] currentPageTable = currentPageTableArr.toIntVector();
        currentPageTableArr.close();

        // Free blocks that were allocated after the checkpoint (refcount-aware:
        // shared blocks only decrement, never return to the pool while referenced)
        for (int b = cp.numAllocatedBlocks; b < currentAllocated; b++) {
            if (b < currentPageTable.length && currentPageTable[b] >= 0) {
                cache.freeBlock(currentPageTable[b]);
                log.trace("Freed post-checkpoint block {} (physical={})", b, currentPageTable[b]);
            }
        }

        // Restore page table
        for (int b = 0; b < cache.pageTables[seqIdx].length; b++) {
            if (b < cp.pageTableSnapshot.length) {
                cache.pageTables[seqIdx][b] = cp.pageTableSnapshot[b];
            } else {
                cache.pageTables[seqIdx][b] = -1;
            }
        }

        // Restore context length
        cache.seqLengths[seqIdx] = cp.contextLength;

        rollbackCount++;
        log.trace("Rolled back paged checkpoint {} for seq {}: blocks {} -> {}, len {} -> {}",
                checkpointId, seqIdx, currentAllocated, cp.numAllocatedBlocks,
                cache.getSequenceLength(seqIdx), cp.contextLength);
    }

    /**
     * Compact a paged KV cache to keep only the accepted path's blocks.
     *
     * <p>For tree-structured speculative decoding, multiple candidate branches
     * may have been allocated in the KV cache. After verification determines
     * the accepted path, this method frees blocks that belong to rejected branches
     * and compacts the accepted path into contiguous blocks.</p>
     *
     * @param cache             the paged KV cache
     * @param seqIdx            the sequence index
     * @param acceptedPositions the token positions in the accepted path (sorted ascending)
     */
    public void compactAcceptedPath(PagedKVCache cache, int seqIdx, int[] acceptedPositions) {
        if (acceptedPositions == null || acceptedPositions.length == 0) {
            return;
        }

        // Determine which blocks contain accepted positions
        java.util.Set<Integer> acceptedBlocks = new java.util.HashSet<>();
        for (int pos : acceptedPositions) {
            int logicalBlock = pos / cache.getBlockSize();
            acceptedBlocks.add(logicalBlock);
        }

        // Free blocks that don't contain any accepted positions (refcount-aware)
        int numAllocated = cache.getNumAllocatedBlocks(seqIdx);
        for (int b = 0; b < numAllocated; b++) {
            if (!acceptedBlocks.contains(b)) {
                int blockId = cache.pageTables[seqIdx][b];
                if (blockId >= 0) {
                    cache.freeBlock(blockId);
                    cache.pageTables[seqIdx][b] = -1;
                    log.trace("Freed rejected branch block {} (logical={}, physical={})", b, b, blockId);
                }
            }
        }

        // Update sequence length to match accepted positions
        if (acceptedPositions.length > 0) {
            cache.seqLengths[seqIdx] = acceptedPositions[acceptedPositions.length - 1] + 1;
        }

        log.trace("Compacted accepted path for seq {}: {} accepted blocks of {} total",
                seqIdx, acceptedBlocks.size(), numAllocated);
    }

    /**
     * Check if a checkpoint ID exists (either static or paged).
     *
     * @param checkpointId the checkpoint ID
     * @return true if the checkpoint exists
     */
    public boolean hasCheckpoint(int checkpointId) {
        return staticCheckpoints.containsKey(checkpointId) || pagedCheckpoints.containsKey(checkpointId);
    }

    /**
     * Discard a checkpoint without rolling back.
     *
     * @param checkpointId the checkpoint to discard
     */
    public void discardCheckpoint(int checkpointId) {
        staticCheckpoints.remove(checkpointId);
        pagedCheckpoints.remove(checkpointId);
    }

    /**
     * Get the number of active checkpoints.
     */
    public int getActiveCheckpointCount() {
        return staticCheckpoints.size() + pagedCheckpoints.size();
    }

    /**
     * Get performance statistics.
     */
    public String getStats() {
        return String.format(
                "SpeculativeKVCacheManager[active=%d, checkpoints=%d, accepts=%d, rollbacks=%d]",
                getActiveCheckpointCount(), checkpointCount, acceptCount, rollbackCount);
    }

    @Override
    public String toString() {
        return getStats();
    }
}
