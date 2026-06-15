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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Host-side offloader for evicted KV cache blocks with pinned memory and LRU eviction.
 *
 * <p>When GPU memory is constrained, KV cache blocks from idle or low-priority sequences
 * can be evicted to pinned host memory. This class manages a pool of pinned host buffers
 * and provides async-capable copy operations for eviction and restoration.</p>
 *
 * <h3>Two-tier memory hierarchy</h3>
 * <pre>
 * GPU (hot):   [block pool] ← active sequences, fast access
 *                   ↕ evict/restore (async DMA)
 * Host (warm):  [pinned buffers] ← idle sequences, async restore overlapped with decode
 * </pre>
 *
 * <h3>Eviction policy</h3>
 * <p>Uses LRU (Least Recently Used) to select which host-side blocks to discard when
 * the host buffer pool is full. Host eviction is less costly than GPU eviction since
 * the blocks can always be recomputed from the token sequence.</p>
 *
 * <h3>Async overlap</h3>
 * <p>Restoration of a KV cache block from host to GPU is initiated before the sequence
 * needs it (when the scheduler decides to resume a previously-paused sequence).
 * The host-to-GPU DMA transfer overlaps with other sequences' decode steps.</p>
 *
 * <h3>Usage with PagedKVCache</h3>
 * <pre>
 * // Evict blocks when GPU is low
 * for (int blockId : blocksToEvict) {
 *     INDArray gpuKey = pagedCache.getKeyBlock(blockId);
 *     INDArray gpuVal = pagedCache.getValueBlock(blockId);
 *     offloader.evictBlock(blockId, gpuKey, gpuVal);
 *     pagedCache.freeBlock(blockId);
 * }
 *
 * // Restore blocks when sequence resumes
 * for (int blockId : blocksToRestore) {
 *     int newGpuBlockId = pagedCache.allocateBlock();
 *     KVBlockPair restored = offloader.restoreBlock(blockId);
 *     pagedCache.writeBlock(newGpuBlockId, restored.key, restored.value);
 * }
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see PagedKVCache
 * @see KVCachePrefixTree
 */
@Slf4j
public class KVCacheHostOffloader implements AutoCloseable {

    /** Default maximum number of blocks to keep in host memory. */
    public static final int DEFAULT_MAX_HOST_BLOCKS = 1024;

    /** Block dimensions: [blockSize, numKvHeads, headDim]. */
    @Getter
    private final int blockSize;

    @Getter
    private final int numKvHeads;

    @Getter
    private final int headDim;

    @Getter
    private final DataType dataType;

    /** Maximum host blocks before LRU eviction kicks in. */
    @Getter
    private final int maxHostBlocks;

    /**
     * LRU-ordered map of physical block ID to host-side KV block pair.
     * Least-recently-used entries are at the head (insertion order with access reordering).
     */
    private final LinkedHashMap<Integer, KVBlockPair> hostBlocks;

    /** Pool of pre-allocated pinned host buffers for reuse. */
    private final Deque<INDArray> freeKeyBuffers;
    private final Deque<INDArray> freeValueBuffers;

    /** Total number of blocks currently in host memory. */
    @Getter
    private int hostBlockCount;

    // Statistics
    private long evictionCount;
    private long restorationCount;
    private long hostEvictionCount;
    private long evictionTimeMs;
    private long restorationTimeMs;

    /**
     * A key-value block pair stored in host memory.
     */
    public static class KVBlockPair {
        @Getter
        private final INDArray key;

        @Getter
        private final INDArray value;

        /** Original physical block ID on GPU (for tracking). */
        @Getter
        private final int originalBlockId;

        public KVBlockPair(INDArray key, INDArray value, int originalBlockId) {
            this.key = key;
            this.value = value;
            this.originalBlockId = originalBlockId;
        }

        /** Close and free both host arrays. */
        public void close() {
            if (key != null && !key.wasClosed()) {
                key.close();
            }
            if (value != null && !value.wasClosed()) {
                value.close();
            }
        }
    }

    /**
     * Create a KV cache host offloader.
     *
     * @param blockSize     tokens per KV cache block
     * @param numKvHeads    number of KV attention heads
     * @param headDim       dimension per head
     * @param dataType      data type for KV tensors
     * @param maxHostBlocks maximum blocks to keep in host memory
     * @param preallocate   number of host buffer pairs to pre-allocate
     */
    public KVCacheHostOffloader(int blockSize, int numKvHeads, int headDim,
                                DataType dataType, int maxHostBlocks, int preallocate) {
        this.blockSize = blockSize;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.dataType = dataType;
        this.maxHostBlocks = maxHostBlocks;
        this.hostBlockCount = 0;

        // Use access-ordered LinkedHashMap for LRU behavior
        this.hostBlocks = new LinkedHashMap<>(16, 0.75f, true);

        this.freeKeyBuffers = new ArrayDeque<>(preallocate);
        this.freeValueBuffers = new ArrayDeque<>(preallocate);

        // Pre-allocate pinned host buffers
        for (int i = 0; i < preallocate; i++) {
            freeKeyBuffers.push(allocateHostBuffer());
            freeValueBuffers.push(allocateHostBuffer());
        }

        log.info("KVCacheHostOffloader: maxBlocks={}, preallocated={}, blockShape=[{},{},{}], dtype={}",
                maxHostBlocks, preallocate, blockSize, numKvHeads, headDim, dataType);
    }

    /**
     * Create with default settings.
     */
    public KVCacheHostOffloader(int blockSize, int numKvHeads, int headDim, DataType dataType) {
        this(blockSize, numKvHeads, headDim, dataType, DEFAULT_MAX_HOST_BLOCKS, 0);
    }

    /**
     * Evict a KV cache block from GPU to host memory.
     *
     * <p>Copies the key and value tensors for the given block from GPU to pinned host
     * buffers. The caller is responsible for freeing the GPU block after this call.</p>
     *
     * <p>If the host pool is full, the least-recently-used host block is discarded
     * to make room.</p>
     *
     * @param blockId   the physical block ID being evicted
     * @param gpuKey    GPU key tensor for this block, shape [blockSize, numKvHeads, headDim]
     * @param gpuValue  GPU value tensor for this block, same shape
     */
    public void evictBlock(int blockId, INDArray gpuKey, INDArray gpuValue) {
        long startMs = System.currentTimeMillis();

        // Evict LRU host block if at capacity
        if (hostBlockCount >= maxHostBlocks) {
            evictLruHostBlock();
        }

        // Get or allocate host buffers
        INDArray hostKey = getOrAllocateKeyBuffer();
        INDArray hostValue = getOrAllocateValueBuffer();

        // Copy GPU -> Host (this is synchronous; for true async, use CUDA streams)
        hostKey.assign(gpuKey);
        hostValue.assign(gpuValue);

        // Store in LRU map
        KVBlockPair pair = new KVBlockPair(hostKey, hostValue, blockId);
        KVBlockPair existing = hostBlocks.put(blockId, pair);
        if (existing != null) {
            // Block was already in host -- recycle the old buffers
            returnKeyBuffer(existing.getKey());
            returnValueBuffer(existing.getValue());
        } else {
            hostBlockCount++;
        }

        evictionCount++;
        evictionTimeMs += System.currentTimeMillis() - startMs;

        log.trace("Evicted block {} to host (hostBlocks={})", blockId, hostBlockCount);
    }

    /**
     * Restore a KV cache block from host memory back to GPU.
     *
     * <p>Returns the host-side key and value tensors for the given block. The caller
     * should copy these into the GPU block pool and then call {@link #releaseBlock(int)}
     * to return the host buffers to the pool.</p>
     *
     * @param blockId the physical block ID to restore
     * @return the host-side key-value pair, or null if the block is not in host memory
     */
    public KVBlockPair restoreBlock(int blockId) {
        long startMs = System.currentTimeMillis();

        KVBlockPair pair = hostBlocks.get(blockId);
        if (pair == null) {
            log.trace("Block {} not found in host memory", blockId);
            return null;
        }

        restorationCount++;
        restorationTimeMs += System.currentTimeMillis() - startMs;

        log.trace("Restored block {} from host", blockId);
        return pair;
    }

    /**
     * Release a block from host memory after it has been restored to GPU.
     *
     * <p>Returns the host buffers to the free pool for reuse.</p>
     *
     * @param blockId the physical block ID to release from host
     */
    public void releaseBlock(int blockId) {
        KVBlockPair pair = hostBlocks.remove(blockId);
        if (pair != null) {
            returnKeyBuffer(pair.getKey());
            returnValueBuffer(pair.getValue());
            hostBlockCount--;
            log.trace("Released host block {} (hostBlocks={})", blockId, hostBlockCount);
        }
    }

    /**
     * Check whether a block is currently stored in host memory.
     *
     * @param blockId the physical block ID
     * @return true if the block is in host memory
     */
    public boolean hasBlock(int blockId) {
        return hostBlocks.containsKey(blockId);
    }

    /**
     * Get performance statistics.
     */
    public String getStats() {
        double avgEvictMs = evictionCount > 0 ? (double) evictionTimeMs / evictionCount : 0;
        double avgRestoreMs = restorationCount > 0 ? (double) restorationTimeMs / restorationCount : 0;
        return String.format(
                "KVCacheHostOffloader[hosted=%d/%d, evictions=%d(avg=%.1fms), restorations=%d(avg=%.1fms), hostEvictions=%d, freeBuffers=%d]",
                hostBlockCount, maxHostBlocks, evictionCount, avgEvictMs,
                restorationCount, avgRestoreMs, hostEvictionCount,
                freeKeyBuffers.size());
    }

    @Override
    public void close() {
        // Close all hosted blocks
        for (KVBlockPair pair : hostBlocks.values()) {
            pair.close();
        }
        hostBlocks.clear();

        // Close free buffer pools
        while (!freeKeyBuffers.isEmpty()) {
            freeKeyBuffers.pop().close();
        }
        while (!freeValueBuffers.isEmpty()) {
            freeValueBuffers.pop().close();
        }

        hostBlockCount = 0;
        log.info("KVCacheHostOffloader closed. Final stats: {}", getStats());
    }

    // ========== Internal Helpers ==========

    /**
     * Evict the least-recently-used block from host memory.
     */
    private void evictLruHostBlock() {
        if (hostBlocks.isEmpty()) return;

        // LinkedHashMap with access-order: first entry is the LRU
        Map.Entry<Integer, KVBlockPair> lruEntry = hostBlocks.entrySet().iterator().next();
        int lruBlockId = lruEntry.getKey();
        KVBlockPair lruPair = lruEntry.getValue();

        hostBlocks.remove(lruBlockId);
        returnKeyBuffer(lruPair.getKey());
        returnValueBuffer(lruPair.getValue());
        hostBlockCount--;
        hostEvictionCount++;

        log.trace("Host-evicted LRU block {} (hostBlocks={})", lruBlockId, hostBlockCount);
    }

    /**
     * Allocate a pinned host buffer for one block's key or value data.
     */
    private INDArray allocateHostBuffer() {
        // In production, this would use cudaMallocHost for truly pinned memory.
        // With ND4J, Nd4j.createUninitialized on the host is the closest equivalent.
        return Nd4j.zeros(dataType, blockSize, numKvHeads, headDim);
    }

    private INDArray getOrAllocateKeyBuffer() {
        return freeKeyBuffers.isEmpty() ? allocateHostBuffer() : freeKeyBuffers.pop();
    }

    private INDArray getOrAllocateValueBuffer() {
        return freeValueBuffers.isEmpty() ? allocateHostBuffer() : freeValueBuffers.pop();
    }

    private void returnKeyBuffer(INDArray buffer) {
        if (buffer != null && !buffer.wasClosed()) {
            freeKeyBuffers.push(buffer);
        }
    }

    private void returnValueBuffer(INDArray buffer) {
        if (buffer != null && !buffer.wasClosed()) {
            freeValueBuffers.push(buffer);
        }
    }
}
