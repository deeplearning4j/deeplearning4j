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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Paged KV cache for efficient memory management during autoregressive generation.
 *
 * <p>Instead of pre-allocating dense [batch, maxSeqLen, heads, headDim] buffers per sequence,
 * this cache uses fixed-size blocks that are allocated on demand from a shared pool.
 * Each sequence maintains a page table mapping logical block indices to physical blocks.</p>
 *
 * <h3>Block layout</h3>
 * <p>Each block stores {@code blockSize} token positions for all KV heads:
 * {@code [blockSize, numKvHeads, headDim]} for both key and value.
 * The block pool is a single pre-allocated tensor:
 * {@code [numBlocks, blockSize, numKvHeads, headDim]} for keys and values separately.</p>
 *
 * <h3>Page table</h3>
 * <p>Each sequence has an {@code int[]} mapping logical block index → physical block ID.
 * Logical block 0 covers tokens 0..blockSize-1, block 1 covers blockSize..2*blockSize-1, etc.
 * The attention kernel uses this table to gather K/V from scattered physical blocks.</p>
 *
 * <h3>Benefits</h3>
 * <ul>
 *   <li>No per-sequence pre-allocation of maxSeqLen — blocks allocated as tokens arrive</li>
 *   <li>Finished sequences return blocks to the pool — immediate reuse without copying</li>
 *   <li>Slot reassignment for continuous batching is O(1) — just swap page tables</li>
 *   <li>Memory fragmentation is bounded by block size (typically 64 tokens)</li>
 * </ul>
 */
@Slf4j
public class PagedKVCache implements AutoCloseable {

    /** Default block size (tokens per block). Trades granularity vs overhead. */
    public static final int DEFAULT_BLOCK_SIZE = 64;

    @Getter private final int blockSize;
    @Getter private final int numKvHeads;
    @Getter private final int headDim;
    @Getter private final int maxBatchSize;
    @Getter private final int numBlocks;
    @Getter private final DataType dataType;

    // Block pool: pre-allocated GPU tensors holding all blocks
    // Shape: [numBlocks, blockSize, numKvHeads, headDim]
    private final INDArray keyBlockPool;
    private final INDArray valueBlockPool;

    // Free block IDs (LIFO for cache locality)
    protected final Deque<Integer> freeBlocks;

    // Per-sequence page tables: pageTable[seqIdx] = int[] of physical block IDs
    // -1 means no block allocated for that logical position
    protected final int[][] pageTables;

    // Per-sequence current length (number of valid tokens)
    protected final int[] seqLengths;

    // Maximum logical blocks per sequence
    private final int maxBlocksPerSeq;

    /**
     * Create a paged KV cache.
     *
     * @param maxBatchSize   maximum number of concurrent sequences
     * @param maxSeqLen      maximum sequence length (determines max blocks per sequence)
     * @param numKvHeads     number of KV heads (supports GQA)
     * @param headDim        dimension per head
     * @param blockSize      tokens per block
     * @param dataType       data type for KV tensors
     * @param poolSizeFactor multiplier on total needed blocks for pool headroom (e.g., 1.2)
     */
    public PagedKVCache(int maxBatchSize, int maxSeqLen, int numKvHeads, int headDim,
                        int blockSize, DataType dataType, double poolSizeFactor) {
        this.blockSize = blockSize;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.maxBatchSize = maxBatchSize;
        this.dataType = dataType;

        this.maxBlocksPerSeq = (maxSeqLen + blockSize - 1) / blockSize;

        // Pool size: enough for all sequences at max length, with headroom
        int minBlocks = maxBatchSize * maxBlocksPerSeq;
        this.numBlocks = (int) Math.ceil(minBlocks * poolSizeFactor);

        log.info("PagedKVCache: {} blocks of {}×{}×{} ({} per block), pool total {} MB",
                numBlocks, blockSize, numKvHeads, headDim,
                blockSize * numKvHeads * headDim,
                2L * numBlocks * blockSize * numKvHeads * headDim * dataType.width() / (1024 * 1024));

        // Allocate block pools
        this.keyBlockPool = Nd4j.zeros(dataType, numBlocks, blockSize, numKvHeads, headDim);
        this.valueBlockPool = Nd4j.zeros(dataType, numBlocks, blockSize, numKvHeads, headDim);

        // Initialize free list (all blocks free)
        this.freeBlocks = new ArrayDeque<>(numBlocks);
        for (int i = numBlocks - 1; i >= 0; i--) {
            freeBlocks.push(i);
        }

        // Initialize page tables
        this.pageTables = new int[maxBatchSize][];
        this.seqLengths = new int[maxBatchSize];
        for (int i = 0; i < maxBatchSize; i++) {
            pageTables[i] = new int[maxBlocksPerSeq];
            java.util.Arrays.fill(pageTables[i], -1);
        }
    }

    /**
     * Create with default block size and 1.2x pool headroom.
     */
    public PagedKVCache(int maxBatchSize, int maxSeqLen, int numKvHeads, int headDim, DataType dataType) {
        this(maxBatchSize, maxSeqLen, numKvHeads, headDim, DEFAULT_BLOCK_SIZE, dataType, 1.2);
    }

    /**
     * Append new KV entries for a sequence.
     *
     * @param seqIdx   batch index of the sequence
     * @param newKeys  new key tensor [1, newLen, numKvHeads, headDim] or [newLen, numKvHeads, headDim]
     * @param newValues new value tensor (same shape as newKeys)
     */
    public void append(int seqIdx, INDArray newKeys, INDArray newValues) {
        if (seqIdx < 0 || seqIdx >= maxBatchSize) {
            throw new IllegalArgumentException("seqIdx out of range: " + seqIdx);
        }

        long newLen = newKeys.rank() == 4 ? newKeys.size(1) : newKeys.size(0);
        int startPos = seqLengths[seqIdx];

        for (int t = 0; t < newLen; t++) {
            int pos = startPos + t;
            int logicalBlock = pos / blockSize;
            int offsetInBlock = pos % blockSize;

            if (logicalBlock >= maxBlocksPerSeq) {
                throw new IllegalStateException("Sequence " + seqIdx + " exceeded max blocks (" + maxBlocksPerSeq + ")");
            }

            // Allocate block if needed
            if (pageTables[seqIdx][logicalBlock] < 0) {
                pageTables[seqIdx][logicalBlock] = allocateBlock();
            }

            int physicalBlock = pageTables[seqIdx][logicalBlock];

            // Copy single token's KV into the block
            // Source: newKeys[t, :, :] or newKeys[0, t, :, :]
            INDArray keyToken;
            INDArray valToken;
            if (newKeys.rank() == 4) {
                keyToken = newKeys.get(NDArrayIndex.point(0), NDArrayIndex.point(t),
                        NDArrayIndex.all(), NDArrayIndex.all());
                valToken = newValues.get(NDArrayIndex.point(0), NDArrayIndex.point(t),
                        NDArrayIndex.all(), NDArrayIndex.all());
            } else {
                keyToken = newKeys.get(NDArrayIndex.point(t), NDArrayIndex.all(), NDArrayIndex.all());
                valToken = newValues.get(NDArrayIndex.point(t), NDArrayIndex.all(), NDArrayIndex.all());
            }

            // Dest: blockPool[physicalBlock, offsetInBlock, :, :]
            INDArray keyDest = keyBlockPool.get(NDArrayIndex.point(physicalBlock),
                    NDArrayIndex.point(offsetInBlock), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray valDest = valueBlockPool.get(NDArrayIndex.point(physicalBlock),
                    NDArrayIndex.point(offsetInBlock), NDArrayIndex.all(), NDArrayIndex.all());

            keyDest.assign(keyToken);
            valDest.assign(valToken);
        }

        seqLengths[seqIdx] += (int) newLen;
    }

    /**
     * Free all blocks for a sequence, returning them to the pool.
     * Call when a sequence finishes generation.
     *
     * @param seqIdx batch index of the finished sequence
     */
    public void freeSequence(int seqIdx) {
        if (seqIdx < 0 || seqIdx >= maxBatchSize) return;

        for (int b = 0; b < maxBlocksPerSeq; b++) {
            int blockId = pageTables[seqIdx][b];
            if (blockId >= 0) {
                freeBlocks.push(blockId);
                pageTables[seqIdx][b] = -1;
            }
        }
        seqLengths[seqIdx] = 0;
    }

    /**
     * Get the raw page table array for a sequence.
     *
     * @param seqIdx batch index
     * @return int[] of physical block IDs (includes -1 for unallocated slots)
     */
    protected int[] getRawPageTable(int seqIdx) {
        return pageTables[seqIdx];
    }

    /**
     * Get the page table for a sequence as an INDArray (for passing to native ops).
     *
     * @param seqIdx batch index
     * @return int32 tensor of shape [numAllocatedBlocks] with physical block IDs
     */
    public INDArray getPageTableArray(int seqIdx) {
        int numAllocated = getNumAllocatedBlocks(seqIdx);
        int[] table = new int[numAllocated];
        System.arraycopy(pageTables[seqIdx], 0, table, 0, numAllocated);
        return Nd4j.createFromArray(table);
    }

    /**
     * Get all page tables for the active batch as a padded 2D tensor.
     *
     * @param batchSize number of active sequences
     * @return int32 tensor [batchSize, maxBlocksPerSeq] with physical block IDs (-1 for unallocated)
     */
    public INDArray getAllPageTables(int batchSize) {
        int[][] tables = new int[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            tables[i] = pageTables[i].clone();
        }
        return Nd4j.createFromArray(tables);
    }

    /**
     * Get context lengths for the active batch.
     *
     * @param batchSize number of active sequences
     * @return int32 tensor [batchSize] with current sequence lengths
     */
    public INDArray getContextLengths(int batchSize) {
        int[] lens = new int[batchSize];
        System.arraycopy(seqLengths, 0, lens, 0, batchSize);
        return Nd4j.createFromArray(lens);
    }

    /**
     * Get the key block pool tensor (for passing to native attention kernel).
     */
    public INDArray getKeyBlockPool() {
        return keyBlockPool;
    }

    /**
     * Get the value block pool tensor.
     */
    public INDArray getValueBlockPool() {
        return valueBlockPool;
    }

    /**
     * Current sequence length for a batch index.
     */
    public int getSequenceLength(int seqIdx) {
        return seqLengths[seqIdx];
    }

    /**
     * Number of free blocks remaining in the pool.
     */
    public int getNumFreeBlocks() {
        return freeBlocks.size();
    }

    /**
     * Number of blocks allocated for a sequence.
     */
    public int getNumAllocatedBlocks(int seqIdx) {
        int count = 0;
        for (int b = 0; b < maxBlocksPerSeq; b++) {
            if (pageTables[seqIdx][b] >= 0) count++;
            else break;  // Blocks are allocated contiguously
        }
        return count;
    }

    /**
     * Reassign a slot from a finished sequence to a new sequence.
     * Frees the old sequence's blocks and resets state for the new one.
     *
     * @param slotIdx the batch slot to reassign
     */
    public void reassignSlot(int slotIdx) {
        freeSequence(slotIdx);
        // Slot is now clean — ready for append()
    }

    protected int allocateBlock() {
        if (freeBlocks.isEmpty()) {
            throw new IllegalStateException("PagedKVCache: no free blocks (pool exhausted). "
                    + "Consider increasing maxSeqLen or poolSizeFactor.");
        }
        return freeBlocks.pop();
    }

    protected void freeBlock(int blockId) {
        freeBlocks.push(blockId);
    }

    protected void setSequenceLength(int seqIdx, int length) {
        seqLengths[seqIdx] = length;
    }

    public void copyBlock(int srcBlockId, int dstBlockId) {
        INDArray srcKey = keyBlockPool.get(NDArrayIndex.point(srcBlockId));
        INDArray dstKey = keyBlockPool.get(NDArrayIndex.point(dstBlockId));
        dstKey.assign(srcKey);
        INDArray srcVal = valueBlockPool.get(NDArrayIndex.point(srcBlockId));
        INDArray dstVal = valueBlockPool.get(NDArrayIndex.point(dstBlockId));
        dstVal.assign(srcVal);
    }

    @Override
    public void close() {
        keyBlockPool.close();
        valueBlockPool.close();
    }

    @Override
    public String toString() {
        int usedBlocks = numBlocks - freeBlocks.size();
        return String.format("PagedKVCache[blocks=%d/%d used, blockSize=%d, heads=%d, headDim=%d, %s]",
                usedBlocks, numBlocks, blockSize, numKvHeads, headDim, dataType);
    }
}
