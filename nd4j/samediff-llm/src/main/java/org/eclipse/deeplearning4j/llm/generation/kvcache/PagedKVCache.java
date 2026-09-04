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
import org.nd4j.linalg.api.ops.impl.transforms.custom.PagedKvAppend;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayDeque;
import java.util.Arrays;
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
 * <h3>Native append path</h3>
 * <p>Appends route through the batched native {@code paged_kv_append} op: one kernel
 * scatter per append instead of per-token view/assign dispatches. Only the page-table
 * bookkeeping (block allocation) stays on the host. Inputs are normalized to a
 * C-contiguous [1, newLen, numKvHeads, headDim] view before the op call.</p>
 *
 * <h3>Slack pool</h3>
 * <p>Recently freed blocks are kept in a small {@link #reservedBlocks} band (analogous to
 * kvcached's reserved-page list) and are reused first, improving device cache locality for
 * the hot free/alloc churn of continuous batching. Overflow beyond
 * {@link #reservedBlockLimit} spills to the general LIFO free list. Reserved blocks still
 * count as free capacity in {@link #getNumFreeBlocks()}.</p>
 *
 * <h3>Shared blocks (zero-copy prefix sharing)</h3>
 * <p>Physical blocks carry a reference count. {@link #sharePrefixBlocks(int, int, int)}
 * points a destination sequence's leading page-table entries at a source sequence's
 * physical blocks without copying any data. Freeing a sequence decrements counts; a block
 * only returns to the free pool when its count reaches zero. This is the substrate for
 * cross-request prefix sharing and O(1) beam forks.</p>
 *
 * <h3>Benefits</h3>
 * <ul>
 *   <li>No per-sequence pre-allocation of maxSeqLen — blocks allocated as tokens arrive</li>
 *   <li>Finished sequences return blocks to the pool — immediate reuse without copying</li>
 *   <li>Slot reassignment for continuous batching is O(1) — just swap page tables</li>
 *   <li>Memory fragmentation is bounded by block size (typically 64 tokens)</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class PagedKVCache implements AutoCloseable {

    /** Default block size (tokens per block). Trades granularity vs overhead. */
    public static final int DEFAULT_BLOCK_SIZE = 64;

    /** Default limit on the reserved (slack) block band. 0 disables the band. */
    public static final int DEFAULT_RESERVED_BLOCK_LIMIT = 32;

    @Getter private final int blockSize;
    @Getter private final int numKvHeads;
    @Getter private final int headDim;
    @Getter private final int maxBatchSize;
    @Getter private final int numBlocks;
    @Getter private final DataType dataType;

    // Block pool: pre-allocated GPU tensors holding all blocks
    // Shape: [numBlocks, blockSize, numKvHeads, headDim]
    @Getter private final INDArray keyBlockPool;
    @Getter private final INDArray valueBlockPool;

    // Free block IDs (LIFO for cache locality)
    protected final Deque<Integer> freeBlocks;

    // Recently-freed blocks kept hot for immediate reuse (slack band). Invariant:
    // a block is in freeBlocks or reservedBlocks if and only if refCounts[id] == 0.
    protected final Deque<Integer> reservedBlocks;

    /**
     * Limit on {@link #reservedBlocks} size; overflow drains oldest entries into
     * {@link #freeBlocks}. Settable at runtime.
     */
    @Getter @Setter
    private int reservedBlockLimit = DEFAULT_RESERVED_BLOCK_LIMIT;

    // Reference counts per physical block. 0 = free, >= 1 = owned by one or more
    // sequences (multi-owner only through sharePrefixBlocks).
    private final int[] refCounts;

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

        this.reservedBlocks = new ArrayDeque<>();
        this.refCounts = new int[numBlocks];

        // Initialize page tables
        this.pageTables = new int[maxBatchSize][];
        this.seqLengths = new int[maxBatchSize];
        for (int i = 0; i < maxBatchSize; i++) {
            pageTables[i] = new int[maxBlocksPerSeq];
            Arrays.fill(pageTables[i], -1);
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
     * <p>Block allocation is host-side page-table bookkeeping; the token data itself is
     * scattered into the block pools by a single native {@code paged_kv_append} kernel
     * call (one launch per append instead of per-token view/assign dispatches).</p>
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
        if (newLen == 0) {
            return;
        }
        int startPos = seqLengths[seqIdx];

        // Host-side page-table bookkeeping: allocate any missing blocks up front.
        int startLogicalBlock = startPos / blockSize;
        int endLogicalBlock = (startPos + (int) newLen - 1) / blockSize;
        if (endLogicalBlock >= maxBlocksPerSeq) {
            throw new IllegalStateException("Sequence " + seqIdx + " exceeded max blocks (" + maxBlocksPerSeq + ")");
        }
        for (int logicalBlock = startLogicalBlock; logicalBlock <= endLogicalBlock; logicalBlock++) {
            if (pageTables[seqIdx][logicalBlock] < 0) {
                pageTables[seqIdx][logicalBlock] = allocateBlock();
            }
        }

        // Normalize inputs to contiguous rank-4 [1, newLen, numKvHeads, headDim]:
        // the native kernel indexes flat C-order coordinates.
        boolean[] keysCopied = new boolean[1];
        boolean[] valuesCopied = new boolean[1];
        INDArray keys = normalizeForNativeAppend(newKeys, newLen, keysCopied);
        INDArray values = normalizeForNativeAppend(newValues, newLen, valuesCopied);

        try {
            // Single native kernel scatter into both block pools via the page table.
            // The op's output is a view of keyBlockPool (ARRAY_COPY_OFFSET_INPUT_0) —
            // the pools are written in place, no new KV storage is allocated.
            INDArray tablesArg = Nd4j.createFromArray(new int[][]{pageTables[seqIdx]});
            INDArray lensArg = Nd4j.createFromArray(new int[]{startPos});
            try {
                Nd4j.exec(new PagedKvAppend(keyBlockPool, valueBlockPool, keys, values,
                        tablesArg, lensArg, blockSize));
            } finally {
                tablesArg.close();
                lensArg.close();
            }
        } finally {
            // Close only arrays WE created (dup). Reshape views share the caller's
            // buffer — closing those would release the caller's data out from under it.
            if (keysCopied[0]) {
                keys.close();
            }
            if (valuesCopied[0]) {
                values.close();
            }
        }

        seqLengths[seqIdx] += (int) newLen;
    }

    /**
     * Normalize an input tensor to a C-contiguous [1, newLen, numKvHeads, headDim] view
     * suitable for the native append kernel. Sets {@code copiedOut[0]} to true only when
     * a NEW buffer was materialized (dup) and is safe to close after the op runs.
     */
    private INDArray normalizeForNativeAppend(INDArray arr, long newLen, boolean[] copiedOut) {
        copiedOut[0] = false;
        if (arr.rank() != 3 && arr.rank() != 4) {
            throw new IllegalArgumentException("append inputs must be rank 3 or 4, got rank " + arr.rank());
        }
        if (arr.rank() == 4 && arr.size(0) != 1) {
            throw new IllegalArgumentException(
                    "append inputs are single-sequence; got batch dim " + arr.size(0));
        }
        if (arr.size(arr.rank() - 2) != numKvHeads || arr.size(arr.rank() - 1) != headDim) {
            throw new IllegalArgumentException("append input tail dims [" + arr.size(arr.rank() - 2)
                    + ", " + arr.size(arr.rank() - 1) + "] do not match cache [" + numKvHeads + ", " + headDim + "]");
        }

        INDArray out = arr;
        if (!(out.ordering() == 'c' && Shape.strideDescendingCAscendingF(out))) {
            // Non-contiguous (permute/slice views): materialize a C-order copy
            out = out.dup('c');
            copiedOut[0] = true;
        }
        if (out.rank() == 3) {
            // Contiguous reshape is a view — never close it (shares the source buffer)
            out = out.reshape(1, newLen, numKvHeads, headDim);
        }
        return out;
    }

    /**
     * Free all blocks for a sequence, returning them to the pool.
     * Call when a sequence finishes generation.
     *
     * <p>Shared blocks (reference count &gt; 1) are not released to the free pool; only the
     * reference count is decremented. See {@link #sharePrefixBlocks(int, int, int)}.</p>
     *
     * @param seqIdx batch index of the finished sequence
     */
    public void freeSequence(int seqIdx) {
        if (seqIdx < 0 || seqIdx >= maxBatchSize) return;

        for (int b = 0; b < maxBlocksPerSeq; b++) {
            int blockId = pageTables[seqIdx][b];
            if (blockId >= 0) {
                freeBlock(blockId);
                pageTables[seqIdx][b] = -1;
            }
        }
        seqLengths[seqIdx] = 0;
    }

    /**
     * Evict the oldest {@code numBlocksToEvict} physical blocks from a sequence and shift
     * the remaining page-table entries down so logical position 0 stays the oldest surviving
     * token. Sequence length decreases by {@code numBlocksToEvict * blockSize} (capped at the
     * current length's block count). This is the correct sliding-window primitive: page-table
     * entries are cleared and shifted, so no block can later be double-freed.
     *
     * @param seqIdx            batch index
     * @param numBlocksToEvict  number of oldest physical blocks to evict
     * @return the number of blocks actually evicted
     */
    public int evictOldestBlocks(int seqIdx, int numBlocksToEvict) {
        if (seqIdx < 0 || seqIdx >= maxBatchSize) {
            throw new IllegalArgumentException("seqIdx out of range: " + seqIdx);
        }
        int[] table = pageTables[seqIdx];
        int allocated = getNumAllocatedBlocks(seqIdx);
        int n = Math.min(numBlocksToEvict, allocated);
        if (n <= 0) {
            return 0;
        }

        for (int i = 0; i < n; i++) {
            freeBlock(table[i]);
            table[i] = -1;
        }
        // Shift surviving entries left so logical block 0 is again the oldest token
        for (int i = n; i < allocated; i++) {
            table[i - n] = table[i];
            table[i] = -1;
        }
        seqLengths[seqIdx] = Math.max(0, seqLengths[seqIdx] - n * blockSize);

        if (log.isTraceEnabled()) {
            log.trace("Evicted {} oldest blocks from seq {}: length now {}", n, seqIdx, seqLengths[seqIdx]);
        }
        return n;
    }

    /**
     * Zero-copy prefix sharing: point the first {@code numBlocks} page-table entries of
     * {@code toSeqIdx} at the physical blocks of the first {@code numBlocks} entries of
     * {@code fromSeqIdx}. No KV data is copied; the destination sequence's reference counts
     * keep the shared blocks alive until every referencing sequence frees them.
     *
     * <p>The destination sequence must be empty. Typical use: fork a beam or serve a new
     * request that shares a cached prompt prefix.</p>
     *
     * @param fromSeqIdx sequence owning the prefix blocks
     * @param numBlocks  number of leading logical blocks to share
     * @param toSeqIdx   destination sequence (must be empty)
     * @return the number of blocks shared
     */
    public int sharePrefixBlocks(int fromSeqIdx, int numBlocks, int toSeqIdx) {
        if (fromSeqIdx < 0 || fromSeqIdx >= maxBatchSize || toSeqIdx < 0 || toSeqIdx >= maxBatchSize) {
            throw new IllegalArgumentException("seqIdx out of range: " + fromSeqIdx + ", " + toSeqIdx);
        }
        if (fromSeqIdx == toSeqIdx) {
            throw new IllegalArgumentException("Cannot share a sequence's blocks with itself");
        }
        int[] src = pageTables[fromSeqIdx];
        int[] dst = pageTables[toSeqIdx];
        for (int b = 0; b < maxBlocksPerSeq; b++) {
            if (dst[b] >= 0) {
                throw new IllegalStateException(
                        "Destination sequence " + toSeqIdx + " must be empty to share a prefix");
            }
        }
        int n = Math.min(numBlocks, maxBlocksPerSeq);
        for (int b = 0; b < n; b++) {
            if (src[b] < 0) {
                throw new IllegalStateException(
                        "Source sequence " + fromSeqIdx + " does not have block " + b + " allocated");
            }
            dst[b] = src[b];
            refCounts[src[b]]++;
        }
        int sharedTokens = Math.min(n * blockSize, seqLengths[fromSeqIdx]);
        seqLengths[toSeqIdx] = sharedTokens;

        if (log.isTraceEnabled()) {
            log.trace("Shared {} prefix blocks from seq {} to seq {} ({} tokens)", n, fromSeqIdx, toSeqIdx, sharedTokens);
        }
        return n;
    }

    /**
     * Reference count of a physical block. 0 = free, 1 = owned by one sequence,
     * &gt; 1 = shared by multiple sequences (zero-copy).
     */
    public int getBlockRefCount(int physicalBlockId) {
        if (physicalBlockId < 0 || physicalBlockId >= numBlocks) {
            throw new IllegalArgumentException("blockId out of range: " + physicalBlockId);
        }
        return refCounts[physicalBlockId];
    }

    /**
     * Number of blocks currently sitting in the slack (reserved) band.
     */
    public int getNumReservedBlocks() {
        return reservedBlocks.size();
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
     * Current sequence length for a batch index.
     */
    public int getSequenceLength(int seqIdx) {
        return seqLengths[seqIdx];
    }

    /**
     * Number of free blocks remaining in the pool. Includes blocks in the slack band —
     * they are free capacity, just kept hot for reuse.
     */
    public int getNumFreeBlocks() {
        return freeBlocks.size() + reservedBlocks.size();
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

    /**
     * Allocate a free block, taking from the slack band first (hottest reuse).
     * The returned block has reference count 1 (owned by the caller).
     */
    protected int allocateBlock() {
        Integer blockId = reservedBlocks.isEmpty() ? freeBlocks.poll() : reservedBlocks.poll();
        if (blockId == null) {
            throw new IllegalStateException("PagedKVCache: no free blocks (pool exhausted). "
                    + "Consider increasing maxSeqLen or poolSizeFactor.");
        }
        refCounts[blockId] = 1;
        return blockId;
    }

    /**
     * Decrement a block's reference count; return it to the pool (slack band first)
     * when the count reaches zero.
     */
    protected void freeBlock(int blockId) {
        if (blockId < 0 || blockId >= numBlocks) {
            throw new IllegalArgumentException("blockId out of range: " + blockId);
        }
        if (--refCounts[blockId] <= 0) {
            refCounts[blockId] = 0;
            releaseBlock(blockId);
        }
    }

    /**
     * Return a zero-reference block to the pool via the slack band, draining the
     * oldest overflow into the general free list.
     */
    private void releaseBlock(int blockId) {
        reservedBlocks.push(blockId);
        while (reservedBlocks.size() > Math.max(0, reservedBlockLimit)) {
            Integer oldest = reservedBlocks.pollLast();
            if (oldest == null) {
                break;
            }
            freeBlocks.push(oldest);
        }
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
        int usedBlocks = numBlocks - getNumFreeBlocks();
        return String.format("PagedKVCache[blocks=%d/%d used, reserved=%d, blockSize=%d, heads=%d, headDim=%d, %s]",
                usedBlocks, numBlocks, reservedBlocks.size(), blockSize, numKvHeads, headDim, dataType);
    }
}
