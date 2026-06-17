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
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Quantized paged KV cache for memory-efficient autoregressive generation.
 *
 * <p>Extends {@link PagedKVCache} by storing KV data in quantized format (INT8, FP8, or INT4)
 * rather than full precision. This reduces memory usage by 2-4x with minimal accuracy loss,
 * enabling longer context lengths and larger batch sizes.</p>
 *
 * <h3>Quantization formats</h3>
 * <ul>
 *   <li>{@link QuantFormat#INT8}: 8-bit integer, symmetric per-row quantization (2x compression)</li>
 *   <li>{@link QuantFormat#FP8_E4M3}: 8-bit float with 4-bit exponent (2x compression)</li>
 *   <li>{@link QuantFormat#FP8_E5M2}: 8-bit float with 5-bit exponent (2x compression)</li>
 *   <li>{@link QuantFormat#INT4}: 4-bit integer, packed 2 per byte (4x compression)</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * var cache = new QuantizedPagedKVCache(
 *     batchSize, maxSeqLen, numKvHeads, headDim,
 *     QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT);
 *
 * // Append quantizes automatically
 * cache.append(seqIdx, newKeys, newValues);
 *
 * // Dequantize a block for attention computation
 * INDArray[] kv = cache.dequantizeBlock(blockId);
 * INDArray keys = kv[0];   // FLOAT32
 * INDArray values = kv[1]; // FLOAT32
 * }</pre>
 *
 * @author Adam Gibson
 */
@Slf4j
public class QuantizedPagedKVCache implements AutoCloseable {

    /** Default block size (tokens per block). */
    public static final int DEFAULT_BLOCK_SIZE = 64;

    /**
     * Supported quantization formats for KV cache compression.
     */
    public enum QuantFormat {
        /** 8-bit integer symmetric quantization (2x memory reduction) */
        INT8(0),
        /** FP8 with 4-bit exponent, 3-bit mantissa (2x memory reduction) */
        FP8_E4M3(1),
        /** FP8 with 5-bit exponent, 2-bit mantissa (2x memory reduction) */
        FP8_E5M2(2),
        /** 4-bit integer, packed 2 per byte (4x memory reduction) */
        INT4(3);

        @Getter
        private final int nativeCode;

        QuantFormat(int nativeCode) {
            this.nativeCode = nativeCode;
        }
    }

    @Getter private final int blockSize;
    @Getter private final int numKvHeads;
    @Getter private final int headDim;
    @Getter private final int maxBatchSize;
    @Getter private final int numBlocks;
    @Getter private final DataType originalDataType;
    @Getter private final QuantFormat quantFormat;

    // Quantized block pools: [numBlocks, blockSize, numKvHeads, headDim] in INT8
    @Getter private final INDArray keyBlockPool;
    @Getter private final INDArray valueBlockPool;

    // Per-block scale pools: [numBlocks, blockSize, numKvHeads] in FLOAT32
    // One scale per row (a row = one token position across all head dims)
    @Getter private final INDArray keyScalePool;
    @Getter private final INDArray valueScalePool;

    // Free block IDs (LIFO for cache locality)
    private final Deque<Integer> freeBlocks;

    // Per-sequence page tables
    private final int[][] pageTables;
    private final int[] seqLengths;
    private final int maxBlocksPerSeq;

    /**
     * Create a quantized paged KV cache.
     *
     * @param maxBatchSize   maximum number of concurrent sequences
     * @param maxSeqLen      maximum sequence length
     * @param numKvHeads     number of KV heads (supports GQA)
     * @param headDim        dimension per head
     * @param quantFormat    quantization format
     * @param blockSize      tokens per block
     * @param originalDtype  original data type of KV values (for dequantization output)
     * @param poolSizeFactor multiplier for pool headroom (e.g., 1.2)
     */
    public QuantizedPagedKVCache(int maxBatchSize, int maxSeqLen, int numKvHeads, int headDim,
                                  QuantFormat quantFormat, int blockSize,
                                  DataType originalDtype, double poolSizeFactor) {
        this.blockSize = blockSize;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.maxBatchSize = maxBatchSize;
        this.originalDataType = originalDtype;
        this.quantFormat = quantFormat;

        this.maxBlocksPerSeq = (maxSeqLen + blockSize - 1) / blockSize;
        int minBlocks = maxBatchSize * maxBlocksPerSeq;
        this.numBlocks = (int) Math.ceil(minBlocks * poolSizeFactor);

        long quantBytes = (long) numBlocks * blockSize * numKvHeads * headDim;
        long scaleBytes = (long) numBlocks * blockSize * numKvHeads * 4;  // FLOAT32
        long totalMB = 2 * (quantBytes + scaleBytes) / (1024 * 1024);

        log.info("QuantizedPagedKVCache: {} blocks of {}x{}x{}, format={}, pool total ~{} MB (vs ~{} MB unquantized)",
                numBlocks, blockSize, numKvHeads, headDim, quantFormat,
                totalMB,
                2L * numBlocks * blockSize * numKvHeads * headDim * originalDtype.width() / (1024 * 1024));

        // Allocate quantized block pools (INT8 dtype)
        this.keyBlockPool = Nd4j.zeros(DataType.INT8, numBlocks, blockSize, numKvHeads, headDim);
        this.valueBlockPool = Nd4j.zeros(DataType.INT8, numBlocks, blockSize, numKvHeads, headDim);

        // Allocate scale pools: one scale per (token position, head) = per row in quantization
        // Shape: [numBlocks, blockSize, numKvHeads]
        this.keyScalePool = Nd4j.zeros(DataType.FLOAT, numBlocks, blockSize, numKvHeads);
        this.valueScalePool = Nd4j.zeros(DataType.FLOAT, numBlocks, blockSize, numKvHeads);

        // Initialize free list
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
    public QuantizedPagedKVCache(int maxBatchSize, int maxSeqLen, int numKvHeads, int headDim,
                                  QuantFormat quantFormat, DataType originalDtype) {
        this(maxBatchSize, maxSeqLen, numKvHeads, headDim, quantFormat,
                DEFAULT_BLOCK_SIZE, originalDtype, 1.2);
    }

    /**
     * Append new KV entries for a sequence, quantizing before storage.
     *
     * <p>The input keys and values are in original floating-point format.
     * They are quantized per-token-per-head before being stored in the block pool.</p>
     *
     * @param seqIdx    batch index of the sequence
     * @param newKeys   new key tensor [1, newLen, numKvHeads, headDim] or [newLen, numKvHeads, headDim]
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

            // Extract single token KV: shape [numKvHeads, headDim]
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

            // Quantize the token data: [numKvHeads, headDim] -> quantized + scales
            // The native op treats the last dim as the row length, so each KV head is a row.
            KVCacheQuantize keyQuantOp = new KVCacheQuantize(keyToken, quantFormat.getNativeCode());
            INDArray[] keyQuantResult = Nd4j.exec(keyQuantOp);
            INDArray quantizedKey = keyQuantResult[0];   // [numKvHeads, headDim] INT8
            INDArray keyScales = keyQuantResult[1];       // [numKvHeads] FLOAT32

            KVCacheQuantize valQuantOp = new KVCacheQuantize(valToken, quantFormat.getNativeCode());
            INDArray[] valQuantResult = Nd4j.exec(valQuantOp);
            INDArray quantizedVal = valQuantResult[0];
            INDArray valScales = valQuantResult[1];

            // Store quantized data into block pool
            INDArray keyDest = keyBlockPool.get(NDArrayIndex.point(physicalBlock),
                    NDArrayIndex.point(offsetInBlock), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray valDest = valueBlockPool.get(NDArrayIndex.point(physicalBlock),
                    NDArrayIndex.point(offsetInBlock), NDArrayIndex.all(), NDArrayIndex.all());
            keyDest.assign(quantizedKey);
            valDest.assign(quantizedVal);

            // Store scales
            INDArray keyScaleDest = keyScalePool.get(NDArrayIndex.point(physicalBlock),
                    NDArrayIndex.point(offsetInBlock), NDArrayIndex.all());
            INDArray valScaleDest = valueScalePool.get(NDArrayIndex.point(physicalBlock),
                    NDArrayIndex.point(offsetInBlock), NDArrayIndex.all());
            keyScaleDest.assign(keyScales);
            valScaleDest.assign(valScales);
        }

        seqLengths[seqIdx] += (int) newLen;
    }

    /**
     * Dequantize a physical block's KV data back to floating point.
     *
     * @param blockId physical block ID
     * @return array of two INDArrays: [dequantizedKeys, dequantizedValues],
     *         each of shape [blockSize, numKvHeads, headDim] in FLOAT32
     */
    public INDArray[] dequantizeBlock(int blockId) {
        if (blockId < 0 || blockId >= numBlocks) {
            throw new IllegalArgumentException("blockId out of range: " + blockId);
        }

        // Get quantized block data: [blockSize, numKvHeads, headDim]
        INDArray quantizedKeys = keyBlockPool.get(NDArrayIndex.point(blockId),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray quantizedVals = valueBlockPool.get(NDArrayIndex.point(blockId),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

        // Get scales: [blockSize, numKvHeads]
        INDArray keyScales = keyScalePool.get(NDArrayIndex.point(blockId),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray valScales = valueScalePool.get(NDArrayIndex.point(blockId),
                NDArrayIndex.all(), NDArrayIndex.all());

        // Dequantize using the native op
        KVCacheDequantize keyDeqOp = new KVCacheDequantize(quantizedKeys, keyScales, quantFormat.getNativeCode());
        INDArray[] keyResult = Nd4j.exec(keyDeqOp);

        KVCacheDequantize valDeqOp = new KVCacheDequantize(quantizedVals, valScales, quantFormat.getNativeCode());
        INDArray[] valResult = Nd4j.exec(valDeqOp);

        return new INDArray[]{keyResult[0], valResult[0]};
    }

    /**
     * Free all blocks for a sequence, returning them to the pool.
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
     * Get the page table for a sequence as an INDArray.
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
            else break;
        }
        return count;
    }

    /**
     * Reassign a slot from a finished sequence to a new one.
     *
     * @param slotIdx the batch slot to reassign
     */
    public void reassignSlot(int slotIdx) {
        freeSequence(slotIdx);
    }

    private int allocateBlock() {
        if (freeBlocks.isEmpty()) {
            throw new IllegalStateException("QuantizedPagedKVCache: no free blocks (pool exhausted). "
                    + "Consider increasing maxSeqLen or poolSizeFactor.");
        }
        return freeBlocks.pop();
    }

    @Override
    public void close() {
        keyBlockPool.close();
        valueBlockPool.close();
        keyScalePool.close();
        valueScalePool.close();
    }

    @Override
    public String toString() {
        int usedBlocks = numBlocks - freeBlocks.size();
        return String.format("QuantizedPagedKVCache[blocks=%d/%d used, blockSize=%d, heads=%d, headDim=%d, format=%s]",
                usedBlocks, numBlocks, blockSize, numKvHeads, headDim, quantFormat);
    }
}
