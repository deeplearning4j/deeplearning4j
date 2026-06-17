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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.Getter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * KVCache - Key-Value Cache for efficient autoregressive generation.
 *
 * This class manages past key-value tensors to avoid recomputation during
 * token-by-token LLM inference. It supports:
 * - Standard MHA (Multi-Head Attention)
 * - GQA (Grouped Query Attention) - multiple query heads share KV heads
 * - MQA (Multi-Query Attention) - all query heads share single KV head
 *
 * Usage example:
 * <pre>
 * // Create cache for a model with 8 layers, 32 heads, 128 dim, max 2048 tokens
 * KVCache cache = new KVCache(2048, 8, 128, 1, DataType.FLOAT);
 *
 * // During generation:
 * for (int pos = 0; pos < maxLen; pos++) {
 *     INDArray newK = ...; // [1, 1, 8, 128]
 *     INDArray newV = ...; // [1, 1, 8, 128]
 *     cache.update(newK, newV, pos);
 *
 *     // Get all cached KV for attention
 *     INDArray k = cache.getKeys(pos + 1);
 *     INDArray v = cache.getValues(pos + 1);
 *
 *     // Compute attention with cached KV...
 * }
 * </pre>
 *
 * @author Adam Gibson
 */
public class KVCache implements AutoCloseable {

    @Getter
    private final long maxSeqLen;
    @Getter
    private final long numKvHeads;
    @Getter
    private final long headDim;
    @Getter
    private final long batchSize;
    @Getter
    private final DataType dataType;

    @Getter
    private INDArray keyCache;
    @Getter
    private INDArray valueCache;
    @Getter
    private long currentLength;

    /**
     * Create a new KV cache.
     *
     * @param maxSeqLen Maximum sequence length the cache can hold
     * @param numKvHeads Number of key-value heads
     * @param headDim Dimension of each attention head
     * @param batchSize Batch size
     * @param dataType Data type for cache tensors
     */
    public KVCache(long maxSeqLen, long numKvHeads, long headDim,
                   long batchSize, DataType dataType) {
        this.maxSeqLen = maxSeqLen;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.batchSize = batchSize;
        this.dataType = dataType;
        this.currentLength = 0;

        // Allocate cache tensors [batch, maxSeqLen, numKvHeads, headDim]
        this.keyCache = Nd4j.zeros(dataType, batchSize, maxSeqLen, numKvHeads, headDim);
        this.valueCache = Nd4j.zeros(dataType, batchSize, maxSeqLen, numKvHeads, headDim);
    }

    /**
     * Create a KV cache with default FLOAT32 datatype.
     */
    public KVCache(long maxSeqLen, long numKvHeads, long headDim, long batchSize) {
        this(maxSeqLen, numKvHeads, headDim, batchSize, DataType.FLOAT);
    }

    /**
     * Update the cache with new key-value pairs.
     *
     * @param newKeys New keys to add [batch, newSeqLen, numKvHeads, headDim]
     * @param newValues New values to add [batch, newSeqLen, numKvHeads, headDim]
     * @param position Starting position in the cache
     */
    public void update(INDArray newKeys, INDArray newValues, long position) {
        long newSeqLen = newKeys.size(1);

        if (position + newSeqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                "Cannot update beyond maximum sequence length: position=" + position +
                ", newSeqLen=" + newSeqLen + ", maxSeqLen=" + maxSeqLen);
        }

        // Use KVCacheUpdate op for efficient update
        INDArray[] results = Nd4j.exec(new KVCacheUpdate(keyCache, valueCache, newKeys, newValues, (int) position));
        keyCache.assign(results[0]);
        valueCache.assign(results[1]);

        // Update current length
        currentLength = Math.max(currentLength, position + newSeqLen);
    }

    /**
     * Get keys up to a certain sequence length.
     *
     * @param seqLen Number of positions to retrieve (0 = all valid positions)
     * @return View of cached keys [batch, seqLen, numKvHeads, headDim]
     */
    public INDArray getKeys(long seqLen) {
        if (seqLen <= 0 || seqLen > currentLength) {
            seqLen = currentLength;
        }

        if (seqLen == maxSeqLen) {
            return keyCache;
        }

        // Return a view/slice of the cache
        return keyCache.get(
            NDArrayIndex.all(),
            NDArrayIndex.interval(0, seqLen),
            NDArrayIndex.all(),
            NDArrayIndex.all()
        );
    }

    /**
     * Get values up to a certain sequence length.
     *
     * @param seqLen Number of positions to retrieve (0 = all valid positions)
     * @return View of cached values [batch, seqLen, numKvHeads, headDim]
     */
    public INDArray getValues(long seqLen) {
        if (seqLen <= 0 || seqLen > currentLength) {
            seqLen = currentLength;
        }

        if (seqLen == maxSeqLen) {
            return valueCache;
        }

        // Return a view/slice of the cache
        return valueCache.get(
            NDArrayIndex.all(),
            NDArrayIndex.interval(0, seqLen),
            NDArrayIndex.all(),
            NDArrayIndex.all()
        );
    }

    /**
     * Reset the cache (clear all stored values).
     */
    public void reset() {
        keyCache.assign(0);
        valueCache.assign(0);
        currentLength = 0;
    }

    @Override
    public void close() {
        if (keyCache != null) {
            keyCache.close();
            keyCache = null;
        }
        if (valueCache != null) {
            valueCache.close();
            valueCache = null;
        }
    }
}
