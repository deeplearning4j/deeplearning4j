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

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Cross-layer shared KV cache for Multi-head Latent Attention (MLA).
 * <p>
 * MLA, as used in DeepSeek-V2, stores a single compressed latent representation
 * shared across all transformer layers, rather than separate K and V caches per layer.
 * Each layer has its own learned down-projection weights ({@code kvDownProj}) that
 * decompress the shared latent cache into layer-specific K and V tensors on the fly.
 * <p>
 * Memory savings example (DeepSeek-V2 numbers):
 * <ul>
 *   <li>Standard GQA: 60 layers x 2 (K+V) x numKvHeads x headDim per token</li>
 *   <li>MLA: 1 x latentDim per token (shared) + per-layer weights (constants)</li>
 *   <li>Typical compression ratio: 4-8x reduction in KV cache memory</li>
 * </ul>
 * <p>
 * Usage:
 * <pre>{@code
 * MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT);
 *
 * // Set per-layer decompression weights (loaded from model)
 * for (int layer = 0; layer < numLayers; layer++) {
 *     cache.setLayerWeights(layer, model.getKvDownProj(layer));
 * }
 *
 * // During generation, append new latent representations
 * cache.append(newLatent, newRope, cachePosition);
 *
 * // Use with MLAAttention op per layer
 * INDArray output = Nd4j.exec(new MLAAttention(
 *     query, cache.getLatentCache(), cache.getKvDownProj(layerIdx),
 *     cache.getRopeCache(), null,
 *     numHeads, numKvHeads, headDim, latentDim, ropeHeadDim, seqLen, 0.0f));
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class MLAKVCache implements AutoCloseable {

    private final int batchSize;
    private final int maxSeqLen;
    private final int latentDim;
    private final int ropeHeadDim;
    private final int numLayers;
    private final DataType dataType;

    /**
     * Shared latent KV cache: [batchSize, maxSeqLen, latentDim].
     * This single tensor is shared across ALL transformer layers.
     */
    private final INDArray latentCache;

    /**
     * Optional separate RoPE cache: [batchSize, maxSeqLen, ropeHeadDim].
     * Some MLA variants store the uncompressed RoPE portion separately.
     * Null if ropeHeadDim == 0.
     */
    private final INDArray ropeCache;

    /**
     * Per-layer decompression weights: kvDownProj[layer] has shape
     * [latentDim, 2 * numKvHeads * headDim].
     * These are model constants, not cached activations.
     */
    private final INDArray[] kvDownProj;

    /**
     * Creates a new MLA KV cache.
     *
     * @param batchSize   batch size
     * @param maxSeqLen   maximum sequence length the cache can hold
     * @param latentDim   compressed latent dimension
     * @param ropeHeadDim RoPE portion dimension (0 for no separate RoPE cache)
     * @param numLayers   number of transformer layers
     * @param dataType    data type for cache tensors (e.g., FLOAT, HALF)
     */
    public MLAKVCache(int batchSize, int maxSeqLen, int latentDim, int ropeHeadDim,
                      int numLayers, DataType dataType) {
        Preconditions.checkArgument(batchSize > 0, "batchSize must be positive, got %s", batchSize);
        Preconditions.checkArgument(maxSeqLen > 0, "maxSeqLen must be positive, got %s", maxSeqLen);
        Preconditions.checkArgument(latentDim > 0, "latentDim must be positive, got %s", latentDim);
        Preconditions.checkArgument(ropeHeadDim >= 0, "ropeHeadDim must be non-negative, got %s", ropeHeadDim);
        Preconditions.checkArgument(numLayers > 0, "numLayers must be positive, got %s", numLayers);

        this.batchSize = batchSize;
        this.maxSeqLen = maxSeqLen;
        this.latentDim = latentDim;
        this.ropeHeadDim = ropeHeadDim;
        this.numLayers = numLayers;
        this.dataType = dataType;

        this.latentCache = Nd4j.zeros(dataType, batchSize, maxSeqLen, latentDim);
        this.ropeCache = ropeHeadDim > 0
            ? Nd4j.zeros(dataType, batchSize, maxSeqLen, ropeHeadDim)
            : null;
        this.kvDownProj = new INDArray[numLayers];
    }

    /**
     * Sets the per-layer KV decompression weight matrix.
     * These are model constants loaded during model initialization.
     *
     * @param layerIdx        layer index (0-based)
     * @param kvDownProjWeight weight matrix [latentDim, 2 * numKvHeads * headDim]
     */
    public void setLayerWeights(int layerIdx, INDArray kvDownProjWeight) {
        Preconditions.checkArgument(layerIdx >= 0 && layerIdx < numLayers,
            "layerIdx must be in [0, %s), got %s", numLayers, layerIdx);
        Preconditions.checkArgument(kvDownProjWeight.rank() == 2,
            "kvDownProjWeight must be 2D, got rank %s", kvDownProjWeight.rank());
        Preconditions.checkArgument(kvDownProjWeight.size(0) == latentDim,
            "kvDownProjWeight dim 0 must be latentDim (%s), got %s",
            latentDim, kvDownProjWeight.size(0));
        this.kvDownProj[layerIdx] = kvDownProjWeight;
    }

    /**
     * Appends new latent representations to the cache at the specified position.
     *
     * @param newLatent     new latent data [batchSize, newSeqLen, latentDim]
     * @param newRope       new RoPE data [batchSize, newSeqLen, ropeHeadDim] (may be null)
     * @param cachePosition starting position in the cache to write to
     */
    public void append(INDArray newLatent, INDArray newRope, int cachePosition) {
        Preconditions.checkNotNull(newLatent, "newLatent must not be null");
        long newSeqLen = newLatent.size(1);
        Preconditions.checkArgument(cachePosition + newSeqLen <= maxSeqLen,
            "Cache overflow: position %s + newSeqLen %s > maxSeqLen %s",
            cachePosition, newSeqLen, maxSeqLen);

        // Copy new latent data into the shared cache
        for (int b = 0; b < batchSize; b++) {
            for (long s = 0; s < newSeqLen; s++) {
                INDArray srcSlice = newLatent.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(s),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
                INDArray dstSlice = latentCache.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(cachePosition + s),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
                dstSlice.assign(srcSlice);
            }
        }

        // Copy RoPE data if present
        if (ropeCache != null && newRope != null) {
            for (int b = 0; b < batchSize; b++) {
                for (long s = 0; s < newSeqLen; s++) {
                    INDArray srcSlice = newRope.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(s),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                    INDArray dstSlice = ropeCache.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(cachePosition + s),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                    dstSlice.assign(srcSlice);
                }
            }
        }
    }

    /**
     * Returns the shared latent KV cache tensor.
     *
     * @return latent cache [batchSize, maxSeqLen, latentDim]
     */
    public INDArray getLatentCache() {
        return latentCache;
    }

    /**
     * Returns the optional RoPE cache tensor.
     *
     * @return RoPE cache [batchSize, maxSeqLen, ropeHeadDim], or null if ropeHeadDim == 0
     */
    public INDArray getRopeCache() {
        return ropeCache;
    }

    /**
     * Returns the KV decompression weight for a specific layer.
     *
     * @param layerIdx layer index (0-based)
     * @return kvDownProj weight [latentDim, 2 * numKvHeads * headDim]
     */
    public INDArray getKvDownProj(int layerIdx) {
        Preconditions.checkArgument(layerIdx >= 0 && layerIdx < numLayers,
            "layerIdx must be in [0, %s), got %s", numLayers, layerIdx);
        Preconditions.checkNotNull(kvDownProj[layerIdx],
            "kvDownProj for layer %s has not been set. Call setLayerWeights() first.", layerIdx);
        return kvDownProj[layerIdx];
    }

    /**
     * Returns the total memory used by this MLA cache in bytes.
     * This includes the shared latent cache and optional RoPE cache,
     * but NOT the per-layer decompression weights (those are model constants).
     *
     * @return memory usage in bytes
     */
    public long getMemoryBytes() {
        long dtBytes = dataType.width();
        long total = (long) batchSize * maxSeqLen * latentDim * dtBytes;
        if (ropeHeadDim > 0) {
            total += (long) batchSize * maxSeqLen * ropeHeadDim * dtBytes;
        }
        return total;
    }

    /**
     * Computes the equivalent memory that a standard per-layer KV cache would use,
     * for comparison purposes.
     *
     * @param numKvHeads number of KV heads per layer
     * @param headDim    dimension per head
     * @return equivalent standard KV cache memory in bytes
     */
    public long getEquivalentStandardMemoryBytes(int numKvHeads, int headDim) {
        long dtBytes = dataType.width();
        // Standard: numLayers * 2 (K+V) * batchSize * maxSeqLen * numKvHeads * headDim
        return (long) numLayers * 2 * batchSize * maxSeqLen * numKvHeads * headDim * dtBytes;
    }

    /**
     * Returns the memory savings ratio compared to standard per-layer KV caching.
     *
     * @param numKvHeads number of KV heads per layer
     * @param headDim    dimension per head
     * @return savings ratio (e.g., 4.0 means 4x less memory)
     */
    public double getMemorySavingsRatio(int numKvHeads, int headDim) {
        long standard = getEquivalentStandardMemoryBytes(numKvHeads, headDim);
        long mla = getMemoryBytes();
        return mla > 0 ? (double) standard / mla : 0.0;
    }

    /**
     * Zeros out all cache tensors, resetting the cache for a new generation.
     */
    public void reset() {
        latentCache.assign(0);
        if (ropeCache != null) {
            ropeCache.assign(0);
        }
    }

    /**
     * Returns the batch size.
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Returns the maximum sequence length.
     */
    public int getMaxSeqLen() {
        return maxSeqLen;
    }

    /**
     * Returns the latent dimension.
     */
    public int getLatentDim() {
        return latentDim;
    }

    /**
     * Returns the RoPE head dimension.
     */
    public int getRopeHeadDim() {
        return ropeHeadDim;
    }

    /**
     * Returns the number of transformer layers.
     */
    public int getNumLayers() {
        return numLayers;
    }

    /**
     * Returns the data type used by the cache.
     */
    public DataType getDataType() {
        return dataType;
    }

    @Override
    public void close() {
        if (latentCache != null) {
            latentCache.close();
        }
        if (ropeCache != null) {
            ropeCache.close();
        }
        // Note: kvDownProj arrays are model constants owned by the model,
        // not closed here.
    }
}
