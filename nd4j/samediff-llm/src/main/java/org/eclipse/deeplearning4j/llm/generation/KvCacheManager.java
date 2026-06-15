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

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * Unified KV cache management interface for the decode loop.
 *
 * <p>Abstracts all KV cache operations (initialization, input preparation, scatter,
 * cleanup) behind a common interface so the decode loop does not need to know which
 * KV cache strategy is in use. Implementations handle the strategy-specific details:</p>
 *
 * <ul>
 *   <li>{@link UnifiedKvCacheManager} - Single unified implementation for all strategies
 *       (STATIC, PAGED, QUANTIZED, TURBOQUANT) with in-graph KvScatter support</li>
 * </ul>
 *
 * <h3>Lifecycle</h3>
 * <ol>
 *   <li>{@link #initializeFromPrefill} - Called once after prefill to set up cache buffers</li>
 *   <li>{@link #prepareInputs} - Called each decode step to add KV cache entries to input map</li>
 *   <li>{@link #scatterNewEntries} - Called each step to scatter decoder's KV outputs into cache</li>
 *   <li>{@link #scatterMultipleEntries} - Called for speculative decode (multi-token scatter)</li>
 *   <li>{@link #close} - Releases all cache buffers</li>
 * </ol>
 *
 * @see KvCacheStrategy
 * @see UnifiedKvCacheManager
 */
public interface KvCacheManager extends AutoCloseable {

    /**
     * Get the KV cache strategy this manager implements.
     */
    KvCacheStrategy getStrategy();

    /**
     * Initialize the KV cache from prefill outputs.
     *
     * <p>Called once after the prefill step. The implementation creates and populates
     * its internal buffers from the prefill's present KV outputs.</p>
     *
     * @param prefillOutputs  map of decoder output names to their tensors from prefill
     * @param kvNames         KV cache output names (present key/value names)
     * @param maxNewTokens    maximum number of new tokens to generate
     * @param prefillSeqLen   number of tokens in the prefill
     */
    void initializeFromPrefill(Map<String, INDArray> prefillOutputs,
                               ModelIOConfig.KVCacheNames kvNames,
                               int maxNewTokens, long prefillSeqLen);

    /**
     * Initialize empty KV cache buffers (no prefill data).
     *
     * <p>Used by draft models in speculative decoding that start with empty KV caches
     * and build up state autoregressively without a prefill step. The buffers are
     * pre-allocated to {@code maxKvLen} but start with cachePosition = 0.</p>
     *
     * @param decoder         the SameDiff decoder model (for inferring KV dimensions)
     * @param decoderInputNames the decoder's input names (to find past_key_values entries)
     * @param maxKvLen        maximum KV cache length to allocate
     * @param hiddenSize      model hidden size (for dimension inference)
     */
    void initializeEmpty(SameDiff decoder, List<String> decoderInputNames,
                         int maxKvLen, long hiddenSize);

    /**
     * Prepare KV cache inputs for one decode step.
     *
     * <p>Adds the appropriate KV cache tensors to the decoder input map. For static KV,
     * this adds the full static buffer (padded mode) or a view (view mode). For paged KV,
     * this would add block pool tensors and page tables.</p>
     *
     * @param decoderInputMap the decoder input map to populate (modified in-place)
     * @param decoder         the SameDiff decoder model (for variable resolution)
     * @param hiddenSize      model hidden size (for fallback creation)
     * @param dspActive       whether DSP (DynamicShapePlan) is active (padded vs view mode)
     */
    void prepareInputs(Map<String, INDArray> decoderInputMap, SameDiff decoder,
                       long hiddenSize, boolean dspActive);

    /**
     * Scatter new KV entries from a single-token decode step.
     *
     * <p>Extracts the new KV entry from the decoder's present KV outputs and writes
     * it into the cache at the current position, then advances the position.</p>
     *
     * @param decoderOutputs  map of decoder output names to tensors
     * @param kvNames         KV cache output names (present key/value names)
     */
    void scatterNewEntries(Map<String, INDArray> decoderOutputs,
                           ModelIOConfig.KVCacheNames kvNames);

    /**
     * Scatter multiple KV entries from a multi-token decode step (speculative decode).
     *
     * @param decoderOutputs  map of decoder output names to tensors
     * @param kvNames         KV cache output names
     * @param numEntries      number of entries to scatter
     */
    void scatterMultipleEntries(Map<String, INDArray> decoderOutputs,
                                ModelIOConfig.KVCacheNames kvNames,
                                int numEntries);

    /**
     * Get the current write position in the cache (number of valid tokens).
     */
    long getCachePosition();

    /**
     * Set the cache position explicitly (e.g., after speculative decode rollback).
     */
    void setCachePosition(long position);

    /**
     * Get the maximum KV cache length (total allocated capacity).
     */
    long getMaxKvLen();

    /**
     * Whether this cache is initialized and ready for use.
     */
    boolean isInitialized();

    /**
     * Whether CUDA graph replay is compatible with this cache strategy.
     *
     * <p>Static KV cache supports CUDA graphs (fixed shapes). Paged attention does not
     * (page tables change each step).</p>
     */
    boolean supportsCudaGraphReplay();

    /**
     * Get the raw static KV buffer map (past_key_values input name to buffer).
     *
     * <p>Returns null for non-static strategies. The buffers are passed directly
     * to the decoder as normal inputs and written to by the ordinary KvScatter op
     * on each decode step — there is no native KV retention configuration.</p>
     */
    Map<String, INDArray> getStaticKvBuffers();

    /**
     * Release all KV cache buffers and resources.
     */
    @Override
    void close();
}
