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

/**
 * Strategy for managing KV cache during autoregressive decoding.
 *
 * <p>Determines how key-value pairs from transformer attention layers are stored,
 * updated, and retrieved during token generation. Different strategies trade off
 * between memory usage, maximum sequence length, and compute overhead.</p>
 *
 * <ul>
 *   <li>{@link #STATIC} - Pre-allocated dense buffers with fixed max sequence length.
 *       Best for single-sequence decode with CUDA graph replay (fixed shapes).</li>
 *   <li>{@link #PAGED} - Block-based allocation via page tables.
 *       Best for continuous batching with variable-length sequences.</li>
 *   <li>{@link #QUANTIZED} - Block-based allocation with INT8/FP8 quantization.
 *       Best for long-context scenarios where memory is the bottleneck.</li>
 * </ul>
 *
 * @see KvCacheManager
 */
public enum KvCacheStrategy {

    /**
     * Static pre-allocated KV cache.
     *
     * <p>Allocates dense [batch, heads, maxKvLen, headDim] tensors at prefill time
     * and scatters new KV entries via the native {@code kv_scatter} op or Java
     * view+assign. Shapes are fixed after prefill, enabling CUDA graph capture
     * and replay for zero-overhead kernel launches.</p>
     *
     * <p>Supports two scatter modes:</p>
     * <ul>
     *   <li>C++ scatter: DSP handles KV retention internally (logits-only output)</li>
     *   <li>Java scatter: Decoder outputs present KV, Java scatters via KvScatter op</li>
     * </ul>
     */
    STATIC,

    /**
     * Paged KV cache using block-based allocation.
     *
     * <p>Uses {@link PagedKVCache} with fixed-size blocks allocated on demand from
     * a shared pool. Each sequence maintains a page table mapping logical block
     * indices to physical blocks. The native {@code paged_attention} kernel gathers
     * K/V from scattered blocks via page table indirection.</p>
     *
     * <p>Benefits:</p>
     * <ul>
     *   <li>No per-sequence pre-allocation (blocks allocated as tokens arrive)</li>
     *   <li>Immediate block reuse when sequences finish</li>
     *   <li>O(1) slot reassignment for continuous batching</li>
     * </ul>
     *
     * <p>Note: CUDA graph replay is not compatible with paged attention because
     * page tables and context lengths change every step.</p>
     */
    PAGED,

    /**
     * Quantized paged KV cache with INT8/FP8/INT4 compression.
     *
     * <p>Uses {@link QuantizedPagedKVCache} to store KV data in quantized format,
     * reducing memory usage by 2-4x. Dequantization happens on the fly during
     * attention computation. Per-row absmax symmetric quantization preserves
     * accuracy for typical attention value distributions.</p>
     *
     * <p>Best for long-context scenarios (32k+ tokens) where GPU memory is the
     * bottleneck rather than compute.</p>
     */
    QUANTIZED
}
