/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_MLA_ATTENTION_H
#define LIBND4J_MLA_ATTENTION_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Configuration for Multi-head Latent Attention (MLA).
 *
 * MLA is the attention mechanism used in DeepSeek-V2 and similar models.
 * Instead of storing separate K and V caches per layer, a single shared
 * latent (compressed) KV cache is stored and decompressed per-layer
 * via learned down-projection weights. This dramatically reduces KV cache
 * memory during inference.
 */
struct MLAConfig {
    int numHeads;        // Number of query heads
    int numKvHeads;      // Number of KV heads (for GQA grouping)
    int headDim;         // Dimension per head
    int latentDim;       // Latent dimension (compressed KV dimension)
    int ropeHeadDim;     // RoPE portion dimension (0 if no separate RoPE)
    float attScale;      // Attention scale (0 = auto: 1/sqrt(headDim))
};

/**
 * MLA forward pass on CPU.
 *
 * Query attends to a compressed latent KV cache that is decompressed
 * per-layer via kvDownProj.
 *
 * Algorithm:
 *   1. Decompress: latentKVCache @ kvDownProj -> K, V
 *   2. Attention:  softmax(Q @ K^T * scale) @ V
 *   3. GQA head mapping if numHeads != numKvHeads
 *
 * @param query          [B, 1, numHeads, headDim]
 * @param latentKVCache  [B, maxSeqLen, latentDim] - shared across layers
 * @param kvDownProj     [latentDim, 2 * numKvHeads * headDim] - per-layer decompression
 * @param ropeCache      [B, maxSeqLen, ropeHeadDim] optional separate RoPE portion
 * @param output         [B, 1, numHeads, headDim]
 * @param seqLen         current valid sequence length in the cache
 * @param config         MLA configuration parameters
 */
SD_LIB_HIDDEN void mlaAttentionCpu(
    NDArray* query,
    NDArray* latentKVCache,
    NDArray* kvDownProj,
    NDArray* ropeCache,
    NDArray* output,
    int seqLen,
    const MLAConfig& config);

#if defined(SD_CUDA)
/**
 * MLA forward pass on CUDA.
 *
 * Same algorithm as CPU but with GPU-optimized kernels:
 * - One block per (batch, head) pair
 * - Shared memory for decompressed K slice
 * - Warp-level reductions for softmax and attention output
 *
 * @param query          [B, 1, numHeads, headDim]
 * @param latentKVCache  [B, maxSeqLen, latentDim] - shared across layers
 * @param kvDownProj     [latentDim, 2 * numKvHeads * headDim] - per-layer decompression
 * @param ropeCache      [B, maxSeqLen, ropeHeadDim] optional separate RoPE portion
 * @param output         [B, 1, numHeads, headDim]
 * @param seqLen         current valid sequence length in the cache
 * @param config         MLA configuration parameters
 * @param context        CUDA launch context
 */
SD_LIB_HIDDEN void mlaAttentionCuda(
    NDArray* query,
    NDArray* latentKVCache,
    NDArray* kvDownProj,
    NDArray* ropeCache,
    NDArray* output,
    int seqLen,
    const MLAConfig& config,
    LaunchContext* context);
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_MLA_ATTENTION_H
