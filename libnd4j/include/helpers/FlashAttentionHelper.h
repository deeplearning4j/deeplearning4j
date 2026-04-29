/* ******************************************************************************
 *
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

//
// @author Adam Gibson
//
// Flash Attention and KV Cache Helper
// Implements memory-efficient attention computation using tiling and online softmax
//

#ifndef LIBND4J_FLASHATTENTIONHELPER_H
#define LIBND4J_FLASHATTENTIONHELPER_H

#include "array/NDArray.h"
#include <vector>
#include <memory>

namespace sd {

/**
 * KVCache - Key-Value Cache for efficient autoregressive generation
 *
 * Stores past key and value tensors to avoid recomputation during inference.
 * Supports:
 * - Standard MHA (Multi-Head Attention)
 * - GQA (Grouped Query Attention) - multiple query heads share KV heads
 * - MQA (Multi-Query Attention) - all query heads share single KV head
 */
class SD_LIB_EXPORT KVCache {
 public:
  /**
   * Create a new KV cache
   *
   * @param maxSeqLen Maximum sequence length the cache can hold
   * @param numKvHeads Number of key-value heads
   * @param headDim Dimension of each attention head
   * @param batchSize Batch size
   * @param dtype Data type for cache tensors
   */
  KVCache(LongType maxSeqLen, LongType numKvHeads, LongType headDim,
          LongType batchSize, DataType dtype = FLOAT32);

  ~KVCache();

  /**
   * Update the cache with new key-value pairs
   *
   * @param newKeys New keys to add [batch, newSeqLen, numKvHeads, headDim]
   * @param newValues New values to add [batch, newSeqLen, numKvHeads, headDim]
   * @param position Starting position in the cache (for incremental decoding)
   */
  void update(NDArray* newKeys, NDArray* newValues, LongType position);

  /**
   * Get keys up to a certain sequence length
   *
   * @param seqLen Number of positions to retrieve (0 = all valid positions)
   * @return View of cached keys [batch, seqLen, numKvHeads, headDim]
   */
  NDArray* getKeys(LongType seqLen = 0);

  /**
   * Get values up to a certain sequence length
   *
   * @param seqLen Number of positions to retrieve (0 = all valid positions)
   * @return View of cached values [batch, seqLen, numKvHeads, headDim]
   */
  NDArray* getValues(LongType seqLen = 0);

  /**
   * Get the current valid sequence length in the cache
   */
  LongType getCurrentLength() const { return currentLength_; }

  /**
   * Get the maximum sequence length the cache can hold
   */
  LongType getMaxLength() const { return maxSeqLen_; }

  /**
   * Reset the cache (clear all stored values)
   */
  void reset();

  /**
   * Clear the cache (alias for reset)
   */
  void clear();

  /**
   * Get the underlying key cache tensor
   */
  NDArray* getKeyCache() { return keyCache_; }

  /**
   * Get the underlying value cache tensor
   */
  NDArray* getValueCache() { return valueCache_; }

 private:
  NDArray* keyCache_;      // [batch, maxSeqLen, numKvHeads, headDim]
  NDArray* valueCache_;    // [batch, maxSeqLen, numKvHeads, headDim]
  LongType maxSeqLen_;
  LongType numKvHeads_;
  LongType headDim_;
  LongType batchSize_;
  LongType currentLength_;
  DataType dtype_;
};

/**
 * FlashAttentionHelper - Memory-efficient attention computation
 *
 * Implements Flash Attention v4 inspired algorithm which:
 * 1. Processes attention in tiles to reduce memory from O(N^2) to O(N)
 * 2. Uses online softmax to maintain numerical stability
 * 3. Supports selective rescaling for efficiency
 * 4. Leverages platform-specific optimizations (OneDNN, cuDNN)
 *
 * Supports both 3D and 4D input tensors:
 * - 3D: [batch, seqLen, dim] - single head attention
 * - 4D: [batch, seqLen, numHeads, headDim] - multi-head attention
 *
 * Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * https://arxiv.org/abs/2205.14135
 *
 * FlashAttention v4 optimizations inspired by:
 * - 5-stage pipeline architecture with warp specialization
 * - Selective rescaling for reduced correction operations
 * - Cubic polynomial exponential approximation
 */
class SD_LIB_EXPORT FlashAttentionHelper {
 public:
  /**
   * Configuration for flash attention computation
   */
  struct Config {
    float scale = 0.0f;         // Attention scale (0 = auto: 1/sqrt(head_dim or dim))
    bool isCausal = true;       // Apply causal mask
    float dropout = 0.0f;       // Dropout probability
    int blockSizeQ = 64;        // Tile size for queries
    int blockSizeKV = 64;       // Tile size for keys/values
    int numHeads = 8;           // Number of query heads (used for 4D, ignored for 3D)
    int numKvHeads = 0;         // Number of KV heads (0 = same as numHeads)
    int windowSize = 0;         // Sliding window size (0 = no window)
    bool returnSoftmax = false; // Return softmax weights (for debugging)
  };

  /**
   * Compute flash attention forward pass
   *
   * Supports both 3D and 4D input tensors:
   * - 3D: query [batch, seqLen, dim], key/value [batch, kvLen, dim]
   *       output [batch, seqLen, dim]
   * - 4D: query [batch, seqLen, numHeads, headDim]
   *       key/value [batch, kvLen, numKvHeads, headDim]
   *       output [batch, seqLen, numHeads, headDim]
   *
   * For 3D inputs, computation is equivalent to single-head attention.
   * Platform-specific optimizations (OneDNN, cuDNN) are used when available.
   *
   * @param query Query tensor (3D or 4D)
   * @param key Key tensor (same rank as query)
   * @param value Value tensor (same rank as query)
   * @param output Output tensor (same rank as query)
   * @param config Attention configuration
   * @param softmaxLse Optional output for log-sum-exp values (needed for backward)
   * @param attentionScores Optional output for attention weights after softmax
   * @param attentionLogits Optional output for pre-softmax scores
   * @param context Launch context
   */
  static void forward(NDArray* query, NDArray* key, NDArray* value,
                      NDArray* output, const Config& config,
                      NDArray* softmaxLse = nullptr,
                      NDArray* attentionScores = nullptr,
                      NDArray* attentionLogits = nullptr,
                      LaunchContext* context = LaunchContext::defaultContext(),
                      NDArray* attentionBias = nullptr);

  /**
   * Compute flash attention backward pass
   *
   * Supports both 3D and 4D tensors (same as forward).
   *
   * @param gradOutput Gradient of loss w.r.t. output (3D or 4D)
   * @param query Query tensor (3D or 4D)
   * @param key Key tensor (same rank as query)
   * @param value Value tensor (same rank as query)
   * @param output Forward pass output (same rank as query)
   * @param softmaxLse Log-sum-exp from forward pass
   * @param gradQuery Gradient w.r.t. query (output)
   * @param gradKey Gradient w.r.t. key (output)
   * @param gradValue Gradient w.r.t. value (output)
   * @param config Attention configuration
   * @param context Launch context
   */
  static void backward(NDArray* gradOutput, NDArray* query, NDArray* key, NDArray* value,
                       NDArray* output, NDArray* softmaxLse,
                       NDArray* gradQuery, NDArray* gradKey, NDArray* gradValue,
                       const Config& config,
                       LaunchContext* context = LaunchContext::defaultContext());

  /**
   * Compute attention with KV cache for autoregressive generation
   *
   * @param query Query for current step [batch, 1, numHeads, headDim] or [batch, seqLen, numHeads, headDim]
   * @param kvCache KV cache containing past keys/values
   * @param newKey New key for current step [batch, 1, numKvHeads, headDim]
   * @param newValue New value for current step [batch, 1, numKvHeads, headDim]
   * @param output Output tensor [batch, querySeqLen, numHeads, headDim]
   * @param position Current position in sequence
   * @param config Attention configuration
   * @param context Launch context
   */
  static void forwardWithKVCache(NDArray* query, KVCache* kvCache,
                                 NDArray* newKey, NDArray* newValue,
                                 NDArray* output, LongType position,
                                 const Config& config,
                                 LaunchContext* context = LaunchContext::defaultContext());

  /**
   * Compute standard attention (non-flash, for comparison/debugging)
   *
   * @param query Query tensor [batch, seqLen, numHeads, headDim]
   * @param key Key tensor [batch, kvLen, numKvHeads, headDim]
   * @param value Value tensor [batch, kvLen, numKvHeads, headDim]
   * @param output Output tensor [batch, seqLen, numHeads, headDim]
   * @param config Attention configuration
   * @param attentionWeights Optional output for attention weights
   * @param context Launch context
   */
  static void standardAttention(NDArray* query, NDArray* key, NDArray* value,
                                NDArray* output, const Config& config,
                                NDArray* attentionWeights = nullptr,
                                LaunchContext* context = LaunchContext::defaultContext());

  /**
   * Repeat KV heads for GQA (Grouped Query Attention)
   *
   * When numHeads > numKvHeads, this expands the KV heads to match query heads.
   *
   * @param kv Input tensor [batch, seqLen, numKvHeads, headDim]
   * @param numHeads Target number of heads
   * @param output Output tensor [batch, seqLen, numHeads, headDim]
   */
  static void repeatKVHeads(NDArray* kv, int numHeads, NDArray* output);

 private:
  /**
   * Online softmax state for flash attention v4
   * Tracks running max and sum for numerical stability with selective rescaling
   */
  struct SoftmaxState {
    float maxVal;       // Running maximum
    float sumExp;       // Running sum of exp(x - max)
    float correction;   // Correction factor for selective rescaling
    bool needsRescale;  // Whether accumulated values need rescaling
  };

  /**
   * Forward implementation for 3D tensors [batch, seq, dim]
   * Uses OneDNN Graph API for fused SDPA when available
   */
  static void forward3D(NDArray* query, NDArray* key, NDArray* value,
                        NDArray* output, const Config& config,
                        NDArray* softmaxLse, NDArray* attentionScores,
                        NDArray* attentionLogits, LaunchContext* context,
                        NDArray* attentionBias);

  /**
   * Forward implementation for 4D tensors [batch, seq, heads, dim]
   * Uses batched matmul with GQA support
   */
  static void forward4D(NDArray* query, NDArray* key, NDArray* value,
                        NDArray* output, const Config& config,
                        NDArray* softmaxLse, NDArray* attentionScores,
                        NDArray* attentionLogits, LaunchContext* context,
                        NDArray* attentionBias);

  /**
   * Decode-optimized forward path for seqQ=1 (autoregressive generation).
   * Uses cuBLAS batched GEMV for Q@K^T and attn@V with FP16 TensorCore.
   * For M=1 decode, this achieves ~2x throughput vs the fused kernel.
   */
  static void forward4DDecode(NDArray* query, NDArray* key, NDArray* value,
                              NDArray* output, float scale, bool isCausal,
                              LaunchContext* context, NDArray* attentionBias,
                              NDArray* softmaxLse);

  /**
   * Backward implementation for 3D tensors
   */
  static void backward3D(NDArray* gradOutput, NDArray* query, NDArray* key, NDArray* value,
                         NDArray* output, NDArray* softmaxLse,
                         NDArray* gradQuery, NDArray* gradKey, NDArray* gradValue,
                         const Config& config, LaunchContext* context);

  /**
   * Backward implementation for 4D tensors
   */
  static void backward4D(NDArray* gradOutput, NDArray* query, NDArray* key, NDArray* value,
                         NDArray* output, NDArray* softmaxLse,
                         NDArray* gradQuery, NDArray* gradKey, NDArray* gradValue,
                         const Config& config, LaunchContext* context);

  /**
   * Batched attention implementation using MmulHelper (for short sequences)
   * Uses OneDNN/MKL acceleration through MmulHelper
   */
  static void forwardBatched(NDArray* query, NDArray* key, NDArray* value,
                             NDArray* output, const Config& config,
                             NDArray* softmaxLse,
                             LaunchContext* context);

  /**
   * Flash Attention v4 tiled implementation (for long sequences)
   * Uses online softmax with selective rescaling to avoid O(N^2) memory
   *
   * Key optimizations from FlashAttention v4:
   * - Streaming/online softmax with running max tracking
   * - Selective rescaling (only when max changes significantly)
   * - Tiled computation for cache efficiency
   */
  static void forwardTiled(NDArray* query, NDArray* key, NDArray* value,
                           NDArray* output, const Config& config,
                           NDArray* softmaxLse,
                           LaunchContext* context);

  /**
   * Update online softmax state with new values (FlashAttention v4 style)
   * Implements selective rescaling - only updates when numerically necessary
   *
   * @param state Current softmax state
   * @param newMax New maximum value from current tile
   * @param newSum New sum of exponentials from current tile
   * @param threshold Threshold for triggering rescale (default: 1.0)
   */
  static void updateSoftmaxState(SoftmaxState& state, float newMax, float newSum,
                                 float threshold = 1.0f);

  /**
   * Compute a single tile of flash attention with online softmax
   *
   * @param queryTile Query data for this tile
   * @param keyTile Key data for this tile
   * @param valueTile Value data for this tile
   * @param outputTile Output accumulator for this tile
   * @param states Softmax states per query position
   * @param tileQ Number of query positions in tile
   * @param tileKV Number of key/value positions in tile
   * @param headDim Head dimension
   * @param scale Attention scale factor
   * @param isCausal Whether to apply causal masking
   * @param queryOffset Global offset for query positions
   * @param keyOffset Global offset for key positions
   */
  static void computeTile(const float* queryTile, const float* keyTile, const float* valueTile,
                          float* outputTile, SoftmaxState* states,
                          int tileQ, int tileKV, int headDim,
                          float scale, bool isCausal, int queryOffset, int keyOffset);

  /**
   * Check if OneDNN Graph SDPA is available and usable for given inputs
   */
  static bool canUseOneDnnSdpa(NDArray* query, NDArray* key, NDArray* value,
                               const Config& config);

  /**
   * Check if cuDNN SDPA is available and usable for given inputs
   */
  static bool canUseCudnnSdpa(NDArray* query, NDArray* key, NDArray* value,
                              const Config& config);
};

// Forward declaration for CUDA fused attention (implemented in cuda/FlashAttentionHelper.cu)
// Uses online softmax algorithm for memory efficiency - no cuDNN required
#if defined(__CUDACC__) || defined(SD_CUDA)
extern void fusedAttentionCuda(NDArray* query, NDArray* key, NDArray* value,
                               NDArray* output, float scale, bool isCausal,
                               LaunchContext* context,
                               NDArray* attentionBias = nullptr);

// Fused attention that also outputs attention scores and logits
extern void fusedAttentionCudaWithScores(NDArray* query, NDArray* key, NDArray* value,
                                         NDArray* output, NDArray* attentionLogits,
                                         NDArray* attentionScores, float scale, bool isCausal,
                                         LaunchContext* context);

// In-place causal mask application - replaces create+nullify+fill+add with single kernel
extern void applyCausalMaskCuda(NDArray* scores, LaunchContext* context);

// Fused causal mask + softmax - replaces mask kernel + softmax kernel with single kernel
extern void fusedCausalMaskSoftmaxCuda(NDArray* input, NDArray* output, NDArray* logitsOut,
                                        bool isCausal, LaunchContext* context);

// Fused GQA decode attention - single kernel for Q@K^T + softmax + attn@V with GQA head mapping.
// Takes 4D BSHD inputs directly, no permute/tile needed.
// Q: [batch, 1, numQHeads, headDim], K/V: [batch, seqKV, numKvHeads, headDim]
// Output: [batch, 1, numQHeads, headDim]
extern void fusedGQADecodeCuda(NDArray* query, NDArray* key, NDArray* value,
                                NDArray* output, float scale,
                                LaunchContext* context,
                                NDArray* attentionBias = nullptr);
#endif

}  // namespace sd

#endif  // LIBND4J_FLASHATTENTIONHELPER_H
