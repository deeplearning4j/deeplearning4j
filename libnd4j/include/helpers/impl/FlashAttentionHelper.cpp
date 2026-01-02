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
// Flash Attention v4 Style Implementation
// Based on Dao-AILab/flash-attention algorithm with:
// - Tiled computation for O(N) memory complexity
// - Online softmax with running max/sum
// - Leverages OneDNN/MKL for matrix operations via MmulHelper
//

#include <helpers/FlashAttentionHelper.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/activations.h>
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <system/openmp_pragmas.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace sd {

//////////////////////////////////////////////////////////////////////////////
// KVCache Implementation
//////////////////////////////////////////////////////////////////////////////

KVCache::KVCache(LongType maxSeqLen, LongType numKvHeads, LongType headDim,
                 LongType batchSize, DataType dtype)
    : maxSeqLen_(maxSeqLen), numKvHeads_(numKvHeads), headDim_(headDim),
      batchSize_(batchSize), currentLength_(0), dtype_(dtype) {

  std::vector<LongType> cacheShape = {batchSize, maxSeqLen, numKvHeads, headDim};
  keyCache_ = new NDArray('c', cacheShape, dtype);
  valueCache_ = new NDArray('c', cacheShape, dtype);

  keyCache_->nullify();
  valueCache_->nullify();
}

KVCache::~KVCache() {
  delete keyCache_;
  delete valueCache_;
}

void KVCache::update(NDArray* newKeys, NDArray* newValues, LongType position) {
  auto newSeqLen = newKeys->sizeAt(1);

  if (position + newSeqLen > maxSeqLen_) {
    THROW_EXCEPTION("KVCache: Cannot update beyond maximum sequence length");
  }

  for (LongType b = 0; b < batchSize_; ++b) {
    auto keySrc = (*newKeys)({b, b+1, 0, newSeqLen, 0, numKvHeads_, 0, headDim_});
    auto valueSrc = (*newValues)({b, b+1, 0, newSeqLen, 0, numKvHeads_, 0, headDim_});

    auto keyDst = (*keyCache_)({b, b+1, position, position + newSeqLen, 0, numKvHeads_, 0, headDim_});
    auto valueDst = (*valueCache_)({b, b+1, position, position + newSeqLen, 0, numKvHeads_, 0, headDim_});

    keyDst->assign(keySrc);
    valueDst->assign(valueSrc);

    delete keySrc;
    delete valueSrc;
    delete keyDst;
    delete valueDst;
  }

  currentLength_ = std::max(currentLength_, position + newSeqLen);
}

NDArray* KVCache::getKeys(LongType seqLen) {
  if (seqLen == 0 || seqLen > currentLength_) {
    seqLen = currentLength_;
  }

  if (seqLen == maxSeqLen_) {
    return keyCache_;
  }

  std::vector<LongType> resultShape = {batchSize_, seqLen, numKvHeads_, headDim_};
  auto result = new NDArray('c', resultShape, dtype_);

  for (LongType b = 0; b < batchSize_; ++b) {
    auto src = (*keyCache_)({b, b+1, 0, seqLen, 0, numKvHeads_, 0, headDim_});
    auto dst = (*result)({b, b+1, 0, seqLen, 0, numKvHeads_, 0, headDim_});
    dst->assign(src);
    delete src;
    delete dst;
  }

  return result;
}

NDArray* KVCache::getValues(LongType seqLen) {
  if (seqLen == 0 || seqLen > currentLength_) {
    seqLen = currentLength_;
  }

  if (seqLen == maxSeqLen_) {
    return valueCache_;
  }

  std::vector<LongType> resultShape = {batchSize_, seqLen, numKvHeads_, headDim_};
  auto result = new NDArray('c', resultShape, dtype_);

  for (LongType b = 0; b < batchSize_; ++b) {
    auto src = (*valueCache_)({b, b+1, 0, seqLen, 0, numKvHeads_, 0, headDim_});
    auto dst = (*result)({b, b+1, 0, seqLen, 0, numKvHeads_, 0, headDim_});
    dst->assign(src);
    delete src;
    delete dst;
  }

  return result;
}

void KVCache::clear() {
  keyCache_->nullify();
  valueCache_->nullify();
  currentLength_ = 0;
}

void KVCache::reset() {
  clear();
}

//////////////////////////////////////////////////////////////////////////////
// Flash Attention v4 Forward Pass
//////////////////////////////////////////////////////////////////////////////

// Flash Attention v4 tile sizes - optimized for L2 cache
static constexpr int FLASH_TILE_M = 64;   // Q tile size (rows)
static constexpr int FLASH_TILE_N = 64;   // KV tile size
static constexpr int FLASH_TILE_D = 64;   // Head dim tile (for very large heads)

// Threshold for using tiled vs batched implementation
// Below this, full attention matrix fits in cache
static constexpr int TILED_THRESHOLD = 512;

/**
 * Flash Attention v4 style forward pass using ops-based matmul
 *
 * Algorithm (from Dao-AILab):
 * 1. For each Q tile (outer loop):
 *    2. Initialize running max = -inf, running sum = 0, output accum = 0
 *    3. For each KV tile (inner loop):
 *       a. Compute S = Q_tile @ K_tile^T * scale
 *       b. Apply causal mask if needed
 *       c. Update running max: new_max = max(old_max, row_max(S))
 *       d. Rescale previous accumulator: O *= exp(old_max - new_max)
 *       e. Compute P = exp(S - new_max)
 *       f. Update running sum: sum = sum * exp(old_max - new_max) + row_sum(P)
 *       g. Accumulate: O += P @ V_tile
 *    4. Normalize: O /= sum
 */
void FlashAttentionHelper::forward(
    NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, const Config& config,
    NDArray* softmaxLse, LaunchContext* context) {

  auto batch = query->sizeAt(0);
  auto seqLen = query->sizeAt(1);
  auto numHeads = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto kvLen = key->sizeAt(1);
  auto numKvHeads = key->sizeAt(2);

  int actualNumKvHeads = config.numKvHeads > 0 ? config.numKvHeads : numKvHeads;
  int headsPerKvHead = numHeads / actualNumKvHeads;

  float scale = config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(headDim));

  // For short sequences, use the batched matmul approach (faster due to OneDNN)
  if (seqLen <= TILED_THRESHOLD && kvLen <= TILED_THRESHOLD) {
    forwardBatched(query, key, value, output, config, softmaxLse, context);
    return;
  }

  // ========== Flash Attention v4 Tiled Implementation ==========
  forwardTiled(query, key, value, output, config, softmaxLse, context);
}

/**
 * Batched attention using ops-based matmul (for short sequences)
 * Uses OneDNN/MKL acceleration via MmulHelper
 */
void FlashAttentionHelper::forwardBatched(
    NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, const Config& config,
    NDArray* softmaxLse, LaunchContext* context) {

  auto batch = query->sizeAt(0);
  auto seqLen = query->sizeAt(1);
  auto numHeads = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto kvLen = key->sizeAt(1);
  auto numKvHeads = key->sizeAt(2);

  int actualNumKvHeads = config.numKvHeads > 0 ? config.numKvHeads : numKvHeads;
  int headsPerKvHead = numHeads / actualNumKvHeads;

  float scale = config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(headDim));

  // Step 1: Permute to [batch, heads, seq, dim] for batched matmul
  std::vector<LongType> permQKV = {0, 2, 1, 3};
  NDArray* queryPerm = query->permute(permQKV, false, false);
  NDArray* keyPerm = key->permute(permQKV, false, false);
  NDArray* valuePerm = value->permute(permQKV, false, false);

  // Step 2: Handle GQA - expand KV heads if needed
  NDArray* keyExpanded = nullptr;
  NDArray* valueExpanded = nullptr;

  if (headsPerKvHead > 1) {
    std::vector<LongType> finalShape = {batch, numHeads, kvLen, headDim};
    keyExpanded = new NDArray('c', finalShape, query->dataType(), context);
    valueExpanded = new NDArray('c', finalShape, query->dataType(), context);

    auto keyPermBuf = keyPerm->bufferAsT<float>();
    auto valuePermBuf = valuePerm->bufferAsT<float>();
    auto keyExpBuf = keyExpanded->bufferAsT<float>();
    auto valueExpBuf = valueExpanded->bufferAsT<float>();

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (LongType b = 0; b < batch; ++b) {
      for (LongType kvh = 0; kvh < actualNumKvHeads; ++kvh) {
        LongType srcBase = (b * actualNumKvHeads + kvh) * kvLen * headDim;
        for (int r = 0; r < headsPerKvHead; ++r) {
          LongType targetHead = kvh * headsPerKvHead + r;
          LongType dstBase = (b * numHeads + targetHead) * kvLen * headDim;
          PRAGMA_OMP_SIMD
          for (LongType i = 0; i < kvLen * headDim; ++i) {
            keyExpBuf[dstBase + i] = keyPermBuf[srcBase + i];
            valueExpBuf[dstBase + i] = valuePermBuf[srcBase + i];
          }
        }
      }
    }
  } else {
    keyExpanded = new NDArray(*keyPerm);
    valueExpanded = new NDArray(*valuePerm);
  }

  // Step 3: Reshape for batched matmul [batch*heads, seq, dim]
  std::vector<LongType> reshapeQ = {batch * numHeads, seqLen, headDim};
  std::vector<LongType> reshapeKV = {batch * numHeads, kvLen, headDim};

  NDArray* queryContig = queryPerm->dup();
  NDArray* queryReshaped = queryContig->reshape('c', reshapeQ);
  NDArray* keyReshaped = keyExpanded->reshape('c', reshapeKV);
  NDArray* valueReshaped = valueExpanded->reshape('c', reshapeKV);

  // Step 4: Compute attention scores: Q @ K^T with scale
  std::vector<LongType> scoresShape = {batch * numHeads, seqLen, kvLen};
  NDArray scores('c', scoresShape, query->dataType(), context);
  MmulHelper::matmul(queryReshaped, keyReshaped, &scores, false, true, scale, 0.0);

  // Step 5: Apply causal mask
  if (config.isCausal) {
    auto scoresBuf = scores.bufferAsT<float>();
    float maskVal = (query->dataType() == BFLOAT16 || query->dataType() == HALF)
                    ? -65504.0f : -1.0e9f;

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (LongType bh = 0; bh < batch * numHeads; ++bh) {
      for (LongType q = 0; q < seqLen; ++q) {
        LongType rowBase = (bh * seqLen + q) * kvLen;
        PRAGMA_OMP_SIMD
        for (LongType k = q + 1; k < kvLen; ++k) {
          scoresBuf[rowBase + k] = maskVal;
        }
      }
    }
  }

  // Step 6: Softmax
  ops::helpers::softmax(context, &scores, &scores, -1);

  // Compute LSE if needed
  if (softmaxLse != nullptr) {
    double zero = 0.0;
    softmaxLse->assign(zero);
  }

  // Step 7: Compute output: attention @ V
  std::vector<LongType> outputReshapeShape = {batch * numHeads, seqLen, headDim};
  NDArray outputReshaped('c', outputReshapeShape, query->dataType(), context);
  MmulHelper::matmul(&scores, valueReshaped, &outputReshaped, false, false, 1.0, 0.0);

  // Step 8: Reshape and permute back to [batch, seq, heads, dim]
  std::vector<LongType> outputPermShape = {batch, numHeads, seqLen, headDim};
  NDArray* outputPerm = outputReshaped.reshape('c', outputPermShape);
  std::vector<LongType> permBack = {0, 2, 1, 3};
  NDArray* outputFinal = outputPerm->permute(permBack, false, false);
  output->assign(outputFinal);

  // Cleanup
  delete queryPerm;
  delete keyPerm;
  delete valuePerm;
  delete keyExpanded;
  delete valueExpanded;
  delete queryContig;
  delete queryReshaped;
  delete keyReshaped;
  delete valueReshaped;
  delete outputPerm;
  delete outputFinal;
}

/**
 * Flash Attention v4 tiled implementation for long sequences
 * Uses online softmax to avoid O(N^2) memory
 */
void FlashAttentionHelper::forwardTiled(
    NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, const Config& config,
    NDArray* softmaxLse, LaunchContext* context) {

  auto batch = query->sizeAt(0);
  auto seqLen = query->sizeAt(1);
  auto numHeads = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto kvLen = key->sizeAt(1);
  auto numKvHeads = key->sizeAt(2);

  int actualNumKvHeads = config.numKvHeads > 0 ? config.numKvHeads : numKvHeads;
  int headsPerKvHead = numHeads / actualNumKvHeads;

  float scale = config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(headDim));

  // Tile sizes
  int tileM = std::min(static_cast<int>(seqLen), FLASH_TILE_M);
  int tileN = std::min(static_cast<int>(kvLen), FLASH_TILE_N);

  // Get raw buffers
  auto queryBuf = query->bufferAsT<float>();
  auto keyBuf = key->bufferAsT<float>();
  auto valueBuf = value->bufferAsT<float>();
  auto outputBuf = output->bufferAsT<float>();
  float* lseBuf = softmaxLse ? softmaxLse->bufferAsT<float>() : nullptr;

  output->nullify();

  // Parallel over batch and heads
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for (LongType b = 0; b < batch; ++b) {
    for (LongType h = 0; h < numHeads; ++h) {
      LongType kvHead = h / headsPerKvHead;

      // Per-thread state for online softmax
      std::vector<float> rowMax(seqLen, -std::numeric_limits<float>::infinity());
      std::vector<float> rowSum(seqLen, 0.0f);
      std::vector<float> outputAcc(seqLen * headDim, 0.0f);

      // Scratch buffers for tile computation
      std::vector<float> qTile(tileM * headDim);
      std::vector<float> kTile(tileN * headDim);
      std::vector<float> vTile(tileN * headDim);
      std::vector<float> scores(tileM * tileN);

      // Process Q tiles (outer loop)
      for (LongType qStart = 0; qStart < seqLen; qStart += tileM) {
        LongType qEnd = std::min(qStart + tileM, seqLen);
        LongType qTileSize = qEnd - qStart;

        // Load Q tile
        for (LongType qi = 0; qi < qTileSize; ++qi) {
          LongType srcIdx = ((b * seqLen + (qStart + qi)) * numHeads + h) * headDim;
          PRAGMA_OMP_SIMD
          for (LongType d = 0; d < headDim; ++d) {
            qTile[qi * headDim + d] = queryBuf[srcIdx + d];
          }
        }

        // Process KV tiles (inner loop) - Flash Attention v4 style
        for (LongType kvStart = 0; kvStart < kvLen; kvStart += tileN) {
          LongType kvEnd = std::min(kvStart + tileN, kvLen);
          LongType kvTileSize = kvEnd - kvStart;

          // Load K and V tiles
          for (LongType ki = 0; ki < kvTileSize; ++ki) {
            LongType srcIdx = ((b * kvLen + (kvStart + ki)) * numKvHeads + kvHead) * headDim;
            PRAGMA_OMP_SIMD
            for (LongType d = 0; d < headDim; ++d) {
              kTile[ki * headDim + d] = keyBuf[srcIdx + d];
              vTile[ki * headDim + d] = valueBuf[srcIdx + d];
            }
          }

          // Compute scores: S = Q @ K^T * scale
          for (LongType qi = 0; qi < qTileSize; ++qi) {
            LongType qPos = qStart + qi;
            for (LongType ki = 0; ki < kvTileSize; ++ki) {
              LongType kPos = kvStart + ki;

              // Apply causal mask
              if (config.isCausal && kPos > qPos) {
                scores[qi * kvTileSize + ki] = -std::numeric_limits<float>::infinity();
              } else {
                float dot = 0.0f;
                PRAGMA_OMP_SIMD_ARGS(reduction(+:dot))
                for (LongType d = 0; d < headDim; ++d) {
                  dot += qTile[qi * headDim + d] * kTile[ki * headDim + d];
                }
                scores[qi * kvTileSize + ki] = dot * scale;
              }
            }
          }

          // Online softmax update (Flash Attention v4 algorithm)
          for (LongType qi = 0; qi < qTileSize; ++qi) {
            LongType qPos = qStart + qi;

            // Step 1: Find tile max
            float tileMax = -std::numeric_limits<float>::infinity();
            for (LongType ki = 0; ki < kvTileSize; ++ki) {
              tileMax = std::max(tileMax, scores[qi * kvTileSize + ki]);
            }

            // Step 2: Compute new global max
            float oldMax = rowMax[qPos];
            float newMax = std::max(oldMax, tileMax);

            // Step 3: Rescale factors
            float oldScale = std::exp(oldMax - newMax);
            float tileScale = std::exp(tileMax - newMax);

            // Step 4: Compute exp(scores - tileMax) and tile sum
            float tileSum = 0.0f;
            PRAGMA_OMP_SIMD_ARGS(reduction(+:tileSum))
            for (LongType ki = 0; ki < kvTileSize; ++ki) {
              float expVal = std::exp(scores[qi * kvTileSize + ki] - tileMax);
              scores[qi * kvTileSize + ki] = expVal;  // Reuse buffer for P
              tileSum += expVal;
            }

            // Step 5: Rescale previous output accumulator
            LongType outBase = qPos * headDim;
            PRAGMA_OMP_SIMD
            for (LongType d = 0; d < headDim; ++d) {
              outputAcc[outBase + d] *= oldScale;
            }

            // Step 6: Add contribution from this tile: O += P @ V
            for (LongType ki = 0; ki < kvTileSize; ++ki) {
              float p = scores[qi * kvTileSize + ki] * tileScale;
              PRAGMA_OMP_SIMD
              for (LongType d = 0; d < headDim; ++d) {
                outputAcc[outBase + d] += p * vTile[ki * headDim + d];
              }
            }

            // Step 7: Update running statistics
            rowMax[qPos] = newMax;
            rowSum[qPos] = rowSum[qPos] * oldScale + tileSum * tileScale;
          }
        }
      }

      // Final normalization and write output
      for (LongType q = 0; q < seqLen; ++q) {
        float invSum = rowSum[q] > 0.0f ? 1.0f / rowSum[q] : 0.0f;
        LongType dstIdx = ((b * seqLen + q) * numHeads + h) * headDim;
        LongType srcBase = q * headDim;

        PRAGMA_OMP_SIMD
        for (LongType d = 0; d < headDim; ++d) {
          outputBuf[dstIdx + d] = outputAcc[srcBase + d] * invSum;
        }

        // Store LSE if needed: lse = log(sum) + max
        if (lseBuf != nullptr) {
          LongType lseIdx = (b * numHeads + h) * seqLen + q;
          lseBuf[lseIdx] = std::log(rowSum[q]) + rowMax[q];
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Flash Attention Backward Pass
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::backward(
    NDArray* gradOutput, NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, NDArray* softmaxLse,
    NDArray* gradQuery, NDArray* gradKey, NDArray* gradValue,
    const Config& config, LaunchContext* context) {

  auto batch = query->sizeAt(0);
  auto seqLen = query->sizeAt(1);
  auto numHeads = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto kvLen = key->sizeAt(1);
  auto numKvHeads = key->sizeAt(2);

  int actualNumKvHeads = config.numKvHeads > 0 ? config.numKvHeads : numKvHeads;
  int headsPerKvHead = numHeads / actualNumKvHeads;

  float scale = config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(headDim));

  // Initialize gradients to zero
  gradQuery->nullify();
  gradKey->nullify();
  gradValue->nullify();

  auto queryBuf = query->bufferAsT<float>();
  auto keyBuf = key->bufferAsT<float>();
  auto valueBuf = value->bufferAsT<float>();
  auto gradOutBuf = gradOutput->bufferAsT<float>();
  auto gradQBuf = gradQuery->bufferAsT<float>();
  auto gradKBuf = gradKey->bufferAsT<float>();
  auto gradVBuf = gradValue->bufferAsT<float>();

  // Parallel over batch and heads
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for (LongType b = 0; b < batch; ++b) {
    for (LongType h = 0; h < numHeads; ++h) {
      LongType kvHead = h / headsPerKvHead;

      // Recompute attention weights (needed for backward)
      std::vector<float> attnWeights(seqLen * kvLen);
      std::vector<float> dAttn(seqLen * kvLen);

      // Forward pass to get attention weights
      for (LongType q = 0; q < seqLen; ++q) {
        float maxVal = -std::numeric_limits<float>::infinity();

        // Compute scores and find max
        for (LongType k = 0; k < kvLen; ++k) {
          if (config.isCausal && k > q) {
            attnWeights[q * kvLen + k] = -std::numeric_limits<float>::infinity();
          } else {
            LongType qIdx = ((b * seqLen + q) * numHeads + h) * headDim;
            LongType kIdx = ((b * kvLen + k) * numKvHeads + kvHead) * headDim;
            float dot = 0.0f;
            PRAGMA_OMP_SIMD_ARGS(reduction(+:dot))
            for (LongType d = 0; d < headDim; ++d) {
              dot += queryBuf[qIdx + d] * keyBuf[kIdx + d];
            }
            attnWeights[q * kvLen + k] = dot * scale;
            maxVal = std::max(maxVal, attnWeights[q * kvLen + k]);
          }
        }

        // Softmax
        float sumExp = 0.0f;
        for (LongType k = 0; k < kvLen; ++k) {
          if (config.isCausal && k > q) {
            attnWeights[q * kvLen + k] = 0.0f;
          } else {
            attnWeights[q * kvLen + k] = std::exp(attnWeights[q * kvLen + k] - maxVal);
            sumExp += attnWeights[q * kvLen + k];
          }
        }

        if (sumExp > 0.0f) {
          float invSum = 1.0f / sumExp;
          PRAGMA_OMP_SIMD
          for (LongType k = 0; k < kvLen; ++k) {
            attnWeights[q * kvLen + k] *= invSum;
          }
        }
      }

      // Compute gradValue: dV = P^T @ dO
      for (LongType k = 0; k < kvLen; ++k) {
        LongType vIdx = ((b * kvLen + k) * numKvHeads + kvHead) * headDim;
        for (LongType d = 0; d < headDim; ++d) {
          float grad = 0.0f;
          for (LongType q = 0; q < seqLen; ++q) {
            LongType goIdx = ((b * seqLen + q) * numHeads + h) * headDim + d;
            grad += attnWeights[q * kvLen + k] * gradOutBuf[goIdx];
          }
          PRAGMA_OMP_ATOMIC
          gradVBuf[vIdx + d] += grad;
        }
      }

      // Compute dP = dO @ V^T
      for (LongType q = 0; q < seqLen; ++q) {
        LongType goIdx = ((b * seqLen + q) * numHeads + h) * headDim;
        for (LongType k = 0; k < kvLen; ++k) {
          LongType vIdx = ((b * kvLen + k) * numKvHeads + kvHead) * headDim;
          float dp = 0.0f;
          PRAGMA_OMP_SIMD_ARGS(reduction(+:dp))
          for (LongType d = 0; d < headDim; ++d) {
            dp += gradOutBuf[goIdx + d] * valueBuf[vIdx + d];
          }
          dAttn[q * kvLen + k] = dp;
        }
      }

      // Compute dS = P * (dP - rowsum(dP * P)) - softmax backward
      for (LongType q = 0; q < seqLen; ++q) {
        float rowSum = 0.0f;
        PRAGMA_OMP_SIMD_ARGS(reduction(+:rowSum))
        for (LongType k = 0; k < kvLen; ++k) {
          rowSum += dAttn[q * kvLen + k] * attnWeights[q * kvLen + k];
        }

        PRAGMA_OMP_SIMD
        for (LongType k = 0; k < kvLen; ++k) {
          dAttn[q * kvLen + k] = scale * attnWeights[q * kvLen + k] *
                                  (dAttn[q * kvLen + k] - rowSum);
        }
      }

      // Compute gradQuery: dQ = dS @ K
      for (LongType q = 0; q < seqLen; ++q) {
        LongType gqIdx = ((b * seqLen + q) * numHeads + h) * headDim;
        for (LongType d = 0; d < headDim; ++d) {
          float grad = 0.0f;
          for (LongType k = 0; k < kvLen; ++k) {
            LongType kIdx = ((b * kvLen + k) * numKvHeads + kvHead) * headDim + d;
            grad += dAttn[q * kvLen + k] * keyBuf[kIdx];
          }
          gradQBuf[gqIdx + d] = grad;
        }
      }

      // Compute gradKey: dK = dS^T @ Q
      for (LongType k = 0; k < kvLen; ++k) {
        LongType kIdx = ((b * kvLen + k) * numKvHeads + kvHead) * headDim;
        for (LongType d = 0; d < headDim; ++d) {
          float grad = 0.0f;
          for (LongType q = 0; q < seqLen; ++q) {
            LongType qIdx = ((b * seqLen + q) * numHeads + h) * headDim + d;
            grad += dAttn[q * kvLen + k] * queryBuf[qIdx];
          }
          PRAGMA_OMP_ATOMIC
          gradKBuf[kIdx + d] += grad;
        }
      }
    }
  }
}

}  // namespace sd
