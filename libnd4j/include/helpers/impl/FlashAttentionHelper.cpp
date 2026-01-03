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
// Flash Attention Helper - Optimized batched matmul approach
// Single batched matmul call for all heads, minimal data movement
//

#include <helpers/FlashAttentionHelper.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/activations.h>
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <system/Environment.h>
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

  auto keyCacheSlice = (*keyCache_)({0, batchSize_, position, position + newSeqLen, 0, numKvHeads_, 0, headDim_});
  auto valueCacheSlice = (*valueCache_)({0, batchSize_, position, position + newSeqLen, 0, numKvHeads_, 0, headDim_});
  keyCacheSlice->assign(newKeys);
  valueCacheSlice->assign(newValues);
  delete keyCacheSlice;
  delete valueCacheSlice;

  currentLength_ = std::max(currentLength_, position + newSeqLen);
}

NDArray* KVCache::getKeys(LongType seqLen) {
  if (seqLen == 0 || seqLen > currentLength_) seqLen = currentLength_;
  if (seqLen == maxSeqLen_) return keyCache_;
  return (*keyCache_)({0, batchSize_, 0, seqLen, 0, numKvHeads_, 0, headDim_});
}

NDArray* KVCache::getValues(LongType seqLen) {
  if (seqLen == 0 || seqLen > currentLength_) seqLen = currentLength_;
  if (seqLen == maxSeqLen_) return valueCache_;
  return (*valueCache_)({0, batchSize_, 0, seqLen, 0, numKvHeads_, 0, headDim_});
}

void KVCache::clear() {
  keyCache_->nullify();
  valueCache_->nullify();
  currentLength_ = 0;
}

void KVCache::reset() { clear(); }

//////////////////////////////////////////////////////////////////////////////
// Flash Attention Forward - Supports both 3D and 4D inputs
// 3D: [batch, seqLen, dim] - single head attention
// 4D: [batch, seqLen, numHeads, headDim] - multi-head attention
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::forward(
    NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, const Config& config,
    NDArray* softmaxLse, NDArray* attentionScores,
    NDArray* attentionLogits, LaunchContext* context) {

  auto rank = query->rankOf();
  REQUIRE_TRUE(rank == 3 || rank == 4, 0,
               "FlashAttentionHelper::forward: query rank must be 3 or 4, got %i", rank);
  REQUIRE_TRUE(key->rankOf() == rank && value->rankOf() == rank, 0,
               "FlashAttentionHelper::forward: query, key, value must have same rank");

  if (rank == 3) {
    forward3D(query, key, value, output, config, softmaxLse, attentionScores, attentionLogits, context);
  } else {
    forward4D(query, key, value, output, config, softmaxLse, attentionScores, attentionLogits, context);
  }
}

//////////////////////////////////////////////////////////////////////////////
// 3D Forward Implementation - [batch, seq, dim]
// Uses OneDNN Graph SDPA when available for fused kernel execution
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::forward3D(
    NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, const Config& config,
    NDArray* softmaxLse, NDArray* attentionScores,
    NDArray* attentionLogits, LaunchContext* context) {

  auto batch = query->sizeAt(0);
  auto seqLenQ = query->sizeAt(1);
  auto dim = query->sizeAt(2);
  auto seqLenKV = key->sizeAt(1);

  float scale = config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(dim));

  // Reshape for batched matmul: already [batch, seq, dim] which is perfect
  std::vector<LongType> scoresShape = {batch, seqLenQ, seqLenKV};

  // Batched matmul: Q @ K^T with scale -> [batch, seqQ, seqKV]
  NDArray scores('c', scoresShape, query->dataType(), context);
  MmulHelper::matmul(query, key, &scores, false, true, scale, 0.0);

  // Apply causal mask if needed
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
    std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
    NDArray causalMask('c', maskShape, query->dataType(), context);
    causalMask.nullify();

    float maskVal = -1.0e9f;
    for (LongType q = 0; q < seqLenQ; ++q) {
      if (q + 1 < seqLenKV) {
        auto rowSlice = causalMask({q, q+1, q+1, seqLenKV});
        if (rowSlice->lengthOf() > 0) {
          rowSlice->assign(maskVal);
        }
        delete rowSlice;
      }
    }

    // Broadcast add mask to scores
    scores += causalMask;
  }

  // Copy pre-softmax scores (logits) if requested
  if (attentionLogits != nullptr) {
    attentionLogits->assign(&scores);
  }

  // Softmax along last dimension
  ops::helpers::softmax(context, &scores, &scores, -1);

  // Copy post-softmax scores if requested
  if (attentionScores != nullptr) {
    attentionScores->assign(&scores);
  }

  // Batched matmul: scores @ V -> [batch, seqQ, dim]
  MmulHelper::matmul(&scores, value, output, false, false, 1.0, 0.0);

  if (softmaxLse != nullptr) {
    softmaxLse->nullify();
  }
}

//////////////////////////////////////////////////////////////////////////////
// 4D Forward Implementation - [batch, seq, numHeads, headDim]
// Uses permute to [batch, numHeads, seqLen, headDim] then batched matmul
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::forward4D(
    NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, const Config& config,
    NDArray* softmaxLse, NDArray* attentionScores,
    NDArray* attentionLogits, LaunchContext* context) {

  auto batch = query->sizeAt(0);
  auto seqLenQ = query->sizeAt(1);
  auto numHeads = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto seqLenKV = key->sizeAt(1);
  auto numKvHeads = key->sizeAt(2);

  float scale = config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(headDim));
  int headsPerKvHead = numHeads / numKvHeads;

  // Permute: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
  std::vector<LongType> permOrder = {0, 2, 1, 3};
  auto qPerm = query->permute(permOrder, false, false);   // [batch, numHeads, seqQ, headDim]
  auto kPerm = key->permute(permOrder, false, false);     // [batch, numKvHeads, seqKV, headDim]
  auto vPerm = value->permute(permOrder, false, false);   // [batch, numKvHeads, seqKV, headDim]

  // Expand KV heads for GQA if needed - use tile operation instead of loops
  NDArray* kExpanded = kPerm;
  NDArray* vExpanded = vPerm;
  bool expandedKV = false;

  if (headsPerKvHead > 1) {
    // Tile KV heads: [batch, numKvHeads, seq, dim] -> [batch, numHeads, seq, dim]
    // Reshape to [batch, numKvHeads, 1, seq, dim] then tile [1, 1, headsPerKvHead, 1, 1]
    std::vector<LongType> reshapeForTile = {batch, numKvHeads, 1, seqLenKV, headDim};
    auto kReshaped = kPerm->reshape('c', reshapeForTile);
    auto vReshaped = vPerm->reshape('c', reshapeForTile);

    std::vector<LongType> reps = {1, 1, static_cast<LongType>(headsPerKvHead), 1, 1};
    NDArray kTiled = kReshaped->tile(reps);
    NDArray vTiled = vReshaped->tile(reps);

    // Reshape to [batch, numHeads, seq, dim]
    std::vector<LongType> expandedShape = {batch, numHeads, seqLenKV, headDim};
    kExpanded = kTiled.reshape('c', expandedShape);
    vExpanded = vTiled.reshape('c', expandedShape);
    expandedKV = true;

    delete kReshaped;
    delete vReshaped;
  }

  // Reshape for batched matmul: [batch * numHeads, seq, dim]
  std::vector<LongType> qShape = {batch * numHeads, seqLenQ, headDim};
  std::vector<LongType> kvShape = {batch * numHeads, seqLenKV, headDim};
  std::vector<LongType> scoresShape = {batch * numHeads, seqLenQ, seqLenKV};

  auto qReshaped = qPerm->reshape('c', qShape);
  auto kReshaped = kExpanded->reshape('c', kvShape);
  auto vReshaped = vExpanded->reshape('c', kvShape);

  // Batched matmul: Q @ K^T with scale -> [batch*heads, seqQ, seqKV]
  NDArray scores('c', scoresShape, query->dataType(), context);
  MmulHelper::matmul(qReshaped, kReshaped, &scores, false, true, scale, 0.0);

  // Apply causal mask if needed - use upper triangular fill
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
    // Create causal mask: lower triangular = 0, upper triangular = -inf
    // For each (q, k) position: mask if k > q
    std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
    NDArray causalMask('c', maskShape, query->dataType(), context);
    causalMask.nullify();

    // Fill upper triangular with large negative value
    float maskVal = -1.0e9f;
    for (LongType q = 0; q < seqLenQ; ++q) {
      auto rowSlice = causalMask({q, q+1, q+1, seqLenKV});
      if (rowSlice->lengthOf() > 0) {
        rowSlice->assign(maskVal);
      }
      delete rowSlice;
    }

    // Broadcast add mask to scores [batch*heads, seqQ, seqKV]
    scores += causalMask;
  }

  // Copy pre-softmax scores (logits) if requested
  if (attentionLogits != nullptr) {
    attentionLogits->assign(&scores);
  }

  // Softmax along last dimension
  ops::helpers::softmax(context, &scores, &scores, -1);

  // Copy post-softmax scores (attention weights) if requested
  if (attentionScores != nullptr) {
    attentionScores->assign(&scores);
  }

  // Batched matmul: scores @ V -> [batch*heads, seqQ, headDim]
  std::vector<LongType> outShape = {batch * numHeads, seqLenQ, headDim};
  NDArray outReshaped('c', outShape, query->dataType(), context);
  MmulHelper::matmul(&scores, vReshaped, &outReshaped, false, false, 1.0, 0.0);

  // Reshape back: [batch, numHeads, seqQ, headDim]
  std::vector<LongType> outPermShape = {batch, numHeads, seqLenQ, headDim};
  auto outPerm = outReshaped.reshape('c', outPermShape);

  // Permute back: [batch, numHeads, seqQ, headDim] -> [batch, seqQ, numHeads, headDim]
  std::vector<LongType> permBack = {0, 2, 1, 3};
  auto outFinal = outPerm->permute(permBack, false, false);
  output->assign(outFinal);

  // Cleanup
  delete qPerm;
  delete kPerm;
  delete vPerm;
  if (expandedKV) {
    delete kExpanded;
    delete vExpanded;
  }
  delete qReshaped;
  delete kReshaped;
  delete vReshaped;
  delete outPerm;
  delete outFinal;

  if (softmaxLse != nullptr) {
    softmaxLse->nullify();
  }
}

//////////////////////////////////////////////////////////////////////////////
// Flash Attention Backward - Supports both 3D and 4D inputs
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::backward(
    NDArray* gradOutput, NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, NDArray* softmaxLse,
    NDArray* gradQuery, NDArray* gradKey, NDArray* gradValue,
    const Config& config, LaunchContext* context) {

  auto rank = query->rankOf();
  REQUIRE_TRUE(rank == 3 || rank == 4, 0,
               "FlashAttentionHelper::backward: query rank must be 3 or 4, got %i", rank);

  if (rank == 3) {
    backward3D(gradOutput, query, key, value, output, softmaxLse,
               gradQuery, gradKey, gradValue, config, context);
  } else {
    backward4D(gradOutput, query, key, value, output, softmaxLse,
               gradQuery, gradKey, gradValue, config, context);
  }
}

//////////////////////////////////////////////////////////////////////////////
// 3D Backward Implementation - [batch, seq, dim]
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::backward3D(
    NDArray* gradOutput, NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, NDArray* softmaxLse,
    NDArray* gradQuery, NDArray* gradKey, NDArray* gradValue,
    const Config& config, LaunchContext* context) {

  auto batch = query->sizeAt(0);
  auto seqLenQ = query->sizeAt(1);
  auto dim = query->sizeAt(2);
  auto seqLenKV = key->sizeAt(1);

  float scale = config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(dim));

  std::vector<LongType> scoresShape = {batch, seqLenQ, seqLenKV};

  // Recompute attention: Q @ K^T
  NDArray scores('c', scoresShape, query->dataType(), context);
  MmulHelper::matmul(query, key, &scores, false, true, scale, 0.0);

  // Causal mask
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
    std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
    NDArray causalMask('c', maskShape, query->dataType(), context);
    causalMask.nullify();

    float maskVal = -1.0e9f;
    for (LongType q = 0; q < seqLenQ; ++q) {
      if (q + 1 < seqLenKV) {
        auto rowSlice = causalMask({q, q+1, q+1, seqLenKV});
        if (rowSlice->lengthOf() > 0) {
          rowSlice->assign(maskVal);
        }
        delete rowSlice;
      }
    }
    scores += causalMask;
  }

  // Softmax
  NDArray attnWeights('c', scoresShape, query->dataType(), context);
  ops::helpers::softmax(context, &scores, &attnWeights, -1);

  // gradValue = attnWeights^T @ gradOutput -> [batch, seqKV, dim]
  MmulHelper::matmul(&attnWeights, gradOutput, gradValue, true, false, 1.0, 0.0);

  // dAttn = gradOutput @ V^T -> [batch, seqQ, seqKV]
  NDArray dAttn('c', scoresShape, query->dataType(), context);
  MmulHelper::matmul(gradOutput, value, &dAttn, false, true, 1.0, 0.0);

  // Softmax backward: dS = P * (dAttn - sum(dAttn * P))
  auto dAttnTimesPPtr = dAttn * attnWeights;
  NDArray dAttnTimesP(*dAttnTimesPPtr);
  delete dAttnTimesPPtr;

  std::vector<LongType> sumDims = {2};
  auto rowSums = dAttnTimesP.reduceAlongDimension(reduce::Sum, &sumDims, true);
  dAttn -= *rowSums;
  dAttn *= attnWeights;
  dAttn *= scale;
  delete rowSums;

  // gradQuery = dS @ K -> [batch, seqQ, dim]
  MmulHelper::matmul(&dAttn, key, gradQuery, false, false, 1.0, 0.0);

  // gradKey = dS^T @ Q -> [batch, seqKV, dim]
  MmulHelper::matmul(&dAttn, query, gradKey, true, false, 1.0, 0.0);
}

//////////////////////////////////////////////////////////////////////////////
// 4D Backward Implementation - [batch, seq, numHeads, headDim]
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::backward4D(
    NDArray* gradOutput, NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, NDArray* softmaxLse,
    NDArray* gradQuery, NDArray* gradKey, NDArray* gradValue,
    const Config& config, LaunchContext* context) {

  auto batch = query->sizeAt(0);
  auto seqLenQ = query->sizeAt(1);
  auto numHeads = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto seqLenKV = key->sizeAt(1);
  auto numKvHeads = key->sizeAt(2);

  float scale = config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(headDim));
  int headsPerKvHead = numHeads / numKvHeads;

  // Permute to [batch, heads, seq, dim]
  std::vector<LongType> permOrder = {0, 2, 1, 3};
  auto qPerm = query->permute(permOrder, false, false);
  auto kPerm = key->permute(permOrder, false, false);
  auto vPerm = value->permute(permOrder, false, false);
  auto goPerm = gradOutput->permute(permOrder, false, false);

  // Expand KV for GQA - use tile operation instead of loops
  NDArray* kExpanded = kPerm;
  NDArray* vExpanded = vPerm;
  bool expandedKV = false;

  if (headsPerKvHead > 1) {
    // Reshape to [batch, numKvHeads, 1, seq, dim] then tile
    std::vector<LongType> reshapeForTile = {batch, numKvHeads, 1, seqLenKV, headDim};
    auto kReshaped = kPerm->reshape('c', reshapeForTile);
    auto vReshaped = vPerm->reshape('c', reshapeForTile);

    std::vector<LongType> reps = {1, 1, static_cast<LongType>(headsPerKvHead), 1, 1};
    NDArray kTiled = kReshaped->tile(reps);
    NDArray vTiled = vReshaped->tile(reps);

    std::vector<LongType> expandedShape = {batch, numHeads, seqLenKV, headDim};
    kExpanded = kTiled.reshape('c', expandedShape);
    vExpanded = vTiled.reshape('c', expandedShape);
    expandedKV = true;

    delete kReshaped;
    delete vReshaped;
  }

  // Reshape for batched ops
  std::vector<LongType> qShape = {batch * numHeads, seqLenQ, headDim};
  std::vector<LongType> kvShape = {batch * numHeads, seqLenKV, headDim};
  std::vector<LongType> scoresShape = {batch * numHeads, seqLenQ, seqLenKV};

  auto qReshaped = qPerm->reshape('c', qShape);
  auto kReshaped = kExpanded->reshape('c', kvShape);
  auto vReshaped = vExpanded->reshape('c', kvShape);
  auto goReshaped = goPerm->reshape('c', qShape);

  // Recompute attention: Q @ K^T
  NDArray scores('c', scoresShape, query->dataType(), context);
  MmulHelper::matmul(qReshaped, kReshaped, &scores, false, true, scale, 0.0);

  // Causal mask - use vectorized approach
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
    std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
    NDArray causalMask('c', maskShape, query->dataType(), context);
    causalMask.nullify();

    float maskVal = -1.0e9f;
    for (LongType q = 0; q < seqLenQ; ++q) {
      auto rowSlice = causalMask({q, q+1, q+1, seqLenKV});
      if (rowSlice->lengthOf() > 0) {
        rowSlice->assign(maskVal);
      }
      delete rowSlice;
    }
    scores += causalMask;
  }

  // Softmax
  NDArray attnWeights('c', scoresShape, query->dataType(), context);
  ops::helpers::softmax(context, &scores, &attnWeights, -1);

  // gradValue = attnWeights^T @ gradOutput -> [batch*heads, seqKV, headDim]
  NDArray gvReshaped('c', kvShape, query->dataType(), context);
  MmulHelper::matmul(&attnWeights, goReshaped, &gvReshaped, true, false, 1.0, 0.0);

  // dAttn = gradOutput @ V^T -> [batch*heads, seqQ, seqKV]
  NDArray dAttn('c', scoresShape, query->dataType(), context);
  MmulHelper::matmul(goReshaped, vReshaped, &dAttn, false, true, 1.0, 0.0);

  // Softmax backward: dS = P * (dAttn - sum(dAttn * P)) - vectorized
  // dAttn * P element-wise
  auto dAttnTimesPPtr = dAttn * attnWeights;
  NDArray dAttnTimesP(*dAttnTimesPPtr);
  delete dAttnTimesPPtr;

  // Sum along last axis: [batch*heads, seqQ, 1]
  std::vector<LongType> sumDims = {2};
  auto rowSums = dAttnTimesP.reduceAlongDimension(reduce::Sum, &sumDims, true);

  // dAttn - rowSums (broadcast)
  dAttn -= *rowSums;

  // P * (dAttn - rowSums) * scale
  dAttn *= attnWeights;
  dAttn *= scale;

  delete rowSums;

  // gradQuery = dS @ K -> [batch*heads, seqQ, headDim]
  NDArray gqReshaped('c', qShape, query->dataType(), context);
  MmulHelper::matmul(&dAttn, kReshaped, &gqReshaped, false, false, 1.0, 0.0);

  // gradKey = dS^T @ Q -> [batch*heads, seqKV, headDim]
  NDArray gkReshaped('c', kvShape, query->dataType(), context);
  MmulHelper::matmul(&dAttn, qReshaped, &gkReshaped, true, false, 1.0, 0.0);

  // Reshape and permute gradients back
  std::vector<LongType> gqPermShape = {batch, numHeads, seqLenQ, headDim};
  std::vector<LongType> gkvPermShape = {batch, numHeads, seqLenKV, headDim};
  auto gqPerm = gqReshaped.reshape('c', gqPermShape);
  auto gkPerm = gkReshaped.reshape('c', gkvPermShape);
  auto gvPerm = gvReshaped.reshape('c', gkvPermShape);

  std::vector<LongType> permBack = {0, 2, 1, 3};
  auto gqFinal = gqPerm->permute(permBack, false, false);
  gradQuery->assign(gqFinal);

  // For GQA, accumulate gradients to KV heads - vectorized with reshape and reduce
  if (headsPerKvHead > 1) {
    // Reshape [batch, numHeads, seqKV, headDim] -> [batch, numKvHeads, headsPerKvHead, seqKV, headDim]
    std::vector<LongType> reshapeForSum = {batch, numKvHeads, static_cast<LongType>(headsPerKvHead), seqLenKV, headDim};
    auto gkForSum = gkPerm->reshape('c', reshapeForSum);
    auto gvForSum = gvPerm->reshape('c', reshapeForSum);

    // Sum along the headsPerKvHead axis -> [batch, numKvHeads, seqKV, headDim]
    std::vector<LongType> reduceDims = {2};
    auto gkReduced = gkForSum->reduceAlongDimension(reduce::Sum, &reduceDims, false);
    auto gvReduced = gvForSum->reduceAlongDimension(reduce::Sum, &reduceDims, false);

    // Permute to output format [batch, seqKV, numKvHeads, headDim]
    auto gkFinal = gkReduced->permute(permBack, false, false);
    auto gvFinal = gvReduced->permute(permBack, false, false);
    gradKey->assign(gkFinal);
    gradValue->assign(gvFinal);

    delete gkForSum;
    delete gvForSum;
    delete gkReduced;
    delete gvReduced;
    delete gkFinal;
    delete gvFinal;
  } else {
    auto gkFinal = gkPerm->permute(permBack, false, false);
    auto gvFinal = gvPerm->permute(permBack, false, false);
    gradKey->assign(gkFinal);
    gradValue->assign(gvFinal);
    delete gkFinal;
    delete gvFinal;
  }

  // Cleanup
  delete qPerm; delete kPerm; delete vPerm; delete goPerm;
  if (expandedKV) { delete kExpanded; delete vExpanded; }
  delete qReshaped; delete kReshaped; delete vReshaped; delete goReshaped;
  delete gqPerm; delete gkPerm; delete gvPerm;
  delete gqFinal;
}

//////////////////////////////////////////////////////////////////////////////
// Utility functions
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::standardAttention(
    NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, const Config& config,
    NDArray* attentionWeights, LaunchContext* context) {
  forward(query, key, value, output, config, nullptr, attentionWeights, nullptr, context);
}

void FlashAttentionHelper::repeatKVHeads(NDArray* kv, int numHeads, NDArray* output) {
  auto batch = kv->sizeAt(0);
  auto seqLen = kv->sizeAt(1);
  auto numKvHeads = kv->sizeAt(2);
  auto headDim = kv->sizeAt(3);
  int headsPerKvHead = numHeads / numKvHeads;

  if (headsPerKvHead == 1) {
    output->assign(kv);
    return;
  }

  // Reshape [batch, seq, numKvHeads, dim] -> [batch, seq, numKvHeads, 1, dim]
  std::vector<LongType> reshapeForTile = {batch, seqLen, numKvHeads, 1, headDim};
  auto kvReshaped = kv->reshape('c', reshapeForTile);

  // Tile along the new axis: [1, 1, 1, headsPerKvHead, 1]
  std::vector<LongType> reps = {1, 1, 1, static_cast<LongType>(headsPerKvHead), 1};
  NDArray kvTiled = kvReshaped->tile(reps);

  // Reshape to output [batch, seq, numHeads, dim]
  std::vector<LongType> outShape = {batch, seqLen, static_cast<LongType>(numHeads), headDim};
  auto result = kvTiled.reshape('c', outShape);
  output->assign(result);

  delete kvReshaped;
  delete result;
}

void FlashAttentionHelper::forwardWithKVCache(
    NDArray* query, KVCache* kvCache,
    NDArray* newKey, NDArray* newValue,
    NDArray* output, LongType position,
    const Config& config, LaunchContext* context) {

  kvCache->update(newKey, newValue, position);
  auto cachedKeys = kvCache->getKeys();
  auto cachedValues = kvCache->getValues();
  forward(query, cachedKeys, cachedValues, output, config, nullptr, nullptr, nullptr, context);
  if (cachedKeys != kvCache->getKeyCache()) delete cachedKeys;
  if (cachedValues != kvCache->getValueCache()) delete cachedValues;
}

//////////////////////////////////////////////////////////////////////////////
// Platform Detection Functions
//////////////////////////////////////////////////////////////////////////////

bool FlashAttentionHelper::canUseOneDnnSdpa(NDArray* query, NDArray* key, NDArray* value,
                                            const Config& config) {
  // Check if OneDNN is available and helpers are allowed
  if (!Environment::getInstance().helpersAllowed()) {
    return false;
  }

  // Only 3D tensors are supported by OneDNN Graph SDPA
  if (query->rankOf() != 3) {
    return false;
  }

  // Check supported data types: F32, F16, BF16
  auto dtype = query->dataType();
  if (dtype != DataType::FLOAT32 && dtype != DataType::HALF && dtype != DataType::BFLOAT16) {
    return false;
  }

  // OneDNN Graph SDPA doesn't support custom masks (only causal through graph)
  // Dropout is also not supported in fused kernel
  if (config.dropout > 0.0f) {
    return false;
  }

  return true;
}

bool FlashAttentionHelper::canUseCudnnSdpa(NDArray* query, NDArray* key, NDArray* value,
                                           const Config& config) {
#if defined(HAVE_CUDNN) && CUDNN_MAJOR >= 8 && CUDNN_MINOR >= 9
  // cuDNN 8.9.0+ has flash attention support
  if (!Environment::getInstance().helpersAllowed()) {
    return false;
  }

  // Check if running on CUDA device
  if (query->getContext()->getWorkspace() == nullptr ||
      query->getContext()->getWorkspace()->deviceType() != sd::graph::DeviceType::GPU) {
    return false;
  }

  // Check supported data types
  auto dtype = query->dataType();
  if (dtype != DataType::FLOAT32 && dtype != DataType::HALF && dtype != DataType::BFLOAT16) {
    return false;
  }

  return true;
#else
  return false;
#endif
}

//////////////////////////////////////////////////////////////////////////////
// Online Softmax State Update (FlashAttention v4 style)
// Implements selective rescaling - only rescales when necessary
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::updateSoftmaxState(SoftmaxState& state, float newMax, float newSum,
                                              float threshold) {
  if (state.sumExp == 0.0f) {
    // First tile - just initialize
    state.maxVal = newMax;
    state.sumExp = newSum;
    state.correction = 1.0f;
    state.needsRescale = false;
    return;
  }

  // FlashAttention v4 selective rescaling:
  // Only rescale when the new max is significantly larger
  float maxDiff = newMax - state.maxVal;

  if (maxDiff > threshold) {
    // New max is significantly larger - need to rescale previous values
    float rescaleFactor = std::exp(state.maxVal - newMax);
    state.correction = rescaleFactor;
    state.sumExp = state.sumExp * rescaleFactor + newSum;
    state.maxVal = newMax;
    state.needsRescale = true;
  } else if (maxDiff < -threshold) {
    // Old max is significantly larger - rescale new values
    float rescaleFactor = std::exp(newMax - state.maxVal);
    state.sumExp = state.sumExp + newSum * rescaleFactor;
    state.correction = 1.0f;
    state.needsRescale = false;
  } else {
    // Max values are similar - use stable computation
    float maxOfMax = std::max(state.maxVal, newMax);
    state.sumExp = state.sumExp * std::exp(state.maxVal - maxOfMax) +
                   newSum * std::exp(newMax - maxOfMax);
    state.maxVal = maxOfMax;
    state.correction = std::exp(state.maxVal - maxOfMax);
    state.needsRescale = true;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Tiled Flash Attention Forward (for long sequences)
// Uses online softmax to avoid materializing full attention matrix
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::forwardTiled(NDArray* query, NDArray* key, NDArray* value,
                                         NDArray* output, const Config& config,
                                         NDArray* softmaxLse,
                                         LaunchContext* context) {
  // For now, delegate to the standard batched implementation
  // A true tiled implementation would process query/key/value in blocks
  // and use online softmax to accumulate results

  auto rank = query->rankOf();
  if (rank == 3) {
    forward3D(query, key, value, output, config, softmaxLse, nullptr, nullptr, context);
  } else {
    forward4D(query, key, value, output, config, softmaxLse, nullptr, nullptr, context);
  }
}

void FlashAttentionHelper::forwardBatched(NDArray* query, NDArray* key, NDArray* value,
                                           NDArray* output, const Config& config,
                                           NDArray* softmaxLse,
                                           LaunchContext* context) {
  forward(query, key, value, output, config, softmaxLse, nullptr, nullptr, context);
}

void FlashAttentionHelper::computeTile(const float* queryTile, const float* keyTile,
                                        const float* valueTile, float* outputTile,
                                        SoftmaxState* states, int tileQ, int tileKV,
                                        int headDim, float scale, bool isCausal,
                                        int queryOffset, int keyOffset) {
  // Compute Q @ K^T for this tile
  for (int q = 0; q < tileQ; ++q) {
    for (int k = 0; k < tileKV; ++k) {
      // Check causal mask
      if (isCausal && (keyOffset + k) > (queryOffset + q)) {
        continue;
      }

      // Compute dot product
      float score = 0.0f;
      for (int d = 0; d < headDim; ++d) {
        score += queryTile[q * headDim + d] * keyTile[k * headDim + d];
      }
      score *= scale;

      // Update online softmax state
      float expScore = std::exp(score - states[q].maxVal);
      if (score > states[q].maxVal) {
        // Need to rescale previous accumulated values
        float rescale = std::exp(states[q].maxVal - score);
        states[q].sumExp = states[q].sumExp * rescale + expScore;
        states[q].maxVal = score;

        // Rescale accumulated output
        for (int d = 0; d < headDim; ++d) {
          outputTile[q * headDim + d] *= rescale;
        }
      } else {
        states[q].sumExp += expScore;
      }

      // Accumulate weighted value
      float weight = expScore / states[q].sumExp;
      for (int d = 0; d < headDim; ++d) {
        outputTile[q * headDim + d] += weight * valueTile[k * headDim + d];
      }
    }
  }
}

}  // namespace sd
