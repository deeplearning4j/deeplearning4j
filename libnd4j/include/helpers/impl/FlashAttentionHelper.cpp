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
// Flash Attention Helper - Optimized with workspace buffers to eliminate allocation overhead
// Key optimizations:
// 1. Workspace buffer pool - reuses memory across calls
// 2. Minimized permute/reshape operations
// 3. Direct output to user buffers where possible
//

#include <helpers/FlashAttentionHelper.h>
#include <helpers/AttentionWorkspace.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/activations.h>
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <system/Environment.h>
#include <system/type_boilerplate.h>
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
// OPTIMIZED: Uses workspace buffers, minimal allocations
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

#if defined(SD_CUDA)
  // Use fused CUDA kernel only when no scores output needed
  // cuBLAS matmuls are faster than custom kernels when we need intermediate results
  bool supportedType = (query->dataType() == DataType::FLOAT32 ||
                        query->dataType() == DataType::DOUBLE ||
                        query->dataType() == DataType::HALF);
  bool needScores = (attentionScores != nullptr && !attentionScores->isEmpty());
  bool needLogits = (attentionLogits != nullptr && !attentionLogits->isEmpty());

  if (supportedType && !needScores && !needLogits) {
    // Use fast fused kernel when no scores output needed
    fusedAttentionCuda(query, key, value, output, scale, config.isCausal, context);
    if (softmaxLse != nullptr) {
      softmaxLse->nullify();
    }
    return;
  }
  // Fall through to cuBLAS path when scores are needed
#endif

  // Get workspace for scores buffer - REUSED across calls
  auto workspace = AttentionWorkspace::getInstance();
  std::vector<LongType> scoresShape = {batch, seqLenQ, seqLenKV};

  // Determine which array to use for computation
  NDArray* logitsBuffer = nullptr;
  NDArray* workBuffer = nullptr;

  if (attentionLogits != nullptr && !attentionLogits->isEmpty()) {
    logitsBuffer = attentionLogits;
  }

  // Use attentionScores if provided, otherwise get from workspace
  if (attentionScores != nullptr && !attentionScores->isEmpty()) {
    workBuffer = attentionScores;
  } else {
    // Get reusable buffer from workspace - NO ALLOCATION if shape matches previous call
    workBuffer = workspace->getBuffer("forward3d_scores", scoresShape, query->dataType(), context);
  }

  // Batched matmul: Q @ K^T with scale -> [batch, seqQ, seqKV]
  MmulHelper::matmul(query, key, workBuffer, false, true, scale, 0.0);

#if defined(SD_CUDA)
  // Fused causal mask + softmax (single kernel instead of mask + softmax)
  fusedCausalMaskSoftmaxCuda(workBuffer, workBuffer, logitsBuffer, config.isCausal, context);
#else
  // CPU fallback: separate causal mask and softmax
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
    std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
    // Get mask from workspace
    NDArray* causalMask = workspace->getBuffer("forward3d_mask", maskShape, query->dataType(), context);
    causalMask->nullify();
    BUILD_SINGLE_SELECTOR(query->dataType(), causalMask->fillAsTriangular, (-1.0e9f, 1, 0, *causalMask, 'u', false), SD_COMMON_TYPES);
    *workBuffer += *causalMask;
  }
  if (logitsBuffer != nullptr) {
    logitsBuffer->assign(workBuffer);
  }
  ops::helpers::softmax(context, workBuffer, workBuffer, -1);
#endif

  // Batched matmul: scores @ V -> [batch, seqQ, dim]
  MmulHelper::matmul(workBuffer, value, output, false, false, 1.0, 0.0);

  if (softmaxLse != nullptr) {
    softmaxLse->nullify();
  }
}

//////////////////////////////////////////////////////////////////////////////
// 4D Forward Implementation - [batch, seq, numHeads, headDim]
// OPTIMIZED: Uses workspace buffers, minimizes permute/reshape
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

  auto workspace = AttentionWorkspace::getInstance();

#if defined(SD_CUDA)
  // Use fused CUDA kernel only when no scores output needed
  bool supportedType = (query->dataType() == DataType::FLOAT32 ||
                        query->dataType() == DataType::DOUBLE ||
                        query->dataType() == DataType::HALF);
  bool noGQA = (headsPerKvHead == 1);
  bool needScores = (attentionScores != nullptr && !attentionScores->isEmpty());
  bool needLogits = (attentionLogits != nullptr && !attentionLogits->isEmpty());

  // Only use fused kernel when no scores needed - cuBLAS is faster for matmuls
  if (supportedType && noGQA && !needScores && !needLogits) {
    // Use workspace for permuted arrays - CRITICAL: this eliminates malloc/free per call
    std::vector<LongType> qPermShape = {batch, numHeads, seqLenQ, headDim};
    std::vector<LongType> kvPermShape = {batch, numKvHeads, seqLenKV, headDim};
    std::vector<LongType> permOrder = {0, 2, 1, 3};

    NDArray* qPermBuffer = workspace->getBuffer("forward4d_qPerm", qPermShape, query->dataType(), context);
    NDArray* kPermBuffer = workspace->getBuffer("forward4d_kPerm", kvPermShape, key->dataType(), context);
    NDArray* vPermBuffer = workspace->getBuffer("forward4d_vPerm", kvPermShape, value->dataType(), context);

    // Permute into workspace buffers using permute() which returns a view
    auto qPerm = query->permute(permOrder, false, false);
    auto kPerm = key->permute(permOrder, false, false);
    auto vPerm = value->permute(permOrder, false, false);
    qPermBuffer->assign(qPerm);
    kPermBuffer->assign(kPerm);
    vPermBuffer->assign(vPerm);
    delete qPerm;
    delete kPerm;
    delete vPerm;

    // Reshape to 3D: [batch*heads, seq, dim]
    std::vector<LongType> shape3D_Q = {batch * numHeads, seqLenQ, headDim};
    std::vector<LongType> shape3D_KV = {batch * numKvHeads, seqLenKV, headDim};

    qPermBuffer->reshapei(shape3D_Q);
    kPermBuffer->reshapei(shape3D_KV);
    vPermBuffer->reshapei(shape3D_KV);

    // Get output buffer from workspace
    NDArray* outFlat = workspace->getBuffer("forward4d_outFlat", shape3D_Q, query->dataType(), context);

    // Fast path: fused kernel
    fusedAttentionCuda(qPermBuffer, kPermBuffer, vPermBuffer, outFlat, scale, config.isCausal, context);

    // Reshape back to 4D: [batch, heads, seq, dim]
    std::vector<LongType> shape4D = {batch, numHeads, seqLenQ, headDim};
    outFlat->reshapei(shape4D);

    // Permute back and assign to output: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
    auto outPerm = outFlat->permute(permOrder, false, false);
    output->assign(outPerm);
    delete outPerm;

    // Restore workspace buffer shapes for next call
    qPermBuffer->reshapei(qPermShape);
    kPermBuffer->reshapei(kvPermShape);
    vPermBuffer->reshapei(kvPermShape);

    if (softmaxLse != nullptr) softmaxLse->nullify();
    return;
  }
#endif

  // Fallback path with workspace optimization
  std::vector<LongType> permOrder = {0, 2, 1, 3};

  // Workspace buffers for permuted tensors
  std::vector<LongType> qPermShape = {batch, numHeads, seqLenQ, headDim};
  std::vector<LongType> kvPermShape = {batch, numKvHeads, seqLenKV, headDim};

  NDArray* qPermBuffer = workspace->getBuffer("forward4d_qPerm", qPermShape, query->dataType(), context);
  NDArray* kPermBuffer = workspace->getBuffer("forward4d_kPerm", kvPermShape, key->dataType(), context);
  NDArray* vPermBuffer = workspace->getBuffer("forward4d_vPerm", kvPermShape, value->dataType(), context);

  // Permute Q, K, V into workspace
  auto qPerm = query->permute(permOrder, false, false);
  auto kPerm = key->permute(permOrder, false, false);
  auto vPerm = value->permute(permOrder, false, false);
  qPermBuffer->assign(qPerm);
  kPermBuffer->assign(kPerm);
  vPermBuffer->assign(vPerm);
  delete qPerm;
  delete kPerm;
  delete vPerm;

  // Handle GQA: expand KV heads if needed
  NDArray* kExpanded = kPermBuffer;
  NDArray* vExpanded = vPermBuffer;

  if (headsPerKvHead > 1) {
    // Tile KV heads using workspace buffer
    std::vector<LongType> expandedShape = {batch, numHeads, seqLenKV, headDim};
    // Get workspace buffers with tiled shape for direct write
    std::vector<LongType> tiledShape = {batch, numKvHeads, static_cast<LongType>(headsPerKvHead), seqLenKV, headDim};
    NDArray* kTiledBuf = workspace->getBuffer("forward4d_kTiled", tiledShape, key->dataType(), context);
    NDArray* vTiledBuf = workspace->getBuffer("forward4d_vTiled", tiledShape, value->dataType(), context);

    // Reshape for tiling: [batch, numKvHeads, 1, seq, dim]
    std::vector<LongType> reshapeForTile = {batch, numKvHeads, 1, seqLenKV, headDim};
    kPermBuffer->reshapei(reshapeForTile);
    vPermBuffer->reshapei(reshapeForTile);

    // Tile directly into workspace buffers (no allocation)
    std::vector<LongType> reps = {1, 1, static_cast<LongType>(headsPerKvHead), 1, 1};
    kPermBuffer->tile(reps, *kTiledBuf);
    vPermBuffer->tile(reps, *vTiledBuf);

    // Reshape tiled buffers to expanded shape
    kTiledBuf->reshapei(expandedShape);
    vTiledBuf->reshapei(expandedShape);

    kExpanded = kTiledBuf;
    vExpanded = vTiledBuf;

    // Restore kPermBuffer shape
    kPermBuffer->reshapei(kvPermShape);
    vPermBuffer->reshapei(kvPermShape);
  }

  // Reshape for batched matmul: [batch * numHeads, seq, dim]
  std::vector<LongType> qShape = {batch * numHeads, seqLenQ, headDim};
  std::vector<LongType> kvShape = {batch * numHeads, seqLenKV, headDim};
  std::vector<LongType> scoresShape = {batch * numHeads, seqLenQ, seqLenKV};

  qPermBuffer->reshapei(qShape);
  kExpanded->reshapei(kvShape);
  vExpanded->reshapei(kvShape);

  // Use stack-allocated NDArrays for computation buffers.
  // These change shape between prefill (seqLenQ=79) and decode (seqLenQ=1),
  // so workspace caching can trigger CUDA allocations during graph capture (error 901).
  NDArray workBuffer('c', scoresShape, query->dataType(), context);

  // Batched matmul: Q @ K^T with scale
  MmulHelper::matmul(qPermBuffer, kExpanded, &workBuffer, false, true, scale, 0.0);

  // Logits buffer for pre-softmax scores
  NDArray logitsBuf('c', scoresShape, query->dataType(), context);
  NDArray* logitsBuffer = nullptr;
  bool wantLogits = (attentionLogits != nullptr && !attentionLogits->isEmpty());
  if (wantLogits) {
    logitsBuffer = &logitsBuf;
  }

#if defined(SD_CUDA)
  fusedCausalMaskSoftmaxCuda(&workBuffer, &workBuffer, logitsBuffer, config.isCausal, context);
#else
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
    std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
    NDArray causalMask('c', maskShape, query->dataType(), context);
    causalMask.nullify();
    BUILD_SINGLE_SELECTOR(query->dataType(), causalMask.fillAsTriangular, (-1.0e9f, 1, 0, causalMask, 'u', false), SD_COMMON_TYPES);
    workBuffer += causalMask;
  }
  if (logitsBuffer != nullptr) {
    logitsBuffer->assign(&workBuffer);
  }
  ops::helpers::softmax(context, &workBuffer, &workBuffer, -1);
#endif

  // Output buffer
  std::vector<LongType> outShape = {batch * numHeads, seqLenQ, headDim};
  NDArray outReshaped('c', outShape, query->dataType(), context);

  // Batched matmul: scores @ V
  MmulHelper::matmul(&workBuffer, vExpanded, &outReshaped, false, false, 1.0, 0.0);

  // Reshape and permute back to output
  std::vector<LongType> outPermShape = {batch, numHeads, seqLenQ, headDim};
  outReshaped.reshapei(outPermShape);
  auto outPerm = outReshaped.permute(permOrder, false, false);
  output->assign(outPerm);
  delete outPerm;

  // Copy results to output buffers (4D shape expected by caller)
  std::vector<LongType> scores4dShape = {batch, numHeads, seqLenQ, seqLenKV};
  if (attentionScores != nullptr && !attentionScores->isEmpty() &&
      attentionScores->lengthOf() == batch * numHeads * seqLenQ * seqLenKV) {
    workBuffer.reshapei(scores4dShape);
    attentionScores->assign(&workBuffer);
  }
  if (wantLogits && logitsBuffer != nullptr &&
      attentionLogits->lengthOf() == batch * numHeads * seqLenQ * seqLenKV) {
    logitsBuffer->reshapei(scores4dShape);
    attentionLogits->assign(logitsBuffer);
  }

  // Restore workspace buffer shapes for next call (qPerm/kPerm/vPerm only)
  qPermBuffer->reshapei(qPermShape);
  if (headsPerKvHead == 1) {
    kExpanded->reshapei(kvPermShape);
    vExpanded->reshapei(kvPermShape);
  }

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
// OPTIMIZED: Uses workspace buffers
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

  auto workspace = AttentionWorkspace::getInstance();
  std::vector<LongType> scoresShape = {batch, seqLenQ, seqLenKV};

  // Recompute attention: Q @ K^T
  NDArray* scores = workspace->getBuffer("backward3d_scores", scoresShape, query->dataType(), context);
  MmulHelper::matmul(query, key, scores, false, true, scale, 0.0);

  // Apply causal mask if needed
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
#if defined(SD_CUDA)
    applyCausalMaskCuda(scores, context);
#else
    std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
    NDArray* causalMask = workspace->getBuffer("backward3d_mask", maskShape, query->dataType(), context);
    causalMask->nullify();
    BUILD_SINGLE_SELECTOR(query->dataType(), causalMask->fillAsTriangular, (-1.0e9f, 1, 0, *causalMask, 'u', false), SD_COMMON_TYPES);
    *scores += *causalMask;
#endif
  }

  // Softmax
  NDArray* attnWeights = workspace->getBuffer("backward3d_attnWeights", scoresShape, query->dataType(), context);
  ops::helpers::softmax(context, scores, attnWeights, -1);

  // gradValue = attnWeights^T @ gradOutput -> [batch, seqKV, dim]
  MmulHelper::matmul(attnWeights, gradOutput, gradValue, true, false, 1.0, 0.0);

  // dAttn = gradOutput @ V^T -> [batch, seqQ, seqKV]
  NDArray* dAttn = workspace->getBuffer("backward3d_dAttn", scoresShape, query->dataType(), context);
  MmulHelper::matmul(gradOutput, value, dAttn, false, true, 1.0, 0.0);

  // Softmax backward: dS = P * (dAttn - sum(dAttn * P))
  NDArray* dAttnTimesP = workspace->getBuffer("backward3d_dAttnTimesP", scoresShape, query->dataType(), context);
  dAttn->applyPairwiseTransform(pairwise::Multiply, attnWeights, dAttnTimesP);

  // Use workspace buffer for reduction result (keepDims=true -> [batch, seqQ, 1])
  std::vector<LongType> sumDims = {2};
  std::vector<LongType> rowSumsShape = {scoresShape[0], scoresShape[1], 1};
  NDArray* rowSums = workspace->getBuffer("backward3d_rowSums", rowSumsShape, query->dataType(), context);
  dAttnTimesP->reduceAlongDimension(reduce::Sum, rowSums, &sumDims, true);
  *dAttn -= *rowSums;
  dAttn->applyPairwiseTransform(pairwise::Multiply, attnWeights, dAttn);
  *dAttn *= scale;

  // gradQuery = dS @ K -> [batch, seqQ, dim]
  MmulHelper::matmul(dAttn, key, gradQuery, false, false, 1.0, 0.0);

  // gradKey = dS^T @ Q -> [batch, seqKV, dim]
  MmulHelper::matmul(dAttn, query, gradKey, true, false, 1.0, 0.0);
}

//////////////////////////////////////////////////////////////////////////////
// 4D Backward Implementation - [batch, seq, numHeads, headDim]
// OPTIMIZED: Uses workspace buffers
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

  auto workspace = AttentionWorkspace::getInstance();
  std::vector<LongType> permOrder = {0, 2, 1, 3};

  // Workspace for permuted tensors
  std::vector<LongType> qPermShape = {batch, numHeads, seqLenQ, headDim};
  std::vector<LongType> kvPermShape = {batch, numKvHeads, seqLenKV, headDim};
  std::vector<LongType> goPermShape = {batch, numHeads, seqLenQ, headDim};

  NDArray* qPermBuffer = workspace->getBuffer("backward4d_qPerm", qPermShape, query->dataType(), context);
  NDArray* kPermBuffer = workspace->getBuffer("backward4d_kPerm", kvPermShape, key->dataType(), context);
  NDArray* vPermBuffer = workspace->getBuffer("backward4d_vPerm", kvPermShape, value->dataType(), context);
  NDArray* goPermBuffer = workspace->getBuffer("backward4d_goPerm", goPermShape, gradOutput->dataType(), context);

  // Permute into workspace
  auto qPerm = query->permute(permOrder, false, false);
  auto kPerm = key->permute(permOrder, false, false);
  auto vPerm = value->permute(permOrder, false, false);
  auto goPerm = gradOutput->permute(permOrder, false, false);

  qPermBuffer->assign(qPerm);
  kPermBuffer->assign(kPerm);
  vPermBuffer->assign(vPerm);
  goPermBuffer->assign(goPerm);

  delete qPerm;
  delete kPerm;
  delete vPerm;
  delete goPerm;

  // Expand KV for GQA
  NDArray* kExpanded = kPermBuffer;
  NDArray* vExpanded = vPermBuffer;

  if (headsPerKvHead > 1) {
    std::vector<LongType> expandedShape = {batch, numHeads, seqLenKV, headDim};
    std::vector<LongType> tiledShape = {batch, numKvHeads, static_cast<LongType>(headsPerKvHead), seqLenKV, headDim};
    NDArray* kTiledBuf = workspace->getBuffer("backward4d_kTiled", tiledShape, key->dataType(), context);
    NDArray* vTiledBuf = workspace->getBuffer("backward4d_vTiled", tiledShape, value->dataType(), context);

    std::vector<LongType> reshapeForTile = {batch, numKvHeads, 1, seqLenKV, headDim};
    kPermBuffer->reshapei(reshapeForTile);
    vPermBuffer->reshapei(reshapeForTile);

    // Tile directly into workspace buffers (no allocation)
    std::vector<LongType> reps = {1, 1, static_cast<LongType>(headsPerKvHead), 1, 1};
    kPermBuffer->tile(reps, *kTiledBuf);
    vPermBuffer->tile(reps, *vTiledBuf);

    // Reshape tiled buffers to expanded shape
    kTiledBuf->reshapei(expandedShape);
    vTiledBuf->reshapei(expandedShape);

    kExpanded = kTiledBuf;
    vExpanded = vTiledBuf;

    kPermBuffer->reshapei(kvPermShape);
    vPermBuffer->reshapei(kvPermShape);
  }

  // Reshape for batched ops
  std::vector<LongType> qShape = {batch * numHeads, seqLenQ, headDim};
  std::vector<LongType> kvShape = {batch * numHeads, seqLenKV, headDim};
  std::vector<LongType> scoresShape = {batch * numHeads, seqLenQ, seqLenKV};

  qPermBuffer->reshapei(qShape);
  kExpanded->reshapei(kvShape);
  vExpanded->reshapei(kvShape);
  goPermBuffer->reshapei(qShape);

  // Recompute attention: Q @ K^T
  NDArray* scores = workspace->getBuffer("backward4d_scores", scoresShape, query->dataType(), context);
  MmulHelper::matmul(qPermBuffer, kExpanded, scores, false, true, scale, 0.0);

  // Apply causal mask
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
#if defined(SD_CUDA)
    applyCausalMaskCuda(scores, context);
#else
    std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
    NDArray* causalMask = workspace->getBuffer("backward4d_mask", maskShape, query->dataType(), context);
    causalMask->nullify();
    BUILD_SINGLE_SELECTOR(query->dataType(), causalMask->fillAsTriangular, (-1.0e9f, 1, 0, *causalMask, 'u', false), SD_COMMON_TYPES);
    *scores += *causalMask;
#endif
  }

  // Softmax
  NDArray* attnWeights = workspace->getBuffer("backward4d_attnWeights", scoresShape, query->dataType(), context);
  ops::helpers::softmax(context, scores, attnWeights, -1);

  // gradValue = attnWeights^T @ gradOutput
  NDArray* gvReshaped = workspace->getBuffer("backward4d_gvReshaped", kvShape, query->dataType(), context);
  MmulHelper::matmul(attnWeights, goPermBuffer, gvReshaped, true, false, 1.0, 0.0);

  // dAttn = gradOutput @ V^T
  NDArray* dAttn = workspace->getBuffer("backward4d_dAttn", scoresShape, query->dataType(), context);
  MmulHelper::matmul(goPermBuffer, vExpanded, dAttn, false, true, 1.0, 0.0);

  // Softmax backward
  NDArray* dAttnTimesP = workspace->getBuffer("backward4d_dAttnTimesP", scoresShape, query->dataType(), context);
  dAttn->applyPairwiseTransform(pairwise::Multiply, attnWeights, dAttnTimesP);

  // Use workspace buffer for reduction result (keepDims=true -> [..., 1])
  std::vector<LongType> sumDims = {2};
  std::vector<LongType> rowSumsShape = {scoresShape[0], scoresShape[1], 1};
  NDArray* rowSums = workspace->getBuffer("backward4d_rowSums", rowSumsShape, query->dataType(), context);
  dAttnTimesP->reduceAlongDimension(reduce::Sum, rowSums, &sumDims, true);
  *dAttn -= *rowSums;
  dAttn->applyPairwiseTransform(pairwise::Multiply, attnWeights, dAttn);
  *dAttn *= scale;

  // gradQuery = dS @ K
  NDArray* gqReshaped = workspace->getBuffer("backward4d_gqReshaped", qShape, query->dataType(), context);
  MmulHelper::matmul(dAttn, kExpanded, gqReshaped, false, false, 1.0, 0.0);

  // gradKey = dS^T @ Q
  NDArray* gkReshaped = workspace->getBuffer("backward4d_gkReshaped", kvShape, query->dataType(), context);
  MmulHelper::matmul(dAttn, qPermBuffer, gkReshaped, true, false, 1.0, 0.0);

  // Reshape and permute gradients back
  std::vector<LongType> gqPermShape = {batch, numHeads, seqLenQ, headDim};
  std::vector<LongType> gkvPermShape = {batch, numHeads, seqLenKV, headDim};

  gqReshaped->reshapei(gqPermShape);
  auto gqPerm = gqReshaped->permute(permOrder, false, false);
  gradQuery->assign(gqPerm);
  delete gqPerm;
  gqReshaped->reshapei(qShape);

  // For GQA, accumulate gradients to KV heads
  if (headsPerKvHead > 1) {
    std::vector<LongType> reshapeForSum = {batch, numKvHeads, static_cast<LongType>(headsPerKvHead), seqLenKV, headDim};
    gkReshaped->reshapei(reshapeForSum);
    gvReshaped->reshapei(reshapeForSum);

    // Reduce along the headsPerKvHead dimension (keepDims=false -> [batch, numKvHeads, seqLenKV, headDim])
    std::vector<LongType> reduceDims = {2};
    std::vector<LongType> reducedShape = {batch, numKvHeads, seqLenKV, headDim};
    NDArray* gkReduced = workspace->getBuffer("backward4d_gkReduced", reducedShape, query->dataType(), context);
    NDArray* gvReduced = workspace->getBuffer("backward4d_gvReduced", reducedShape, query->dataType(), context);
    gkReshaped->reduceAlongDimension(reduce::Sum, gkReduced, &reduceDims, false);
    gvReshaped->reduceAlongDimension(reduce::Sum, gvReduced, &reduceDims, false);

    auto gkPerm = gkReduced->permute(permOrder, false, false);
    auto gvPerm = gvReduced->permute(permOrder, false, false);
    gradKey->assign(gkPerm);
    gradValue->assign(gvPerm);
    delete gkPerm;
    delete gvPerm;

    gkReshaped->reshapei(kvShape);
    gvReshaped->reshapei(kvShape);
  } else {
    gkReshaped->reshapei(gkvPermShape);
    gvReshaped->reshapei(gkvPermShape);
    auto gkPerm = gkReshaped->permute(permOrder, false, false);
    auto gvPerm = gvReshaped->permute(permOrder, false, false);
    gradKey->assign(gkPerm);
    gradValue->assign(gvPerm);
    delete gkPerm;
    delete gvPerm;
    gkReshaped->reshapei(kvShape);
    gvReshaped->reshapei(kvShape);
  }

  // Restore workspace shapes
  qPermBuffer->reshapei(qPermShape);
  goPermBuffer->reshapei(goPermShape);
  if (headsPerKvHead == 1) {
    kExpanded->reshapei(kvPermShape);
    vExpanded->reshapei(kvPermShape);
  }
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

  // Use workspace buffers for intermediates
  auto workspace = AttentionWorkspace::getInstance();
  std::vector<LongType> reshapeForTile = {batch, seqLen, numKvHeads, 1, headDim};
  NDArray* kvReshaped = workspace->getBuffer("repeatKV_reshaped", reshapeForTile, kv->dataType());

  kvReshaped->assign(kv);
  kvReshaped->reshapei(reshapeForTile);

  // Tile directly into workspace buffer (no allocation)
  std::vector<LongType> reps = {1, 1, 1, static_cast<LongType>(headsPerKvHead), 1};
  std::vector<LongType> tiledShape = {batch, seqLen, numKvHeads, static_cast<LongType>(headsPerKvHead), headDim};
  NDArray* kvTiledBuf = workspace->getBuffer("repeatKV_tiled", tiledShape, kv->dataType());
  kvReshaped->tile(reps, *kvTiledBuf);

  std::vector<LongType> outShape = {batch, seqLen, static_cast<LongType>(numHeads), headDim};
  kvTiledBuf->reshapei(outShape);
  output->assign(kvTiledBuf);

  // Restore shapes for next call
  kvReshaped->reshapei({batch, seqLen, numKvHeads, headDim});
  kvTiledBuf->reshapei(tiledShape);
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
  if (!Environment::getInstance().helpersAllowed()) {
    return false;
  }
  if (query->rankOf() != 3) {
    return false;
  }
  auto dtype = query->dataType();
  if (dtype != DataType::FLOAT32 && dtype != DataType::HALF && dtype != DataType::BFLOAT16) {
    return false;
  }
  if (config.dropout > 0.0f) {
    return false;
  }
  return true;
}

bool FlashAttentionHelper::canUseCudnnSdpa(NDArray* query, NDArray* key, NDArray* value,
                                           const Config& config) {
#if defined(HAVE_CUDNN) && CUDNN_MAJOR >= 8 && CUDNN_MINOR >= 9
  if (!Environment::getInstance().helpersAllowed()) {
    return false;
  }
  if (query->getContext()->getWorkspace() == nullptr ||
      query->getContext()->getWorkspace()->deviceType() != sd::graph::DeviceType::GPU) {
    return false;
  }
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
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::updateSoftmaxState(SoftmaxState& state, float newMax, float newSum,
                                              float threshold) {
  if (state.sumExp == 0.0f) {
    state.maxVal = newMax;
    state.sumExp = newSum;
    state.correction = 1.0f;
    state.needsRescale = false;
    return;
  }

  float maxDiff = newMax - state.maxVal;

  if (maxDiff > threshold) {
    float rescaleFactor = std::exp(state.maxVal - newMax);
    state.correction = rescaleFactor;
    state.sumExp = state.sumExp * rescaleFactor + newSum;
    state.maxVal = newMax;
    state.needsRescale = true;
  } else if (maxDiff < -threshold) {
    float rescaleFactor = std::exp(newMax - state.maxVal);
    state.sumExp = state.sumExp + newSum * rescaleFactor;
    state.correction = 1.0f;
    state.needsRescale = false;
  } else {
    float maxOfMax = std::max(state.maxVal, newMax);
    state.sumExp = state.sumExp * std::exp(state.maxVal - maxOfMax) +
                   newSum * std::exp(newMax - maxOfMax);
    state.maxVal = maxOfMax;
    state.correction = std::exp(state.maxVal - maxOfMax);
    state.needsRescale = true;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Tiled Flash Attention Forward
//////////////////////////////////////////////////////////////////////////////

void FlashAttentionHelper::forwardTiled(NDArray* query, NDArray* key, NDArray* value,
                                         NDArray* output, const Config& config,
                                         NDArray* softmaxLse,
                                         LaunchContext* context) {
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
  for (int q = 0; q < tileQ; ++q) {
    for (int k = 0; k < tileKV; ++k) {
      if (isCausal && (keyOffset + k) > (queryOffset + q)) {
        continue;
      }

      float score = 0.0f;
      for (int d = 0; d < headDim; ++d) {
        score += queryTile[q * headDim + d] * keyTile[k * headDim + d];
      }
      score *= scale;

      float expScore = std::exp(score - states[q].maxVal);
      if (score > states[q].maxVal) {
        float rescale = std::exp(states[q].maxVal - score);
        states[q].sumExp = states[q].sumExp * rescale + expScore;
        states[q].maxVal = score;

        for (int d = 0; d < headDim; ++d) {
          outputTile[q * headDim + d] *= rescale;
        }
      } else {
        states[q].sumExp += expScore;
      }

      float weight = expScore / states[q].sumExp;
      for (int d = 0; d < headDim; ++d) {
        outputTile[q * headDim + d] += weight * valueTile[k * headDim + d];
      }
    }
  }
}

}  // namespace sd
