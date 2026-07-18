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
#include <math/templatemath.h>
#include <graph/DspDeviceDispatch.h>
#include <graph/DspDiagnostics.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace sd {

namespace {

template <typename T>
static void cpuDecode4D_(NDArray* query, NDArray* key, NDArray* value,
                         NDArray* output, double scale,
                         NDArray* attentionBias,
                         NDArray* attentionScores,
                         NDArray* attentionLogits) {
  const LongType batch = query->sizeAt(0);
  const LongType numHeads = query->sizeAt(2);
  const LongType headDim = query->sizeAt(3);
  const LongType seqLenKV = key->sizeAt(1);
  const LongType numKvHeads = key->sizeAt(2);
  const LongType headsPerKvHead = numHeads / numKvHeads;

  const T* qBuf = query->bufferAsT<T>();
  const T* kBuf = key->bufferAsT<T>();
  const T* vBuf = value->bufferAsT<T>();
  T* outBuf = output->bufferAsT<T>();

  T* scoresBuf = (attentionScores != nullptr && !attentionScores->isEmpty() &&
                  attentionScores->dataType() == query->dataType())
                     ? attentionScores->bufferAsT<T>() : nullptr;
  T* logitsBuf = (attentionLogits != nullptr && !attentionLogits->isEmpty() &&
                  attentionLogits->dataType() == query->dataType())
                     ? attentionLogits->bufferAsT<T>() : nullptr;
  const T* biasBuf = (attentionBias != nullptr && !attentionBias->isEmpty() &&
                      attentionBias->dataType() == query->dataType())
                       ? attentionBias->bufferAsT<T>() : nullptr;

  const LongType qS0 = query->strideAt(0);
  const LongType qS2 = query->strideAt(2);
  const LongType qS3 = query->strideAt(3);
  const LongType kS0 = key->strideAt(0);
  const LongType kS1 = key->strideAt(1);
  const LongType kS2 = key->strideAt(2);
  const LongType kS3 = key->strideAt(3);
  const LongType vS0 = value->strideAt(0);
  const LongType vS1 = value->strideAt(1);
  const LongType vS2 = value->strideAt(2);
  const LongType vS3 = value->strideAt(3);
  const LongType oS0 = output->strideAt(0);
  const LongType oS2 = output->strideAt(2);
  const LongType oS3 = output->strideAt(3);

  LongType scoreS0 = 0, scoreS1 = 0, scoreS3 = 0;
  if (scoresBuf != nullptr) {
    scoreS0 = attentionScores->strideAt(0);
    scoreS1 = attentionScores->strideAt(1);
    scoreS3 = attentionScores->strideAt(3);
  }
  LongType logitsS0 = 0, logitsS1 = 0, logitsS3 = 0;
  if (logitsBuf != nullptr) {
    logitsS0 = attentionLogits->strideAt(0);
    logitsS1 = attentionLogits->strideAt(1);
    logitsS3 = attentionLogits->strideAt(3);
  }

  LongType biasS0 = 0, biasS1 = 0, biasS3 = 0;
  int biasRank = 0;
  if (biasBuf != nullptr) {
    biasRank = attentionBias->rankOf();
    if (biasRank == 4) {
      biasS0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
      biasS1 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
      biasS3 = attentionBias->strideAt(3);
    } else if (biasRank == 3) {
      biasS0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
      biasS3 = attentionBias->strideAt(2);
    } else if (biasRank == 2) {
      biasS3 = attentionBias->strideAt(1);
    }
  }

  std::vector<double> logits(static_cast<size_t>(seqLenKV));

  for (LongType b = 0; b < batch; ++b) {
    for (LongType h = 0; h < numHeads; ++h) {
      const LongType kvHead = h / headsPerKvHead;
      const T* q = qBuf + b * qS0 + h * qS2;
      double maxLogit = -std::numeric_limits<double>::infinity();

      for (LongType t = 0; t < seqLenKV; ++t) {
        const T* k = kBuf + b * kS0 + t * kS1 + kvHead * kS2;
        double dot = 0.0;
        for (LongType d = 0; d < headDim; ++d) {
          dot += static_cast<double>(q[d * qS3]) * static_cast<double>(k[d * kS3]);
        }
        dot *= scale;
        if (biasBuf != nullptr) {
          LongType biasOffset = 0;
          if (biasRank == 4) biasOffset = b * biasS0 + h * biasS1 + t * biasS3;
          else if (biasRank == 3) biasOffset = b * biasS0 + t * biasS3;
          else if (biasRank == 2) biasOffset = t * biasS3;
          dot += static_cast<double>(biasBuf[biasOffset]);
        }
        logits[static_cast<size_t>(t)] = dot;
        if (logitsBuf != nullptr) {
          logitsBuf[b * logitsS0 + h * logitsS1 + t * logitsS3] = static_cast<T>(dot);
        }
        if (dot > maxLogit) maxLogit = dot;
      }

      double sumExp = 0.0;
      for (LongType t = 0; t < seqLenKV; ++t) {
        double w = std::exp(logits[static_cast<size_t>(t)] - maxLogit);
        logits[static_cast<size_t>(t)] = w;
        sumExp += w;
      }
      const double invSum = sumExp > 0.0 ? 1.0 / sumExp : 0.0;

      T* out = outBuf + b * oS0 + h * oS2;
      for (LongType d = 0; d < headDim; ++d) {
        double acc = 0.0;
        for (LongType t = 0; t < seqLenKV; ++t) {
          double w = logits[static_cast<size_t>(t)] * invSum;
          const T* v = vBuf + b * vS0 + t * vS1 + kvHead * vS2;
          acc += w * static_cast<double>(v[d * vS3]);
          if (d == 0 && scoresBuf != nullptr) {
            scoresBuf[b * scoreS0 + h * scoreS1 + t * scoreS3] = static_cast<T>(w);
          }
        }
        out[d * oS3] = static_cast<T>(acc);
      }
    }
  }

  output->tickWriteHost();
  if (attentionScores != nullptr && !attentionScores->isEmpty()) attentionScores->tickWriteHost();
  if (attentionLogits != nullptr && !attentionLogits->isEmpty()) attentionLogits->tickWriteHost();
}

}  // namespace

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
    NDArray* attentionLogits, LaunchContext* context,
    NDArray* attentionBias) {

  auto rank = query->rankOf();
  REQUIRE_TRUE(rank == 3 || rank == 4, 0,
               "FlashAttentionHelper::forward: query rank must be 3 or 4, got %i", rank);
  REQUIRE_TRUE(key->rankOf() == rank && value->rankOf() == rank, 0,
               "FlashAttentionHelper::forward: query, key, value must have same rank");

  if (rank == 3) {
    forward3D(query, key, value, output, config, softmaxLse, attentionScores, attentionLogits, context,
              attentionBias);
  } else {
    forward4D(query, key, value, output, config, softmaxLse, attentionScores, attentionLogits, context,
              attentionBias);
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
    NDArray* attentionLogits, LaunchContext* context,
    NDArray* attentionBias) {

  auto batch = query->sizeAt(0);
  auto seqLenQ = query->sizeAt(1);
  auto dim = query->sizeAt(2);
  auto seqLenKV = key->sizeAt(1);

  // Compute scale in double to avoid float32 truncation of 1/sqrt(dim).
  // This is critical for DOUBLE inputs: 1/sqrt(3) in float32 loses ~1e-8 relative error.
  // The kernel receives scale as double and applies it as static_cast<T>(scale) inside.
  double scale = config.scale > 0.0f ? static_cast<double>(config.scale) : 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(dim));

  if (sd::graph::dspIsCudaBuild()) {
    // Use fused CUDA kernel - now supports attention bias!
    // cuBLAS matmuls are only faster when we need intermediate score results
    bool supportedType = (query->dataType() == DataType::FLOAT32 ||
                          query->dataType() == DataType::DOUBLE ||
                          query->dataType() == DataType::HALF);
    bool needScores = (attentionScores != nullptr && !attentionScores->isEmpty());
    bool needLogits = (attentionLogits != nullptr && !attentionLogits->isEmpty());

    if (supportedType && !needScores && !needLogits) {
      // Use fast fused kernel - handles attention bias internally
      fusedAttentionCuda(query, key, value, output, scale, config.isCausal, context, attentionBias);
      if (softmaxLse != nullptr) {
        softmaxLse->nullify();
      }
      return;
    }
    // Fall through to cuBLAS path when scores are needed
  }

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

  if (attentionBias != nullptr && !attentionBias->isEmpty()) {
    NDArray* biasForAdd = attentionBias;
    std::unique_ptr<NDArray> biasPermutedOwner;
    std::unique_ptr<NDArray> biasReshapedOwner;

    // Some ONNX exports provide additive attention bias as [..., seqKV, seqQ]
    // (source,target) instead of [..., seqQ, seqKV]. Normalize by swapping the
    // trailing dimensions when needed.
    if (seqLenQ != seqLenKV &&
        biasForAdd->rankOf() >= 2 &&
        biasForAdd->sizeAt(biasForAdd->rankOf() - 2) == seqLenKV &&
        biasForAdd->sizeAt(biasForAdd->rankOf() - 1) == seqLenQ) {
      std::vector<LongType> perm(static_cast<size_t>(biasForAdd->rankOf()));
      for (int i = 0; i < biasForAdd->rankOf(); i++) perm[static_cast<size_t>(i)] = i;
      std::swap(perm[perm.size() - 2], perm[perm.size() - 1]);
      biasPermutedOwner.reset(biasForAdd->permute(perm, false, false));
      biasForAdd = biasPermutedOwner.get();
    }

    // Common ONNX case for 3D attention bias: [batch,1,seqQ,seqKV] -> [batch,seqQ,seqKV]
    // MUST be a no-copy VIEW: this helper's kernels are captured into CUDA graphs and
    // replayed without re-running this host code. A reshape COPY freezes the bias at
    // capture-time content in a buffer that is freed after this call — replays keep
    // reading it while per-step mask updates land in the original buffer, silently
    // diverging decode from the first replayed step.
    if (biasForAdd->rankOf() == 4 && biasForAdd->sizeAt(1) == 1) {
      std::vector<LongType> biasShape3d = {biasForAdd->sizeAt(0), biasForAdd->sizeAt(2), biasForAdd->sizeAt(3)};
      biasReshapedOwner.reset(biasForAdd->reshape('c', biasShape3d, false));
      biasForAdd = biasReshapedOwner.get();
    }

    workBuffer->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), biasForAdd, workBuffer, false);
  }

  if (sd::graph::dspIsCudaBuild()) {
    // Fused causal mask + softmax (single kernel instead of mask + softmax)
    fusedCausalMaskSoftmaxCuda(workBuffer, workBuffer, logitsBuffer, config.isCausal, context);
  } else {
    // CPU fallback: separate causal mask and softmax
    if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
      std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
      // Get mask from workspace
      NDArray* causalMask = workspace->getBuffer("forward3d_mask", maskShape, query->dataType(), context);
      causalMask->nullify();
      int causalLower = (seqLenKV > seqLenQ) ? static_cast<int>(seqLenKV - seqLenQ) : 0;
      // Fill strict upper triangle (row < col - causalLower) with -1e9
      // to prevent attending to future tokens.
      BUILD_SINGLE_SELECTOR(query->dataType(), causalMask->fillAsTriangular,
                            (-1.0e9f, causalLower, 0, *causalMask, 'u', false), SD_COMMON_TYPES);
      workBuffer->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), causalMask, workBuffer, false);
    }
    if (logitsBuffer != nullptr) {
      logitsBuffer->assign(workBuffer);
    }
    // IMPORTANT: Must use explicit positive dimension for softmax.
    // The TAD helper treats -1 as sentinel meaning "all dimensions" (entire array as one TAD),
    // NOT as "last dimension". Using -1 produces all-1.0 output instead of proper softmax.
    int softmaxDim3D = workBuffer->rankOf() - 1;
    ops::helpers::softmax(context, workBuffer, workBuffer, softmaxDim3D);
  }

  // Batched matmul: scores @ V -> [batch, seqQ, dim]
  // IMPORTANT: Do NOT write directly to `output`. When the DSP allocator produces
  // a non-contiguous output buffer (e.g. a 1D oversized buffer reshaped in-place),
  // MmulHelper detects the non-contiguous stride layout and silently allocates a
  // fresh temporary array for the result, leaving the actual `output` buffer all-zeros.
  // Use a workspace buffer for the matmul and then assign back to guarantee that the
  // result always lands in the real output buffer regardless of its stride layout.
  // (Same fix pattern as onnx_multi_head_attention.cpp lines 360-383.)
  std::vector<LongType> outShape3d = {batch, seqLenQ, dim};
  NDArray* outBuffer3d = workspace->getBuffer("forward3d_out", outShape3d, query->dataType(), context);
  MmulHelper::matmul(workBuffer, value, outBuffer3d, false, false, 1.0, 0.0);
  output->assign(outBuffer3d);

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
    NDArray* attentionLogits, LaunchContext* context,
    NDArray* attentionBias) {

  auto batch = query->sizeAt(0);
  auto seqLenQ = query->sizeAt(1);
  auto numHeads = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto seqLenKV = key->sizeAt(1);
  auto numKvHeads = key->sizeAt(2);

  // Use double for scale computation — critical for DOUBLE inputs.
  double scale = config.scale > 0.0f ? static_cast<double>(config.scale) : 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(headDim));
  bool validGqaLayout = numKvHeads > 0 && numHeads % numKvHeads == 0;
  int headsPerKvHead = validGqaLayout
      ? static_cast<int>(numHeads / numKvHeads)
      : 0;

  auto workspace = AttentionWorkspace::getInstance();

  if (sd::graph::dspIsCudaBuild()) {
    // Use fused CUDA kernel - supports attention bias in the kernel itself
    bool supportedType = (query->dataType() == DataType::FLOAT32 ||
                          query->dataType() == DataType::DOUBLE ||
                          query->dataType() == DataType::HALF);
    bool noGQA = (headsPerKvHead == 1);
    bool needScores = (attentionScores != nullptr && !attentionScores->isEmpty());
    bool needLogits = (attentionLogits != nullptr && !attentionLogits->isEmpty());
    bool hasAttentionBias = (attentionBias != nullptr && !attentionBias->isEmpty());
    bool hasCurrentWindow =
        config.currentKeyWindow != nullptr || config.currentValueWindow != nullptr
        || config.currentKvPosition != nullptr;
    bool directCurrentWindowLayout = !hasCurrentWindow;
    if (config.currentKeyWindow != nullptr && config.currentValueWindow != nullptr
        && config.currentKvPosition != nullptr
        && config.currentKeyWindow->rankOf() == 4
        && config.currentValueWindow->rankOf() == 4) {
      directCurrentWindowLayout =
          config.currentKeyWindow->sizeAt(0) == batch
          && config.currentKeyWindow->sizeAt(1) == seqLenQ
          && config.currentKeyWindow->sizeAt(2) == numKvHeads
          && config.currentKeyWindow->sizeAt(3) == headDim
          && config.currentValueWindow->sizeAt(0) == batch
          && config.currentValueWindow->sizeAt(1) == seqLenQ
          && config.currentValueWindow->sizeAt(2) == numKvHeads
          && config.currentValueWindow->sizeAt(3) == headDim
          && config.currentKeyWindow->dataType() == query->dataType()
          && config.currentValueWindow->dataType() == query->dataType();
    }

    // Rank-4 GQA prefill fast path with required auxiliary outputs. The op's
    // training/backward contract materializes both logits and softmax scores, so
    // use those arrays as the kernel's intermediates and keep Q/K/V in BSHD.
    // Unusual auxiliary or bias layouts fall through to the general workspace path.
    bool directAuxLayout =
        needScores && needLogits
        && attentionScores->rankOf() == 4
        && attentionLogits->rankOf() == 4
        && attentionScores->sizeAt(0) == batch
        && attentionScores->sizeAt(1) == numHeads
        && attentionScores->sizeAt(2) == seqLenQ
        && attentionScores->sizeAt(3) == seqLenKV
        && attentionLogits->sizeAt(0) == batch
        && attentionLogits->sizeAt(1) == numHeads
        && attentionLogits->sizeAt(2) == seqLenQ
        && attentionLogits->sizeAt(3) == seqLenKV
        && attentionScores->dataType() == query->dataType()
        && attentionLogits->dataType() == query->dataType()
        && output->dataType() == query->dataType()
        && key->dataType() == query->dataType()
        && value->dataType() == query->dataType();

    bool directBiasLayout = !hasAttentionBias;
    if (hasAttentionBias && attentionBias->dataType() == query->dataType()) {
      const int biasRank = attentionBias->rankOf();
      if (biasRank == 4) {
        directBiasLayout =
            (attentionBias->sizeAt(0) == 1 || attentionBias->sizeAt(0) == batch)
            && (attentionBias->sizeAt(1) == 1 || attentionBias->sizeAt(1) == numHeads)
            && (attentionBias->sizeAt(2) == 1 || attentionBias->sizeAt(2) == seqLenQ)
            && (attentionBias->sizeAt(3) == 1 || attentionBias->sizeAt(3) == seqLenKV);
      } else if (biasRank == 3) {
        directBiasLayout =
            (attentionBias->sizeAt(0) == 1 || attentionBias->sizeAt(0) == batch)
            && (attentionBias->sizeAt(1) == 1 || attentionBias->sizeAt(1) == seqLenQ)
            && (attentionBias->sizeAt(2) == 1 || attentionBias->sizeAt(2) == seqLenKV);
      } else if (biasRank == 2) {
        directBiasLayout =
            (attentionBias->sizeAt(0) == 1 || attentionBias->sizeAt(0) == seqLenQ)
            && (attentionBias->sizeAt(1) == 1 || attentionBias->sizeAt(1) == seqLenKV);
      }
    }

    if (DSP_DIAG_ENABLED(KV_CACHE)) {
      DSP_DIAG(KV_CACHE,
               "fa4d_gqa_path qSeq=%lld kvSeq=%lld qHeads=%lld kvHeads=%lld hpk=%d "
               "supported=%d scores=%d logits=%d aux=%d bias=%d hasCurrent=%d currentLayout=%d "
               "currentKRank=%d currentKSeq=%lld currentKHeads=%lld currentKDim=%lld",
               (long long)seqLenQ, (long long)seqLenKV,
               (long long)numHeads, (long long)numKvHeads, headsPerKvHead,
               supportedType ? 1 : 0, needScores ? 1 : 0, needLogits ? 1 : 0,
               directAuxLayout ? 1 : 0, directBiasLayout ? 1 : 0,
               hasCurrentWindow ? 1 : 0, directCurrentWindowLayout ? 1 : 0,
               config.currentKeyWindow != nullptr ? config.currentKeyWindow->rankOf() : -1,
               config.currentKeyWindow != nullptr && config.currentKeyWindow->rankOf() > 1
                   ? (long long)config.currentKeyWindow->sizeAt(1) : -1LL,
               config.currentKeyWindow != nullptr && config.currentKeyWindow->rankOf() > 2
                   ? (long long)config.currentKeyWindow->sizeAt(2) : -1LL,
               config.currentKeyWindow != nullptr && config.currentKeyWindow->rankOf() > 3
                   ? (long long)config.currentKeyWindow->sizeAt(3) : -1LL);
    }

    // Keep scalar decode and multi-row target verification on one reduction path.
    // A distinct scalar fallback introduces small per-layer deltas that can amplify
    // across a deep recurrent decode and violate speculative losslessness.
    if (supportedType && validGqaLayout && headsPerKvHead > 1
        && directAuxLayout && directBiasLayout && directCurrentWindowLayout) {
      fusedGQAAttentionCudaWithScores(
          query, key, value, output, attentionLogits, attentionScores,
          scale, config.isCausal, context,
          hasAttentionBias ? attentionBias : nullptr,
          config.currentKeyWindow, config.currentValueWindow,
          config.currentKvPosition);
      if (softmaxLse != nullptr) softmaxLse->nullify();
      return;
    }

    // Direct online-softmax path: one block per (batch, q-head, query-row).
    // Decode remains on its existing output-only kernel, while multi-row GQA uses
    // the same reduction order without materializing repeated K/V heads.
    bool isDecode = (seqLenQ == 1);
    bool directKernelTypes =
        output->dataType() == query->dataType()
        && key->dataType() == query->dataType()
        && value->dataType() == query->dataType();
    bool directGqaWindow =
        seqLenQ > 1 && validGqaLayout && headsPerKvHead > 1
        && directKernelTypes && directBiasLayout && directCurrentWindowLayout;
    if (supportedType && ((isDecode && directCurrentWindowLayout) || directGqaWindow)
        && !needScores && !needLogits) {
      fusedGQADecodeCuda(
          query, key, value, output, scale, config.isCausal,
          context, hasAttentionBias ? attentionBias : nullptr,
          config.currentKeyWindow, config.currentValueWindow,
          config.currentKvPosition);
      if (softmaxLse != nullptr) softmaxLse->nullify();
      return;
    }

    // Use fused kernel when no scores needed - cuBLAS is only faster when we need intermediate results
    // The fused kernel now handles attention bias internally for maximum performance
    if (supportedType && noGQA && !needScores && !needLogits) {
      // Use workspace for permuted arrays -  this eliminates malloc/free per call
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

      // Get output buffer from workspace and zero it — the fused kernel may not write
      // all positions (e.g., causal mask), and stale data from a previous call would leak.
      NDArray* outFlat = workspace->getBuffer("forward4d_outFlat", shape3D_Q, query->dataType(), context);
      outFlat->nullify();

      // Prepare attention bias if present - broadcast+reshape to [batch*heads, seqQ, seqKV]
      //  The fused CUDA kernel templates on query dtype (float32/float64/half).
      // If the bias is a different dtype (e.g. LONG from ONNX mask), we MUST cast it
      // to match the query type, otherwise the kernel reads raw bytes as wrong type.
      //
      // CAPTURE SAFETY: the cast/broadcast intermediates MUST live in the persistent
      // AttentionWorkspace, never in call-local temporaries. This helper's kernels are
      // captured into CUDA graphs during warmup and replayed every decode step WITHOUT
      // re-running this host code — a call-local temp is freed after capture while the
      // captured broadcast + attention kernels keep reading/writing its address on every
      // replay. As the freed block gets reused, the effective bias decays to garbage and
      // masked columns leak into attention (progressive logits corruption from the first
      // replayed step). Workspace buffers keep a stable address across steps by design.
      NDArray* biasFlat = nullptr;
      std::unique_ptr<NDArray> biasReshapedOwner;
      if (hasAttentionBias) {
        NDArray* biasToUse = attentionBias;

        // Cast bias to query dtype if mismatched (LONG->FLOAT32 is common for ONNX masks)
        if (attentionBias->dataType() != query->dataType()) {
          std::vector<LongType> biasShapeVec(
              attentionBias->shapeOf(), attentionBias->shapeOf() + attentionBias->rankOf());
          NDArray* biasCastBuf = workspace->getBuffer(
              "forward4d_biasCast", biasShapeVec, query->dataType(), context);
          biasCastBuf->assign(attentionBias);
          biasToUse = biasCastBuf;
        }

        auto biasElements = biasToUse->lengthOf();
        auto targetElements = batch * numHeads * seqLenQ * seqLenKV;

        if (biasElements == targetElements) {
          // Direct reshape: [batch, numHeads, seqQ, seqKV] -> [batch*numHeads, seqQ, seqKV]
          // (view of the live bias buffer — the view object is call-local but the
          // captured kernels hold the underlying buffer address, which stays live)
          std::vector<LongType> biasShape3D = {batch * numHeads, seqLenQ, seqLenKV};
          biasReshapedOwner.reset(biasToUse->reshape('c', biasShape3D, false));
          biasFlat = biasReshapedOwner.get();
        } else {
          // Bias needs broadcasting (e.g. [1,1,1,seqKV] -> [batch,numHeads,seqQ,seqKV])
          std::vector<LongType> targetShape4D = {batch, numHeads, seqLenQ, seqLenKV};
          NDArray* biasBcastBuf = workspace->getBuffer(
              "forward4d_biasBcast", targetShape4D, query->dataType(), context);
          biasToUse->applyTrueBroadcast(BroadcastOpsTuple::Assign(), biasBcastBuf,
                                            biasBcastBuf, false);
          std::vector<LongType> biasShape3D = {batch * numHeads, seqLenQ, seqLenKV};
          biasReshapedOwner.reset(biasBcastBuf->reshape('c', biasShape3D, false));
          biasFlat = biasReshapedOwner.get();
        }
      }

      // Fast path: fused kernel with optional bias
      fusedAttentionCuda(qPermBuffer, kPermBuffer, vPermBuffer, outFlat, scale, config.isCausal, context, biasFlat);

      // Reshape back to 4D: [batch, heads, seq, dim]
      std::vector<LongType> shape4D = {batch, numHeads, seqLenQ, headDim};
      outFlat->reshapei(shape4D);

      // Permute back and assign to output: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
      auto outPerm = outFlat->permute(permOrder, false, false);
      output->assign(outPerm);
      delete outPerm;

      // Restore workspace buffer shapes for next call.
      // Keeping the original shape avoids workspace reallocation for the same key
      // in later attention ops captured into the same CUDA graph segment.
      outFlat->reshapei(shape3D_Q);
      qPermBuffer->reshapei(qPermShape);
      kPermBuffer->reshapei(kvPermShape);
      vPermBuffer->reshapei(kvPermShape);

      if (softmaxLse != nullptr) softmaxLse->nullify();
      return;
    }
  }

  if (sd::graph::dspIsHostBuild() && seqLenQ == 1 && numKvHeads > 0 && numHeads % numKvHeads == 0) {
    BUILD_SINGLE_SELECTOR(query->dataType(), cpuDecode4D_,
                          (query, key, value, output, scale, attentionBias,
                           attentionScores, attentionLogits),
                          SD_FLOAT_TYPES);
    if (softmaxLse != nullptr) softmaxLse->nullify();
    return;
  }

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

  std::vector<LongType> tiledShape;
  if (headsPerKvHead > 1) {
    // Tile KV heads using workspace buffer
    std::vector<LongType> expandedShape = {batch, numHeads, seqLenKV, headDim};
    // Get workspace buffers with tiled shape for direct write
    tiledShape = {batch, numKvHeads, static_cast<LongType>(headsPerKvHead), seqLenKV, headDim};
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

  // Use persistent workspace buffers for intermediates.
  // CUDA graph replay captures raw buffer addresses; stack-local NDArrays are
  // destroyed after capture and leave stale addresses for replay.
  NDArray* workBuffer = workspace->getBuffer("forward4d_scores", scoresShape, query->dataType(), context);

  // Batched matmul: Q @ K^T with scale
  MmulHelper::matmul(qPermBuffer, kExpanded, workBuffer, false, true, scale, 0.0);

  if (attentionBias != nullptr && !attentionBias->isEmpty()) {
    NDArray* biasForAdd = attentionBias;
    std::unique_ptr<NDArray> biasPermutedOwner;
    std::unique_ptr<NDArray> biasExpandedOwner;
    std::vector<LongType> scores4dShape = {batch, numHeads, seqLenQ, seqLenKV};

    // Normalize ONNX source/target ordering [..., seqKV, seqQ] -> [..., seqQ, seqKV].
    if (seqLenQ != seqLenKV &&
        biasForAdd->rankOf() >= 2 &&
        biasForAdd->sizeAt(biasForAdd->rankOf() - 2) == seqLenKV &&
        biasForAdd->sizeAt(biasForAdd->rankOf() - 1) == seqLenQ) {
      std::vector<LongType> perm(static_cast<size_t>(biasForAdd->rankOf()));
      for (int i = 0; i < biasForAdd->rankOf(); i++) perm[static_cast<size_t>(i)] = i;
      std::swap(perm[perm.size() - 2], perm[perm.size() - 1]);
      biasPermutedOwner.reset(biasForAdd->permute(perm, false, false));
      biasForAdd = biasPermutedOwner.get();
    }

    // Rank-3 additive bias from ONNX: [batch, seqQ, seqKV] -> [batch, 1, seqQ, seqKV]
    if (biasForAdd->rankOf() == 3 &&
        biasForAdd->sizeAt(0) == batch &&
        biasForAdd->sizeAt(1) == seqLenQ &&
        biasForAdd->sizeAt(2) == seqLenKV) {
      std::vector<LongType> expandedShape = {batch, 1, seqLenQ, seqLenKV};
      biasExpandedOwner.reset(biasForAdd->reshape('c', expandedShape));
      biasForAdd = biasExpandedOwner.get();
    }

    workBuffer->reshapei(scores4dShape);
    workBuffer->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), biasForAdd, workBuffer, false);
    workBuffer->reshapei(scoresShape);
  }

  // Logits buffer for pre-softmax scores
  NDArray* logitsBuffer = nullptr;
  bool wantLogits = (attentionLogits != nullptr && !attentionLogits->isEmpty());
  if (wantLogits) {
    logitsBuffer = workspace->getBuffer("forward4d_logits", scoresShape, query->dataType(), context);
  }

  if (sd::graph::dspIsCudaBuild()) {
    fusedCausalMaskSoftmaxCuda(workBuffer, workBuffer, logitsBuffer, config.isCausal, context);
  } else {
    // FP16 softmax overflow prevention: cast scores to FLOAT32 before softmax.
    // In HALF, exp(x) overflows for x > ~11.09. With headDim=128 and scale ~0.088,
    // a raw Q·K dot product of ~126 (common in FP16 models) overflows after scaling,
    // producing NaN via inf/inf in softmax. FP32 has range up to ~88.7 for exp().
    // GPU paths (cuBLAS, OneDNN) accumulate in FP32 internally; CPU must do it explicitly.
    bool needsFp32Softmax = (workBuffer->dataType() == DataType::HALF || workBuffer->dataType() == DataType::BFLOAT16);
    NDArray* softmaxBuffer = workBuffer;
    NDArray* fp32Buffer = nullptr;
    if (needsFp32Softmax) {
      fp32Buffer = workspace->getBuffer("forward4d_fp32_scores", scoresShape, DataType::FLOAT32, context);
      fp32Buffer->assign(workBuffer);
      softmaxBuffer = fp32Buffer;
    }

    if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
      std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
      NDArray causalMask('c', maskShape, softmaxBuffer->dataType(), context);
      causalMask.nullify();
      // For standard causal attention (seqQ == seqKV): causalOffset = 0.
      // For cross-attention decode (seqKV > seqQ): offset the mask to align positions.
      int causalLower = (seqLenKV > seqLenQ) ? static_cast<int>(seqLenKV - seqLenQ) : 0;
      // Fill strict upper triangle (row < col - causalLower) with -1e9
      // to prevent attending to future tokens.
      BUILD_SINGLE_SELECTOR(softmaxBuffer->dataType(), causalMask.fillAsTriangular,
                            (-1.0e9f, causalLower, 0, causalMask, 'u', false), SD_COMMON_TYPES);
      softmaxBuffer->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), &causalMask, softmaxBuffer, false);
    }
    if (logitsBuffer != nullptr) {
      logitsBuffer->assign(softmaxBuffer);
    }
    // IMPORTANT: Must use explicit positive dimension for softmax.
    // The TAD helper treats -1 as sentinel meaning "all dimensions",
    // NOT as "last dimension".
    int softmaxDim4D = softmaxBuffer->rankOf() - 1;
    ops::helpers::softmax(context, softmaxBuffer, softmaxBuffer, softmaxDim4D);

    // Cast back to original dtype for the output matmul
    if (needsFp32Softmax) {
      workBuffer->assign(softmaxBuffer);
    }
  }

  // Output buffer
  std::vector<LongType> outShape = {batch * numHeads, seqLenQ, headDim};
  NDArray* outReshaped = workspace->getBuffer("forward4d_outReshaped", outShape, query->dataType(), context);

  // Batched matmul: scores @ V
  MmulHelper::matmul(workBuffer, vExpanded, outReshaped, false, false, 1.0, 0.0);

  // Reshape and permute back to output
  std::vector<LongType> outPermShape = {batch, numHeads, seqLenQ, headDim};
  outReshaped->reshapei(outPermShape);
  auto outPerm = outReshaped->permute(permOrder, false, false);
  output->assign(outPerm);
  delete outPerm;

  // Copy results to output buffers (4D shape expected by caller)
  std::vector<LongType> scores4dShape = {batch, numHeads, seqLenQ, seqLenKV};
  if (attentionScores != nullptr && !attentionScores->isEmpty() &&
      attentionScores->lengthOf() == batch * numHeads * seqLenQ * seqLenKV) {
    workBuffer->reshapei(scores4dShape);
    attentionScores->assign(workBuffer);
    workBuffer->reshapei(scoresShape);
  }
  if (wantLogits && logitsBuffer != nullptr &&
      attentionLogits->lengthOf() == batch * numHeads * seqLenQ * seqLenKV) {
    logitsBuffer->reshapei(scores4dShape);
    attentionLogits->assign(logitsBuffer);
    logitsBuffer->reshapei(scoresShape);
  }

  // Restore output intermediate shape for workspace reuse
  outReshaped->reshapei(outShape);

  // Restore workspace buffer shapes for next call (qPerm/kPerm/vPerm only)
  qPermBuffer->reshapei(qPermShape);
  if (headsPerKvHead == 1) {
    kExpanded->reshapei(kvPermShape);
    vExpanded->reshapei(kvPermShape);
  } else {
    // Restore tiled workspace buffers to requested shape for deterministic reuse
    // across multiple attention ops in captured CUDA graph segments.
    kExpanded->reshapei(tiledShape);
    vExpanded->reshapei(tiledShape);
  }

  if (softmaxLse != nullptr) {
    softmaxLse->nullify();
  }
}

//////////////////////////////////////////////////////////////////////////////
// Decode-Optimized Forward Path (seqQ=1) - Uses cuBLAS Batched GEMV
//////////////////////////////////////////////////////////////////////////////
void FlashAttentionHelper::forward4DDecode(
    NDArray* query, NDArray* key, NDArray* value,
    NDArray* output, float scale, bool isCausal,
    LaunchContext* context, NDArray* attentionBias,
    NDArray* softmaxLse) {
  // Decode phase: seqQ=1, using cuBLAS batched GEMV for TensorCore utilization.
  // Q: [batch, 1, numHeads, dim], K: [batch, seqKV, numKvHeads, dim], V: same
  // For M=1, Q@K^T is a GEMV: [1,dim] × [seqKV,dim]^T → [1,seqKV] per head
  
  auto batch = query->sizeAt(0);
  auto numHeads = query->sizeAt(2);
  auto headDim = query->sizeAt(3);
  auto seqKV = key->sizeAt(1);
  auto numKvHeads = key->sizeAt(2);
  int headsPerKvHead = numHeads / numKvHeads;
  
  auto workspace = AttentionWorkspace::getInstance();
  
  if (sd::graph::dspIsCudaBuild()) {
    // Permute Q, K, V to [batch*heads, seq, dim] layout for cuBLAS
    std::vector<LongType> permOrder = {0, 2, 1, 3};
    std::vector<LongType> qShape = {batch * numHeads, 1, headDim};
    std::vector<LongType> kvShape = {batch * numKvHeads, seqKV, headDim};
    std::vector<LongType> scoresShape = {batch * numHeads, 1, seqKV};

    NDArray* qPerm = workspace->getBuffer("decode_qPerm", qShape, query->dataType(), context);
    NDArray* kPerm = workspace->getBuffer("decode_kPerm", kvShape, key->dataType(), context);
    NDArray* vPerm = workspace->getBuffer("decode_vPerm", kvShape, value->dataType(), context);

    // Permute and assign
    auto qView = query->permute(permOrder, false, false);
    auto kView = key->permute(permOrder, false, false);
    auto vView = value->permute(permOrder, false, false);
    qPerm->assign(qView);
    kPerm->assign(kView);
    vPerm->assign(vView);
    delete qView;
    delete kView;
    delete vView;

    // Handle GQA: expand KV heads if needed
    NDArray* kExpanded = kPerm;
    NDArray* vExpanded = vPerm;

    if (headsPerKvHead > 1) {
      std::vector<LongType> expandedShape = {batch * numHeads, seqKV, headDim};
      NDArray* kExpBuf = workspace->getBuffer("decode_kExp", expandedShape, key->dataType(), context);
      NDArray* vExpBuf = workspace->getBuffer("decode_vExp", expandedShape, value->dataType(), context);

      // Tile KV heads
      std::vector<LongType> tileShape = {batch, numKvHeads, 1, seqKV, headDim};
      kPerm->reshapei(tileShape);
      vPerm->reshapei(tileShape);
      std::vector<LongType> reps = {1, 1, static_cast<LongType>(headsPerKvHead), 1, 1};
      kPerm->tile(reps, *kExpBuf);
      vPerm->tile(reps, *vExpBuf);
      kPerm->reshapei(kvShape);
      vPerm->reshapei(kvShape);

      kExpBuf->reshapei(expandedShape);
      vExpBuf->reshapei(expandedShape);
      kExpanded = kExpBuf;
      vExpanded = vExpBuf;
    }

    // Workspace for scores [batch*heads, 1, seqKV]
    NDArray* scores = workspace->getBuffer("decode_scores", scoresShape, query->dataType(), context);

    // Q @ K^T: [batch*heads, 1, dim] × [batch*heads, dim, seqKV] → [batch*heads, 1, seqKV]
    // For seqQ=1, this is batched GEMV, not GEMM
    MmulHelper::matmul(qPerm, kExpanded, scores, false, true, scale, 0.0);

    // Add attention bias if present
    if (attentionBias != nullptr && !attentionBias->isEmpty()) {
      // Bias shape: [batch, seqQ, seqKV] or [batch, numHeads, seqQ, seqKV]
      if (attentionBias->rankOf() == 3) {
        // [batch, 1, seqKV] -> broadcast to [batch*heads, 1, seqKV]
        scores->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), attentionBias, scores, false);
      } else if (attentionBias->rankOf() == 4) {
        // Bias may be [batch, numHeads, seqQ, seqKV] (full) or broadcastable
        // (e.g. [1, 1, 1, seqKV]). Broadcast to full shape first, then reshape.
        auto biasElements = attentionBias->lengthOf();
        auto targetElements = batch * numHeads * 1 * seqKV;
        std::vector<LongType> biasShape3D = {batch * numHeads, 1, seqKV};
        if (biasElements == targetElements) {
          NDArray* biasFlat = attentionBias->reshape('c', biasShape3D, false);
          scores->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), biasFlat, scores, false);
          delete biasFlat;
        } else {
          // Broadcast [1,1,1,seqKV] -> [batch,numHeads,1,seqKV] then reshape
          std::vector<LongType> fullShape = {batch, numHeads, 1, seqKV};
          NDArray biasFull('c', fullShape, attentionBias->dataType(), context);
          attentionBias->applyTrueBroadcast(sd::BroadcastOpsTuple::Assign(), &biasFull, &biasFull, false);
          NDArray* biasFlat = biasFull.reshape('c', biasShape3D, false);
          scores->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), biasFlat, scores, false);
          delete biasFlat;
        }
      }
    }

    // NOTE: Causal mask is NOT needed for decode (seqQ=1).
    // During decode, we attend to all past tokens (positions 0..cachePos),
    // and there are no "future" positions to mask out.
    // Causal masking only applies during prefill (seqQ > 1).

    // Fused softmax with warp-shuffle reductions — more efficient than generic softmax
    // for decode shape [batch*numHeads, 1, seqKV]. isCausal=false since seqQ=1 has nothing to mask.
    fusedCausalMaskSoftmaxCuda(scores, scores, nullptr, false, context);

    // attn @ V: [batch*heads, 1, seqKV] × [batch*heads, seqKV, dim] → [batch*heads, 1, dim]
    NDArray* outFlat = workspace->getBuffer("decode_out", qShape, query->dataType(), context);
    MmulHelper::matmul(scores, vExpanded, outFlat, false, false, 1.0, 0.0);

    // Permute back to [batch, 1, numHeads, dim] -> [batch, 1, numHeads, dim]
    outFlat->reshapei({batch, numHeads, 1, headDim});
    auto outPerm = outFlat->permute(permOrder, false, false);
    output->assign(outPerm);
    delete outPerm;

    // Restore workspace shapes
    qPerm->reshapei(qShape);
    kPerm->reshapei(kvShape);
    if (headsPerKvHead > 1) {
      kExpanded->reshapei({batch * numHeads, seqKV, headDim});
    }
  } else {
    // CPU fallback - use standard forward4D path
    Config config;
    config.scale = scale;
    config.isCausal = isCausal;
    forward4D(query, key, value, output, config, softmaxLse, nullptr, nullptr, context, attentionBias);
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

  float scale = config.scale > 0.0f ? config.scale : 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(dim));

  auto workspace = AttentionWorkspace::getInstance();
  std::vector<LongType> scoresShape = {batch, seqLenQ, seqLenKV};

  // Recompute attention: Q @ K^T
  NDArray* scores = workspace->getBuffer("backward3d_scores", scoresShape, query->dataType(), context);
  MmulHelper::matmul(query, key, scores, false, true, scale, 0.0);

  // Apply causal mask if needed
  if (config.isCausal && seqLenQ > 0 && seqLenKV > 0) {
    if (sd::graph::dspIsCudaBuild()) {
      applyCausalMaskCuda(scores, context);
    } else {
      std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
      NDArray* causalMask = workspace->getBuffer("backward3d_mask", maskShape, query->dataType(), context);
      causalMask->nullify();
      int causalLower = (seqLenKV > seqLenQ) ? static_cast<int>(seqLenKV - seqLenQ) : 0;
      // Fill strict upper triangle (row < col - causalLower) with -1e9
      // to prevent attending to future tokens.
      BUILD_SINGLE_SELECTOR(query->dataType(), causalMask->fillAsTriangular,
                            (-1.0e9f, causalLower, 0, *causalMask, 'u', false), SD_COMMON_TYPES);
      scores->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), causalMask, scores, false);
    }
  }

  // Softmax - use explicit positive dimension (TAD treats -1 as "all dimensions")
  NDArray* attnWeights = workspace->getBuffer("backward3d_attnWeights", scoresShape, query->dataType(), context);
  int bwSoftmaxDim3D = scores->rankOf() - 1;
  ops::helpers::softmax(context, scores, attnWeights, bwSoftmaxDim3D);

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
  dAttn->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), rowSums, dAttn, false);
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

  float scale = config.scale > 0.0f ? config.scale : 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(headDim));
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
    if (sd::graph::dspIsCudaBuild()) {
      applyCausalMaskCuda(scores, context);
    } else {
      std::vector<LongType> maskShape = {seqLenQ, seqLenKV};
      NDArray* causalMask = workspace->getBuffer("backward4d_mask", maskShape, query->dataType(), context);
      causalMask->nullify();
      int causalLower = (seqLenKV > seqLenQ) ? static_cast<int>(seqLenKV - seqLenQ) : 0;
      // Fill strict upper triangle (row < col - causalLower) with -1e9
      // to prevent attending to future tokens.
      BUILD_SINGLE_SELECTOR(query->dataType(), causalMask->fillAsTriangular,
                            (-1.0e9f, causalLower, 0, *causalMask, 'u', false), SD_COMMON_TYPES);
      scores->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), causalMask, scores, false);
    }
  }

  // Softmax - use explicit positive dimension (TAD treats -1 as "all dimensions")
  NDArray* attnWeights = workspace->getBuffer("backward4d_attnWeights", scoresShape, query->dataType(), context);
  int bwSoftmaxDim4D = scores->rankOf() - 1;
  ops::helpers::softmax(context, scores, attnWeights, bwSoftmaxDim4D);

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
  dAttn->applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), rowSums, dAttn, false);
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
    float rescaleFactor = sd::math::sd_exp<float, float>(state.maxVal - newMax);
    state.correction = rescaleFactor;
    state.sumExp = state.sumExp * rescaleFactor + newSum;
    state.maxVal = newMax;
    state.needsRescale = true;
  } else if (maxDiff < -threshold) {
    float rescaleFactor = sd::math::sd_exp<float, float>(newMax - state.maxVal);
    state.sumExp = state.sumExp + newSum * rescaleFactor;
    state.correction = 1.0f;
    state.needsRescale = false;
  } else {
    float maxOfMax = std::max(state.maxVal, newMax);
    state.sumExp = state.sumExp * sd::math::sd_exp<float, float>(state.maxVal - maxOfMax) +
                   newSum * sd::math::sd_exp<float, float>(newMax - maxOfMax);
    state.maxVal = maxOfMax;
    state.correction = sd::math::sd_exp<float, float>(state.maxVal - maxOfMax);
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
    forward3D(query, key, value, output, config, softmaxLse, nullptr, nullptr, context, nullptr);
  } else {
    forward4D(query, key, value, output, config, softmaxLse, nullptr, nullptr, context, nullptr);
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

      float expScore = sd::math::sd_exp<float, float>(score - states[q].maxVal);
      if (score > states[q].maxVal) {
        float rescale = sd::math::sd_exp<float, float>(states[q].maxVal - score);
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
