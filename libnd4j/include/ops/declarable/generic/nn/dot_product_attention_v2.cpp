/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Paul Dubs
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dot_product_attention_v2)

#include <math/templatemath.h>
#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/reverse.h>
#include <graph/DspDeviceDispatch.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <ops/declarable/helpers/kv_cache_quantize.h>
#include <graph/DspDiagnostics.h>
#include <helpers/AttentionHelper.h>
#include <helpers/FlashAttentionHelper.h>
#include <helpers/AttentionWorkspace.h>
#include <cmath>
#include <memory>

namespace sd {
namespace ops {

namespace {

// A rank-4 BSHD attention that goes through AttentionHelper is evaluated as `batch * heads` independent rank-3
// attentions, the layout AttentionHelper works on. The forward and the backward move tensors into that head-major
// layout and back through the helpers below, so the two cannot drift apart.

// Copies a BSHD tensor [batch, seq, heads, dim] into the head-major [batch * heads * group, seq, dim]: row
// (b * heads + h) * group + g is head h of batch b, repeated `group` times in a row. That is the grouped-query layout:
// with `heads` KV heads and group = queryHeads / kvHeads, the row of query head q reads KV head q / group. The caller
// owns the result, a C-contiguous array.
NDArray* toHeadMajor3d(NDArray* bshd, const LongType group, LaunchContext* context) {
  const LongType batch = bshd->sizeAt(0);
  const LongType seq = bshd->sizeAt(1);
  const LongType heads = bshd->sizeAt(2);
  const LongType dim = bshd->sizeAt(3);

  std::vector<LongType> permOrder = {0, 2, 1, 3};
  NDArray* headMajorView = bshd->permute(permOrder, false, false);  // [batch, heads, seq, dim], a view of bshd
  NDArray* headMajor = headMajorView->dup('c');                     // C-contiguous copy of that view
  delete headMajorView;

  if (group > 1) {
    std::vector<LongType> unitGroupShape = {batch, heads, 1, seq, dim};
    headMajor->reshapei(unitGroupShape);
    std::vector<LongType> groupedShape = {batch, heads, group, seq, dim};
    NDArray* grouped = new NDArray('c', groupedShape, bshd->dataType(), context);
    std::vector<LongType> reps = {1, 1, group, 1, 1};
    headMajor->tile(reps, *grouped);
    // The tile runs on headMajor's stream and reads its buffer asynchronously on CUDA: flush that stream before the
    // buffer is freed.
    headMajor->synchronizeExecStream("dpa_v2 head-major temp free");
    delete headMajor;
    headMajor = grouped;
  }

  std::vector<LongType> flatShape = {batch * heads * group, seq, dim};
  headMajor->reshapei(flatShape);
  return headMajor;
}

// The inverse of toHeadMajor3d for a gradient: writes the head-major [batch * heads * group, seq, dim] array into the
// BSHD tensor `bshd` [batch, seq, heads, dim], summing the `group` copies of each head (the gradient of a KV head
// shared by `group` query heads is the sum over them).
void fromHeadMajor3d(NDArray* headMajor, NDArray* bshd, const LongType group, LaunchContext* context) {
  const LongType batch = bshd->sizeAt(0);
  const LongType seq = bshd->sizeAt(1);
  const LongType heads = bshd->sizeAt(2);
  const LongType dim = bshd->sizeAt(3);

  NDArray* perHead = nullptr;     // [batch, heads, seq, dim]
  NDArray* summed = nullptr;      // owned, only when copies had to be summed
  NDArray* groupedView = nullptr;
  if (group > 1) {
    std::vector<LongType> groupedShape = {batch, heads, group, seq, dim};
    groupedView = headMajor->reshape('c', groupedShape, false);
    std::vector<LongType> perHeadShape = {batch, heads, seq, dim};
    summed = new NDArray('c', perHeadShape, headMajor->dataType(), context);
    std::vector<LongType> sumDims = {2};
    groupedView->reduceAlongDimension(reduce::Sum, summed, &sumDims, false);
    perHead = summed;
  } else {
    std::vector<LongType> perHeadShape = {batch, heads, seq, dim};
    perHead = headMajor->reshape('c', perHeadShape, false);  // a view of the C-contiguous head-major array
  }

  std::vector<LongType> permOrder = {0, 2, 1, 3};
  NDArray* bshdView = perHead->permute(permOrder, false, false);  // [batch, seq, heads, dim]
  bshd->assign(bshdView);
  // The kernels above read the temporaries asynchronously on CUDA (the sum on the head-major gradient's stream, the
  // copy on the stream of its source): flush the streams before the temporaries are freed.
  perHead->synchronizeExecStream("dpa_v2 head-major gradient temp free (source)");
  bshd->synchronizeExecStream("dpa_v2 head-major gradient temp free (target)");
  delete bshdView;
  delete groupedView;
  delete summed;
  if (group <= 1) delete perHead;
}

// The head-major attention has `batch * heads` rows, so a [batch, length] mask of a multi-head, multi-batch attention
// has to be expanded to [batch * heads, length]: row b * heads + h is the mask row of batch b. Every other mask already
// broadcasts against the head-major rows and is left alone: a batch-shared [1, length] mask, and the [1, 1, 1, length]
// views graphs hand over (AttentionHelper squeezes those). Returns the expanded mask in the attention's floating point
// type (BOOL and integer masks are accepted), owned by the caller, or nullptr when `mask` needs no expansion.
NDArray* expandMaskPerHead(NDArray* mask, const LongType batch, const LongType heads, const LongType length,
                           const DataType floatType, LaunchContext* context) {
  if (mask == nullptr || heads <= 1 || mask->rankOf() != 2 || mask->sizeAt(0) == 1) return nullptr;
  REQUIRE_TRUE(mask->sizeAt(0) == batch && mask->sizeAt(1) == length, 0,
               "dot_product_attention_v2: a [batch, length] mask of a rank-4 attention must be [%lld, %lld], got "
               "[%lld, %lld]",
               static_cast<long long>(batch), static_cast<long long>(length), static_cast<long long>(mask->sizeAt(0)),
               static_cast<long long>(mask->sizeAt(1)));

  NDArray* numeric = mask->cast(floatType);  // a C-contiguous float copy
  std::vector<LongType> unitHeadShape = {batch, 1, length};
  numeric->reshapei(unitHeadShape);
  std::vector<LongType> tiledShape = {batch, heads, length};
  NDArray* tiled = new NDArray('c', tiledShape, floatType, context);
  std::vector<LongType> reps = {1, heads, 1};
  numeric->tile(reps, *tiled);
  // The tile runs on numeric's stream and reads its buffer asynchronously on CUDA: flush that stream before the buffer
  // is freed.
  numeric->synchronizeExecStream("dpa_v2 per-head mask temp free");
  delete numeric;
  std::vector<LongType> flatShape = {batch * heads, length};
  tiled->reshapei(flatShape);
  return tiled;
}

}  // namespace

CUSTOM_OP_IMPL(dot_product_attention_v2, -2, -1, false, -2, -2) {
  auto queriesOrig = INPUT_VARIABLE(0);
  auto valuesOrig = INPUT_VARIABLE(1);

  REQUIRE_TRUE(queriesOrig->rankOf() >= 2 && queriesOrig->rankOf() <= 4, 0,
               "dot_product_attention_v2: Input rank must be 2, 3, or 4, got %i", queriesOrig->rankOf());
  REQUIRE_TRUE(queriesOrig->isR(), 0,
               "dot_product_attention_v2: queries must be floating-point/real type, got %i",
               static_cast<int>(queriesOrig->dataType()));
  REQUIRE_TRUE(valuesOrig->isR(), 0,
               "dot_product_attention_v2: values must be floating-point/real type, got %i",
               static_cast<int>(valuesOrig->dataType()));

  // Track reshaped arrays for cleanup
  NDArray* queries = nullptr;
  NDArray* values = nullptr;
  NDArray* keys = nullptr;
  NDArray* qMask = nullptr;
  NDArray* vMask = nullptr;
  bool reshapedQ = false;
  bool promotedV = false;
  // promotedKPtr captures the allocated reshape result so cleanup can free it
  // even after `keys` is overwritten by `keys = keyCache` in the KV-cache path.
  NDArray* promotedKPtr = nullptr;
  // Sliced attention bias (allocated when prefill bias is wider than K seq dim)
  NDArray* slicedBiasOwner = nullptr;

  // Auto-promote V from 3D to 4D when Q is 4D (GGUF models may pass flat V before head split)
  if (queriesOrig->rankOf() == 4 && valuesOrig->rankOf() == 3) {
    auto headDim = queriesOrig->sizeAt(3);
    auto vDim = valuesOrig->sizeAt(2);
    REQUIRE_TRUE(vDim % headDim == 0, 0,
                 "dot_product_attention_v2: V dim %lld must be divisible by Q headDim %lld for auto-reshape",
                 (long long)vDim, (long long)headDim);
    auto numKvHeads = vDim / headDim;
    std::vector<sd::LongType> vShape4d = {valuesOrig->sizeAt(0), valuesOrig->sizeAt(1), numKvHeads, headDim};
    valuesOrig = valuesOrig->reshape('c', vShape4d);
    promotedV = true;
  }

  REQUIRE_TRUE(queriesOrig->rankOf() == valuesOrig->rankOf(), 0,
               "dot_product_attention_v2: Queries and values must have same rank, got %i vs %i",
               queriesOrig->rankOf(), valuesOrig->rankOf());

  bool isRank4 = (queriesOrig->rankOf() == 4);

  // Handle rank 2 inputs by adding batch dimension
  if(queriesOrig->rankOf() == 2) {
    reshapedQ = true;
    std::vector<sd::LongType> qShape = {1, queriesOrig->sizeAt(0), queriesOrig->sizeAt(1)};
    std::vector<sd::LongType> vShape = {1, valuesOrig->sizeAt(0), valuesOrig->sizeAt(1)};
    queries = queriesOrig->reshape('c', qShape);
    values = valuesOrig->reshape('c', vShape);
  } else {
    queries = queriesOrig;
    values = valuesOrig;
  }

  // Handle keys - defaults to values if not provided
  auto keysOrig = block.width() > 2 ? INPUT_VARIABLE(2) : valuesOrig;
  if(reshapedQ && block.width() > 2) {
    std::vector<sd::LongType> kShape = {1, keysOrig->sizeAt(0), keysOrig->sizeAt(1)};
    keys = keysOrig->reshape('c', kShape);
  } else if(reshapedQ) {
    keys = values;  // keys defaults to values
  } else {
    keys = keysOrig;
  }

  // Auto-promote K from 3D to 4D when Q is 4D (symmetric with V promotion)
  bool promotedK = false;
  if (isRank4 && keys->rankOf() == 3) {
    auto headDim = queriesOrig->sizeAt(3);
    auto kDim = keys->sizeAt(2);
    if (kDim % headDim == 0) {
      auto numKvHeads = kDim / headDim;
      std::vector<sd::LongType> kShape4d = {keys->sizeAt(0), keys->sizeAt(1), numKvHeads, headDim};
      keys = keys->reshape('c', kShape4d);
      promotedK = true;
      // Save the reshape allocation immediately. `keys` may be overwritten later
      // (e.g. by `keys = keyCache` in the in-place KV-cache path). Cleanup MUST
      // free this pointer, not whatever `keys` holds at cleanup time.
      promotedKPtr = keys;
    }
  }

  REQUIRE_TRUE(keys->isR(), 0,
               "dot_product_attention_v2: keys must be floating-point/real type, got %i",
               static_cast<int>(keys->dataType()));

  // Handle masks - check for empty arrays as well as nullptr
  auto qMaskOrig = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  auto vMaskOrig = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

  // Treat empty arrays as no mask
  if(qMaskOrig != nullptr && qMaskOrig->isEmpty()) {
    qMaskOrig = nullptr;
  }
  if(vMaskOrig != nullptr && vMaskOrig->isEmpty()) {
    vMaskOrig = nullptr;
  }

  // Reshape masks if needed
  if(qMaskOrig != nullptr && reshapedQ) {
    std::vector<sd::LongType> qmShape = {1, qMaskOrig->sizeAt(0), qMaskOrig->sizeAt(1)};
    qMask = qMaskOrig->reshape('c', qmShape);
  } else {
    qMask = qMaskOrig;
  }

  if(vMaskOrig != nullptr && reshapedQ) {
    std::vector<sd::LongType> vmShape = {1, vMaskOrig->sizeAt(0), vMaskOrig->sizeAt(1)};
    vMask = vMaskOrig->reshape('c', vmShape);
  } else {
    vMask = vMaskOrig;
  }

  // Optional additive attention bias (input 5) for ONNX-style relative position bias / attn mask.
  // We intentionally infer this at runtime from tensor shape only; importer-time .arr/.shape is unreliable.
  // If input 6 is present, input 5 is treated as KV cache input instead.
  NDArray* attentionBias = nullptr;
  auto extraInput = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
  auto extraInput2 = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;
  // Rank guard: empty/rank-0 arrays may lose the EMPTY flag through DSP wiring
  if (extraInput != nullptr && extraInput->rankOf() < 2) extraInput = nullptr;
  if (extraInput2 != nullptr && extraInput2->rankOf() < 2) extraInput2 = nullptr;
  if (extraInput != nullptr && !extraInput->isEmpty() &&
      (extraInput2 == nullptr || extraInput2->isEmpty())) {
    auto tq = queries->sizeAt(1);
    auto tv = values->sizeAt(1);
    bool looksLikeBias = false;
    if (extraInput->rankOf() >= 2) {
      auto d0 = extraInput->sizeAt(extraInput->rankOf() - 2);
      auto d1 = extraInput->sizeAt(extraInput->rankOf() - 1);
      // Accept both [..., Tq, Tv] and [..., Tv, Tq].
      // Some ONNX exports use (source, target) ordering for attention bias.
      looksLikeBias = (d0 == tq && d1 == tv) || (d0 == tv && d1 == tq);
    }
    if (looksLikeBias) {
      attentionBias = extraInput;
    }
  }

  // In-place KV cache write: when cache_position (input 7) is present along with
  // keyCache (input 5) and valueCache (input 6), write current K/V at that position
  // in the cache buffers and use the full buffers for attention.
  auto cachePosInput = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;
  if (cachePosInput != nullptr && (cachePosInput->isEmpty() || cachePosInput->lengthOf() == 0)) cachePosInput = nullptr;

  bool useInPlaceKv = false;
  NDArray* keyCache = nullptr;
  NDArray* valueCache = nullptr;
  NDArray* currentKeyWindow = nullptr;
  NDArray* currentValueWindow = nullptr;
  const void* currentKvPosition = nullptr;

  if (extraInput != nullptr && !extraInput->isEmpty() && extraInput->rankOf() >= 2 &&
      extraInput2 != nullptr && !extraInput2->isEmpty() && extraInput2->rankOf() >= 2) {
    keyCache = extraInput;
    valueCache = extraInput2;
    useInPlaceKv = (cachePosInput != nullptr);
  }

  // V2 QUANTIZED path: detect INT8 key cache → quantised-on-write + quantised attention read.
  // The quantised scale caches are threaded through input[9] (keyScale) and input[10] (valScale).
  // When keyCache is INT8, the float K/V tensors (keys, values) are quantised into the INT8
  // caches at cachePosition, then the quantised GQA decode kernel is used for attention.
  NDArray* keyScaleCache = block.width() > 9 ? INPUT_VARIABLE(9) : nullptr;
  NDArray* valScaleCache = block.width() > 10 ? INPUT_VARIABLE(10) : nullptr;
  if (keyScaleCache != nullptr && keyScaleCache->isEmpty()) keyScaleCache = nullptr;
  if (valScaleCache != nullptr && valScaleCache->isEmpty()) valScaleCache = nullptr;

  // Determine if this is a V2 quantised call: keyCache is INT8.
  // ADR 0107 V2 ROW-INLINE SCALE: with inputs 9/10 absent, every INT8 cache row is
  // [headDim values | FLOAT32 scale] (last dim headDim + 4), so the scale travels with its row.
  // Non-empty inputs 9/10 are separate FLOAT32 [batch, maxKvLen, kvHeads] scale caches over an
  // INT8 [batch, maxKvLen, kvHeads, headDim] cache.
  bool useQuantisedKv = (keyCache != nullptr && keyCache->dataType() == DataType::INT8);

  // ADR 0107 V2 diagnosis: record whether the quantised-KV path is taken at each call.
  // At CUDA-graph capture the taken branch is baked into the replayed graph, so a capture-time
  // useQuant=0 on an INT8 cache means the frozen plan will forever run the float-on-INT8 path.
  if (DSP_DIAG_ENABLED(KV_CACHE)) {
    int kcDt = keyCache != nullptr ? (int)keyCache->dataType() : -1;
    int in9 = block.width() > 9 ? (INPUT_VARIABLE(9)->isEmpty() ? 0 : 1) : -1;
    int in10 = block.width() > 10 ? (INPUT_VARIABLE(10)->isEmpty() ? 0 : 1) : -1;
    DSP_DIAG(KV_CACHE,
             "dpa_v2 width=%d keyCacheDt=%d in9=%d in10=%d keyScale=%s valScale=%s useQuant=%d",
             (int)block.width(), kcDt, in9, in10,
             keyScaleCache ? "set" : "null", valScaleCache ? "set" : "null",
             useQuantisedKv ? 1 : 0);
  }

  if (useInPlaceKv) {
    // CUDA graph compatible: kvInPlaceWriteBSHD / kvInPlaceWriteQuantisedBSHD reads
    // cache_position from a device-side pointer. The pointer ADDRESS is baked into the
    // graph at capture time; only the VALUE changes between replays (updated via D2D staging).
    currentKvPosition = sd::graph::dspBufferConst(cachePosInput);
    const void* cachePosPtr = currentKvPosition;

    // Keep the producer tensors available to the direct GQA attention kernel.
    // The cache scatter remains the write-back for future invocations, while this
    // invocation reads its own current window without a read-after-write dependency.
    if (!useQuantisedKv && isRank4 && keys->rankOf() == 4 && values->rankOf() == 4) {
      currentKeyWindow = keys;
      currentValueWindow = values;
    }
    if (DSP_DIAG_ENABLED(KV_CACHE)) {
      DSP_DIAG(KV_CACHE,
               "dpa_current_window inPlace=%d rank4=%d quant=%d pos=%d "
               "qSeq=%lld currentKRank=%d currentKSeq=%lld currentKHeads=%lld currentKDim=%lld "
               "cacheSeq=%lld cacheHeads=%lld cacheDim=%lld",
               useInPlaceKv ? 1 : 0, isRank4 ? 1 : 0, useQuantisedKv ? 1 : 0,
               currentKvPosition != nullptr ? 1 : 0,
               (long long)queries->sizeAt(1),
               currentKeyWindow != nullptr ? currentKeyWindow->rankOf() : -1,
               currentKeyWindow != nullptr && currentKeyWindow->rankOf() > 1
                   ? (long long)currentKeyWindow->sizeAt(1) : -1LL,
               currentKeyWindow != nullptr && currentKeyWindow->rankOf() > 2
                   ? (long long)currentKeyWindow->sizeAt(2) : -1LL,
               currentKeyWindow != nullptr && currentKeyWindow->rankOf() > 3
                   ? (long long)currentKeyWindow->sizeAt(3) : -1LL,
               (long long)keyCache->sizeAt(1),
               keyCache->rankOf() > 2 ? (long long)keyCache->sizeAt(2) : -1LL,
               keyCache->rankOf() > 3 ? (long long)keyCache->sizeAt(3) : -1LL);
    }

    if (useQuantisedKv) {
      // V2 QUANTIZED: quantise the current-step K/V rows (model dtype, any float type) into the
      // INT8 caches at cachePosition. keys/values keep pointing at those rows: the quantised read
      // below takes them as its current window instead of reading back the row just scattered.
      REQUIRE_TRUE(isRank4 && keys->rankOf() == 4 && values->rankOf() == 4 && queries->sizeAt(1) == 1 &&
                       keys->sizeAt(1) == 1 && values->sizeAt(1) == 1,
                   0,
                   "dot_product_attention_v2: the INT8 KV cache supports rank-4 single-token decode only "
                   "([batch, 1, heads, headDim] queries/keys/values); got query rank %i seq %lld, key rank %i "
                   "seq %lld, value rank %i seq %lld",
                   queries->rankOf(), (long long)queries->sizeAt(1), keys->rankOf(), (long long)keys->sizeAt(1),
                   values->rankOf(), (long long)values->sizeAt(1));
      helpers::kvInPlaceWriteQuantisedBSHD(keyCache, keyScaleCache, keys, cachePosPtr, block.launchContext());
      helpers::kvInPlaceWriteQuantisedBSHD(valueCache, valScaleCache, values, cachePosPtr, block.launchContext());
    } else {
      // Float path (unchanged)
      // Write current K/V at cache_position in the buffers (in-place).
      // kvInPlaceWriteBSHD handles both rank-4 BSHD and rank-3 BSF layouts.
      helpers::kvInPlaceWriteBSHD(keyCache, keys, cachePosPtr, block.launchContext());
      helpers::kvInPlaceWriteBSHD(valueCache, values, cachePosPtr, block.launchContext());
    }

    // Use full cache buffers as K/V for attention (float path only; quantised handled below)
    if (!useQuantisedKv) {
      keys = keyCache;
      values = valueCache;
    }

    // Check for attention bias at input[8] when KV cache is active.
    // This is separate from the input[5] bias path (which is skipped when input[5] is keyCache).
    auto kvCacheBias = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;
    if (kvCacheBias != nullptr && kvCacheBias->isEmpty()) kvCacheBias = nullptr;
    if (kvCacheBias != nullptr) {
      attentionBias = kvCacheBias;
    }
  }

  // Fallback: when KV caches are empty (prefill) but attention bias is at input[8],
  // it was never read above because the useInPlaceKv block was skipped.
  // This happens during GGUF prefill: empty KV cache placeholders at input[5,6],
  // cache_position at input[7], and causal mask at input[8].
  if (attentionBias == nullptr && block.width() > 8) {
    auto prefillBias = INPUT_VARIABLE(8);
    if (prefillBias != nullptr && !prefillBias->isEmpty() && prefillBias->rankOf() >= 2) {
      // The bias at input[8] is sized for the full KV cache (maxKvLen), but during
      // prefill the raw K has only seqLen positions (no cache). Check if the bias
      // last dim matches K's seq dim — if not, slice it or fall back to built-in causal.
      auto biasLastDim = prefillBias->sizeAt(prefillBias->rankOf() - 1);
      auto kSeqDim = keys->sizeAt(isRank4 ? 1 : 1);
      if (biasLastDim == kSeqDim) {
        attentionBias = prefillBias;
      } else if (biasLastDim > kSeqDim) {
        // Bias is wider than K (designed for full cache) — slice to [.., Tq, kSeqDim]
        // Use subarray to take only the first kSeqDim columns
        std::vector<LongType> sliceIdx;
        for (int d = 0; d < prefillBias->rankOf() - 1; d++) {
          sliceIdx.push_back(0);
          sliceIdx.push_back(prefillBias->sizeAt(d));
        }
        sliceIdx.push_back(0);
        sliceIdx.push_back(kSeqDim);
        // Preserve batch/head/query axes, including the single-token [1,1,1,1] case.
        auto* slicedBias = (*prefillBias)(sliceIdx, true);
        if (slicedBias->dataType() != queries->dataType()) {
          // Keep only the view here: the stride-aware assign below converts directly
          // into the persistent, C-contiguous dpa_v2_biasCast workspace buffer.
          // Duplicating first would allocate a redundant full-size source-dtype bias.
          // Retain the view until op cleanup; its storage belongs to prefillBias.
          slicedBiasOwner = slicedBias;
        } else {
          // Preserve the existing contiguous preparation for same-dtype bias.
          slicedBiasOwner = new NDArray(slicedBias->dup());
          delete slicedBias;
        }
        attentionBias = slicedBiasOwner;
      }
      // else biasLastDim < kSeqDim: unexpected, skip bias
    }
  }

  // Get arguments - T_ARG order: scale, dropout
  auto scale = block.numT() > 0 ? T_ARG(0) : 1.0;
  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

  // Auto scale when scale <= 0: 1/sqrt(headDim or dim)
  if (scale <= 0.0) {
    auto dim = isRank4 ? queries->sizeAt(3) : queries->sizeAt(2);
    scale = 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(dim));
  }

  // B_ARG order: useCausalMask, training, useFlashAttention
  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;
  auto useFlashAttention = block.numB() > 2 ? B_ARG(2) : true;

  // Get output variables. The DSP executor may leave inference-only auxiliary
  // outputs null when their logical values are not consumed; preserve output
  // numbering while allowing the fused path to avoid quadratic allocations.
  auto applyScoresOut = OUTPUT_VARIABLE(0);
  NDArray* attentionScores = block.width() > 1 ? OUTPUT_VARIABLE(1) : nullptr;
  NDArray* attentionLogits = block.width() > 2 ? OUTPUT_VARIABLE(2) : nullptr;
  auto dropoutMask = dropout > 0.0 && block.width() > 3 ? OUTPUT_VARIABLE(3) : nullptr;

  // Reshape outputs for rank 2 case
  if(reshapedQ) {
    applyScoresOut->reshapei('c', {1, applyScoresOut->sizeAt(0), applyScoresOut->sizeAt(1)});
    if (attentionLogits != nullptr) {
      attentionLogits->reshapei('c', {1, attentionLogits->sizeAt(0), attentionLogits->sizeAt(1)});
    }
    if (attentionScores != nullptr) {
      attentionScores->reshapei('c', {1, attentionScores->sizeAt(0), attentionScores->sizeAt(1)});
    }
  }

  // Setup FlashAttentionHelper config
  FlashAttentionHelper::Config config;
  config.scale = static_cast<float>(scale);
  config.isCausal = useCausalMask;
  config.dropout = 0.0f;
  if (isRank4) {
    // Rank 4: [batch, seq, numHeads, headDim] (BSHD format)
    config.numHeads = queries->sizeAt(2);
    config.numKvHeads = keys->sizeAt(2);
  } else {
    config.numHeads = 1;
    config.numKvHeads = 1;
  }
  config.currentKeyWindow = currentKeyWindow;
  config.currentValueWindow = currentValueWindow;
  config.currentKvPosition = currentKvPosition;

  // Treat empty or scalar arrays as no mask
  // SameDiff may create empty placeholders or rank-0 scalar arrays for null inputs
  if(qMask != nullptr && (qMask->isEmpty() || qMask->rankOf() == 0)) {
    qMask = nullptr;
  }
  if(vMask != nullptr && (vMask->isEmpty() || vMask->rankOf() == 0)) {
    vMask = nullptr;
  }

  bool hasInputMasks = (qMask != nullptr) || (vMask != nullptr);
  bool hasAttentionBias = (attentionBias != nullptr && !attentionBias->isEmpty());

  // CAPTURE SAFETY: Additive bias/mask can arrive as BOOL/INT from importer graphs and must
  // be cast to query dtype for arithmetic in the helper path. A call-local unique_ptr would
  // be freed after this block, but CUDA graph capture records the device address. On every
  // replay that address would be stale (freed memory). Use a persistent AttentionWorkspace
  // buffer instead — its device address is stable across capture/replay cycles. The buffer is
  // only reallocated when the shape or dtype changes (workspace internally checks shape-key).
  if (hasAttentionBias && attentionBias->dataType() != queries->dataType()) {
    auto* workspace = AttentionWorkspace::getInstance();
    std::vector<sd::LongType> biasShapeVec(
        attentionBias->shapeOf(), attentionBias->shapeOf() + attentionBias->rankOf());
    NDArray* biasCastBuf = workspace->getBuffer(
        "dpa_v2_biasCast", biasShapeVec, queries->dataType(), block.launchContext());
    biasCastBuf->assign(attentionBias);
    attentionBias = biasCastBuf;
  }

  // V2 QUANTIZED decode read: the INT8 caches were written above. Q, the current K/V window, the
  // bias and the output stay in the query dtype; only the cache rows are INT8, dequantized with
  // their FLOAT32 scales inside the kernel. This runs before the K/V auto-cast below so a window
  // cast can use persistent workspace buffers (same capture-safety rule as the bias cast above).
  if (useQuantisedKv && useInPlaceKv) {
    REQUIRE_TRUE(applyScoresOut->dataType() == queries->dataType(), 0,
                 "dot_product_attention_v2: the INT8 KV cache path writes the query dtype %i, got output dtype %i",
                 static_cast<int>(queries->dataType()), static_cast<int>(applyScoresOut->dataType()));
    auto* workspace = AttentionWorkspace::getInstance();
    auto windowInQueryType = [&](NDArray* window, const char* bufferKey) -> NDArray* {
      if (window->dataType() == queries->dataType()) return window;
      std::vector<sd::LongType> windowShape(window->shapeOf(), window->shapeOf() + window->rankOf());
      NDArray* castBuf = workspace->getBuffer(bufferKey, windowShape, queries->dataType(), block.launchContext());
      castBuf->assign(window);
      return castBuf;
    };
    NDArray* keyWindow = windowInQueryType(keys, "dpa_v2_quantKeyWindowCast");
    NDArray* valueWindow = windowInQueryType(values, "dpa_v2_quantValueWindowCast");
    NDArray* quantBias = hasAttentionBias ? attentionBias : nullptr;

#if defined(__CUDACC__) || defined(SD_CUDA)
    fusedGQADecodeQuantisedCuda(queries, keyCache, keyScaleCache, valueCache, valScaleCache, applyScoresOut, scale,
                                block.launchContext(), quantBias, keyWindow, valueWindow, currentKvPosition,
                                attentionScores, attentionLogits);
#else
    helpers::fusedGQADecodeQuantisedCpu(queries, keyCache, keyScaleCache, valueCache, valScaleCache, applyScoresOut,
                                        scale, quantBias, block.launchContext(), keyWindow, valueWindow,
                                        currentKvPosition, attentionScores, attentionLogits);
#endif

    // The rank-4 single-token requirement above rules out the rank-2 reshape and K/V casts, so
    // only the auto-promotion and sliced-bias allocations can be live here.
    if (promotedV) delete valuesOrig;
    if (promotedK) delete promotedKPtr;
    if (slicedBiasOwner != nullptr) delete slicedBiasOwner;
    return sd::Status::OK;
  }

  // Auto-cast K/V to match Q dtype when they differ (e.g. FusedRoPE promotes
  // Q/K to FLOAT while V stays HALF, or GraphOptimizer strips type casts).
  // This mirrors how MmulHelper handles mixed dtypes via pickPairwiseResultType.
  // Without this, CUDA kernels would reinterpret_cast the wrong dtype → silent corruption.
  NDArray* keysCastOwner = nullptr;
  NDArray* valuesCastOwner = nullptr;
  // Save pre-cast pointers for reshape cleanup (reshapedQ path).
  // After cast, `keys`/`values` will point to new allocations, but the
  // reshapedQ cleanup must still delete the original reshape results.
  NDArray* keysPreCast = keys;
  NDArray* valuesPreCast = values;
  if (keys->dataType() != queries->dataType()) {
    keysCastOwner = keys->cast(queries->dataType());
    keys = keysCastOwner;
  }
  if (values->dataType() != queries->dataType()) {
    valuesCastOwner = values->cast(queries->dataType());
    values = valuesCastOwner;
  }

  // Fast flash path: explicitly enabled + no masks + no dropout
  // The fused CUDA kernel now handles attention bias internally
  bool canUseFlashFast = useFlashAttention && !hasInputMasks && dropout == 0.0;

  if (canUseFlashFast) {
    // Materialize auxiliary arrays when demanded by the graph. The DSP executor supplies
    // shape-preserving empty placeholders for dead inference-only outputs; the fused helper
    // treats those placeholders as not requested and avoids quadratic storage.
    FlashAttentionHelper::forward(queries, keys, values, applyScoresOut, config,
                                  nullptr, attentionScores, attentionLogits,
                                  block.launchContext(), attentionBias);
  } else if (!hasInputMasks && dropout == 0.0) {
    // Non-flash or debug path: still use helper implementation so additive attention bias
    // remains supported. Auxiliary outputs are materialized when demanded.
    FlashAttentionHelper::forward(queries, keys, values, applyScoresOut, config,
                                  nullptr, attentionScores, attentionLogits,
                                  block.launchContext(), attentionBias);
  } else {
    REQUIRE_TRUE(attentionScores != nullptr && attentionLogits != nullptr, 0,
                 "dot_product_attention_v2: scores/logits outputs are required for masked, fallback, or dropout execution");
    REQUIRE_TRUE(!hasAttentionBias, 0,
                 "dot_product_attention_v2: additive attention bias with query/value masks or dropout is not "
                 "supported in this path yet");
    REQUIRE_TRUE(dropout >= 0.0 && dropout <= 1.0, 0,
                 "dot_product_attention_v2: the dropout probability must be in [0, 1], got %f", dropout);

    // Fallback to AttentionHelper for masks/dropout support.
    // AttentionHelper::doAttention expects 3D [batch*heads, seq, dim] format.
    // For rank-4 BSHD inputs, we must reshape to 3D and handle GQA (KV head expansion).

    // Dropout draws its mask from this op's own generator: one draw of it names the seed of the dropout op and the
    // generator advances, so every execution drops different weights (and the draws follow whatever state the executor
    // gave the context). block.randomSeed() is not used: nothing ever assigns it.
    int dropoutSeed = 0;
    if (dropout > 0.0 && training) {
      auto& rng = block.randomGenerator();
      dropoutSeed = rng.relativeInt(0) & 0x7fffffff;
      if (dropoutSeed == 0) dropoutSeed = 1;
      rng.rewindH(1);
    }

    std::vector<sd::NDArray*> inputs;
    // Note: mask nullification already done above for hasInputMasks check
    std::vector<sd::NDArray*> masks = {qMask, vMask};

    NDArray* q3d = nullptr;
    NDArray* k3d = nullptr;
    NDArray* v3d = nullptr;
    NDArray* qMaskPerHead = nullptr;
    NDArray* vMaskPerHead = nullptr;
    std::vector<sd::LongType> scoresShape3d;

    // Save 4D dimensions BEFORE any modifications (they may be corrupted by doAttention)
    sd::LongType batch4d = 0, seqQ4d = 0, numHeads4d = 0, headDim4d = 0, seqKV4d = 0;
    if (isRank4) {
      batch4d = queries->sizeAt(0);
      seqQ4d = queries->sizeAt(1);
      numHeads4d = queries->sizeAt(2);
      headDim4d = queries->sizeAt(3);
      seqKV4d = keys->sizeAt(1);
    }

    if (isRank4) {
      const sd::LongType numKvHeads = keys->sizeAt(2);
      REQUIRE_TRUE(numKvHeads > 0 && numHeads4d % numKvHeads == 0 && keys->rankOf() == 4 && values->rankOf() == 4 &&
                       values->sizeAt(1) == seqKV4d && values->sizeAt(2) == numKvHeads &&
                       keys->sizeAt(3) == headDim4d && values->sizeAt(3) == headDim4d,
                   0,
                   "dot_product_attention_v2: a rank-4 attention with masks or dropout needs keys and values "
                   "[batch, seqKV, kvHeads, headDim] with the query head count a multiple of kvHeads and the same "
                   "headDim as the queries (%lld heads, %lld head dim); got keys rank %i and values rank %i",
                   static_cast<long long>(numHeads4d), static_cast<long long>(headDim4d), keys->rankOf(),
                   values->rankOf());
      const sd::LongType group = numHeads4d / numKvHeads;

      // Head-major [batch*heads, seq, dim] copies; KV heads are repeated `group` times for grouped-query attention.
      // The backward builds exactly the same layout through the same helpers.
      q3d = toHeadMajor3d(queries, 1, block.launchContext());
      k3d = toHeadMajor3d(keys, group, block.launchContext());
      v3d = toHeadMajor3d(values, group, block.launchContext());
      inputs = {q3d, v3d, k3d};

      // A [batch, length] mask has one row per batch but the head-major attention has one per (batch, head).
      qMaskPerHead = expandMaskPerHead(qMask, batch4d, numHeads4d, seqQ4d, queries->dataType(), block.launchContext());
      vMaskPerHead = expandMaskPerHead(vMask, batch4d, numHeads4d, seqKV4d, queries->dataType(), block.launchContext());
      masks = {qMaskPerHead != nullptr ? qMaskPerHead : qMask, vMaskPerHead != nullptr ? vMaskPerHead : vMask};

      scoresShape3d = {batch4d * numHeads4d, seqQ4d, seqKV4d};

      // Reshape output tensors to 3D for doAttention
      applyScoresOut->reshapei({batch4d * numHeads4d, seqQ4d, headDim4d});
      attentionLogits->reshapei(scoresShape3d);
      attentionScores->reshapei(scoresShape3d);
      if (dropoutMask != nullptr) {
        dropoutMask->reshapei(scoresShape3d);
      }
    } else {
      inputs = {queries, values, keys};
    }

    AttentionHelper::doAttention(inputs, masks, training, useCausalMask, dropout, scale, attentionScores,
                                 dropoutSeed, applyScoresOut, attentionLogits, dropoutMask);

    // Restore 4D shapes after doAttention (use saved dimensions, not from arrays)
    if (isRank4) {
      // The weights, logits and dropout mask are head-major [batch*heads, seqQ, seqKV]: row b * heads + h, so
      // [batch, heads, seqQ, seqKV] in memory.
      attentionLogits->reshapei({batch4d, numHeads4d, seqQ4d, seqKV4d});
      attentionScores->reshapei({batch4d, numHeads4d, seqQ4d, seqKV4d});
      if (dropoutMask != nullptr) {
        dropoutMask->reshapei({batch4d, numHeads4d, seqQ4d, seqKV4d});
      }

      // The attention output is head-major as well: [batch*heads, seqQ, headDim] is [batch, heads, seqQ, headDim] in
      // memory. View it as that (the output still carries its 3D shape), permute it back to BSHD
      // [batch, seqQ, heads, headDim], and write it to the output with its BSHD shape restored.
      // outPerm is a strided VIEW over applyScoresOut's own buffer — assigning it
      // directly back is an in-place transpose (aliased read+write through different
      // index maps) and races on CUDA: nondeterministic corruption/NaN per run.
      // Materialize the permuted order into a fresh buffer first.
      std::vector<sd::LongType> headMajorShape = {batch4d, numHeads4d, seqQ4d, headDim4d};
      applyScoresOut->reshapei(headMajorShape);
      std::vector<sd::LongType> permBack = {0, 2, 1, 3};
      auto outPerm = applyScoresOut->permute(permBack, false, false);
      auto outPermDup = outPerm->dup('c');
      delete outPerm;
      std::vector<sd::LongType> bshdShape = {batch4d, seqQ4d, numHeads4d, headDim4d};
      applyScoresOut->reshapei(bshdShape);
      applyScoresOut->assign(outPermDup);

      // The kernels doAttention enqueued (QK matmul, softmax, PV matmul, and the
      // assign above) read q3d/k3d/v3d/outPermDup ASYNCHRONOUSLY. Deleting these
      // temps before those reads complete frees their DataBuffers (cudaFreeAsync on
      // the DSP free-stream) → the pool recycles the blocks and later ops overwrite
      // them mid-kernel → nondeterministic garbage/NaN attention rows (BGE NaN root).
      // Flush the exec streams first; capture-aware (no-op during capture, where the
      // recorded kernels have not run yet; no-op on CPU). q3d covers the QK/PV GEMM
      // stream (input-lineage context), applyScoresOut covers the output assign.
      q3d->synchronizeExecStream("dpa_v2 BSHD temp free (inputs)");
      applyScoresOut->synchronizeExecStream("dpa_v2 BSHD temp free (output)");
      delete outPermDup;

      // Cleanup temporary arrays — the head-major copies and the per-head masks are new NDArray objects
      delete q3d;
      delete k3d;
      delete v3d;
      delete qMaskPerHead;  // nullptr when the mask needed no expansion
      delete vMaskPerHead;
    }
  }

  // Cleanup reshaped arrays and restore output shapes.
  // Use pre-cast pointers: if K/V were auto-cast, `keys`/`values` now point to
  // the cast copies (freed below as keysCastOwner/valuesCastOwner), while
  // keysPreCast/valuesPreCast still hold the reshape results that must be freed here.
  if(reshapedQ) {
    delete queries;
    delete valuesPreCast;
    if(block.width() > 2) {
      delete keysPreCast;
    }
    if(qMaskOrig != nullptr) {
      delete qMask;
    }
    if(vMaskOrig != nullptr) {
      delete vMask;
    }

    // Restore original shapes for outputs
    applyScoresOut->reshapei('c', {applyScoresOut->sizeAt(1), applyScoresOut->sizeAt(2)});
    attentionLogits->reshapei('c', {attentionLogits->sizeAt(1), attentionLogits->sizeAt(2)});
    attentionScores->reshapei('c', {attentionScores->sizeAt(1), attentionScores->sizeAt(2)});
  }

  // Cleanup auto-promoted V/K arrays.
  // IMPORTANT: Use promotedKPtr, NOT `keys`. The `keys` pointer may have been
  // overwritten by `keys = keyCache` in the in-place KV-cache path, which would
  // cause us to delete a live INPUT_VARIABLE and corrupt subsequent executions.
  if (promotedV) delete valuesOrig;
  if (promotedK) delete promotedKPtr;
  if (slicedBiasOwner != nullptr) delete slicedBiasOwner;
  delete keysCastOwner;
  delete valuesCastOwner;

  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention_v2) {
  getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
  // Cache-form attention mutates K/V inputs even when no cache is requested
  // as an output. Prefill/bias-only forms do not activate these write groups.
  getOpDescriptor()->addInputWrites({5, 6}, {{5, 2}, {6, 2}, {7, 0}});
  getOpDescriptor()->addInputWrites({9, 10}, {{5, 2}, {6, 2}, {7, 0}}, 5, DataType::INT8);
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})                  // queries
      ->setAllowedInputTypes(1, {ALL_FLOATS})                  // values
      ->setAllowedInputTypes(2, {ALL_FLOATS})                  // keys
      ->setAllowedInputTypes(3, {ALL_FLOATS, ALL_INTS, BOOL})  // queryMask (optional)
      ->setAllowedInputTypes(4, {ALL_FLOATS, ALL_INTS, BOOL})  // valueMask (optional)
      ->setAllowedInputTypes(5, {ALL_FLOATS, ALL_INTS, BOOL, INT8})  // attentionBias/keyCache (V2: INT8)
      ->setAllowedInputTypes(6, {ALL_FLOATS, ALL_INTS, BOOL, INT8})  // valueCache (V2: INT8)
      ->setAllowedInputTypes(7, {DataType::INT64})             // cache_position (optional); kernels read int64
      ->setAllowedInputTypes(8, {ALL_FLOATS, ALL_INTS, BOOL})  // attention bias with KV cache (optional)
      ->setAllowedInputTypes(9, {DataType::FLOAT32})           // V2: separate key scale cache (optional)
      ->setAllowedInputTypes(10, {DataType::FLOAT32})          // V2: separate value scale cache (optional)
      ->setAllowedOutputTypes({ALL_FLOATS})
      ;
}

DECLARE_SHAPE_FN(dot_product_attention_v2) {
  auto queriesType = ArrayOptions::dataType(inputShape->at(0));
  // Output/scores dtype MUST equal the QUERIES dtype. The runtime (CUSTOM_OP_IMPL) auto-casts
  // K and V to the queries dtype — NOT to the widest of Q/K/V — so the fused CUDA kernels and
  // the cuBLAS/matmul fallback both compute in and write queries-typed results. Allocating the
  // output as the widest type (e.g. DOUBLE when Q=FLOAT but K/V default to DOUBLE) made a FLOAT
  // kernel write into a DOUBLE buffer: the 4-byte writes land in the first half of each 8-byte
  // slot, so the result reads back as reinterpreted garbage/denormals for the first half of the
  // elements and untouched zeros for the rest (manifested as an all-near-zero attention output).
  // Keep this in lockstep with the K/V auto-cast in CUSTOM_OP_IMPL.
  auto firstInputType = queriesType;
  auto queriesShape = inputShape->at(0);
  auto valuesShape = inputShape->at(1);
  auto keysShape = block.width() > 2 ? inputShape->at(2) : valuesShape;

  auto dropout = block.numT() > 1 ? block.getTArguments()->at(1) : 0.0;

  // Check for in-place KV cache mode: when cache_position (input 7) is present
  // with keyCache (input 5) and valueCache (input 6), Tv = cache seq dim.
  // Guard with rankOf() >= 2: empty/rank-0 arrays may lose the EMPTY flag through
  // DSP slot wiring, so isEmpty() alone is insufficient.
  auto input5Shape = block.width() > 5 ? inputShape->at(5) : nullptr;
  bool input5Valid = (input5Shape != nullptr && !shape::isEmpty(input5Shape) && shape::rank(input5Shape) >= 2);
  bool hasInPlaceKv = (block.width() > 7 && input5Valid);
  auto keyCacheShapePtr = input5Valid ? input5Shape : nullptr;

  // For rank 4: [batch, seq_len, numHeads, headDim] (BSHD)
  // For rank 3: [batch, seq_len, features]
  // For rank 2: [seq_len, features] - treated as batch=1
  std::vector<sd::LongType> outShape;
  std::vector<sd::LongType> scoresShape;

  if(shape::rank(queriesShape) == 4) {
    // Rank 4: [batch, Tq, numHeads, headDim] (BSHD format)
    sd::LongType batchSize = shape::sizeAt(queriesShape, static_cast<sd::LongType>(0));
    sd::LongType tq = shape::sizeAt(queriesShape, static_cast<sd::LongType>(1));
    sd::LongType numHeads = shape::sizeAt(queriesShape, static_cast<sd::LongType>(2));
    sd::LongType headDim = shape::sizeAt(queriesShape, static_cast<sd::LongType>(3));
    sd::LongType tv = (hasInPlaceKv && keyCacheShapePtr != nullptr)
                       ? shape::sizeAt(keyCacheShapePtr, static_cast<sd::LongType>(1))
                       : shape::sizeAt(valuesShape, static_cast<sd::LongType>(1));

    // Output shape: [batch, Tq, numHeads, headDim] (same as query)
    outShape = {batchSize, tq, numHeads, headDim};
    // Attention scores shape: [batch, numHeads, Tq, Tv] (per-head scores)
    scoresShape = {batchSize, numHeads, tq, tv};
  } else if(shape::rank(queriesShape) == 3) {
    sd::LongType batchSize = shape::sizeAt(queriesShape, static_cast<sd::LongType>(0));
    sd::LongType tq = shape::sizeAt(queriesShape, static_cast<sd::LongType>(1));
    sd::LongType tv = (hasInPlaceKv && keyCacheShapePtr != nullptr)
                       ? shape::sizeAt(keyCacheShapePtr, static_cast<sd::LongType>(1))
                       : shape::sizeAt(valuesShape, static_cast<sd::LongType>(1));
    sd::LongType dim = shape::sizeAt(valuesShape, static_cast<sd::LongType>(2));

    outShape = {batchSize, tq, dim};
    scoresShape = {batchSize, tq, tv};
  } else {
    sd::LongType tq = shape::sizeAt(queriesShape, static_cast<sd::LongType>(0));
    sd::LongType tv = (hasInPlaceKv && keyCacheShapePtr != nullptr)
                       ? shape::sizeAt(keyCacheShapePtr, static_cast<sd::LongType>(0))
                       : shape::sizeAt(valuesShape, static_cast<sd::LongType>(0));
    sd::LongType dim = shape::sizeAt(valuesShape, static_cast<sd::LongType>(1));

    outShape = {tq, dim};
    scoresShape = {tq, tv};
  }

  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(firstInputType, 'c', outShape);
  auto attentionScoresShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(firstInputType, 'c', scoresShape);
  auto attentionLogitsShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(firstInputType, 'c', scoresShape);

  if(dropout > 0) {
    auto dropoutMaskShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(firstInputType, 'c', scoresShape);
    return SHAPELIST(outputShapeInfo, attentionScoresShapeInfo, attentionLogitsShapeInfo, dropoutMaskShapeInfo);
  } else {
    return SHAPELIST(outputShapeInfo, attentionScoresShapeInfo, attentionLogitsShapeInfo);
  }
}

CUSTOM_OP_IMPL(dot_product_attention_v2_bp, -2, 3, false, 0, -2) {
  auto queriesOrig = INPUT_VARIABLE(0);
  auto valuesOrig = INPUT_VARIABLE(1);
  auto keysOrig = INPUT_VARIABLE(2);

  // Track reshaped arrays for cleanup
  NDArray* queries = nullptr;
  NDArray* values = nullptr;
  NDArray* keys = nullptr;
  NDArray* qMask = nullptr;
  NDArray* vMask = nullptr;
  bool reshapedQ = false;

  // Handle rank 2 inputs by adding batch dimension
  if(queriesOrig->rankOf() == 2) {
    reshapedQ = true;
    std::vector<sd::LongType> qShape = {1, queriesOrig->sizeAt(0), queriesOrig->sizeAt(1)};
    std::vector<sd::LongType> vShape = {1, valuesOrig->sizeAt(0), valuesOrig->sizeAt(1)};
    std::vector<sd::LongType> kShape = {1, keysOrig->sizeAt(0), keysOrig->sizeAt(1)};
    queries = queriesOrig->reshape('c', qShape);
    values = valuesOrig->reshape('c', vShape);
    keys = keysOrig->reshape('c', kShape);
  } else {
    queries = queriesOrig;
    values = valuesOrig;
    keys = keysOrig;
  }

  auto attentionScoresOut = INPUT_VARIABLE(3);
  auto attentionScoresWeights = INPUT_VARIABLE(4);
  auto attentionScoreLogits = INPUT_VARIABLE(5);

  if(reshapedQ) {
    attentionScoresOut->reshapei('c', {1, attentionScoresOut->sizeAt(0), attentionScoresOut->sizeAt(1)});
    attentionScoreLogits->reshapei('c', {1, attentionScoreLogits->sizeAt(0), attentionScoreLogits->sizeAt(1)});
    attentionScoresWeights->reshapei('c', {1, attentionScoresWeights->sizeAt(0), attentionScoresWeights->sizeAt(1)});
  }

  auto eps = INPUT_VARIABLE(6);
  if(reshapedQ) {
    eps->reshapei('c', {1, eps->sizeAt(0), eps->sizeAt(1)});
  }

  // Handle dropout mask - check for empty array
  auto dropoutMaskOrig = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;
  NDArray* dropoutMask = nullptr;
  if(dropoutMaskOrig != nullptr && !dropoutMaskOrig->isEmpty()) {
    dropoutMask = dropoutMaskOrig;
  }

  // Handle masks - check for empty arrays
  auto qMaskOrig = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;
  auto vMaskOrig = block.width() > 9 ? INPUT_VARIABLE(9) : nullptr;

  // Treat empty or rank-0 arrays as no mask, exactly as the forward op does: SameDiff and the DSP wiring can hand an
  // absent optional input over as a rank-0 placeholder, and a rank-0 value mask would otherwise be broadcast over the
  // whole gradient (all of dQ and dK zero for a placeholder of 0).
  if(qMaskOrig != nullptr && (qMaskOrig->isEmpty() || qMaskOrig->rankOf() == 0)) {
    qMaskOrig = nullptr;
  }
  if(vMaskOrig != nullptr && (vMaskOrig->isEmpty() || vMaskOrig->rankOf() == 0)) {
    vMaskOrig = nullptr;
  }

  // The masks are prepared exactly as the forward prepares them before AttentionHelper expands them (the same way in
  // both directions, AttentionHelper::doAttention and doAttentionBp): unbatched inputs get a leading batch dimension.
  if(qMaskOrig != nullptr && reshapedQ) {
    std::vector<sd::LongType> qmShape = {1, qMaskOrig->sizeAt(0), qMaskOrig->sizeAt(1)};
    qMask = qMaskOrig->reshape('c', qmShape);
  } else {
    qMask = qMaskOrig;
  }

  if(vMaskOrig != nullptr && reshapedQ) {
    std::vector<sd::LongType> vmShape = {1, vMaskOrig->sizeAt(0), vMaskOrig->sizeAt(1)};
    vMask = vMaskOrig->reshape('c', vmShape);
  } else {
    vMask = vMaskOrig;
  }

  auto dLdq = OUTPUT_VARIABLE(0);
  auto dLdv = OUTPUT_VARIABLE(1);
  auto dLdk = OUTPUT_VARIABLE(2);

  if(reshapedQ) {
    dLdq->reshapei('c', {1, dLdq->sizeAt(0), dLdq->sizeAt(1)});
    dLdv->reshapei('c', {1, dLdv->sizeAt(0), dLdv->sizeAt(1)});
    dLdk->reshapei('c', {1, dLdk->sizeAt(0), dLdk->sizeAt(1)});
  }

  // Get arguments - T_ARG order: scale, dropout (same as forward pass)
  auto scale = block.numT() > 0 ? T_ARG(0) : 1.0;
  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

  // Mirror the forward's auto-scale logic: when scale <= 0, compute 1/sqrt(headDim).
  // The forward stores the original T_ARG (may be 0) without updating it, so the backward
  // must replicate the same auto-scale resolution to apply the correct chain-rule factor.
  if (scale <= 0.0) {
    auto dim = queries->rankOf() == 4 ? queries->sizeAt(3) : queries->sizeAt(2);
    scale = 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(dim));
  }

  // B_ARG order: useCausalMask, training, useFlashAttention (third arg is forward-only)
  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;

  int seed = block.randomSeed();

  // The backward differentiates what the forward recorded: its weights (input 4, the softmax output the result was
  // formed from, after dropout) and logits (input 5). Those carry every mask, the causal mask, dropout and an additive
  // bias alike, so AttentionHelper's backward is right whichever path the forward took: rank 4 runs it on the
  // head-major [batch * heads, seq, dim] layout the forward's helper path uses, through the same layout helpers. Only
  // when the forward's weights were not materialized does the rank-4 backward recompute them the flash way, which
  // knows no masks or dropout (or bias).
  const bool haveForwardWeights = attentionScoresWeights != nullptr && !attentionScoresWeights->isEmpty() &&
                                  attentionScoreLogits != nullptr && !attentionScoreLogits->isEmpty();
  if (queries->rankOf() == 4 && !haveForwardWeights) {
    REQUIRE_TRUE(qMask == nullptr && vMask == nullptr && dropout == 0.0, 0,
                 "dot_product_attention_v2_bp: query/value masks or dropout need the forward's attention weights and "
                 "logits (outputs 1 and 2)");
    FlashAttentionHelper::Config config;
    config.scale = static_cast<float>(scale);
    config.isCausal = useCausalMask;
    config.dropout = 0.0f;
    config.numHeads = static_cast<int>(queries->sizeAt(2));
    config.numKvHeads = static_cast<int>(keys->sizeAt(2));
    FlashAttentionHelper::backward(eps, queries, keys, values,
                                   attentionScoresOut, nullptr,
                                   dLdq, dLdk, dLdv,
                                   config, block.launchContext());
  } else if (queries->rankOf() == 4) {
    const sd::LongType batch = queries->sizeAt(0);
    const sd::LongType seqQ = queries->sizeAt(1);
    const sd::LongType heads = queries->sizeAt(2);
    const sd::LongType headDim = queries->sizeAt(3);
    const sd::LongType seqKV = keys->sizeAt(1);
    const sd::LongType kvHeads = keys->sizeAt(2);
    REQUIRE_TRUE(kvHeads > 0 && heads % kvHeads == 0, 0,
                 "dot_product_attention_v2_bp: the query head count %lld must be a multiple of the KV head count %lld",
                 static_cast<long long>(heads), static_cast<long long>(kvHeads));
    const sd::LongType group = heads / kvHeads;
    auto context = block.launchContext();

    NDArray* q3d = toHeadMajor3d(queries, 1, context);
    NDArray* k3d = toHeadMajor3d(keys, group, context);
    NDArray* v3d = toHeadMajor3d(values, group, context);
    NDArray* eps3d = toHeadMajor3d(eps, 1, context);

    // The weights, logits and dropout multiplier are head-major [batch, heads, seqQ, seqKV] already.
    std::vector<sd::LongType> scores3d = {batch * heads, seqQ, seqKV};
    auto asScores3d = [&](NDArray* scores) -> NDArray* {
      if (scores == nullptr) return nullptr;
      NDArray* view = scores->reshape('c', scores3d, false);
      if (view == nullptr) {
        view = scores->dup('c');
        view->reshapei('c', scores3d);
      }
      return view;
    };
    NDArray* weights3d = asScores3d(attentionScoresWeights);
    NDArray* logits3d = asScores3d(attentionScoreLogits);
    NDArray* dropout3d = asScores3d(dropoutMask);

    NDArray* qMaskPerHead = expandMaskPerHead(qMask, batch, heads, seqQ, queries->dataType(), context);
    NDArray* vMaskPerHead = expandMaskPerHead(vMask, batch, heads, seqKV, queries->dataType(), context);

    std::vector<sd::LongType> qShape3d = {batch * heads, seqQ, headDim};
    std::vector<sd::LongType> kvShape3d = {batch * heads, seqKV, headDim};
    NDArray dLdq3d('c', qShape3d, dLdq->dataType(), context);
    NDArray dLdk3d('c', kvShape3d, dLdk->dataType(), context);
    NDArray dLdv3d('c', kvShape3d, dLdv->dataType(), context);

    // attentionScoresOut is passed through untouched: the backward does not read the forward's result.
    std::vector<NDArray*> inputs3d = {q3d, v3d, k3d, attentionScoresOut, weights3d, logits3d, eps3d, dropout3d};
    std::vector<NDArray*> masks3d = {qMaskPerHead != nullptr ? qMaskPerHead : qMask,
                                     vMaskPerHead != nullptr ? vMaskPerHead : vMask};
    AttentionHelper::doAttentionBp(inputs3d, masks3d, training, useCausalMask, dropout, scale,
                                   {&dLdq3d, &dLdv3d, &dLdk3d}, seed);

    // Back to BSHD; a KV head's gradient sums the copies its `group` query heads used. fromHeadMajor3d fences the
    // streams before it returns, so the temporaries can go.
    fromHeadMajor3d(&dLdq3d, dLdq, 1, context);
    fromHeadMajor3d(&dLdk3d, dLdk, group, context);
    fromHeadMajor3d(&dLdv3d, dLdv, group, context);
    dLdq->synchronizeExecStream("dpa_v2_bp head-major temp free");
    delete q3d;
    delete k3d;
    delete v3d;
    delete eps3d;
    delete weights3d;
    delete logits3d;
    delete dropout3d;
    delete qMaskPerHead;
    delete vMaskPerHead;
  } else {
    std::vector<NDArray*> inputs3d = {queries, values, keys, attentionScoresOut, attentionScoresWeights,
                                      attentionScoreLogits, eps, dropoutMask};
    std::vector<NDArray*> masks = {qMask, vMask};
    AttentionHelper::doAttentionBp(inputs3d, masks, training, useCausalMask, dropout, scale, {dLdq, dLdv, dLdk}, seed);
  }

  // Cleanup and restore shapes
  if(reshapedQ) {
    delete queries;
    delete values;
    delete keys;
    if(qMaskOrig != nullptr && qMask != qMaskOrig) {
      delete qMask;
    }
    if(vMaskOrig != nullptr && vMask != vMaskOrig) {
      delete vMask;
    }

    dLdq->reshapei('c', {dLdq->sizeAt(1), dLdq->sizeAt(2)});
    dLdv->reshapei('c', {dLdv->sizeAt(1), dLdv->sizeAt(2)});
    dLdk->reshapei('c', {dLdk->sizeAt(1), dLdk->sizeAt(2)});
    eps->reshapei('c', {eps->sizeAt(1), eps->sizeAt(2)});

    // Restore attention tensors shapes
    attentionScoresOut->reshapei('c', {attentionScoresOut->sizeAt(1), attentionScoresOut->sizeAt(2)});
    attentionScoreLogits->reshapei('c', {attentionScoreLogits->sizeAt(1), attentionScoreLogits->sizeAt(2)});
    attentionScoresWeights->reshapei('c', {attentionScoresWeights->sizeAt(1), attentionScoresWeights->sizeAt(2)});
  }

  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention_v2_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  // The masks may be boolean or integer, as the forward's masks may: they are cast where they are used.
  getOpDescriptor()->setAllowedInputTypes(8, {ALL_FLOATS, ALL_INTS, BOOL});  // queryMask (optional)
  getOpDescriptor()->setAllowedInputTypes(9, {ALL_FLOATS, ALL_INTS, BOOL});  // valueMask (optional)
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

DECLARE_SHAPE_FN(dot_product_attention_v2_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)), CONSTANT(inputShape->at(2)));
}

}  // namespace ops
}  // namespace sd

#endif
