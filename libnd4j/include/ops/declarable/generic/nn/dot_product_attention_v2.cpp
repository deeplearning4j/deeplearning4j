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

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/reverse.h>
#include <helpers/AttentionHelper.h>
#include <helpers/FlashAttentionHelper.h>
#include <cmath>
#include <memory>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dot_product_attention_v2, -2, -1, false, -2, -2) {
  auto queriesOrig = INPUT_VARIABLE(0);
  auto valuesOrig = INPUT_VARIABLE(1);

  REQUIRE_TRUE(queriesOrig->rankOf() == valuesOrig->rankOf(), 0,
               "dot_product_attention_v2: Queries and values must have same rank, got %i vs %i",
               queriesOrig->rankOf(), valuesOrig->rankOf());
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
  if (cachePosInput != nullptr && cachePosInput->isEmpty()) cachePosInput = nullptr;

  bool useInPlaceKv = false;
  NDArray* keyCache = nullptr;
  NDArray* valueCache = nullptr;

  if (extraInput != nullptr && !extraInput->isEmpty() &&
      extraInput2 != nullptr && !extraInput2->isEmpty()) {
    keyCache = extraInput;
    valueCache = extraInput2;
    useInPlaceKv = (cachePosInput != nullptr);
  }

  if (useInPlaceKv) {
    LongType cachePosVal = cachePosInput->e<LongType>(0);
    LongType maxSeq = keyCache->sizeAt(isRank4 ? 1 : 1);  // seq dim is always 1 for both rank 3 and 4

    REQUIRE_TRUE(cachePosVal >= 0 && cachePosVal < maxSeq, 0,
                 "dot_product_attention_v2: cache_position %lld must be in [0, %lld)",
                 (long long)cachePosVal, (long long)maxSeq);

    // Write current K/V at cache_position in the buffers (in-place)
    if (isRank4) {
      // BSHD: keyCache[batch, maxSeq, heads, dim], keys[batch, 1, heads, dim]
      auto batch = keys->sizeAt(0);
      auto numKvHeads = keys->sizeAt(2);
      auto headDim = keys->sizeAt(3);
      std::vector<LongType> writeIdx = {0, batch, cachePosVal, cachePosVal + 1, 0, numKvHeads, 0, headDim};
      auto* kSlice = (*keyCache)(writeIdx);
      auto* vSlice = (*valueCache)(writeIdx);
      kSlice->assign(keys);
      vSlice->assign(values);
      kSlice->syncToDevice();
      vSlice->syncToDevice();
      delete kSlice;
      delete vSlice;
    } else {
      // BSF: keyCache[batch, maxSeq, features], keys[batch, 1, features]
      auto batch = keys->sizeAt(0);
      auto features = keys->sizeAt(2);
      std::vector<LongType> writeIdx = {0, batch, cachePosVal, cachePosVal + 1, 0, features};
      auto* kSlice = (*keyCache)(writeIdx);
      auto* vSlice = (*valueCache)(writeIdx);
      kSlice->assign(keys);
      vSlice->assign(values);
      kSlice->syncToDevice();
      vSlice->syncToDevice();
      delete kSlice;
      delete vSlice;
    }

    // Use full cache buffers as K/V for attention
    keys = keyCache;
    values = valueCache;
  }

  // Get arguments - T_ARG order: scale, dropout
  auto scale = block.numT() > 0 ? T_ARG(0) : 1.0;
  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

  // Auto scale when scale <= 0: 1/sqrt(headDim or dim)
  if (scale <= 0.0) {
    auto dim = isRank4 ? queries->sizeAt(3) : queries->sizeAt(2);
    scale = 1.0 / std::sqrt(static_cast<double>(dim));
  }

  // B_ARG order: useCausalMask, training, useFlashAttention
  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;
  auto useFlashAttention = block.numB() > 2 ? B_ARG(2) : true;

  // Get output variables
  auto applyScoresOut = OUTPUT_VARIABLE(0);
  auto attentionScores = OUTPUT_VARIABLE(1);
  auto attentionLogits = OUTPUT_VARIABLE(2);
  auto dropoutMask = dropout > 0.0 ? OUTPUT_VARIABLE(3) : nullptr;

  // Reshape outputs for rank 2 case
  if(reshapedQ) {
    applyScoresOut->reshapei('c', {1, applyScoresOut->sizeAt(0), applyScoresOut->sizeAt(1)});
    attentionLogits->reshapei('c', {1, attentionLogits->sizeAt(0), attentionLogits->sizeAt(1)});
    attentionScores->reshapei('c', {1, attentionScores->sizeAt(0), attentionScores->sizeAt(1)});
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
  std::unique_ptr<NDArray> attentionBiasCastOwner;

  // Additive bias/mask can arrive as BOOL/INT from importer graphs.
  // Cast once to query dtype for arithmetic in the helper path.
  if (hasAttentionBias && attentionBias->dataType() != queries->dataType()) {
    attentionBiasCastOwner.reset(attentionBias->cast(queries->dataType()));
    attentionBias = attentionBiasCastOwner.get();
  }

  // Fast flash path: explicitly enabled + no masks + no dropout
  // The fused CUDA kernel now handles attention bias internally
  bool canUseFlashFast = useFlashAttention && !hasInputMasks && dropout == 0.0;

  if (canUseFlashFast) {
    // Pass nullptr for scores/logits to enable the fastest fused CUDA kernel path.
    // The fused kernel handles attention bias internally - no fallback needed.
    FlashAttentionHelper::forward(queries, keys, values, applyScoresOut, config,
                                  nullptr, nullptr, nullptr,
                                  block.launchContext(), attentionBias);
  } else if (!hasInputMasks && dropout == 0.0) {
    // Non-flash or debug path: still use helper implementation so additive attention bias
    // remains supported, and materialize scores/logits outputs for diagnostics.
    FlashAttentionHelper::forward(queries, keys, values, applyScoresOut, config,
                                  nullptr, attentionScores, attentionLogits,
                                  block.launchContext(), attentionBias);
  } else {
    REQUIRE_TRUE(!hasAttentionBias, 0,
                 "dot_product_attention_v2: additive attention bias with query/value masks or dropout is not "
                 "supported in this path yet");
    // Fallback to AttentionHelper for masks/dropout support.
    // AttentionHelper::doAttention expects 3D [batch*heads, seq, dim] format.
    // For rank-4 BSHD inputs, we must reshape to 3D and handle GQA (KV head expansion).
    std::vector<sd::NDArray*> inputs;
    // Note: mask nullification already done above for hasInputMasks check
    std::vector<sd::NDArray*> masks = {qMask, vMask};

    NDArray* q3d = nullptr;
    NDArray* k3d = nullptr;
    NDArray* v3d = nullptr;
    NDArray* qPerm = nullptr;
    NDArray* kPerm = nullptr;
    NDArray* vPerm = nullptr;
    NDArray* kExpanded = nullptr;
    NDArray* vExpanded = nullptr;
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
      auto numKvHeads = keys->sizeAt(2);
      int headsPerKv = numHeads4d / numKvHeads;

      // Permute Q from BSHD [batch, seq, heads, dim] to BHSD [batch, heads, seq, dim]
      std::vector<sd::LongType> permOrder = {0, 2, 1, 3};
      qPerm = queries->permute(permOrder, false, false);
      kPerm = keys->permute(permOrder, false, false);
      vPerm = values->permute(permOrder, false, false);

      // Reshape Q to 3D: [batch*heads, seq, dim]
      std::vector<sd::LongType> qShape3d = {batch4d * numHeads4d, seqQ4d, headDim4d};
      q3d = qPerm->reshape('c', qShape3d);

      // Handle GQA: expand KV heads if needed
      k3d = kPerm;
      v3d = vPerm;
      if (headsPerKv > 1) {
        // Tile KV heads: [batch, numKvHeads, seq, dim] -> [batch, numHeads, seq, dim]
        std::vector<sd::LongType> tiledShape = {batch4d, numKvHeads, static_cast<sd::LongType>(headsPerKv), seqKV4d, headDim4d};
        NDArray* kTiled = new NDArray('c', tiledShape, keys->dataType(), block.launchContext());
        NDArray* vTiled = new NDArray('c', tiledShape, values->dataType(), block.launchContext());

        std::vector<sd::LongType> reshapeForTile = {batch4d, numKvHeads, 1, seqKV4d, headDim4d};
        kPerm->reshapei(reshapeForTile);
        vPerm->reshapei(reshapeForTile);

        std::vector<sd::LongType> reps = {1, 1, static_cast<sd::LongType>(headsPerKv), 1, 1};
        kPerm->tile(reps, *kTiled);
        vPerm->tile(reps, *vTiled);

        std::vector<sd::LongType> expandedShape = {batch4d, numHeads4d, seqKV4d, headDim4d};
        kTiled->reshapei(expandedShape);
        vTiled->reshapei(expandedShape);

        kExpanded = kTiled;
        vExpanded = vTiled;

        // Restore kPerm/vPerm shapes
        kPerm->reshapei({batch4d, numKvHeads, seqKV4d, headDim4d});
        vPerm->reshapei({batch4d, numKvHeads, seqKV4d, headDim4d});

        // Reshape expanded KV to 3D: [batch*heads, seq, dim]
        std::vector<sd::LongType> kvShape3d = {batch4d * numHeads4d, seqKV4d, headDim4d};
        k3d = kExpanded->reshape('c', kvShape3d);
        v3d = vExpanded->reshape('c', kvShape3d);
      } else {
        std::vector<sd::LongType> kvShape3d = {batch4d * numHeads4d, seqKV4d, headDim4d};
        k3d = kPerm->reshape('c', kvShape3d);
        v3d = vPerm->reshape('c', kvShape3d);
      }

      inputs = {q3d, v3d, k3d};
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
                                 block.randomSeed(), applyScoresOut, attentionLogits, dropoutMask);

    // Restore 4D shapes after doAttention (use saved dimensions, not from arrays)
    if (isRank4) {
      // Restore output shapes to 4D BSHD
      applyScoresOut->reshapei({batch4d, seqQ4d, numHeads4d, headDim4d});
      attentionLogits->reshapei({batch4d, numHeads4d, seqQ4d, seqKV4d});
      attentionScores->reshapei({batch4d, numHeads4d, seqQ4d, seqKV4d});
      if (dropoutMask != nullptr) {
        dropoutMask->reshapei({batch4d, numHeads4d, seqQ4d, seqKV4d});
      }

      // Permute applyScoresOut from BHSD back to BSHD
      std::vector<sd::LongType> permBack = {0, 2, 1, 3};
      auto outPerm = applyScoresOut->permute(permBack, false, false);
      applyScoresOut->assign(outPerm);
      delete outPerm;

      // Cleanup temporary arrays — reshape() creates new NDArray objects that must be freed
      delete q3d;   // reshape of qPerm
      delete k3d;   // reshape of kExpanded (GQA) or kPerm (non-GQA)
      delete v3d;   // reshape of vExpanded (GQA) or vPerm (non-GQA)
      delete qPerm; // permute of queries
      delete kPerm; // permute of keys
      delete vPerm; // permute of values
      delete kExpanded;  // nullptr when non-GQA
      delete vExpanded;  // nullptr when non-GQA
    }
  }

  // Cleanup reshaped arrays and restore output shapes
  if(reshapedQ) {
    delete queries;
    delete values;
    if(block.width() > 2) {
      delete keys;
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

  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention_v2) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})                  // queries
      ->setAllowedInputTypes(1, {ALL_FLOATS})                  // values
      ->setAllowedInputTypes(2, {ALL_FLOATS})                  // keys
      ->setAllowedInputTypes(3, {ALL_FLOATS, ALL_INTS, BOOL})  // queryMask (optional)
      ->setAllowedInputTypes(4, {ALL_FLOATS, ALL_INTS, BOOL})  // valueMask (optional)
      ->setAllowedInputTypes(5, {ALL_FLOATS, ALL_INTS, BOOL})  // attentionBias/keyCache (optional)
      ->setAllowedInputTypes(6, {ALL_FLOATS, ALL_INTS, BOOL})  // valueCache (optional)
      ->setAllowedInputTypes(7, {ALL_INTS})                    // cache_position (optional)
      ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_v2) {
  auto firstInputType = INPUT_VARIABLE(0)->dataType();
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = block.width() > 2  ? INPUT_VARIABLE(2) : values;

  auto dropout = block.numT() > 1 ? block.getTArguments()->at(1) : 0.0;

  // Check for in-place KV cache mode: when cache_position (input 7) is present
  // with keyCache (input 5) and valueCache (input 6), Tv = cache seq dim
  bool hasInPlaceKv = (block.width() > 7);
  auto keyCacheShape = (block.width() > 5 && !INPUT_VARIABLE(5)->isEmpty()) ? INPUT_VARIABLE(5) : nullptr;

  // For rank 4: [batch, seq_len, numHeads, headDim] (BSHD)
  // For rank 3: [batch, seq_len, features]
  // For rank 2: [seq_len, features] - treated as batch=1
  std::vector<sd::LongType> outShape;
  std::vector<sd::LongType> scoresShape;

  if(queries->rankOf() == 4) {
    // Rank 4: [batch, Tq, numHeads, headDim] (BSHD format)
    sd::LongType batchSize = queries->sizeAt(0);
    sd::LongType tq = queries->sizeAt(1);
    sd::LongType numHeads = queries->sizeAt(2);
    sd::LongType headDim = queries->sizeAt(3);
    sd::LongType tv = (hasInPlaceKv && keyCacheShape != nullptr)
                       ? keyCacheShape->sizeAt(1) : values->sizeAt(1);

    // Output shape: [batch, Tq, numHeads, headDim] (same as query)
    outShape = {batchSize, tq, numHeads, headDim};
    // Attention scores shape: [batch, numHeads, Tq, Tv] (per-head scores)
    scoresShape = {batchSize, numHeads, tq, tv};
  } else if(queries->rankOf() == 3) {
    sd::LongType batchSize = queries->sizeAt(0);
    sd::LongType tq = queries->sizeAt(1);
    sd::LongType tv = (hasInPlaceKv && keyCacheShape != nullptr)
                       ? keyCacheShape->sizeAt(1) : values->sizeAt(1);
    sd::LongType dim = values->sizeAt(2);

    outShape = {batchSize, tq, dim};
    scoresShape = {batchSize, tq, tv};
  } else {
    sd::LongType tq = queries->sizeAt(0);
    sd::LongType tv = (hasInPlaceKv && keyCacheShape != nullptr)
                       ? keyCacheShape->sizeAt(0) : values->sizeAt(0);
    sd::LongType dim = values->sizeAt(1);

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

  // Treat empty arrays as no mask
  if(qMaskOrig != nullptr && qMaskOrig->isEmpty()) {
    qMaskOrig = nullptr;
  }
  if(vMaskOrig != nullptr && vMaskOrig->isEmpty()) {
    vMaskOrig = nullptr;
  }

  // Reshape masks if needed
  // For 2D masks [batch, seq], reshape to [batch, 1, seq] to broadcast correctly with attention scores [batch, Tq, Tv]
  if(qMaskOrig != nullptr && qMaskOrig->rankOf() == 2) {
    std::vector<sd::LongType> qmShape = {qMaskOrig->sizeAt(0), 1, qMaskOrig->sizeAt(1)};
    qMask = qMaskOrig->reshape('c', qmShape);
  } else {
    qMask = qMaskOrig;
  }

  if(vMaskOrig != nullptr && vMaskOrig->rankOf() == 2) {
    std::vector<sd::LongType> vmShape = {vMaskOrig->sizeAt(0), 1, vMaskOrig->sizeAt(1)};
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

  // B_ARG order: useCausalMask, training, useFlashAttention (third arg is forward-only)
  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;

  int seed = block.randomSeed();
  AttentionHelper::dotProductAttentionBpHelper(queries, keys, values, scale, dLdq, dLdk, dLdv, eps, seed, qMask, vMask,
                                               useCausalMask, dropout, training, attentionScoresWeights,
                                               attentionScoreLogits, dropoutMask);

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
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_v2_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)), CONSTANT(inputShape->at(2)));
}

}  // namespace ops
}  // namespace sd

#endif
