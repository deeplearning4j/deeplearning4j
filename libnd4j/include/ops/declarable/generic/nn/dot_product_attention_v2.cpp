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

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>
#include <helpers/AttentionHelper.h>
#include <helpers/FlashAttentionHelper.h>

namespace sd {
namespace ops {

/**
 * dot_product_attention_v2 - Dot product attention with flash attention optimization
 *
 * Supports:
 * - Flash attention for 4D inputs [batch, seq, heads, dim] (memory efficient)
 * - Standard attention for 2D/3D inputs [Tq, dim] or [batch, Tq, dim]
 * - KV cache for autoregressive generation (4D only)
 *
 * Inputs:
 *   0: queries [Tq, dim], [batch, Tq, dim], or [batch, Tq, heads, dim]
 *   1: values [Tv, dim], [batch, Tv, dim], or [batch, Tv, heads, dim]
 *   2: keys (optional, defaults to values)
 *   3: queryMask (optional)
 *   4: valueMask (optional)
 *   5: keyCache (optional) - for KV caching [batch, maxSeq, heads, dim]
 *   6: valueCache (optional) - for KV caching [batch, maxSeq, heads, dim]
 *
 * T_ARG:
 *   0: scale (default 1.0)
 *   1: dropout (default 0.0)
 *
 * B_ARG:
 *   0: useCausalMask (default false)
 *   1: training (default false)
 *   2: useFlashAttention (default true when applicable)
 *
 * I_ARG:
 *   0: kvCachePosition (for KV cache, default 0)
 */
CUSTOM_OP_IMPL(dot_product_attention_v2, -2, -1, false, -2, -2) {
  auto queriesOrig = INPUT_VARIABLE(0);
  auto valuesOrig = INPUT_VARIABLE(1);

  REQUIRE_TRUE(queriesOrig->rankOf() == valuesOrig->rankOf(), 0,
               "dot_product_attention_v2: Queries and values must have same rank, got %i vs %i",
               queriesOrig->rankOf(), valuesOrig->rankOf());
  REQUIRE_TRUE(queriesOrig->rankOf() >= 2 && queriesOrig->rankOf() <= 4, 0,
               "dot_product_attention_v2: Input rank must be 2, 3, or 4, got %i", queriesOrig->rankOf());

  // Get arguments
  auto scale = block.numT() > 0 ? T_ARG(0) : 1.0;
  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;
  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;
  auto useFlashAttention = block.numB() > 2 ? B_ARG(2) : true;
  auto kvCachePosition = block.numI() > 0 ? INT_ARG(0) : 0;

  // Check for KV cache inputs
  NDArray* keyCache = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
  NDArray* valueCache = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;
  bool useKVCache = (keyCache != nullptr && valueCache != nullptr &&
                     !keyCache->isEmpty() && !valueCache->isEmpty());

  // Check if we can use flash attention:
  // - 4D inputs [batch, seq, heads, dim]
  // - No dropout during inference (or we don't need intermediate weights)
  bool canUseFlash = useFlashAttention &&
                     queriesOrig->rankOf() == 4 &&
                     (dropout == 0.0 || !training);

  if (canUseFlash) {
    // ========== FLASH ATTENTION PATH (4D inputs) ==========
    auto query = queriesOrig;
    auto keysOrig = block.width() > 2 ? INPUT_VARIABLE(2) : valuesOrig;
    auto key = keysOrig;
    auto value = valuesOrig;

    auto output = OUTPUT_VARIABLE(0);

    // Configure flash attention
    FlashAttentionHelper::Config config;
    config.scale = scale > 0.0 ? static_cast<float>(scale) : 0.0f;
    config.isCausal = useCausalMask;
    config.numHeads = query->sizeAt(2);
    config.numKvHeads = key->sizeAt(2);

    if (useKVCache) {
      // With KV cache: update cache and use cached KV
      auto newSeqLen = key->sizeAt(1);
      auto batch = key->sizeAt(0);
      auto numHeads = key->sizeAt(2);
      auto headDim = key->sizeAt(3);

      // Update cache slice
      auto keyCacheSlice = (*keyCache)({0, batch, kvCachePosition, kvCachePosition + newSeqLen, 0, numHeads, 0, headDim});
      auto valueCacheSlice = (*valueCache)({0, batch, kvCachePosition, kvCachePosition + newSeqLen, 0, numHeads, 0, headDim});
      keyCacheSlice->assign(key);
      valueCacheSlice->assign(value);
      delete keyCacheSlice;
      delete valueCacheSlice;

      FlashAttentionHelper::forward(query, keyCache, valueCache, output, config,
                                    nullptr, block.launchContext());
    } else {
      FlashAttentionHelper::forward(query, key, value, output, config,
                                    nullptr, block.launchContext());
    }

    // Flash attention doesn't compute separate attention scores/logits
    double zeroVal = 0.0;
    if (block.outputWidth() > 1) {
      OUTPUT_VARIABLE(1)->assign(zeroVal);
    }
    if (block.outputWidth() > 2) {
      OUTPUT_VARIABLE(2)->assign(zeroVal);
    }

    return sd::Status::OK;
  }

  // ========== STANDARD ATTENTION PATH (2D/3D inputs) ==========
  auto queries = queriesOrig;
  auto values = valuesOrig;

  // Handle rank 2 inputs by adding batch dimension
  bool reshapedQ = false;
  if (queries->rankOf() == 2) {
    reshapedQ = true;
    std::vector<sd::LongType> qShape = {1, queries->sizeAt(0), queries->sizeAt(-1)};
    std::vector<sd::LongType> vShape = {1, values->sizeAt(0), values->sizeAt(-1)};
    queries = queries->reshape('c', qShape);
    values = values->reshape('c', vShape);
  }

  // Handle keys - defaults to values if not provided
  auto keys = block.width() > 2 ? INPUT_VARIABLE(2) : valuesOrig;
  if (reshapedQ && block.width() > 2) {
    std::vector<sd::LongType> kShape = {1, keys->sizeAt(0), keys->sizeAt(-1)};
    keys = keys->reshape('c', kShape);
  } else if (reshapedQ) {
    keys = values;  // keys defaults to values
  }

  // Handle masks - check for empty arrays
  auto qMask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  auto vMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

  if (qMask != nullptr && qMask->isEmpty()) qMask = nullptr;
  if (vMask != nullptr && vMask->isEmpty()) vMask = nullptr;

  // Reshape masks if needed for rank 2 case
  if (qMask != nullptr && reshapedQ) {
    std::vector<sd::LongType> qmShape = {1, qMask->sizeAt(0), qMask->sizeAt(-1)};
    qMask = qMask->reshape('c', qmShape);
  }

  if (vMask != nullptr && reshapedQ) {
    std::vector<sd::LongType> vmShape = {1, vMask->sizeAt(0), vMask->sizeAt(-1)};
    vMask = vMask->reshape('c', vmShape);
  }

  // Prepare inputs and masks
  std::vector<sd::NDArray*> inputs = {queries, values, keys};
  std::vector<sd::NDArray*> masks = {qMask, vMask};

  // Get output variables
  auto applyScoresOut = OUTPUT_VARIABLE(0);
  auto attentionScores = OUTPUT_VARIABLE(1);
  auto attentionLogits = OUTPUT_VARIABLE(2);
  auto dropoutMask = dropout > 0.0 ? OUTPUT_VARIABLE(3) : nullptr;

  // Reshape outputs for rank 2 case
  if (reshapedQ) {
    applyScoresOut->reshapei('c', {1, applyScoresOut->sizeAt(0), applyScoresOut->sizeAt(1)});
    attentionLogits->reshapei('c', {1, attentionLogits->sizeAt(0), attentionLogits->sizeAt(1)});
    attentionScores->reshapei('c', {1, attentionScores->sizeAt(0), attentionScores->sizeAt(1)});
  }

  // Execute attention
  AttentionHelper::doAttention(inputs, masks, training, useCausalMask, dropout, scale, attentionScores,
                               block.randomSeed(), applyScoresOut, attentionLogits, dropoutMask);

  // Cleanup and restore shapes
  if (reshapedQ) {
    delete queries;
    delete values;
    if (block.width() > 2) {
      delete keys;
    }
    if (qMask != nullptr) delete qMask;
    if (vMask != nullptr) delete vMask;

    applyScoresOut->reshapei('c', {applyScoresOut->sizeAt(1), applyScoresOut->sizeAt(-1)});
    attentionLogits->reshapei('c', {attentionLogits->sizeAt(1), attentionLogits->sizeAt(-1)});
    attentionScores->reshapei('c', {attentionScores->sizeAt(1), attentionScores->sizeAt(-1)});
  }

  return sd::Status::OK;
}

DECLARE_TYPES(dot_product_attention_v2) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_v2) {
  auto firstInputType = INPUT_VARIABLE(0)->dataType();
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = block.width() > 2 ? INPUT_VARIABLE(2) : values;

  auto dropout = block.numT() > 1 ? block.getTArguments()->at(1) : 0.0;

  std::vector<sd::LongType> outShape;
  std::vector<sd::LongType> scoresShape;

  if (queries->rankOf() == 4) {
    // Rank 4: [batch, seq, heads, dim] - flash attention format
    sd::LongType batchSize = queries->sizeAt(0);
    sd::LongType tq = queries->sizeAt(1);
    sd::LongType numHeads = queries->sizeAt(2);
    sd::LongType headDim = queries->sizeAt(3);
    sd::LongType tv = values->sizeAt(1);

    outShape = {batchSize, tq, numHeads, headDim};
    scoresShape = {batchSize, numHeads, tq, tv};
  } else if (queries->rankOf() == 3) {
    // Rank 3: [batch, Tq, dim]
    sd::LongType batchSize = queries->sizeAt(0);
    sd::LongType tq = queries->sizeAt(-2);
    sd::LongType tv = values->sizeAt(-2);
    sd::LongType dim = values->sizeAt(-1);

    outShape = {batchSize, tq, dim};
    scoresShape = {batchSize, tq, tv};
  } else {
    // Rank 2: [Tq, dim]
    sd::LongType batchSize = queries->sizeAt(0);
    sd::LongType tv = values->sizeAt(0);
    sd::LongType dim = values->sizeAt(-1);

    outShape = {batchSize, tv};
    scoresShape = {batchSize, dim};
  }

  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(firstInputType, 'c', outShape);
  auto attentionScoresShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(firstInputType, 'c', scoresShape);
  auto attentionLogitsShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(firstInputType, 'c', scoresShape);

  if (dropout > 0) {
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

  // Get arguments
  auto scale = block.numT() > 0 ? T_ARG(0) : 1.0;
  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;
  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;
  auto useFlashAttention = block.numB() > 2 ? B_ARG(2) : true;

  // Check if we should use flash attention backward
  bool canUseFlash = useFlashAttention &&
                     queriesOrig->rankOf() == 4 &&
                     (dropout == 0.0 || !training);

  if (canUseFlash) {
    // ========== FLASH ATTENTION BACKWARD PATH ==========
    auto attentionScoresOut = INPUT_VARIABLE(3);
    auto eps = INPUT_VARIABLE(6);

    auto dLdq = OUTPUT_VARIABLE(0);
    auto dLdv = OUTPUT_VARIABLE(1);
    auto dLdk = OUTPUT_VARIABLE(2);

    FlashAttentionHelper::Config config;
    config.scale = scale > 0.0 ? static_cast<float>(scale) : 0.0f;
    config.isCausal = useCausalMask;
    config.numHeads = queriesOrig->sizeAt(2);
    config.numKvHeads = keysOrig->sizeAt(2);

    // Compute forward pass to get LSE for backward
    auto batch = queriesOrig->sizeAt(0);
    auto seqLen = queriesOrig->sizeAt(1);
    auto numHeads = queriesOrig->sizeAt(2);

    auto queryShapeVec = queriesOrig->getShapeAsVector();
    auto computedOutput = NDArrayFactory::create_<float>('c', *queryShapeVec);
    delete queryShapeVec;
    std::vector<sd::LongType> lseShape = {batch, numHeads, seqLen};
    auto computedLse = NDArrayFactory::create_<float>('c', lseShape);

    FlashAttentionHelper::forward(queriesOrig, keysOrig, valuesOrig, computedOutput, config,
                                  computedLse, block.launchContext());

    FlashAttentionHelper::backward(eps, queriesOrig, keysOrig, valuesOrig, computedOutput, computedLse,
                                   dLdq, dLdk, dLdv, config, block.launchContext());

    delete computedOutput;
    delete computedLse;

    return sd::Status::OK;
  }

  // ========== STANDARD ATTENTION BACKWARD PATH ==========
  auto queries = queriesOrig;
  auto values = valuesOrig;
  auto keys = keysOrig;

  bool reshapedQ = false;
  if (queries->rankOf() == 2) {
    reshapedQ = true;
    std::vector<sd::LongType> qShape = {1, queries->sizeAt(0), queries->sizeAt(-1)};
    std::vector<sd::LongType> vShape = {1, values->sizeAt(0), values->sizeAt(-1)};
    std::vector<sd::LongType> kShape = {1, keys->sizeAt(0), keys->sizeAt(-1)};
    queries = queries->reshape('c', qShape);
    values = values->reshape('c', vShape);
    keys = keys->reshape('c', kShape);
  }

  auto attentionScoresOut = INPUT_VARIABLE(3);
  auto attentionScoresWeights = INPUT_VARIABLE(4);
  auto attentionScoreLogits = INPUT_VARIABLE(5);

  if (reshapedQ) {
    attentionScoresOut->reshapei('c', {1, attentionScoresOut->sizeAt(0), attentionScoresOut->sizeAt(1)});
    attentionScoreLogits->reshapei('c', {1, attentionScoreLogits->sizeAt(0), attentionScoreLogits->sizeAt(1)});
    attentionScoresWeights->reshapei('c', {1, attentionScoresWeights->sizeAt(0), attentionScoresWeights->sizeAt(1)});
  }

  auto eps = INPUT_VARIABLE(6);
  if (reshapedQ) {
    eps->reshapei('c', {1, eps->sizeAt(0), eps->sizeAt(1)});
  }

  // Handle dropout mask - check for empty array
  auto dropoutMask = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;
  if (dropoutMask != nullptr && dropoutMask->isEmpty()) dropoutMask = nullptr;

  // Handle masks
  auto qMask = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;
  auto vMask = block.width() > 9 ? INPUT_VARIABLE(9) : nullptr;

  if (qMask != nullptr && qMask->isEmpty()) qMask = nullptr;
  if (vMask != nullptr && vMask->isEmpty()) vMask = nullptr;

  if (qMask != nullptr && qMask->rankOf() == 2) {
    std::vector<sd::LongType> qmShape = {1, qMask->sizeAt(0), qMask->sizeAt(-1)};
    qMask = qMask->reshape('c', qmShape);
  }

  if (vMask != nullptr && vMask->rankOf() == 2) {
    std::vector<sd::LongType> vmShape = {1, vMask->sizeAt(0), vMask->sizeAt(-1)};
    vMask = vMask->reshape('c', vmShape);
  }

  auto dLdq = OUTPUT_VARIABLE(0);
  auto dLdv = OUTPUT_VARIABLE(1);
  auto dLdk = OUTPUT_VARIABLE(2);

  if (reshapedQ) {
    dLdq->reshapei('c', {1, dLdq->sizeAt(0), dLdq->sizeAt(1)});
    dLdv->reshapei('c', {1, dLdv->sizeAt(0), dLdv->sizeAt(1)});
    dLdk->reshapei('c', {1, dLdk->sizeAt(0), dLdk->sizeAt(1)});
  }

  int seed = block.randomSeed();
  AttentionHelper::dotProductAttentionBpHelper(queries, keys, values, scale, dLdq, dLdk, dLdv, eps, seed, qMask, vMask,
                                               useCausalMask, dropout, training, attentionScoresWeights,
                                               attentionScoreLogits, dropoutMask);

  // Cleanup and restore shapes
  if (reshapedQ) {
    delete queries;
    delete values;
    delete keys;
    if (qMask != nullptr) delete qMask;
    if (vMask != nullptr) delete vMask;

    dLdq->reshapei('c', {dLdq->sizeAt(1), dLdq->sizeAt(2)});
    dLdv->reshapei('c', {dLdv->sizeAt(1), dLdv->sizeAt(2)});
    dLdk->reshapei('c', {dLdk->sizeAt(1), dLdk->sizeAt(2)});
    eps->reshapei('c', {eps->sizeAt(1), eps->sizeAt(2)});

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
