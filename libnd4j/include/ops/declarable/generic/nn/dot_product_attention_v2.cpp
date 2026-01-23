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

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dot_product_attention_v2, -2, -1, false, -2, -2) {
  auto queriesOrig = INPUT_VARIABLE(0);
  auto valuesOrig = INPUT_VARIABLE(1);

  REQUIRE_TRUE(queriesOrig->rankOf() == valuesOrig->rankOf(), 0,
               "dot_product_attention_v2: Queries and values must have same rank, got %i vs %i",
               queriesOrig->rankOf(), valuesOrig->rankOf());
  REQUIRE_TRUE(queriesOrig->rankOf() >= 2 && queriesOrig->rankOf() <= 3, 0,
               "dot_product_attention_v2: Input rank must be 2 or 3, got %i", queriesOrig->rankOf());

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

  // Get arguments - T_ARG order: scale, dropout
  auto scale = block.numT() > 0 ? T_ARG(0) : 1.0;
  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

  // B_ARG order: useCausalMask, training
  auto useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
  auto training = block.numB() > 1 ? B_ARG(1) : false;

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

  // Use FlashAttentionHelper when no custom masks or dropout (faster path)
  // FlashAttentionHelper only supports causal masking, not custom masks
  bool canUseFlash = (qMask == nullptr || qMask->isEmpty()) &&
                     (vMask == nullptr || vMask->isEmpty()) &&
                     dropout == 0.0;

  if (canUseFlash) {
    // Setup FlashAttentionHelper config
    FlashAttentionHelper::Config config;
    config.scale = static_cast<float>(scale);
    config.isCausal = useCausalMask;
    config.dropout = 0.0f;
    config.numHeads = 1;
    config.numKvHeads = 1;

    // Call FlashAttentionHelper::forward() directly with 3D inputs
    // This avoids unnecessary 4D reshape + permute operations
    // forward3D is called when inputs have rank 3
    FlashAttentionHelper::forward(queries, keys, values, applyScoresOut, config,
                                  nullptr, attentionScores, attentionLogits,
                                  block.launchContext());
  } else {
    // Fallback to AttentionHelper for masks/dropout support
    std::vector<sd::NDArray*> inputs = {queries, values, keys};
    std::vector<sd::NDArray*> masks = {qMask, vMask};
    AttentionHelper::doAttention(inputs, masks, training, useCausalMask, dropout, scale, attentionScores,
                                 block.randomSeed(), applyScoresOut, attentionLogits, dropoutMask);
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
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dot_product_attention_v2) {
  auto firstInputType = INPUT_VARIABLE(0)->dataType();
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = block.width() > 2  ? INPUT_VARIABLE(2) : values;

  auto dropout = block.numT() > 1 ? block.getTArguments()->at(1) : 0.0;

  // For rank 3: [batch, seq_len, features]
  // For rank 2: [seq_len, features] - treated as batch=1
  std::vector<sd::LongType> outShape;
  std::vector<sd::LongType> scoresShape;

  if(queries->rankOf() == 3) {
    // Rank 3: [batch, Tq, dim] @ [batch, Tv, dim]^T -> [batch, Tq, Tv] -> softmax -> @ [batch, Tv, dim] -> [batch, Tq, dim]
    sd::LongType batchSize = queries->sizeAt(0);
    sd::LongType tq = queries->sizeAt(1);  // query sequence length
    sd::LongType tv = values->sizeAt(1);   // value sequence length
    sd::LongType dim = values->sizeAt(2);  // feature dimension

    // Output shape: [batch, Tq, dim]
    outShape = {batchSize, tq, dim};
    // Attention scores shape: [batch, Tq, Tv]
    scoresShape = {batchSize, tq, tv};
  } else {
    // Rank 2: [Tq, dim] @ [Tv, dim]^T -> [Tq, Tv] -> softmax -> @ [Tv, dim] -> [Tq, dim]
    sd::LongType tq = queries->sizeAt(0);  // query sequence length
    sd::LongType tv = values->sizeAt(0);   // value sequence length
    sd::LongType dim = values->sizeAt(1);  // feature dimension

    // Output shape: [Tq, dim]
    outShape = {tq, dim};
    // Attention scores shape: [Tq, Tv]
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

  // B_ARG order: useCausalMask, training (same as forward pass)
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
