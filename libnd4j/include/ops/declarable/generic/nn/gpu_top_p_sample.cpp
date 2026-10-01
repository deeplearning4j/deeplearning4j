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
// gpu_top_p_sample - nucleus (top-p) token sampling
//
// Draws one token per row from the smallest set of the most likely tokens that
// holds p of the softmax of the logits scaled by 1 / temperature
// (tokenSampleDraw), and reports the probability of each drawn token under the
// kept distribution. Given the token history, the repetition, frequency and
// presence penalties first rewrite the logits of the sampled position.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_gpu_top_p_sample)
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/sampling_penalties.h>
#include <ops/declarable/helpers/token_sample.h>

#include <vector>

namespace sd {
namespace ops {

// The penalties rewrite the logits of the sampled position (the last one of
// rank-3 logits), so they apply to a copy of that position and the logits stay
// read only. A history of another integer type converts through the central
// assign into the INT64 ids the penalties read.
static NDArray* gpuTopPSamplePenalized(NDArray* logits, NDArray* history, double repetition, double frequency,
                                       double presence, LaunchContext* context) {
  NDArray* position = logits;
  if (logits->rankOf() == 3) {
    const LongType last = logits->sizeAt(1) - 1;
    std::vector<LongType> lastIdx = {0, 0, last, last + 1, 0, 0};
    position = (*logits)(lastIdx, true);
  }
  std::vector<LongType> positionShape(position->shapeOf(), position->shapeOf() + position->rankOf());
  auto* penalized = new NDArray('c', positionShape, logits->dataType(), context);
  penalized->assign(position);
  if (position != logits) delete position;

  NDArray* ids = history;
  const DataType idType = DataTypeUtils::fromT<LongType>();
  if (history->dataType() != idType) {
    std::vector<LongType> historyShape(history->shapeOf(), history->shapeOf() + history->rankOf());
    ids = new NDArray('c', historyShape, idType, context);
    ids->assign(history);
  }
  helpers::applyLogitPenalties(penalized, ids, repetition, frequency, presence, context);
  if (ids != history) MmulHelper::deleteTemporary(ids);
  return penalized;
}

CUSTOM_OP_IMPL(gpu_top_p_sample, 1, 2, false, 0, 0) {
  REQUIRE_TRUE(block.width() <= 3, 0,
               "gpu_top_p_sample: expected the logits, optional uniforms and an optional token history, got %i inputs",
               block.width());
  auto logits = INPUT_VARIABLE(0);
  // Empty uniforms or history inputs stand for absent ones.
  auto uniforms = block.width() > 1 && !INPUT_VARIABLE(1)->isEmpty() ? INPUT_VARIABLE(1) : nullptr;
  auto history = block.width() > 2 && !INPUT_VARIABLE(2)->isEmpty() ? INPUT_VARIABLE(2) : nullptr;
  auto tokens = OUTPUT_VARIABLE(0);
  auto probabilities = OUTPUT_VARIABLE(1);
  const LongType seed = INT_ARG_OR(0, 0);
  const double topP = T_ARG_OR(0, 0.9);
  const double temperature = T_ARG_OR(1, 1.0);
  const double repetition = T_ARG_OR(2, 1.0);
  const double frequency = T_ARG_OR(3, 0.0);
  const double presence = T_ARG_OR(4, 0.0);
  REQUIRE_TRUE(logits->rankOf() >= 1 && logits->rankOf() <= 3, 0,
               "gpu_top_p_sample: logits must have rank 1 to 3, got %i", logits->rankOf());
  if (history != nullptr) {
    const LongType batch = logits->rankOf() == 1 ? 1 : logits->sizeAt(0);
    REQUIRE_TRUE(DataTypeUtils::isZ(history->dataType()), 0,
                 "gpu_top_p_sample: the token history must hold integer ids, got %s",
                 DataTypeUtils::asString(history->dataType()).c_str());
    REQUIRE_TRUE(history->rankOf() <= 1 || (history->rankOf() == 2 && history->sizeAt(0) == batch), 0,
                 "gpu_top_p_sample: the token history must be [seqLen], shared by every row, or [%lld, seqLen], "
                 "got %s",
                 batch, ShapeUtils::shapeAsString(history).c_str());
  }

  NDArray* source = logits;
  const bool penalties = repetition != 1.0 || frequency != 0.0 || presence != 0.0;
  if (history != nullptr && penalties && !logits->isEmpty()) {
    source = gpuTopPSamplePenalized(logits, history, repetition, frequency, presence, block.launchContext());
  }

  // A greedy selection draws nothing, so it takes no entropy.
  const bool greedy = helpers::tokenSampleIsGreedy(temperature, 0, topP);
  helpers::tokenSampleDraw(source, tokens, probabilities, uniforms,
                           greedy ? graph::RandomGenerator(1, 1) : helpers::tokenSampleGenerator(seed), temperature, 0,
                           topP, block.launchContext());
  if (source != logits) MmulHelper::deleteTemporary(source);
  return Status::OK;
}

DECLARE_TYPES(gpu_top_p_sample) {
  // Without a positive seed the draws take fresh entropy.
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_INTS})
      ->setAllowedOutputTypes(0, {INT64})
      ->setAllowedOutputTypes(1, {ALL_FLOATS})
      ->setShapeValueInputs({})
      ->addTraits(OP_TRAIT_FULLY_WRITING | OP_TRAIT_STATEFUL);
}

DECLARE_SHAPE_FN(gpu_top_p_sample) {
  auto logits = inputShape->at(0);
  const int rank = shape::rank(logits);
  REQUIRE_TRUE(rank >= 1 && rank <= 3, 0, "gpu_top_p_sample: logits must have rank 1 to 3, got %i", rank);
  const DataType dtype = ArrayOptions::dataType(logits);
  // One token and its probability per row: [vocab] -> scalars, [batch, ..., vocab] -> [batch].
  if (rank == 1) {
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(INT64),
                     ConstantShapeHelper::getInstance().scalarShapeInfo(dtype));
  }
  // The shape-vector form flags a batch of zero rows as empty.
  const std::vector<LongType> batch = {shape::sizeAt(logits, static_cast<LongType>(0))};
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(INT64, 'c', batch),
                   ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', batch));
}

}  // namespace ops
}  // namespace sd
#endif
