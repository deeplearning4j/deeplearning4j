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
// gpu_top_k_sample - top-k token sampling
//
// Draws one token per row from the softmax of the logits scaled by
// 1 / temperature, truncated to the k most likely tokens (tokenSampleDraw), and
// reports the probability of each drawn token under the kept distribution. The
// draws come from the given uniforms, else from the generator of the seed.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_gpu_top_k_sample)
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/token_sample.h>

#include <vector>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(gpu_top_k_sample, 1, 2, false, 0, 0) {
  REQUIRE_TRUE(block.width() <= 2, 0, "gpu_top_k_sample: expected the logits and optional uniforms, got %i inputs",
               block.width());
  auto logits = INPUT_VARIABLE(0);
  // An empty uniforms input stands for an absent one.
  auto uniforms = block.width() > 1 && !INPUT_VARIABLE(1)->isEmpty() ? INPUT_VARIABLE(1) : nullptr;
  auto tokens = OUTPUT_VARIABLE(0);
  auto probabilities = OUTPUT_VARIABLE(1);
  const int topK = static_cast<int>(INT_ARG_OR(0, 50));
  const LongType seed = INT_ARG_OR(1, 0);
  const double temperature = T_ARG_OR(0, 1.0);

  // A greedy selection draws nothing, so it takes no entropy.
  const bool greedy = helpers::tokenSampleIsGreedy(temperature, topK, 0.0);
  helpers::tokenSampleDraw(logits, tokens, probabilities, uniforms,
                           greedy ? graph::RandomGenerator(1, 1) : helpers::tokenSampleGenerator(seed), temperature,
                           topK, 0.0, block.launchContext());
  return Status::OK;
}

DECLARE_TYPES(gpu_top_k_sample) {
  // Without a positive seed the draws take fresh entropy.
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedOutputTypes(0, {INT64})
      ->setAllowedOutputTypes(1, {ALL_FLOATS})
      ->setShapeValueInputs({})
      ->addTraits(OP_TRAIT_FULLY_WRITING | OP_TRAIT_STATEFUL);
}

DECLARE_SHAPE_FN(gpu_top_k_sample) {
  auto logits = inputShape->at(0);
  const int rank = shape::rank(logits);
  REQUIRE_TRUE(rank >= 1 && rank <= 3, 0, "gpu_top_k_sample: logits must have rank 1 to 3, got %i", rank);
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
