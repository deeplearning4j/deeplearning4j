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

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_per_layer_embedding)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/per_layer_embedding.h>

namespace sd {
namespace ops {

/**
 * per_layer_embedding: output = hidden_states + ple_weight[token_ids] * scale
 *
 * Input 0: hidden_states [batch, seq_len, hidden_dim]
 * Input 1: ple_weight    [vocab_size, hidden_dim]
 * Input 2: token_ids     [batch, seq_len] (INT32 or INT64)
 *
 * T_ARG(0): scale (default 1.0)
 *
 * Output 0: [batch, seq_len, hidden_dim]
 */
CUSTOM_OP_IMPL(per_layer_embedding, 3, 1, false, 0, 0) {
  auto hiddenStates = INPUT_VARIABLE(0);
  auto pleWeight = INPUT_VARIABLE(1);
  auto tokenIds = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  double scale = block.numT() > 0 ? T_ARG(0) : 1.0;

  REQUIRE_TRUE(hiddenStates->rankOf() == 3, 0,
               "per_layer_embedding: hidden_states must be rank 3 [batch, seq_len, hidden_dim], got rank %d",
               hiddenStates->rankOf());
  REQUIRE_TRUE(pleWeight->rankOf() == 2, 0,
               "per_layer_embedding: ple_weight must be rank 2 [vocab_size, hidden_dim], got rank %d",
               pleWeight->rankOf());
  REQUIRE_TRUE(tokenIds->rankOf() == 2, 0,
               "per_layer_embedding: token_ids must be rank 2 [batch, seq_len], got rank %d",
               tokenIds->rankOf());
  REQUIRE_TRUE(hiddenStates->sizeAt(0) == tokenIds->sizeAt(0), 0,
               "per_layer_embedding: batch size mismatch between hidden_states (%lld) and token_ids (%lld)",
               (long long)hiddenStates->sizeAt(0), (long long)tokenIds->sizeAt(0));
  REQUIRE_TRUE(hiddenStates->sizeAt(1) == tokenIds->sizeAt(1), 0,
               "per_layer_embedding: seq_len mismatch between hidden_states (%lld) and token_ids (%lld)",
               (long long)hiddenStates->sizeAt(1), (long long)tokenIds->sizeAt(1));
  REQUIRE_TRUE(hiddenStates->sizeAt(2) == pleWeight->sizeAt(1), 0,
               "per_layer_embedding: hidden_dim mismatch between hidden_states (%lld) and ple_weight (%lld)",
               (long long)hiddenStates->sizeAt(2), (long long)pleWeight->sizeAt(1));

  helpers::perLayerEmbedding(block.launchContext(), hiddenStates, pleWeight, tokenIds, output, scale);

  return Status::OK;
}

DECLARE_TYPES(per_layer_embedding) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_INTS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_VALUE_DEPENDENT_SHAPE);
}

DECLARE_SHAPE_FN(per_layer_embedding) {
  auto hiddenShapeInfo = inputShape->at(0);
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(hiddenShapeInfo),
      shape::order(hiddenShapeInfo),
      shape::rank(hiddenShapeInfo),
      shape::shapeOf(hiddenShapeInfo)));
}

}  // namespace ops
}  // namespace sd

#endif
